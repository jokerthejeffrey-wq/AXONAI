"""Small CPU smoke test for AXON's standalone NumPy Transformer."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import inspect
import runpy
import time
import itertools
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "AXON_Local" / "LLM Train.py"
spec = importlib.util.spec_from_file_location("axon_smoke_trainer", TRAINER)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load AXON trainer")
core = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = core
spec.loader.exec_module(core)

with tempfile.TemporaryDirectory(prefix="axon-smoke-") as folder:
    core.DB_PATH = Path(folder) / "BRAIN.db"
    core.DATA_PATH = Path(folder) / "training_data.txt"
    with core.connection() as db:
        # SQLite-level enforcement is stronger than checking file size in Python.
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        page_cap = db.execute("PRAGMA max_page_count").fetchone()[0]
        assert core.MAX_DB_BYTES == 5_000_000_000
        assert page_cap == core.MAX_DB_BYTES // page_size
        try:
            core.enforce_db_size(db, core.MAX_DB_BYTES)
        except RuntimeError as exc:
            assert "5 GB" in str(exc)
        else:
            raise AssertionError("5 GB database limit is not enforced")
        print(f"SQLite BRAIN.db cap: {page_cap * page_size / 1_000_000_000:.3f} GB")
        governor = core.AdaptiveResources()
        assert governor.workers <= 4
        assert governor.choose_backend("cpu") is None
        print("Resource governor:", governor.describe())

        # Explicit maximum-performance mode should not sleep just because
        # CPU utilization is high. It still keeps a low-memory safety pause.
        max_governor = core.AdaptiveResources("max")
        assert max_governor.workers >= governor.workers
        assert max_governor.choose_backend("cpu") is None
        max_governor.cpu_percent = 99.0
        max_governor.ram_available = 4 * 1024**3
        max_governor.sample = lambda force=False: None
        start = time.monotonic()
        max_governor.throttle()
        assert time.monotonic() - start < 0.3, "Max mode unexpectedly throttled"
        assert core.parse_training_steps("inf") is None
        assert core.parse_training_steps("infinite") is None
        assert core.parse_training_steps("99999") == 99999
        print("Unlimited steps and maximum-use resource profile: PASS")
        core.bootstrap_lessons(db)
        vocab = core.build_vocabulary(db)
        assert len(vocab) > len(core.SPECIAL)
        tokens = core.tokenizer("An unseen Turkish word: nasılsın?")
        assert tokens and all(token in set(vocab) for token in tokens)

        # Verify that the actual neural optimizer reduces next-token error,
        # not merely that it generates random tokens without crashing.
        examples = [("hi", "hello there"), ("how are you", "i am doing well")]
        lookup = {token: i for i, token in enumerate(vocab)}
        batch = core._trf_make_batch(np, examples, lookup)
        params, _, _, _ = core.load_transformer(db, np, vocab)
        params["_active_vocab"] = len(vocab)
        before, _ = core._trf_loss(np, params, *batch, backward=False)
        trained = core.train_student(
            db, np, itertools.cycle(examples), max_steps=8, print_every=8
        )
        params, _, _, _ = core.load_transformer(db, np, vocab)
        params["_active_vocab"] = len(vocab)
        after, _ = core._trf_loss(np, params, *batch, backward=False)
        print(f"Next-token test loss: {before:.3f} -> {after:.3f}")
        assert trained == 8
        assert after < before, "AXON optimizer failed to reduce next-token error"
        assert int(core.get_meta(db, "transformer_steps", 0)) >= 8

        # Confirm unbounded training passes a None step cap through to the
        # real Transformer loop without actually running indefinitely in CI.
        original_train_student = core.train_student
        sent = {}
        core.DATA_PATH.write_text(
            "unique corpus question\\tunique corpus answer\\n".replace(
                "\\\\t", "\\t"
            ).replace("\\\\n", "\\n"),
            encoding="utf-8"
        )
        def record_infinite_training(*args, **kwargs):
            sent.update(kwargs)
            sent["sample"] = list(itertools.islice(args[2], 30))
            return 1
        core.train_student = record_infinite_training
        try:
            assert core.learn_observations(
                db, np, steps=None, device="cpu",
                resource_profile="max", include_corpus=True
            ) == 1
        finally:
            core.train_student = original_train_student
        assert sent.get("max_steps") is None
        assert sent.get("resource_profile") == "max"
        assert any("unique corpus question" in q for q, a in sent["sample"])
        print("Infinite training and full-corpus streaming: PASS")

        # Exercise the alternative accelerator code path without requiring a
        # CUDA GPU on GitHub's normal hosted CI runner.
        class SimulatedArrayBackend:
            def __getattr__(self, name):
                return getattr(np, name)

        original_choose = core.AdaptiveResources.choose_backend
        core.AdaptiveResources.choose_backend = (
            lambda self, device="auto": SimulatedArrayBackend()
        )
        try:
            simulated = core.train_student(
                db, np, itertools.cycle(examples), max_steps=2,
                print_every=2, device="gpu"
            )
        finally:
            core.AdaptiveResources.choose_backend = original_choose
        assert simulated == 2, "Alternative array backend path failed"
        print("Simulated GPU-backend training, optimizer and checkpoint: PASS")

        pieces = []
        original_forward = core.transformer_forward
        forward_calls = [0]
        def count_forward(*args, **kwargs):
            forward_calls[0] += 1
            return original_forward(*args, **kwargs)
        core.transformer_forward = count_forward
        try:
            answer = core.transformer_generate(
                db, "hi", length=8, temperature=0.72, top_k=32, top_p=0.95,
                on_piece=pieces.append,
            )
        finally:
            core.transformer_forward = original_forward
        assert isinstance(answer, str)
        assert pieces, "AXON did not stream any generated tokens"
        assert forward_calls[0] >= 2, "Speculative neural lookahead was not used"
        print(f"AXON neural forward passes for token generation: {forward_calls[0]}")

# Verify minimum 0.5 seconds per complete word, not per BPE fragment.
chat_module = runpy.run_path(str(ROOT / "AXON_Local" / "CHAT.py"), run_name="axon_pacer_test")
pacer = chat_module["WordPacedPrinter"](minimum_seconds=0.5)
output = io.StringIO()
with contextlib.redirect_stdout(output):
    start = time.monotonic()
    pacer.on_piece("hel")
    pacer.on_piece("lo ")
    first = time.monotonic()
    pacer.on_piece("wor")
    pacer.on_piece("ld")
    pacer.finish()
    second = time.monotonic()
assert first - start >= 0.48, "First word appeared before 0.5 seconds"
assert second - first >= 0.48, "Next word appeared before 0.5 seconds"
assert output.getvalue() == "hello world", "BPE fragments failed to assemble into whole words"
print(f"Word intervals: {first - start:.3f}s, {second - first:.3f}s")

CHAT = (ROOT / "AXON_Local" / "CHAT.py").read_text(encoding="utf-8")
MANUAL = (ROOT / "AXON_Local" / "MANUAL Chat.py").read_text(encoding="utf-8")
assert "retrieve_learned_reply" not in CHAT
assert "learned_greeting_examples" not in CHAT
assert "transformer_generate" in CHAT
assert "Teacher(" not in MANUAL
assert "reviewer_model" in inspect.signature(core._chat_worker).parameters
assert "reviewer_model" in inspect.signature(core.agent_dialogues_parallel).parameters
print("PASS: max profile, infinite training, 5 GB cap, CPU/GPU paths, decreasing loss, lookahead and pacing.")
