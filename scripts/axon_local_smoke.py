"""Small CPU smoke test for AXON's standalone NumPy Transformer."""
from __future__ import annotations

import importlib.util
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

        pieces = []
        answer = core.transformer_generate(
            db, "hi", length=8, temperature=0.72, top_k=32, top_p=0.95,
            on_piece=pieces.append,
        )
        assert isinstance(answer, str)
        assert pieces, "AXON did not stream any generated tokens"

CHAT = (ROOT / "AXON_Local" / "CHAT.py").read_text(encoding="utf-8")
MANUAL = (ROOT / "AXON_Local" / "MANUAL Chat.py").read_text(encoding="utf-8")
assert "retrieve_learned_reply" not in CHAT
assert "learned_greeting_examples" not in CHAT
assert "transformer_generate" in CHAT
assert "Teacher(" not in MANUAL
print("PASS: BPE, decreasing neural training loss, autoregressive streaming and AXON-only chat.")
