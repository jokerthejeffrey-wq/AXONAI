"""AXON causal Transformer chat.

Normal chat runs AXON's own NumPy decoder-only Transformer. It does not call
Qwen. Training uses causal multi-head self-attention, positional embeddings,
masked full-softmax cross entropy, Adam, and held-out validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "LLM Train.py"
if not TRAINER.is_file():
    raise SystemExit('Place "LLM Chat.py" beside "LLM Train.py" and "BRAIN.db".')

spec = importlib.util.spec_from_file_location("axon_train", TRAINER)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def show_stats(db, vocab):
    observations = db.execute(
        "SELECT COUNT(*) FROM observations WHERE speaker IN ('Human','Assistant')"
    ).fetchone()[0]
    dialogue_pairs = db.execute(
        "SELECT COUNT(*) FROM lessons WHERE origin='dialogue_quality_v5'"
    ).fetchone()[0]
    step = core.get_meta(db, "transformer_steps", "0")
    validation = core.get_meta(db, "transformer_validation_loss", "not measured")
    print("Architecture: NumPy causal decoder-only Transformer")
    print(f"Parameters: {core.PARAMETERS:,}")
    print(
        f"Layers: {core.TRANSFORMER_LAYERS} | heads: {core.TRANSFORMER_HEADS} | "
        f"dimension: {core.TRANSFORMER_DIM} | context: {core.TRANSFORMER_CONTEXT}"
    )
    print(f"Training batches: {step} | held-out validation loss: {validation}")
    print(f"Vocabulary: {len(vocab):,}/{core.VOCAB_SIZE:,}")
    print(f"Saved messages: {observations:,} | teacher Q&A pairs: {dialogue_pairs:,}")
    print(f"BRAIN.db: {core.DB_PATH.stat().st_size / 1024**2:.1f} MiB / 15360 MiB")
    print("Normal chat uses AXON weights only; Qwen is used only by :observe.")


def main():
    parser = argparse.ArgumentParser(description="Chat with AXON's causal Transformer student")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--length", type=int, default=80)
    args = parser.parse_args()

    if not 0.15 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0.15 and 2.0")
    if not 2 <= args.top_k <= 256:
        parser.error("--top-k must be between 2 and 256")
    if not 8 <= args.length <= 200:
        parser.error("--length must be between 8 and 200")

    np = core.ensure_numpy()
    print("AXON AI | causal Transformer | word-by-word generation")
    print(
        f"{core.TRANSFORMER_LAYERS} layers · {core.TRANSFORMER_HEADS} attention heads · "
        f"{core.TRANSFORMER_DIM}-dimension · {core.TRANSFORMER_CONTEXT}-token context"
    )
    print("Normal chat uses AXON's own weights. Type :help for commands.")

    with core.connection() as db:
        core.bootstrap_lessons(db)
        vocab = core.build_vocabulary(db)
        temperature = args.temperature
        top_k = args.top_k
        length = args.length
        history = []

        trained_steps = int(core.get_meta(db, "transformer_steps", 0))
        if trained_steps:
            print(f"Loaded AXON Transformer checkpoint ({trained_steps:,} training batches).")
        else:
            print('No trained Transformer checkpoint yet. Run "LLM Train.py" first.')

        while True:
            try:
                prompt = input("\nYou > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Checkpoint remains saved.")
                break
            if not prompt:
                continue
            lower = prompt.casefold()

            if lower in (":exit", ":quit", "exit", "quit"):
                break
            if lower == ":help":
                print("Commands:")
                print("  :train 200               Train on saved conversations")
                print("  :observe 4 8 [topic]     Generate and learn from local teacher chats")
                print("  :teach question => answer  Add your own labeled example")
                print("  :notes 10                Show recent human/assistant notes")
                print("  :temperature 0.75        Set sampling temperature (0.15-2.0)")
                print("  :topk 32                 Set candidate-token count (2-256)")
                print("  :length 80               Set max generated tokens (8-200)")
                print("  :stats                   Show architecture, steps, validation loss")
                print("  :exit                    Quit")
                continue

            if lower.startswith(":temperature "):
                try:
                    value = float(prompt.split()[1])
                    if not 0.15 <= value <= 2.0:
                        raise ValueError()
                    temperature = value
                    print(f"Temperature: {temperature:.2f}")
                except (ValueError, IndexError):
                    print("Use :temperature 0.15 through :temperature 2.0.")
                continue

            if lower.startswith(":topk "):
                try:
                    value = int(prompt.split()[1])
                    if not 2 <= value <= 256:
                        raise ValueError()
                    top_k = value
                    print(f"Top-k candidates: {top_k}")
                except (ValueError, IndexError):
                    print("Use :topk 2 through :topk 256.")
                continue

            if lower.startswith(":length "):
                try:
                    value = int(prompt.split()[1])
                    if not 8 <= value <= 200:
                        raise ValueError()
                    length = value
                    print(f"Maximum generated tokens: {length}")
                except (ValueError, IndexError):
                    print("Use :length 8 through :length 200.")
                continue

            if lower == ":stats":
                vocab = core.build_vocabulary(db)
                show_stats(db, vocab)
                continue

            if lower.startswith(":notes"):
                try:
                    limit = int(prompt.split()[1])
                except (ValueError, IndexError):
                    limit = 10
                for speaker, receiver, message in core.recent_notes(db, limit):
                    print(f"{speaker} -> {receiver}: {message}")
                continue

            if lower.startswith(":teach "):
                pair = prompt[7:].split("=>", 1)
                if len(pair) != 2 or not pair[0].strip() or not pair[1].strip():
                    print("Use :teach What is 1 + 1? => 1 + 1 = 2.")
                    continue
                question, answer = (part.strip() for part in pair)
                core.store_lesson(db, question, answer, "user")
                core.observe_word_patterns(db, answer, prefix=question)
                core.extend_vocabulary(db, question + " " + answer)
                db.commit()
                print("Training Transformer on your example...")
                try:
                    core.train_student(
                        db, np, itertools.repeat((question, answer), 100),
                        max_steps=30, print_every=10
                    )
                    vocab = core.build_vocabulary(db)
                    print("Example trained and checkpoint saved.")
                except KeyboardInterrupt:
                    print("\nTraining interrupted; checkpoint was saved.")
                continue

            if lower.startswith(":train"):
                try:
                    steps = int(prompt.split()[1])
                except (ValueError, IndexError):
                    steps = 200
                if not 1 <= steps <= 100000:
                    print("Choose 1 through 100,000 training batches.")
                    continue
                try:
                    core.learn_observations(db, np, steps=steps)
                    vocab = core.build_vocabulary(db)
                except KeyboardInterrupt:
                    print("\nTraining interrupted; last checkpoint was saved.")
                continue

            if lower.startswith(":observe"):
                bits = prompt.split()
                try:
                    chats = int(bits[1]) if len(bits) > 1 else 4
                    turns = int(bits[2]) if len(bits) > 2 else 8
                except ValueError:
                    print("Usage: :observe 4 8 [optional topic]")
                    continue
                if not 1 <= chats <= 12 or not 2 <= turns <= 200:
                    print("Use 1-12 chats and 2-200 messages per chat.")
                    continue
                topic = " ".join(bits[3:]) or None
                try:
                    model = core.ensure_teacher(db)
                    with core.Teacher(model, parallel=chats, role="Human") as human:
                        with core.Teacher(model, parallel=chats, role="Assistant") as assistant:
                            sessions, pairs = core.agent_dialogues_parallel(
                                db, human, assistant, chats=chats,
                                turns=turns, topic=topic
                            )
                    print(f"Saved {pairs} useful Q&A pairs across {sessions} chats.")
                    if pairs:
                        core.learn_observations(db, np, steps=max(40, pairs * 8))
                        vocab = core.build_vocabulary(db)
                except KeyboardInterrupt:
                    print("\nObservation interrupted; saved exchanges remain.")
                except Exception as exc:
                    print(f"Observation failed: {exc}")
                continue

            if prompt.startswith(":"):
                print("Unknown command. Type :help.")
                continue

            if int(core.get_meta(db, "transformer_steps", 0)) <= 0:
                print('AXON has not been trained yet. Run "LLM Train.py" first.')
                continue

            print("AXON > ", end="", flush=True)
            try:
                answer = core.transformer_generate(
                    db, prompt, length=length, temperature=temperature,
                    top_k=top_k, top_p=0.92,
                    on_piece=lambda piece: print(piece, end="", flush=True),
                    history=history
                )
                if not answer.strip():
                    print("[No tokens generated. Train on more examples.]")
                else:
                    print()
                    history.append((prompt, answer))
                    history = history[-3:]
            except Exception as exc:
                print(f"\nAXON generation error: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
