"""AXON manual teacher mode.

Put MANUAL Chat.py beside LLM Train.py and BRAIN.db.
You type as the human; ONE local pretrained Qwen model responds.
The messages are saved as training examples in the existing SQLite brain.
AXON's separate neural student receives real next-token gradient updates.
Python 3.11+; uses the existing AXON trainer dependencies.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "LLM Train.py"
if not TRAINER.is_file():
    raise SystemExit("Put MANUAL Chat.py and LLM Train.py in the same AXON folder.")

spec = importlib.util.spec_from_file_location("axon_trainer", TRAINER)
axon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = axon
spec.loader.exec_module(axon)

SYSTEM = (
    "You are a friendly, knowledgeable local AI having a conversation with a real human. "
    "Reply directly to the human's latest message. For a greeting, greet naturally. "
    "For a question, give a clear and accurate answer with a specific example if useful. "
    "Do not give the same generic help offer after every reply. "
    "Do not imitate a human speaking to an assistant. "
    "Use natural wording; vary sentence structures appropriately."
)


def statistics(db):
    observations = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    answers = db.execute(
        "SELECT COUNT(*) FROM lessons WHERE origin='dialogue_quality_v5'"
    ).fetchone()[0]
    steps = axon.get_meta(db, "student_steps", "0")
    db_size = axon.DB_PATH.stat().st_size / (1024 * 1024)
    print(f"AXON: {answers:,} teacher Q&A lessons, {observations:,} saved notes, "
          f"{steps} neural training batches; BRAIN.db {db_size:.1f}/15360 MiB")


def teach_student(db, np, pairs, steps):
    """Train from recently observed replies; the pretrained AI never chats for AXON."""
    if steps <= 0 or not pairs:
        return
    def examples():
        while True:
            for q, a in pairs:
                yield q, a
    print(f"[AXON learning] {len(pairs)} recent examples; {steps} gradient batches.",
          flush=True)
    axon.train_student(
        db, np, examples(), max_steps=steps,
        print_every=max(10, steps // 4)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Manual human-to-Qwen teacher chat that trains AXON."
    )
    parser.add_argument(
        "--steps", type=int, default=40,
        help="student gradient batches after each teacher reply (0 means save only)"
    )
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be 0 or more")
    if args.max_tokens < 10:
        parser.error("--max-tokens must be at least 10")

    np = axon.ensure_numpy()
    session = "manual-" + uuid.uuid4().hex[:12]
    with axon.connection() as db:
        axon.bootstrap_lessons(db)
        model = axon.ensure_teacher(db)
        history = []
        recent = []
        turn = 0
        print("AXON MANUAL CHAT: YOU + one pretrained local assistant.")
        print("Each teacher answer is saved and trains your own AXON student.")
        print("Commands: :stats   :train 300   :axon your question   :quit")
        with axon.Teacher(model, parallel=1, role="Manual teacher") as teacher:
            while True:
                try:
                    user_text = input("\nYou > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped. Previously saved notes and weights are preserved.")
                    break
                if not user_text:
                    continue
                cmd = user_text.lower()
                if cmd in (":quit", ":exit", ":q"):
                    break
                if cmd == ":stats":
                    statistics(db)
                    continue
                if cmd.startswith(":axon "):
                    question = user_text[len(":axon "):].strip()
                    print("AXON student > ", end="", flush=True)
                    try:
                        axon.student_generate(
                            db, question,
                            on_piece=lambda part: print(part, end="", flush=True)
                        )
                    except Exception as exc:
                        print("AXON student error:", exc)
                    print()
                    continue
                if cmd.startswith(":train"):
                    try:
                        steps = int(user_text.split()[1])
                    except (IndexError, ValueError):
                        steps = 300
                    if steps < 1:
                        print("Use :train 300 (positive number required).")
                        continue
                    examples = list(recent[-25:])
                    if not examples:
                        examples = db.execute(
                            "SELECT prompt,answer FROM lessons "
                            "WHERE origin='dialogue_quality_v5' "
                            "ORDER BY id DESC LIMIT 100"
                        ).fetchall()
                    if not examples:
                        print("No real conversation examples saved yet.")
                    else:
                        teach_student(db, np, examples, steps)
                    continue
                if user_text.startswith(":"):
                    print("Unknown command. Use :stats, :train 300, :axon ... or :quit")
                    continue

                messages = [{"role": "system", "content": SYSTEM}]
                for human, assistant in history[-6:]:
                    messages.extend((
                        {"role": "user", "content": human},
                        {"role": "assistant", "content": assistant},
                    ))
                messages.append({"role": "user", "content": user_text[:1500]})
                print("Pretrained AI > ", end="", flush=True)
                try:
                    response = teacher.complete(
                        messages, max_tokens=args.max_tokens,
                        temperature=0.70, stream=True
                    )
                except Exception as exc:
                    print(f"\nThe teacher failed: {exc}. This turn was not saved.")
                    continue
                response = " ".join(response.split()).strip()
                if not response or any(
                    marker in response
                    for marker in ("<|im_start|>", "<|im_end|>", "\ufffd")
                ):
                    print("Empty or corrupted reply; example not saved.")
                    continue
                try:
                    # Save BOTH speakers and create the training Q&A using the
                    # existing database schema and tokenizer.
                    axon.note_exchange(
                        db, session, turn * 2 + 1, "Human", "Assistant", "", user_text
                    )
                    axon.note_exchange(
                        db, session, turn * 2 + 2,
                        "Assistant", "Human", user_text, response
                    )
                    history.append((user_text, response))
                    recent.append((user_text, response))
                    turn += 1
                    if args.steps:
                        teach_student(db, np, recent[-12:], args.steps)
                    print("AXON recorded this exchange and updated its weights.")
                except KeyboardInterrupt:
                    print("\nInterrupted; previously saved data remains.")
                    break
                except Exception as exc:
                    print(f"Learning error: {exc}. Check BRAIN.db before retrying.")
        statistics(db)


if __name__ == "__main__":
    main()
