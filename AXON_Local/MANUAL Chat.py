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
    "You are a friendly, knowledgeable assistant talking with a real person. "
    "Read the previous conversation turns before responding. "
    "React naturally to what was just said and to the earlier dialogue. "
    "If the user greets you again, acknowledge the repeat casually, for example "
    "'Hey again!' or 'Hi again!', rather than pretending this is the first greeting. "
    "Do NOT repeat the same greeting or generic offer to help in every answer. "
    "If the user refers to a detail mentioned earlier, use it accurately when visible "
    "in the supplied history. If it is not visible, say you do not recall. "
    "Answer questions clearly, use natural variety, and do not invent memories."
)


def get_manual_session(db, fresh=False):
    """Keep one manual conversation across restarts until :new is requested."""
    session = None if fresh else axon.get_meta(db, "manual_chat_active_session_v1")
    if not session and not fresh:
        row = db.execute(
            "SELECT session FROM observations "
            "WHERE session LIKE 'manual-%' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        session = row[0] if row else None
    if not session:
        session = "manual-" + uuid.uuid4().hex[:12]
    axon.set_meta(db, "manual_chat_active_session_v1", session)
    db.commit()
    return session


def load_history(db, session, limit=40):
    """Reload actual words from BRAIN.db, not a paraphrase of the conversation."""
    rows = db.execute(
        "SELECT speaker, message FROM ("
        " SELECT id, speaker, message FROM observations "
        " WHERE session=? AND speaker IN ('Human','Assistant') "
        " ORDER BY id DESC LIMIT ?"
        ") ORDER BY id ASC",
        (session, max(2, limit * 2)),
    ).fetchall()
    pairs = []
    current_human = None
    for speaker, message in rows:
        if speaker == "Human":
            current_human = message
        elif speaker == "Assistant" and current_human is not None:
            pairs.append((current_human, message))
            current_human = None
    return pairs


def conversation_messages(history, user_text):
    """Use recent turns within a bounded context to avoid model context overflow."""
    chosen = []
    characters = 0
    for human, assistant in reversed(history[-12:]):
        human, assistant = human[:750], assistant[:1100]
        cost = len(human) + len(assistant)
        if chosen and characters + cost > 3600:
            break
        if cost > 3600 and not chosen:
            human, assistant = human[-700:], assistant[-1400:]
            cost = len(human) + len(assistant)
        chosen.append((human, assistant))
        characters += cost
    messages = [{"role": "system", "content": SYSTEM}]
    for human, assistant in reversed(chosen):
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_text[:1500]})
    return messages


def show_memory(db, session, limit=10):
    history = load_history(db, session, limit=max(1, min(limit, 30)))
    if not history:
        print("No previous exchanges saved in this conversation.")
        return
    print(f"Last {len(history)} saved exchanges (your words + AI replies):")
    for number, (human, assistant) in enumerate(history, 1):
        print(f"  {number}. You > {human}")
        print(f"     Pretrained AI > {assistant}")


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
    parser.add_argument(
        "--new", action="store_true",
        help="Start a fresh conversation rather than resume the last saved chat"
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be 0 or more")
    if args.max_tokens < 10:
        parser.error("--max-tokens must be at least 10")

    np = axon.ensure_numpy()
    with axon.connection() as db:
        axon.bootstrap_lessons(db)
        session = get_manual_session(db, fresh=args.new)
        history = load_history(db, session, limit=40)
        recent = list(history[-25:])
        turn = db.execute(
            "SELECT COALESCE(MAX(turn),0) FROM observations WHERE session=?",
            (session,)
        ).fetchone()[0] // 2
        print("AXON MANUAL CHAT: YOU + one pretrained local assistant.")
        print(f"Loaded {len(history)} saved exchanges from BRAIN.db." )
        print("Teacher remembers recent chat turns, and AXON learns from both sides.")
        print("Commands: :memory 10  :new  :stats  :train 300  :axon question  :quit")
        model = axon.ensure_teacher(db)
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
                if cmd == ":new":
                    session = get_manual_session(db, fresh=True)
                    history.clear()
                    recent.clear()
                    turn = 0
                    print("Started a fresh chat. Previous transcripts are still stored in BRAIN.db.")
                    continue
                if cmd.startswith(":memory"):
                    try:
                        count = int(user_text.split()[1])
                    except (IndexError, ValueError):
                        count = 10
                    show_memory(db, session, limit=count)
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
                    print("Unknown command. Use :memory 10, :new, :stats, :train 300, :axon ... or :quit")
                    continue

                messages = conversation_messages(history, user_text)
                print("Pretrained AI > ", end="", flush=True)
                try:
                    response = teacher.complete(
                        messages, max_tokens=args.max_tokens,
                        temperature=0.70, stream=True
                    )
                except Exception as exc:
                    print(f"\nThe teacher failed: {exc}. This turn was not saved.")
                    continue
                response = response.strip()
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
                    print("AXON saved your words and the AI reply in BRAIN.db.")
                    if args.steps:
                        print("AXON also updated its own next-word weights.")
                except KeyboardInterrupt:
                    print("\nInterrupted; previously saved data remains.")
                    break
                except Exception as exc:
                    print(f"Learning error: {exc}. Check BRAIN.db before retrying.")
        statistics(db)


if __name__ == "__main__":
    main()
