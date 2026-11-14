"""AXON AI student chat with deliberate, word-by-word decoding.

Each word is chosen by comparing several candidates and looking one or two
tokens ahead. Normal chat uses only AXON's own NumPy-trained model weights
and learned next-word statistics from BRAIN.db; Qwen is NOT used for chat.
Qwen is only used when the user explicitly invokes :observe.

This is bounded next-token search, NOT human-like reasoning. The 25M-parameter
student is not a transformer; fluency is limited by its training and architecture.

Put this file beside "LLM Train.py" and "BRAIN.db".
Python 3.11+.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "LLM Train.py"
if not TRAINER.is_file():
    raise SystemExit('Place "LLM Chat.py" beside "LLM Train.py" and "BRAIN.db".')

spec = importlib.util.spec_from_file_location("axon_train", TRAINER)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

STOP_PUNCT = {".", "!", "?"}
PUNCT = {".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "-", "[", "]"}
BAD_FIRST = PUNCT | {"%", "/"}


class DeliberateDecoder:
    """Receding-horizon token search, with batched neural lookahead.

    The decoder searches possible *words*. It never asks a pretrained model
    what to say and never retrieves a complete stored reply.
    """

    def __init__(self, np, db, weights, temperature=0.7, beam=6, depth=2):
        self.np = np
        self.db = db
        self.vocab = core.build_vocabulary(db)
        self.lookup = {word: i for i, word in enumerate(self.vocab)}
        self.w_in, w_out = weights
        self.w_out = np.ascontiguousarray(w_out[:len(self.vocab)])
        self.position = np.asarray(core.POSITION, dtype=np.float32)
        self.position /= self.position.sum()
        self.temperature = float(temperature)
        self.beam = max(2, min(10, int(beam)))
        self.depth = max(1, min(3, int(depth)))
        self.rng = np.random.default_rng()

    def scores_for(self, prefixes):
        """Compute scores for multiple speculative continuations in one batch."""
        np = self.np
        contexts = np.asarray([core._context(seq) for seq in prefixes], dtype=np.int32)
        hidden = np.einsum(
            "bld,l->bd", self.w_in[contexts], self.position, optimize=True
        )
        scores = float(core.LOGIT_SCALE) * (hidden @ self.w_out.T)
        return np.nan_to_num(
            scores.astype(np.float64),
            nan=-1e8, posinf=1e6, neginf=-1e8
        )

    def options(self, raw_scores, generated, limit=12, pattern_prefix=None):
        """Rank candidate words using neural and learned word-order probabilities."""
        np = self.np
        scores = raw_scores.copy()
        scores[:5] = -1e9  # forbid pad, unk, eos, user, assistant initially
        last = self.vocab[generated[-1]] if generated else ""
        terminal = last in STOP_PUNCT

        # Allow the end token only after some text, preferably a full sentence.
        if len(generated) >= 6 and (terminal or len(generated) >= 12):
            scores[2] = raw_scores[2]
            if not terminal:
                scores[2] -= 2.2
            elif len(generated) >= 10:
                scores[2] += 1.5

        # Disallow initial or consecutive punctuation, not real vocabulary words.
        forbidden = BAD_FIRST if not generated else (PUNCT if last in PUNCT else set())
        if len(generated) < 3:
            forbidden = forbidden | {",", ";", ":"}
        for token in forbidden:
            index = self.lookup.get(token)
            if index is not None:
                scores[index] = -1e9

        # Predict repeats less often, particularly repeated bigrams/trigrams.
        recent = collections.Counter(generated[-14:])
        for index, count in recent.items():
            if index >= 5:
                scores[index] -= min(2.8, 0.6 * count)
        if generated:
            scores[generated[-1]] -= 2.7
        if len(generated) >= 2:
            tail = generated[-2:]
            for j in range(2, len(generated)):
                if generated[j-2:j] == tail:
                    scores[generated[j]] -= 4.0

        neural_count = min(len(self.vocab), max(24, 6 * self.beam))
        pool = set(int(x) for x in np.argpartition(scores, -neural_count)[-neural_count:])
        # The learned word-order model is conditioned on the current question.
        # This avoids selecting every answer's first word from one global pool.
        pattern_tokens = list(pattern_prefix or []) + [self.vocab[i] for i in generated]
        patterns = core.word_pattern_probabilities(
            self.db, pattern_tokens, self.lookup
        )
        patterns = [(int(idx), int(count)) for idx, count in patterns if idx < len(self.vocab)]
        pool.update(idx for idx, _ in patterns[:50])
        if scores[2] > -1e8:
            pool.add(2)
        pool = [idx for idx in pool if scores[idx] > -1e8]
        if not pool:
            return []

        ids = np.asarray(pool, dtype=np.int32)
        logits = scores[ids] / max(0.2, self.temperature)
        logits -= logits.max()
        neural = np.exp(np.clip(logits, -80, 0))
        neural /= max(float(neural.sum()), 1e-12)

        counts = dict(patterns)
        pattern = np.asarray(
            [float(counts.get(int(i), 0)) ** 0.75 for i in ids],
            dtype=np.float64
        )
        if pattern.sum() > 0:
            pattern /= pattern.sum()
            # Word-order patterns help fluency, but do not override the model.
            mix = 0.42
            probs = (1.0 - mix) * neural + mix * pattern
        else:
            probs = neural

        # Apply repetition penalties AFTER blending too, so bigram statistics
        # cannot revive a repeated or punctuation-loop token.
        seen_trigrams = set(zip(generated, generated[1:], generated[2:]))
        for pos, index in enumerate(ids):
            index = int(index)
            if index in recent:
                probs[pos] *= 0.56 ** min(4, recent[index])
            if generated and index == generated[-1]:
                probs[pos] *= 0.06
            if len(generated) >= 2 and (
                generated[-2], generated[-1], index
            ) in seen_trigrams:
                probs[pos] *= 0.06
        total = float(probs.sum())
        if total <= 0:
            return []
        probs /= total
        order = np.argsort(-probs)[:max(1, int(limit))]
        return [(int(ids[k]), float(probs[k])) for k in order if probs[k] > 0]

    def choose(self, prefix, generated, pattern_prefix):
        """Look ahead, choose the first token of the best short continuation."""
        base_logits = self.scores_for([prefix])[0]
        first = self.options(base_logits, generated, limit=self.beam, pattern_prefix=pattern_prefix)
        if not first:
            return 2, []

        quality = [math.log(max(p, 1e-12)) for _, p in first]
        if self.depth > 1:
            future_prefixes = [
                prefix + [idx] for idx, _ in first if idx != 2
            ]
            valid_index = [i for i, (idx, _) in enumerate(first) if idx != 2]
            if future_prefixes:
                future_logits = self.scores_for(future_prefixes)
                second = {}
                for k, i in enumerate(valid_index):
                    idx = first[i][0]
                    second[i] = self.options(
                        future_logits[k], generated + [idx],
                        limit=2 if self.depth >= 3 else 1,
                        pattern_prefix=pattern_prefix
                    )
                if self.depth == 2:
                    for i, choices in second.items():
                        if choices:
                            quality[i] += 0.65 * math.log(max(choices[0][1], 1e-12))
                else:
                    third_prefixes = []
                    third_items = []
                    for i, choices in second.items():
                        for next_idx, next_prob in choices:
                            if next_idx == 2:
                                quality[i] = max(
                                    quality[i] - 0.65,
                                    math.log(max(first[i][1], 1e-12)) +
                                    0.65 * math.log(max(next_prob, 1e-12))
                                )
                                continue
                            third_prefixes.append(
                                prefix + [first[i][0], next_idx]
                            )
                            third_items.append((i, next_idx, next_prob))
                    if third_prefixes:
                        third_logits = self.scores_for(third_prefixes)
                        best_future = {}
                        for row, (i, next_idx, next_prob) in enumerate(third_items):
                            third = self.options(
                                third_logits[row],
                                generated + [first[i][0], next_idx],
                                limit=1, pattern_prefix=pattern_prefix
                            )
                            if third:
                                future = (
                                    0.65 * math.log(max(next_prob, 1e-12)) +
                                    0.35 * math.log(max(third[0][1], 1e-12))
                                )
                                best_future[i] = max(best_future.get(i, -1e6), future)
                        for i, value in best_future.items():
                            quality[i] += value

        # Penalize premature stopping so the search compares words fairly.
        for i, (idx, _) in enumerate(first):
            if idx == 2:
                quality[i] -= 0.8

        # Very small stochastic tie-break prevents permanently identical output.
        adjusted = [
            q + 0.06 * self.temperature * float(self.rng.gumbel())
            for q in quality
        ]
        winner = max(range(len(first)), key=lambda i: adjusted[i])
        alternatives = [
            (self.vocab[idx] if idx != 2 else "<end>", round(q, 3))
            for (idx, _), q in sorted(
                zip(first, quality), key=lambda item: item[1], reverse=True
            )[:min(4, len(first))]
        ]
        return first[winner][0], alternatives

    def generate(self, question, history=None, length=72, show_choices=False):
        # This student was trained on single-turn question -> answer examples.
        # Do not append untrained prose labels such as "Previous user:" that
        # dilute the actual question in its fixed-size context window.
        question_tokens = core.tokenizer(question)
        pattern_prefix = ['<user>'] + question_tokens[-120:] + ['<assistant>']
        prefix = [3] + [
            self.lookup.get(w, 1) for w in question_tokens
        ][-110:] + [4]
        generated = []
        pieces = []
        diagnostics = []
        previous = ""
        try:
            for step in range(max(1, min(200, int(length)))):
                selected, alternatives = self.choose(prefix, generated, pattern_prefix)
                if selected == 2:
                    break
                token = self.vocab[selected]
                piece = core._format_piece(token, previous)
                if previous in STOP_PUNCT and token and token[0].isalpha():
                    piece = " " + token[0].upper() + token[1:]
                print(piece, end="", flush=True)
                pieces.append(piece)
                prefix.append(selected)
                generated.append(selected)
                previous = token
                if show_choices and len(diagnostics) < 18:
                    diagnostics.append((token, alternatives))
                # Avoid filling the answer with a second unrelated sentence.
                if step >= 24 and token in STOP_PUNCT:
                    break
                if len(generated) >= 12 and generated[-6:] == generated[-12:-6]:
                    break
        except KeyboardInterrupt:
            print("\n[Generation interrupted.]", end="", flush=True)
        return "".join(pieces), diagnostics


def show_help():
    print("Commands:")
    print("  :think 1|2|3             Search depth: fast, balanced, slow (default 2)")
    print("  :beam 2..10              Number of candidate first words (default 6)")
    print("  :choices on|off          Display alternatives considered for early words")
    print("  :temperature 0.2..2.0    Creativity / probability smoothing")
    print("  :length 20..200          Maximum number of generated tokens")
    print("  :observe 4 8 [topic]     Generate 4 human/assistant chats, 8 turns each")
    print("  :notes 10                Show saved human/assistant observations")
    print("  :train 200               Train AXON on saved observations")
    print("  :teach question => answer Add a labeled example and train weights")
    print("  :reload                  Reload weights from BRAIN.db")
    print("  :stats                   Show student learning and database statistics")
    print("  :exit                    Quit")


def main():
    parser = argparse.ArgumentParser(description="AXON slower word-by-word search")
    parser.add_argument("--think", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--length", type=int, default=72)
    args = parser.parse_args()
    np = core.ensure_numpy()
    print("AXON AI | neural word-by-word chat with multiple-choice lookahead")
    print("Word decisions use AXON's own weights, not pretrained Qwen.")
    print("Type :help for commands. :think 3 is slower.")
    with core.connection() as db:
        core.bootstrap_lessons(db)
        vocab = core.build_vocabulary(db)
        state = None
        if db.execute("SELECT 1 FROM weights WHERE key='emb_in'").fetchone():
            state = core.load_student(db, np, vocab)
        else:
            print("No trained AXON weights found. Run LLM Train.py first.")
        temperature = 0.7
        beam = max(2, min(10, int(args.beam)))
        depth = args.think
        length = max(20, min(200, int(args.length)))
        choices_on = False
        history = []

        while True:
            try:
                prompt = input("\nYou > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not prompt:
                continue
            lower = prompt.casefold()
            if lower in (":exit", ":quit", "exit", "quit"):
                break
            if lower == ":help":
                show_help()
                continue
            if lower.startswith(":think "):
                try:
                    next_depth = int(prompt.split()[1])
                    if next_depth not in (1, 2, 3):
                        raise ValueError()
                    depth = next_depth
                    print(f"Search depth: {depth}.")
                except (ValueError, IndexError):
                    print("Use :think 1, :think 2 or :think 3.")
                continue
            if lower.startswith(":beam "):
                try:
                    value = int(prompt.split()[1])
                    if not 2 <= value <= 10:
                        raise ValueError()
                    beam = value
                    print(f"Word candidates: {beam}.")
                except (ValueError, IndexError):
                    print("Use :beam 2 through :beam 10.")
                continue
            if lower.startswith(":choices "):
                setting = prompt.split()[1].casefold()
                if setting in ("on", "off"):
                    choices_on = setting == "on"
                    print(f"Candidate display: {setting}.")
                else:
                    print("Use :choices on or :choices off.")
                continue
            if lower.startswith(":length "):
                try:
                    value = int(prompt.split()[1])
                    if not 20 <= value <= 200:
                        raise ValueError()
                    length = value
                    print(f"Maximum tokens: {length}.")
                except (ValueError, IndexError):
                    print("Use :length 20 through :length 200.")
                continue
            if lower.startswith(":temperature "):
                try:
                    value = float(prompt.split()[1])
                    if not 0.2 <= value <= 2.0:
                        raise ValueError()
                    temperature = value
                    print(f"Temperature: {temperature:.2f}.")
                except (ValueError, IndexError):
                    print("Use :temperature 0.2 through :temperature 2.0.")
                continue
            if lower == ":reload":
                vocab = core.build_vocabulary(db)
                state = core.load_student(db, np, vocab)
                print("Reloaded AXON student weights and vocabulary.")
                continue
            if lower.startswith(":notes"):
                try:
                    limit = int(prompt.split()[1])
                except (ValueError, IndexError):
                    limit = 10
                for speaker, receiver, message in core.recent_notes(db, limit):
                    print(f"{speaker} -> {receiver}: {message}")
                continue
            if lower == ":stats":
                observation_count = db.execute(
                    "SELECT COUNT(*) FROM observations "
                    "WHERE speaker IN ('Human','Assistant')"
                ).fetchone()[0]
                lessons = db.execute(
                    "SELECT COUNT(*) FROM lessons WHERE origin='dialogue_quality_v5'"
                ).fetchone()[0]
                print(f"Architecture: {core.STUDENT_ARCH}")
                print(f"Trainable weights: {core.PARAMETERS:,}")
                print(f"Gradient batches: {core.get_meta(db,'student_steps',0)}")
                print(f"Known tokens: {len(vocab):,} / {core.VOCAB_SIZE:,}")
                print(f"Observed messages: {observation_count:,}; quality answer pairs: {lessons:,}")
                print(f"Search: depth {depth}, {beam} first-word candidates.")
                print(f"BRAIN.db: {core.DB_PATH.stat().st_size / 1024**2:.1f} MiB.")
                continue
            if lower.startswith(":observe"):
                bits = prompt.split()
                try:
                    chats = int(bits[1]) if len(bits) > 1 else 4
                    turns = int(bits[2]) if len(bits) > 2 else 8
                except ValueError:
                    print("Usage: :observe 4 8 [topic]")
                    continue
                if not 1 <= chats <= 12 or not 2 <= turns <= 200:
                    print("Use 1-12 chats, 2-200 turns.")
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
                    print(f"Recorded {pairs} Q&A pairs across {sessions} chats.")
                    if pairs:
                        core.learn_observations(db, np, steps=max(80, pairs * 18))
                        vocab = core.build_vocabulary(db)
                        state = core.load_student(db, np, vocab)
                except KeyboardInterrupt:
                    print("\nObservation interrupted; saved notes remain.")
                except Exception as exc:
                    print(f"Observation failed: {exc}")
                continue
            if lower.startswith(":train"):
                try:
                    steps = int(prompt.split()[1])
                except (ValueError, IndexError):
                    steps = 200
                if not 1 <= steps <= 1000000:
                    print("Choose 1 through 1,000,000 gradient batches.")
                    continue
                try:
                    core.learn_observations(db, np, steps=steps)
                    vocab = core.build_vocabulary(db)
                    state = core.load_student(db, np, vocab)
                except KeyboardInterrupt:
                    print("\nTraining interrupted; previously saved weights remain.")
                continue
            if lower.startswith(":teach "):
                pair = prompt[7:].split("=>", 1)
                if len(pair) != 2 or not pair[0].strip() or not pair[1].strip():
                    print("Use :teach How are you? => I am fine, thanks!")
                    continue
                question, answer = (part.strip() for part in pair)
                core.store_lesson(db, question, answer, "user")
                core.observe_word_patterns(db, answer, prefix=question)
                core.extend_vocabulary(db, question + " " + answer)
                db.commit()
                print("Training student weights on your example...")
                try:
                    core.train_student(
                        db, np, itertools.repeat((question, answer), 300),
                        max_steps=45, print_every=15
                    )
                    vocab = core.build_vocabulary(db)
                    state = core.load_student(db, np, vocab)
                    print("Training update saved.")
                except KeyboardInterrupt:
                    print("\nTraining interrupted; previous progress saved.")
                continue
            if prompt.startswith(":"):
                print("Unknown command. Type :help.")
                continue
            if state is None:
                print('No trained weights. Run "LLM Train.py" or use :observe 4 8.')
                continue

            print("AXON > ", end="", flush=True)
            try:
                decoder = DeliberateDecoder(
                    np, db, state, temperature=temperature, beam=beam, depth=depth
                )
                answer, audit = decoder.generate(
                    prompt, history=history, length=length,
                    show_choices=choices_on
                )
                if not answer.strip():
                    print("[No useful words yet; train with more good conversations.]", end="")
                print(flush=True)
                if answer.strip():
                    history.append((prompt, answer))
                    history = history[-3:]
                if choices_on:
                    print("\nCandidate word comparisons (search scores, NOT thoughts):")
                    for chosen, alternatives in audit:
                        comparison = ", ".join(
                            f"{word} ({score:.2f})" for word, score in alternatives
                        )
                        print(f"  Chose {chosen!r}; considered: {comparison}")
            except Exception as exc:
                print(f"\nAXON generation error: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
