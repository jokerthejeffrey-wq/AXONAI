# AXONAI — independent student, teacher-assisted training

This is a new architecture, not an update that can reuse the previous project's adapters. Extract it into a fresh folder and use the new `student_state/` directory.

The student is a **3,423,744-parameter causal transformer initialized randomly**. It has a new UTF-8 byte tokenizer (256 byte tokens plus four control tokens), four attention blocks, and a 512-token context. No pretrained weights or pretrained tokenizer are copied into the student. `student.py` imports PyTorch only; `CHAT.py` now defaults to pretrained-assisted mode; set AXON_CHAT_MODE=student to use only student inference. No pretrained weights are copied into the student.

The separate `TEACHER.py` downloads Qwen2.5-0.5B-Instruct to create short training responses. Contributors train the independent student on that text. This is teacher-generated supervised training, sometimes called sequence-level distillation; it is not logit matching. The owner runs the teacher once per new curriculum example so each volunteer does not need to download and run it.

## Honest expectations

Random initialization is not a smart chatbot. This small student and starter curriculum demonstrate a working training system, not a trained general assistant. It will initially generate nonsense. More volunteers alone do not solve data quality, model capacity or optimization. A four-use cap on every topic severely limits learning; respecting that cap cannot guarantee that the skill is learned. Creating new topic names solely to replay the same lesson defeats your requested rule.

In student-only mode, replies are sampled autoregressively from the student's token probabilities, using the conversation as context, an end-of-turn token, temperature and a small repetition penalty. They are not chosen from a greeting/response table, and that mode has no teacher fallback. Creator questions use explicit community-attribution metadata rather than a guessed model biography. Tokens here are bytes, not whole words. The UI displays each finished reply; it is not a streaming typewriter effect. The context is only 512 bytes/control tokens, so long conversations lose older details. These mechanisms allow learned continuation; they do not guarantee reasoning or good conversation.

## Files

| File | Purpose |
|---|---|
| GPU.py / CPU.py | Console contribution programs with automatic environment setup |
| START_GPU.cmd / START_CPU.cmd | Windows launchers; offer Python 3.12 installation if missing |
| bootstrap.py | Isolated dependency install, one repair retry, hash-checked source repair |
| TEACHER.py | Owner-only pretrained teacher download and lesson generation |
| REVIEW.py | Approve, edit or reject teacher drafts before training |
| student.py | Independent model, tokenizer, gradient training and generation |
| CHAT.py | Assisted or student-only chat, shared by console and website |
| HTML.py | Flask website/API, job assignment and contribution uploads |
| manage.py | Initialize, import, review, publish, inspect budget, reject, rollback |
| vercel/ | Static frontend ready for Vercel |

## Assisted chat while the student learns

Default: `AXON_CHAT_MODE=assisted`. `python HTML.py` installs the inference dependencies and downloads Qwen on the first non-identity chat. The UI labels replies **Qwen-assisted**. The app credits the AXONAI community while truthfully naming the pretrained underlying model. Chatting does not train the student or consume training budgets.

There is NO automatic claim that a low loss means the student is a good chatbot. Evaluate held-out questions, multi-turn follow-ups, accuracy and closing behavior first (see CHAT_READINESS.md). Then set `AXON_CHAT_MODE=student` and restart the backend. Switching back to `assisted` restores pretrained replies. The student remains randomly initialized at creation and learns only through the separate training pipeline.

## Owner setup

Use Python 3.11/3.12 in PyCharm. Keep all files together. The scripts prompt to create isolated environments and install packages. They do not modify your PyCharm or system Python packages.

```sh
python TEACHER.py
python REVIEW.py
python manage.py init
python HTML.py
```

The teacher runs on CPU by default. `python TEACHER.py --gpu` selects a separate CUDA environment. `--limit 10` generates up to ten missing curriculum responses. It resumes by source ID and never regenerates existing records automatically. The supplied training file contains 48 teacher-generated starter responses with AI-assisted corrections; verify them for your own use. Raw drafts are preserved in data/teacher_drafts.jsonl. New teacher outputs go ONLY to the draft file. REVIEW.py requires approval or correction before adding them to train.jsonl; the small teacher can make serious factual errors. Add genuinely new topics to `data/curriculum.json`, rerun the teacher, run REVIEW.py, stop HTML.py, then `python manage.py import` to import only new records without resetting counts.

`manage.py init` generates a random student checkpoint if none exists. It does not load Qwen weights. In student mode, version 0 is deliberately blocked until a training update is published. Default assisted chat works immediately, even before student training. Even after publication, early output will be poor. Terminal chat: `python CHAT.py`, with `/new` and `/quit` commands.

## Contributor installation and repair

Windows users extract the client ZIP and launch START_CPU.cmd or START_GPU.cmd. If Python 3.12 is missing, `start.ps1` offers installation through Microsoft's Windows Package Manager using the exact Python.Python.3.12 package. It does not request an administrator-wide install. If winget is unavailable, install 64-bit Python 3.12 from python.org and rerun. Linux/macOS need Python 3.11/3.12 available first.

```sh
python CPU.py --server https://YOUR-PYTHON-BACKEND --threads 2
# OR:
python GPU.py --server https://YOUR-PYTHON-BACKEND
```

Missing dependencies are installed in `.axon-cpu`, `.axon-gpu`, `.axon-teacher`, or `.axon-teacher-gpu`. Setup tries a forced reinstall once if package installation/import checks fail. Package downloads can be several GB. GPU targets NVIDIA CUDA 12.8 and requires a compatible driver. The installer cannot fix an unsupported driver, unsupported OS, insufficient RAM/VRAM, no internet, full disk, antivirus block or every Python bug. No driver installation is attempted. CPU is the fallback.

Source files are checked against the hashes in `client_manifest.json`. Missing/changed shared files can be restored from your backend after consent, but only if their SHA-256 matches the manifest shipped in the downloaded ZIP. Local edits are overwritten only after approval. If the launcher, bootstrap or manifest itself is missing/broken, re-extract a trusted complete ZIP. The manifest is a local integrity record, not a cryptographic publisher signature; distribute it through a trusted website. A new backend version cannot silently replace an old client's expected source; download a new complete ZIP when hashes differ.

The downloaded client does not scan personal files or collect chats. It receives only shared student weights and teacher-written lessons. One job uses at most eight lessons, one optimizer step each, with batch size 1. `--rounds 3` requests three jobs. `--threads 2` is a thread cap, not a power/percent limit. `--yes` skips the compute consent only, not installation consent. Ctrl+C stops. Local partial work is discarded.

## Four-use rule

The server stores permanent use counters in SQLite, both per normalized conversation prompt and per curated concept group. It reserves a use atomically BEFORE issuing a job. There is no client-side epoch loop or replacement sampling. A topic can be scheduled at most **four times total**, shared across all official clients and jobs.

Hi, hello, greetings, good morning and other tagged greetings share `greetings`. The owner must label semantic variants consistently in `curriculum.json`. The alias rules recognize common standalone English greetings; they cannot perfectly detect every paraphrase or every language. This is explicit topic grouping, not a claim of universal semantic duplicate detection.

A changed answer or reimported normalized prompt does not reset its counter. Rejected, failed, abandoned, stale and expired jobs consume reservations permanently. Rollback does not restore uses. This conservative choice avoids accidentally training a lesson five times after a retry. It can waste scarce budget on failed jobs. The ledger counts ASSIGNED uses; an untrusted volunteer can modify their own program, so the coordinator cannot prove how many gradient steps a remote machine actually ran. Start with trusted contributors.

```sh
python manage.py budget
```

Do not delete the database, change topic labels to evade limits, or reset counters. Back up the whole `student_state/` directory while the server is stopped.

## Publish contributions

Stop HTML.py before modifying owner operations. Use one coordinator process and one state directory.

```sh
python manage.py list
python manage.py merge UPDATE_ID_1 UPDATE_ID_2
python HTML.py
```

Merge accepts only same-version pending checkpoints. It checks tensor shapes, dtypes, finiteness and update magnitude, averages the selected student weights equally, then mixes 50% with the prior student. A separate validation dataset must show nonincreasing loss before publication. This is an experimental synchronous weight-averaging scheme, not guaranteed convergence or robust Byzantine aggregation. Jobs contain up to eight lessons, so equal weighting is per selected update, not sample-weighted.

The small included holdout only tests a few behaviors. It cannot rule out poisoning, backdoors or general regression. Review contributors and data. Larger public deployments need stronger identity, robust aggregation, richer private evaluation and monitoring. Do not automatically accept unknown uploads.

```sh
python manage.py reject UPDATE_ID
python manage.py rollback OLD_VERSION
```

Rollback publishes old weights under a NEW version to invalidate in-flight jobs. It leaves training-use counters untouched. Rolling back to version 0 weights still makes a new version with untrained weights; it is a recovery tool, not training progress.

The protocol uploads safetensors, never Python pickle. Full checkpoints are around 14 MB, and uploads are capped at 32 MiB. Keep `settings.py` identical on all clients. For architecture changes create a new project/state directory; existing weights are incompatible.

## Website and Vercel

Local website: http://127.0.0.1:5000 after `python HTML.py`. The landing page retains your black/white design; Try it ↗ opens chat, and the sun/moon control persists theme. The console ZIP is generated by an explicit allowlist, excluding state, secrets and owner tools.

For a `your-project.vercel.app` frontend, follow **VERCEL.md**. Vercel serves static assets; browser requests go directly to your persistent Python backend over HTTPS. No teacher or student weights are bundled into Vercel functions. The small client ZIP is served statically by Vercel; model checkpoints and uploads go directly to the backend. The backend must remain online and have persistent storage.

## Resource guidance

The student has ~14 MB of float32 weights, plus activations and optimizer state. CPU is sufficient to test this small architecture; speed varies. The teacher is much larger and downloads ~1 GB of weights, with several GB of RAM needed while running. Run teacher generation separately from contribution training on low-memory machines. No GPU performance or memory guarantee is made.

## Privacy and configuration

Chat content stays in the current browser tab and is sent to the Python backend for inference; the application does not save it or add it to training. Infrastructure logs may retain request metadata. Contribution weights, job IDs and topic budgets persist.

Environment variables:
- `AXON_CHAT_MODE`: assisted (default) or student; restart server after changing.
- `AXON_SERVER`: contributor default backend URL.
- `AXON_STATE`: persistent student directory; defaults to student_state beside code.
- `AXON_TRAIN_TOKEN`: optional shared contributor secret; match on server/client. Never place it in Vercel config or source.
- `AXON_ALLOWED_ORIGINS`: comma-separated exact frontend origins allowed by CORS, e.g. https://my-axon.vercel.app.
- `AXON_CHAT_DEVICE`: cpu by default. A CUDA deployment needs a CUDA-enabled backend environment.
- `AXON_THREADS`: inference CPU threads, default 2.
- `HOST` / `PORT`: default 127.0.0.1 / 5000. Waitress is used, not Flask debug mode.

Built-in rate limits use the actual socket IP and intentionally ignore untrusted forwarded headers. Behind a tunnel/proxy, users may share a limit. Configure trusted edge limits if you scale. CORS is not authentication; Python clients can still call public APIs. A shared contributor token helps a trusted pilot but is not a per-person identity system.

## Editing and rebuilding

After deliberate edits to client source or templates, run `python build_project.py`. It regenerates the integrity manifest and copies UI assets into vercel/public. Redistribute the complete new client ZIP; old manifests will reject changed source repair files. Do not run this just to mask unexplained tampering.

Tests: install dependencies, then run `python -m unittest discover -p "test_*.py" -v` in a configured environment. For automatic environment use, run `.axon-cpu/Scripts/python.exe` on Windows or `.axon-cpu/bin/python` on Linux. See TEST_RESULTS.md for what was actually tested.

The code is MIT licensed. Teacher weights download separately and retain the upstream Qwen license. Teacher text should be reviewed for accuracy and permitted use before distribution. The archive contains no pretrained or trained student weights.
