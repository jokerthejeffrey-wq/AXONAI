# AXON AI

**50M experimental model:** The repository contains an isolated, tested 50,224,640-parameter research architecture and legacy checkpoint converter in [docs/50m-research.md](docs/50m-research.md). The current Vercel chat and training workers still use the proven legacy checkpoint format until large-checkpoint storage, aggregation and inference are verified; five volunteer PCs cannot guarantee natural conversation within days.


AXON AI is an independently trained experimental transformer. The website, model-inference API, and training coordination API run on Vercel. Training checkpoints and progress are stored in Vercel Blob and Neon Postgres. The website does **not** need Vercel AI Gateway.

## Windows CPU and GPU trainers

Download `CPU-AXON.exe` or `GPU-AXON.exe` from the website. Running one starts a local console, installs Python and pinned dependencies on first launch, prepares a cached conversational corpus, downloads a local pretrained 0.5B teacher, and trains the AXON transformer on the volunteer's computer. The trainer does not use the pretrained teacher for website chat.

The console displays ASCII progress bars and approximate remaining times when measurable. Setup is cached across launches; installation times depend on bandwidth and hardware. Both trainers use full-gradient PyTorch. GPU requires a supported NVIDIA CUDA driver; CPU uses a CPU-only runtime. Training pauses when the predicted 90% RAM threshold would be exceeded; this is not a hard memory limit. Ctrl+C stops training.

## How data is used

The initial corpus comes from a pinned revision of HuggingFaceTB/smol-smoltalk (Apache-2.0). The measured cache contains approximately 1 GB and 213,571 deduplicated conversations. It does not contain 15 million unique training examples. Adding synthetic duplicates to reach a large line count would not necessarily improve model quality.

Neon reserves non-overlapping corpus ranges across volunteers. The local pretrained teacher proposes a different question-and-answer version from each assigned record; two separately prompted local checks must both return PASS before the example is included. Verified records are cached locally. These two checks reduce some errors but do **not** guarantee factual accuracy, independence, or freedom from model mistakes.

The worker updates all student transformer weights using full backpropagation rather than training only the output layer. Updates are submitted as safetensors, checked for numerical bounds and limited regressions, and merged by the website's training system. The website runs only its own student model for chat. Hardcoded arithmetic and fixed greeting routes are removed.

## Configuration

Deploy the project using Vercel with root directory at the repository root, Node.js 22, output directory `public`. Attach Neon (`DATABASE_URL`) and Vercel Blob storage to the Production deployment. The model can serve basic chat without installing AI Gateway. `AXON_OWNER_TOKEN` can be used for reviewing training updates; keep it private.

The EXEs are configured for https://axonai-theta.vercel.app. Downloads and updates use this domain. `/api/status` reports whether training storage is online and which checkpoint version is active.

## Limits

Ollama is a local model runtime, not a training quality level. AXON's student remains a small model (approximately 3.4 million parameters), far smaller than most capable open models. More training and more participating PCs can help, but do not guarantee monotonic improvement or ChatGPT/Ollama-model-class competence within days. Independent language-quality tests, data quality, sufficient compute, and larger model architectures would be needed for a competitive assistant.

Training jobs have queue and daily limits to protect cloud quotas. More volunteers cannot exceed those safety caps automatically. A local teacher may hallucinate; double-checking using the same model is not equivalent to two independent fact sources. Only train with datasets and models whose licenses and terms permit their use.
