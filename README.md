# AXON AI

**11M and 50M experimental models:** The repository contains isolated 11,240,448- and 50,224,640-parameter research architectures and legacy checkpoint converter in [docs/50m-research.md](docs/50m-research.md). The current Vercel chat and training workers still use the proven legacy checkpoint format until large-checkpoint storage, aggregation and inference are verified; five volunteer PCs cannot guarantee natural conversation within days.


AXON AI is an independently trained experimental transformer. The website, model-inference API, and training coordination API run on Vercel. Training checkpoints and progress are stored in Vercel Blob and Neon Postgres. The website does **not** need Vercel AI Gateway.

## Windows CPU and GPU trainers

Download `CPU-AXON.exe` or `GPU-AXON.exe` from the website. Running one starts a local console, installs Python and pinned dependencies on first launch, downloads only checksum-verified pages containing assigned lessons, and trains the AXON transformer on the volunteer's computer. The optional local pretrained reviewer is enabled with `--teacher`; it is not downloaded by default. Its weights are never copied into the student. Website chat now uses only AXON student checkpoints; before quality approval the chat reports learning status, with an opt-in unverified experimental preview.

The console displays ASCII progress bars and approximate remaining times when measurable. Setup is cached across launches; installation times depend on bandwidth and hardware. Both trainers use full-gradient PyTorch. GPU requires a supported NVIDIA CUDA driver; CPU uses a CPU-only runtime. Training pauses when the predicted 90% RAM threshold would be exceeded; this is not a hard memory limit. Ctrl+C stops training.

## How data is used

The initial corpus comes from a pinned revision of HuggingFaceTB/smol-smoltalk (Apache-2.0). The shared corpus contains 999,994,848 bytes and 213,571 deduplicated conversations. CI prepares it once and publishes 209 compressed pages. Each page is at most 1.56 MB compressed; a volunteer downloads only pages overlapping an assignment and reuses the cache. It does not contain 15 million unique training examples. Adding synthetic duplicates to reach a large line count would not necessarily improve model quality.

Neon reserves non-overlapping corpus ranges across volunteers. When explicitly enabled, the local pretrained teacher proposes a different question-and-answer version from each assigned record; two separately prompted local checks must both return PASS before the example is included. Verified records are cached locally. These two checks reduce some errors but do **not** guarantee factual accuracy, independence, or freedom from model mistakes.

The worker updates all student transformer weights using full backpropagation rather than training only the output layer. Updates are submitted as safetensors, checked for numerical bounds and limited regressions, and merged by the website's training system. The website chat uses only AXON student weights. Training is not evidence of conversational readiness: the default chat remains gated by quality evaluation, while an explicit unverified preview allows testing experimental output. The larger research checkpoints are not yet deployed.

## Configuration

Deploy the project using Vercel with root directory at the repository root, Node.js 22, output directory `public`. Attach Neon (`DATABASE_URL`) and Vercel Blob storage to the Production deployment. Cloud chat no longer uses Vercel AI Gateway or pretrained browser models. Only approved AXON student weights are used for experimental chat inference; default chat remains gated until evaluated. `AXON_OWNER_TOKEN` can be used for reviewing training updates; keep it private.

The EXEs are configured for https://axonai-theta.vercel.app. Training updates use this domain; the direct EXE downloads are hosted in verified GitHub releases. `/api/status` reports whether training storage is online and which checkpoint version is active.

## Limits

Ollama is a local model runtime, not a training quality level. AXON's student remains a small model (approximately 3.4 million parameters), far smaller than most capable open models. More training and more participating PCs can help, but do not guarantee monotonic improvement or ChatGPT/Ollama-model-class competence within days. Independent language-quality tests, data quality, sufficient compute, and larger model architectures would be needed for a competitive assistant.

Training jobs have queue and daily limits to protect cloud quotas. More volunteers cannot exceed those safety caps automatically. A local teacher may hallucinate; double-checking using the same model is not equivalent to two independent fact sources. Only train with datasets and models whose licenses and terms permit their use.

## Scratch-only chat and fast local teacher (November 2026)

The public chat no longer calls Qwen, Vercel AI Gateway or browser-hosted pretrained models. The model is an autoregressive Transformer with a UTF-8 byte tokenizer and real token-by-token sampling. A byte is one token; word/subword tokenization requires a new tokenizer and checkpoint migration, not merely a UI switch. A checkpoint with version 0 is random initialization and cannot answer. Trained-but-unevaluated checkpoints are accessible only using the clearly labelled experimental preview; this output may be nonsensical.

The Windows training helper uses standard-library source validation and conservative input rephrasing by default; there is no pretrained teacher download. Full-gradient PyTorch is still required for student training, including an initial CPU or CUDA package installation. The 11M/50M research tracks remain separate from the currently deployed ~3.4M protocol. Parameter growth and volunteer-hours alone cannot guarantee GPT-3/4-level quality.
