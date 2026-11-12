# AXON AI — adaptive chat and progress

Upload the CONTENTS of `AXONAI_Vercel` to the GitHub repository root used by Vercel.

**Improved chat:** Local model is now Qwen2.5-0.5B-Instruct ONNX Q4. Initial model download is large (~786 MB) and may fail on Hobby function CPU/RAM/time constraints; live Vercel testing required. For better chat quality, set `AXON_OPENROUTER_KEY` in Vercel server environment variables and optionally `AXON_HOSTED_MODEL` (default `qwen/qwen3-8b`). OpenRouter billing/quotas and model availability depend on your account. Keep keys secret.

**Training status and auto-switch:** The protected coordinator ZIP is separate. Deploy it to a persistent VPS and configure `AXON_COORDINATOR_URL=https://your-coordinator.example` and `AXON_INFERENCE_TOKEN=...` on Vercel. Configure the same `AXON_INFERENCE_TOKEN` on the coordinator. Evaluations occur on approved checkpoint updates and daily. Submitted-but-unapproved updates may be evaluated but are never activated. Promotion requires passing score AND a teacher judgement: `AXON_JUDGE_API_KEY` (OpenRouter API key) on the coordinator. Without this token the student remains inactive. `AXON_JUDGE_MODEL` optionally overrides default.

**Ring:** Shows benchmark pass percentage for tested student checkpoint, not a claim of GPT-3 equivalent intelligence. Displays untested when no coordinator or no evaluation exists.

The pretrained helper is always the fallback when the student is unqualified or offline. The coordinator must be running separately; it cannot be persisted by Vercel Functions. Previous CPU/GPU EXEs remain compatible.

**Volunteer download:** workers now fetch the coordinator address automatically from `public/worker-config.json` in the public AXONAI GitHub repository. Leave the file blank until the owner has deployed a real persistent HTTPS training server. Never publish a train token in the public config. When the server is offline the worker exits cleanly before installing packages. See the private coordinator guide.
