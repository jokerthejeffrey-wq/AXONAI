# AXON AI — Website (cloud chat)

This website is built for Vercel, with no home PC required for chat.
The old local model downloader and Transformers/ONNX dependencies have been removed.
The site uses `openai/gpt-4.1-mini` through Vercel AI Gateway for pretrained chat.
No pretrained model weights are installed inside a Vercel Function.

## Deploy (replace the old website)

1. Extract this ZIP and upload **all contents inside `AXONAI_Vercel`** to the actual **Vercel Root Directory** in GitHub. Do not upload the ZIP as a single file.
2. Remove old model files and the older `vercel/` or other legacy application directories if they are still in that deployment root. Older GitHub files are NOT deleted automatically when new files are uploaded.
3. Ensure the root contains this ZIP's `package.json`, `package-lock.json`, `vercel.json`, `api/`, `server/`, `public/`, and `scripts/`.
4. In your Vercel dashboard, enable **AI Gateway** on this project. Vercel OIDC is supported, or you can create an AI Gateway key and set `AI_GATEWAY_API_KEY` in **Project Settings → Environment Variables** (Production), then redeploy. NEVER place this key in GitHub or the website.
5. Deploy from the correct GitHub branch and root folder. In Vercel you can **Redeploy** and disable reuse of the build cache to ensure a clean replacement.
6. Visit `/api/status` on your deployed site. `local_model_installed` must be `false`. `chat_credentials_present` should be `true` if authentication is available. Then open `/chat` and send `hello`.

If `chat_credentials_present` is false, enable AI Gateway or configure the private API key and redeploy. If it is true but chat still fails, check Vercel → Logs → Functions for the sanitized `AXON AI chat backend` code. The error shown in chat will distinguish authentication, billing, and model configuration problems.

**Important:** Vercel Functions are not persistent computers. New deployments replace application code and dependencies, but Vercel manages temporary caches from older deployments. This ZIP can't remotely delete another deployment's model files or your Vercel account data. It only stops shipping/loading the old local model. A hosted pretrained model is called by API; it is not installed on Vercel.

**Training:** CPU/GPU EXEs remain available as downloads, but they need a separately hosted coordinator for actual distributed training. Chat remains pretrained until a coordinator certifies and serves an approved student checkpoint. The 90% RAM admission limit remains in workers; chat does not need to reserve RAM for a local model.

**Security:** Input validation, same-origin restrictions, rate limiting and optional Cloudflare Turnstile are preserved. AI Gateway usage is metered; set budgets and firewall rules before promoting the website.
