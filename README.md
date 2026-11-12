# AXON AI — Vercel Website

## Deploy
1. Upload the **contents** of `AXONAI_Vercel/` to the root of your Vercel GitHub repository.
2. Vercel project root must contain `vercel.json` and `package.json`.
3. In Vercel, open **AI Gateway** and enable it for the project. Vercel provides OIDC credentials to production deployments. Alternatively set `AI_GATEWAY_API_KEY` as a private environment variable in Vercel.
4. Redeploy. The website runs on Vercel with no home PC required.

**Chat:** Calls Vercel AI Gateway to use `openai/gpt-4.1-mini` by default; change `AXON_GATEWAY_MODEL` to another supported model if needed. Usage is metered against your AI Gateway credits. Never put the key in `public/` or GitHub. Chat text is sent to the selected upstream AI provider for processing; no chat history is stored by this app. Vercel and the provider may apply their own retention policies.

**Training:** The CPU/GPU downloads are present, but the separate community training coordinator still needs a persistent cloud host. Only after that coordinator has evaluated and approved a student model can chat switch to it; until then the hosted pretrained AI remains active.

**Protection:** Request validation, same-origin restrictions, best-effort limits and optional Cloudflare Turnstile are included. For large-scale rate limits and bot challenges configure Vercel Firewall. See `SECURITY.md` if present.

**Local validation:** `npm ci && npm test && npm run build`. Tests mock remote AI requests; only a live Vercel deployment can verify real AI Gateway authentication and billing.
