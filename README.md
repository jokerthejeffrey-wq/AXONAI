# AXONAI — Vercel-only cloud chat

This is a separate deployment package. Chat runs the pretrained **SmolLM2-135M-Instruct ONNX Q4 model inside a Vercel Node.js Function**. You do NOT need to leave your PC on, start Flask, run cloudflared, configure AXON_API_URL, or buy an inference API key.

The independent from-scratch student and its distributed training remain a separate project. This package does not pretend a Vercel temporary directory is a persistent training database. The **separately provided private trainer package** includes the coordinator and the CPU/GPU Python sources. This public Vercel ZIP intentionally does not ship those sources or any unbuilt EXE files. Chat works independently of it.

## Deploy — exact settings

1. Extract the ZIP. Put the CONTENTS of the AXONAI_Vercel folder into your GitHub repository, or a separate folder in the repository. Keep your old training code/checkpoints separately.
2. Import the repository into Vercel, or change the existing project's Root Directory to this AXONAI_Vercel folder.
3. Framework preset: **Other**. Node.js version: **22.x**. Enable **Fluid Compute** in project settings; the function needs up to 300 seconds for a cold download/start.
4. Settings from vercel.json:
   - Install command: `node scripts/install.mjs`
   - Build command: `npm run build`
   - Output directory: `public`
5. Remove the old **AXON_API_URL** setting. This build does not use it. No tunnel URL or AXON_ALLOWED_ORIGINS setting is needed for chat.
6. Redeploy with the previous build cache disabled for the first migration. Open `/api/status` on your new .vercel.app site: it should return JSON with `hosting: "vercel"`.
7. Open `/chat`, hard-refresh, and send a normal message. The first request on a new instance downloads about 181 MB of model weights plus tokenizer/config files into `/tmp`, then runs CPU inference. Allow a few minutes.

The package includes api/, server/, scripts/, public/, package.json, package-lock.json and vercel.json. The training workers and coordinator are distributed as a separate PRIVATE artifact. **Do not upload only public/**, and do not use the old `node vercel/build.mjs` command or `vercel/public` output path. The old project is not the same deployment layout.

## What gets installed

Vercel installs the pinned Transformers.js dependency and the bundled ONNX CPU runtime during deployment. `scripts/install.mjs` disables unnecessary CUDA downloads; there is no GPU on the target runtime. Model weights download automatically from the pinned public Hugging Face revision on the first non-identity message. Warm instances reuse the loaded model and cache.

The website and chat API use the same origin. Browser requests no longer depend on your PC, a temporary trycloudflare.com URL, or cross-origin Flask configuration. The model runs on Vercel's server CPU, not in the visitor's browser. There is no third-party inference API call.

## Real limits

The 8 GB shown in a build log describes the build machine; it is not your function's permanent disk or runtime RAM. Vercel documents Hobby Fluid Compute functions with 2 GB RAM and a 300-second maximum invocation. Functions have a read-only deployment filesystem and temporary writable scratch storage. This is why this package uses a small quantized model and `/tmp` caching instead of full PyTorch/Qwen installs.

This is small-scale cloud inference, not an always-running GPU server. Vercel may discard caches between instances, so cold downloads can repeat. The free/Hobby plan has CPU, memory-time, transfer and invocation quotas; hosting is not unlimited free AI compute. Check your Vercel usage dashboard. Concurrent traffic may create separate instances, each loading its own model. The app's rate limiter is best-effort per instance, not a global quota; use Vercel's project controls for a public launch.

SmolLM2 135M is substantially smaller than the previous Qwen 0.5B assistant. Expect limited reasoning, factual errors and short answers. Replies are limited to 128 new tokens and short conversation context. This package credits the AXONAI community and transparently identifies SmolLM2 assistance. It does not claim that the pretrained model itself was trained from scratch by AXONAI.

## Troubleshooting

- **Still 'Failed to fetch':** check that you deployed THIS root, not the old frontend. Hard-refresh. `/api/status` must return JSON from the same website.
- **API route returns HTML/404:** api/ was not deployed, Root Directory is wrong, or the old build/output settings remain.
- **CUDA installer download error:** set Install Command to `node scripts/install.mjs`; this skips GPU binaries. Do not replace it with an unconfigured `npm install`.
- **Timeout:** confirm Fluid Compute and 300-second function duration. Review the Vercel Function logs. An interrupted cold download may need a retry.
- **Model download failure:** check Vercel logs for Hugging Face/network errors. Model files must remain publicly downloadable.
- **Bundle size:** the configuration excludes non-Linux-x64 and GPU native binaries. Weights are downloaded at runtime, not included in the function deployment.
- **Memory/runtime error:** check logs; a local smoke test is not a guarantee of identical Vercel behavior. This package is prepared for deployment but no live project was deployed on your account.
- **Training client cannot connect:** the coordinator is a separately deployed persistent HTTPS service, not this Vercel website. Deploy it from the private trainer ZIP and configure its public URL on the workers.

## Development

Use Node 22. For a clean dependency install, run `node scripts/install.mjs`. Then `npm test` and `npm run build`. Use Vercel CLI's `vercel dev` for local function routing if desired. Local model smoke testing can import `generate` from server/model.js. Set AXON_CACHE_DIR to an accessible temporary directory if your environment has no writable `/tmp`.

No conversation is stored by this application. It still processes prompts on Vercel, and infrastructure may keep request metadata. Model downloads go to Hugging Face; the code does not send prompts to Hugging Face inference services.

## References

- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/functions/runtimes
- https://huggingface.co/docs/transformers.js/tutorials/node
- https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX

Code inherits the AXONAI project MIT license. Separately downloaded pretrained model and dependencies retain their respective licenses; see their model card/package notices.

## November 12 update: public/private split, UI cleanup, and memory admission

- Removed the left/right animated square tile columns from the home hero. The floor wave remains.
- Removed `public/download/client.zip`, which exposed the Python training source to public visitors. Training code and the `.github/workflows/build-windows.yml` build instructions live in the **separate private ZIP**.
- `/downloads` gives download guidance but does not pretend EXEs already exist; set `window.AXON_RELEASE_ASSET_BASE` in `public/config.js` after verified releases (pointing to the GitHub release `/releases/latest/download` base URL).
- `/api/chat` uses Linux cgroup-aware memory accounting where available and refuses new generation when predicted memory use would exceed 90% of the runtime limit. The browser waits and retries when the server responds with `retryable:true`. These are best-effort limits, not guaranteed OS memory caps; per-instance and rate limits still apply.
- The private coordinator runs on an always-on VPS or similar persistent host, not Vercel Functions. It atomically reserves example pairs in SQLite and stores contributor checkpoints for owner review.

See `AXONAI_PRIVATE/README.md` in the separate private trainer package for the full Ubuntu-to-Windows build and coordinator setup.

## Website security (bot and injection protections)

This revision adds the following to the Vercel website:

- Strict server-side message structure and length limits, JSON content-type and 128 KiB request limit; rejects unexpected top-level fields.
- Rejects cross-site browser requests using origin and Fetch Metadata headers. This is defense in depth, **not an authentication method**; HTTP clients may spoof or omit these headers.
- Per-function-instance traffic limits (120 requests/minute and 8 accepted messages/minute per client) before expensive inference. These counters **do not** synchronize across Vercel instances, so they are not a global anti-abuse quota.
- Content Security Policy, framing disabled, MIME sniffing disabled, limited browser permissions, and restrictive referrer policy. The chat UI uses `textContent` for user input and AI output, not HTML rendering.
- Optional Cloudflare Turnstile verification on chat messages. The server verifies **each** single-use token using Cloudflare Siteverify, checks its expected action and hostname and rejects invalid tokens. If only one environment variable is set, requests fail closed rather than silently disabling the check.
- System prompt guidance treats user content as untrusted, and the server enforces user/assistant role alternation. No protection can guarantee that a language model will never follow a malicious prompt injection.

### Activate Turnstile (recommended for a public chat)

1. In the **Cloudflare Turnstile** dashboard create a widget for your real site domain (e.g. `your-project.vercel.app`). You do **not** have to move website hosting to Cloudflare.
2. In **Vercel → Project → Settings → Environment Variables**, add these two server environment variables for Production:
   - `AXON_TURNSTILE_SITE_KEY` = your public site key
   - `AXON_TURNSTILE_SECRET_KEY` = your private secret key (never put this in GitHub or `public/`)
3. Redeploy to apply environment variables. `/api/security` should now report `turnstileEnabled: true` and the public site key, never the secret. The chat page displays a verification widget and its server API validates the resulting token. Enter **both** keys, not just one.
4. Add Preview deployments/hostnames to your Turnstile allowed-hostnames settings if you want verification in preview URLs, or test directly on your final Production hostname. A hostname mismatch is rejected by design.

**Turnstile is OFF until both keys are set.** The basic checks still work without keys, but they cannot reliably stop determined bots.

### Enable Vercel Bot Protection (recommended even with Turnstile)

In your project's **Vercel dashboard → Firewall → Rules → Bot Management**, choose the **Bot Protection** rule. Start with **Log** to observe traffic, then choose **Challenge**, click **Review Changes** and **Publish**. For repeated abusive calls, configure an additional **Firewall** custom rule for `/api/chat` with a rate limit or challenge. These edge rules are more effective at limiting bot costs across all serverless instances than the code-only limit.

Vercel managed Bot Protection is available on all plans, but features/rate-limit usage and charges can change. Do not add a second reverse proxy in front of Vercel if you rely on Vercel bot detection without reviewing its compatibility limitations.

References:
- https://developers.cloudflare.com/turnstile/get-started/
- https://developers.cloudflare.com/turnstile/get-started/server-side-validation/
- https://vercel.com/docs/bot-management
- https://vercel.com/docs/vercel-firewall/vercel-waf/rate-limiting

**Do not treat a WAF, captcha, or string filter as a guarantee against every SQL injection, cross-site scripting, or prompt injection.** This website does not contain a SQL query endpoint; future APIs need their own parameterized database queries, authorization and validation.
