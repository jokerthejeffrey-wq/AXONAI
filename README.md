# AXON AI — one Vercel website + two Windows trainers

This folder is the **entire deployment**. It contains the public website, chat API,
training API in Vercel Functions, starter checkpoint files, and both CPU/GPU EXEs.
You do **not** deploy or keep a separate coordinator, Ubuntu server, or home PC online.

### One-time owner setup

1. Upload the entire `AXONAI_Vercel` folder to your GitHub repo at its root.
2. In Vercel, set **Root Directory = `AXONAI_Vercel`**, Framework Preset **Other**, Node.js **22.x**. Delete any legacy custom **Install Command** (do not use `npm ci` with this version); let it use `npm install`. Build command `npm run build`, Output Directory `public`.
3. Deploy. In **Vercel → Storage / Marketplace**, add **Neon Postgres** to the project, which supplies `DATABASE_URL`. You do not have to write SQL manually; API creates its tables.
4. In **Vercel → Storage**, create a **public Vercel Blob** store and connect it to the project, with the `BLOB_READ_WRITE_TOKEN` variable. This stores uploaded checkpoint pieces.
5. In Vercel → Settings → Environment Variables, create a random `AXON_OWNER_TOKEN` (64 hex digits) for reviewing updates. Do not put it on the website or in GitHub. Enable **AI Gateway** for the Vercel project. This site uses the official AI SDK, so deployed functions normally authenticate automatically with Vercel OIDC; a genuine `AI_GATEWAY_API_KEY` is an optional alternative. AI Gateway requires available credits/billing. Never use a random AXON app secret as an AI Gateway key.
6. Redeploy after connecting storage. Open `https://YOUR-WEBSITE/api/status` and confirm `"training_available":true`.
7. The trainers are already configured for `https://axonai-theta.vercel.app`. No URL needs to be entered by volunteers. Only recompile the EXEs if you change the site's address.

Volunteer experience: download `CPU-AXON.exe` or `GPU-AXON.exe`, open it, and it automatically connects to the AXON AI WEBSITE. It installs its own required runtime on first launch and starts training until closed. CPU uses NumPy without PyTorch; GPU requires NVIDIA/CUDA. Both have a conservative 90% RAM threshold.

### How training works

- `POST /api/job` atomically assigns each worker eight unique lessons in Neon.
- `GET /api/checkpoint/:version` returns a manifest of 2.5 MB chunks. Workers fetch chunks directly from the site's static assets / Vercel Blob and verify their SHA-256 checksum.
- Workers upload a trained checkpoint in small chunks to `/api/update-part`, then finalize via `/api/update`. Each job has its own short-lived random authorization token, not a global shared secret.
- Submissions stay pending owner review. Untrusted volunteers do **not** automatically rewrite the live AI.
- `/api/admin-review` is owner-only. GET lists updates; POST accepts `{ "id": "<job id>", "action": "approve" }` or `"reject"`. Approval validates numeric tensors and limits weight changes before producing a new checkpoint. Use `Authorization: Bearer YOUR_AXON_OWNER_TOKEN`. Approved student checkpoints **do not replace pretrained chat automatically**, since a verified conversation-quality evaluation/inference service is not implemented on Vercel.
- `GET /api/status` exposes non-sensitive progress metrics. Website chat uses AI Gateway and remains separate from training.

### Owner review example

```bash
curl -H "Authorization: Bearer $AXON_OWNER_TOKEN" https://YOUR-WEBSITE/api/admin-review
curl -X POST -H "Authorization: Bearer $AXON_OWNER_TOKEN" -H "Content-Type: application/json" \
  -d '{"id":"SUBMISSION_ID","action":"approve"}' https://YOUR-WEBSITE/api/admin-review
```

### Limits and important constraints

Vercel Functions are short-lived: there is no always-running training process or coordinator daemon. Neon and Blob provide persistent storage, but have provider quotas/charges; monitor storage, transfer, AI Gateway use and firewall. Vercel's request body limit is 4.5 MB, hence chunked transfers. API caps jobs to 96 new claims per UTC day and 12 pending updates to reduce unbounded public spending. Use Vercel Firewall rate limits for `/api/job` and `/api/update-part` before advertising publicly. Avoid JavaScript/CAPTCHA challenges on these EXE endpoints, because Windows workers cannot complete browser challenges.

The starting student is a small randomly initialized model trained from scratch, not a conversational model. Progress counts lessons assigned, not a GPT-3-equivalent intelligence score. Nothing here guarantees that it will become conversational merely from repeated simple lessons.

The supplied EXEs were compiled for Windows x64 here and the CPU protocol was tested against a simulated website endpoint. **The live Vercel + Neon + Blob integration, actual Windows execution, and GPU training were not tested in this environment.** The executables contain Python bytecode and can be reverse-engineered.

### Troubleshooting chat

If chat returns a service error, first check `https://axonai-theta.vercel.app/api/status` and your Vercel Function logs. Enable AI Gateway for the correct project/team and verify its credits. If you configured a random string for `AI_GATEWAY_API_KEY`, remove it or replace it with a genuine key from Vercel AI Gateway, then redeploy. The server uses automatic OIDC when Vercel provides it, so a manual key is normally unnecessary. The ZIP does **not** contain an API key or create Vercel authentication on your behalf.

