# Host the website at your-name.vercel.app

This project uses Vercel for the static frontend and a persistent Python backend for pretrained-assisted/student inference, checkpoints and volunteer training coordination. Vercel automatically assigns a .vercel.app address based on project availability. You do not need to buy a domain to use that address.

Do not upload HTML.py as a Vercel Function for this project: full student uploads are about 14 MB, exceeding the documented 4.5 MB function request limit, and this coordinator relies on persistent SQLite/files. The frontend calls the backend directly, so checkpoints do not pass through Vercel Functions.

## 1. Start the Python backend

On your PC, after setup/training:

```
python HTML.py
```

For a free temporary demo, install cloudflared from the official Cloudflare instructions and run in a second terminal:

```
cloudflared tunnel --url http://localhost:5000
```

Keep both processes and the PC running. The resulting HTTPS trycloudflare.com origin is the backend URL. Quick Tunnel URLs can change on restart and are for development. For an ongoing project, use a persistent VPS and stable HTTPS hostname or a named Cloudflare Tunnel. You still pay for your own computer/electricity or VPS; Vercel frontend hosting does not supply free training GPUs.

## 2. Deploy the frontend

1. Put the extracted project in your own GitHub repository. Do NOT commit student_state, environments, secrets or teacher model caches; .gitignore is supplied.
2. In Vercel, choose Add New Project and import that repository.
3. Set **Root Directory** to the folder containing the top-level vercel.json and HTML.py (the extracted AXONAI_Scratch folder).
4. Framework preset: **Other**. Node version: 22 or newer.
5. Build command: `node vercel/build.mjs`. Output directory: `vercel/public`. These are also declared in vercel.json.
6. Add environment variable **AXON_API_URL** with your backend origin, e.g. `https://YOUR-BACKEND.trycloudflare.com`. No trailing API path, credentials or query string. It is public configuration, not a secret.
7. Deploy. Vercel assigns a `your-project.vercel.app` address. The exact name depends on availability.

The build deliberately fails when AXON_API_URL is missing or invalid, instead of publishing a chat UI with a silently broken connection.

## 3. Allow that frontend on the backend

Stop the Python server, set the exact deployed origin, and restart.

Windows PowerShell:

```powershell
$env:AXON_ALLOWED_ORIGINS="https://your-project.vercel.app"
python HTML.py
```

macOS/Linux:

```sh
export AXON_ALLOWED_ORIGINS="https://your-project.vercel.app"
python HTML.py
```

You may comma-separate several EXACT origins for your preview deployments. Do not use `*` or allow every vercel.app subdomain. For PyCharm, put this variable in the HTML.py Run Configuration environment.

Test the website: the status cards should load; the download button should fetch the bundled static client ZIP; /chat should reach the configured backend. Default chat is Qwen-assisted and works before a student checkpoint exists. Student-only mode requires a published checkpoint.

## 4. What address contributors use

CPU.py / GPU.py need the **Python backend** URL, not the static Vercel frontend address. The public /docs page shows commands using the configured backend URL. Chat users stay on your .vercel.app website; API traffic is sent over HTTPS to the backend.

If your temporary backend URL changes, update AXON_API_URL in Vercel and redeploy. Contributors must use the new backend URL too. A stable backend hostname avoids this.

## Optional VPS Docker deployment

Generate/review teacher data on your own computer first. Copy the project (including reviewed data/train.jsonl) to a Linux host with Docker. The Dockerfile includes CPU and Transformers dependencies for assisted inference and skips interactive bootstrap. Qwen weights download on first use; give the container several GB of RAM.

```sh
docker build -t axon-student .
docker volume create axon-student-state
docker run --rm -v axon-student-state:/app/student_state axon-student python manage.py init
docker run -d --name axon-student --restart unless-stopped -p 127.0.0.1:5000:5000 -e AXON_ALLOWED_ORIGINS=https://YOUR-PROJECT.vercel.app -v axon-student-state:/app/student_state axon-student
```

Place a stable HTTPS tunnel or TLS reverse proxy in front of localhost:5000. Back up the state volume. Stop the website container before owner merges:

```sh
docker stop axon-student
docker run --rm -v axon-student-state:/app/student_state axon-student python manage.py list
# Replace UPDATE_ID with a pending ID:
docker run --rm -v axon-student-state:/app/student_state axon-student python manage.py merge UPDATE_ID
docker start axon-student
```

When you add reviewed training data, rebuild the image, stop the server and run `python manage.py import` in a one-off container using the same state volume. Never run multiple backend processes sharing this SQLite coordinator.

## Official references checked 12 November 2026

- Vercel domains: https://vercel.com/docs/domains/working-with-domains
- Vercel function limits: https://vercel.com/docs/functions/limitations
- Cloudflare Tunnel: https://developers.cloudflare.com/tunnel/get-started/
- PyTorch CPU/CUDA builds: https://pytorch.org/get-started/previous-versions/

This archive prepares deployment; it has not created a Vercel account, deployed a live website or reserved a project name for you.
