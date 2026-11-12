# AXON AI — Chat backend replacement

The old local Qwen/SmolLM2 downloads have been removed from this website package.
The new chat routes to `openai/gpt-4.1-mini` using Vercel AI Gateway; nothing needs to be installed on a home PC.

If your deployed site still says the model cannot generate a reply, you may be deploying the wrong GitHub folder, using an old Vercel deployment, or missing Gateway authentication/credits.
Read UPLOAD_TO_GITHUB.txt and visit `/api/status` after redeploying.

A Vercel deployment is stateless: the old ephemeral model cache cannot be deleted by site code. A fresh deployment replaces the code and no longer downloads those weights. This project intentionally does not include a dangerous remote delete command.
