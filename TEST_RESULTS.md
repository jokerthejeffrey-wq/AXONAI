# AXON AI hosted-chat repair — local verification

- Node.js 22; `npm test`: 25/25 passed.
- `npm run build`: passed with no AI model download.
- Static checks: No ONNX/safetensors/PyTorch files or legacy local-model loader in the new website package.
- Hosted chat API: simulated successful AI Gateway reply passed.
- Missing AI Gateway credentials: returns a specific configuration error without calling a model.
- Public website: CPU and GPU EXE downloads preserved.
- Same-origin request checks, rate limits, input validation, and optional Turnstile tests passed.

**Not verified:** Live Vercel deployment, production OIDC/AI Gateway access, account credits, and Windows EXE execution. An owner must enable AI Gateway or set an API key in Vercel project settings, and deploy to the correct Root Directory.
