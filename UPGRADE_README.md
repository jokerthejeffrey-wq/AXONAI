# AXON AI Chat repair

The earlier Qwen ONNX model downloads 786MB into a temporary Vercel function. This build replaces it with a server-side AI Gateway request (no local model weights). Enable AI Gateway for your Vercel project; it authenticates through Vercel OIDC. You can instead use a private `AI_GATEWAY_API_KEY` environment variable. Select another supported model with `AXON_GATEWAY_MODEL`.

Do **not** set a coordinator URL to your Vercel chat domain. The separate student training coordinator is independent and is not hosted by the site. The pretrained helper stays active until a verified student is approved and a serving backend is configured.
