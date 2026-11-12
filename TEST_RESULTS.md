# Verification

- Clean dependency installation using `node scripts/install.mjs`: passed; CUDA downloads skipped.
- `npm test`: 5/5 API tests passed.
- `npm run build`: passed; browser requests use same-origin API routes.
- Actual quantized model download and CPU text generation: passed locally, about 22.4 seconds including startup and approximately 650 MiB process RSS. The small model produced a weak response; generation success does not establish conversational quality.
- Local dependency tracing with @vercel/nft: approximately 78.6 MiB uncompressed after configured exclusions, below the documented 250 MB function bundle limit. Trace emitted optional Sharp/WASM and license parsing warnings; this estimate is not a completed Vercel deployment build.
- Local runtime: Node 24.22.1 on Linux x64; deployment targets Node 22.x. npm reported the expected local engine mismatch warning.
- No live deployment on your Vercel account was performed. Cold-start behavior, quotas and runtime compatibility must be confirmed after deployment.
- This edition hosts chat only. Distributed training/checkpoint persistence requires separate infrastructure.

## November 12 modifications

- Vercel/API tests: 8/8 passed, including cgroup-aware 90% RAM admission and creator attribution.
- Coordinator tests: 5/5 passed (32 simulated concurrent workers, 256 distinct prompt/answer pairs; persistence, immutable reservations, upload idempotency and checkpoint merge).
- `npm run build` passed with Node 22.16.0.
- Coordinator HTTP protocol and one complete CPU worker training/upload session passed locally; Windows Nuitka build and live Vercel deployment have not been verified yet. No actual `.exe` files are shipped.

## Security revision

The security test suite covers same-origin restrictions, malformed and oversized requests, unexpected payload fields, rate limits, secret non-disclosure, verification success/failure, action/hostname checks, and fail-closed configuration. All baseline website and memory tests are preserved. Dynamic Turnstile challenges and Vercel WAF deployment require live testing on the actual Vercel hostname.
