# AXON AI · local verification

- Rebuilt both Windows x64 console EXEs, now defaulting to https://axonai-theta.vercel.app.
- `npm run build`: passed.
- `npm test`: 32/32 local tests passed using temporary test-only stubs for Neon Postgres and Vercel Blob. No external services were accessed.
- Gateway unit tests include the AI SDK call shape, automatic OIDC eligibility, missing local auth, and structured error codes.
- Checked updated server JavaScript with `node --check`.
- This environment could not install real AI SDK dependencies or test a live AI Gateway/Neon/Blob connection. Vercel deployment, billing, and actual Windows runtime remain unverified.
