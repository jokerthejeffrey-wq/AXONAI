# Local verification

- `node --test test/*.test.mjs`: 29 passing tests using local **test-only stubs** for Neon and Vercel Blob (no cloud access).
- `node scripts/build.mjs`: success.
- Both CPU-AXON.exe and GPU-AXON.exe cross-compiled as Windows x64 console binaries.
- Embedded Python loader smoke tests: CPU and GPU passed.
- Full CPU session against **simulated** website API: downloaded 6 model chunks, trained a real lesson, uploaded 6 chunks and finalized a valid SHA-256 manifest.
- **Not verified:** Vercel deployment, real Neon/Blob connections, EXE execution on Windows, GPU training, multi-user production concurrency.
