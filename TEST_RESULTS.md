# Latest assisted-chat / Vercel update

- All 13 automated tests passed, including community-creator answers, assisted routing, and student-only isolation.
- Real Qwen inference on CPU produced "Hello! How can I assist you today?" through the new chat engine. No student model was loaded for the assisted response.
- Root Vercel build succeeded with a test HTTPS backend origin. The bundled static client ZIP and source manifest are included.
- Wave animation, smooth theme transitions and centered SVG send button are preserved.
- Live Vercel hosting, Windows installer execution, GPU execution and browser visual rendering remain unverified.

The remaining results below describe the separately tested student-training pipeline.

# Validation — from-scratch edition

Tested on Linux CPU with PyTorch 2.7.1, the version selected by the installer.

- Twelve automated tests: authentication, invalid upload rejection, causal attention (future tokens cannot influence earlier predictions), integrity manifest, CORS/download endpoints, four-use greeting group, simultaneous client budget enforcement, preserved budgets after data reimport/answer edits, duplicate/stale upload rejection, student gradients and tensor round trip, absence of teacher fallback, and source-repair hash acceptance/rejection.
- The actual 3,423,744-parameter architecture was initialized randomly and trained for eight optimizer steps using reviewed teacher-generated lessons.
- Full checkpoint upload, validation-gated merge and publication succeeded. Holdout loss changed from 5.52737 to 4.37255 in this small smoke test. This does NOT establish general ability or meaningful conversational quality.
- The published student generated a response through the Flask chat API without importing Transformers/the teacher. Early output was not a capable assistant answer; no trained weights are distributed in this archive.
- Rollback published a new version and preserved all consumed lesson budgets.
- The actual pretrained teacher generated 48 curriculum drafts. AI-assisted review corrected errors before the supplied training examples were finalized. Raw drafts are separate from training data. Newly generated drafts require REVIEW.py approval/editing before import.
- All 48 supplied training examples fit the 512-byte/control-token context.
- Python files compiled, JavaScript syntax checks passed, and the Vercel build generated configuration from a valid HTTPS backend origin.

Not verified here:
- Windows winget/Python installation and native PowerShell execution.
- NVIDIA CUDA execution/driver compatibility.
- Live Vercel deployment, a real public HTTPS tunnel or production concurrency/load.
- Browser visual rendering, which this execution sandbox blocks. No visual QA claim is made.
- Fully unattended installation on every OS/network; bootstrap repair has defined limits documented in README.md.

Run tests in the installed CPU environment:

Windows:
```
.axon-cpu\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```
Linux:
```
.axon-cpu/bin/python -m unittest discover -p "test_*.py" -v
```
