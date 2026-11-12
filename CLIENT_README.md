# AXONAI console trainer — from-scratch student

Windows: extract the WHOLE ZIP, then double-click START_CPU.cmd or START_GPU.cmd. The launcher can offer to install Python 3.12 using winget if missing. It then asks to create an isolated environment and install packages. Allow several GB of free disk space. GPU training requires an NVIDIA driver compatible with the included CUDA 12.8 build; the program does not install drivers.

If Python 3.11/3.12 is already installed (PyCharm/Linux/macOS):

```
python CPU.py --server https://YOUR-BACKEND
# or NVIDIA GPU:
python GPU.py --server https://YOUR-BACKEND
```

Use the Python BACKEND URL, not your static .vercel.app frontend URL. The website's /docs page shows the configured backend address. Local testing: http://127.0.0.1:5000.

Training asks for consent, downloads student weights (~14 MB) and teacher-written lessons, then performs ONE gradient step per assigned lesson. One job has at most eight lessons. CPU defaults to two threads; --threads 4 changes this. --rounds 3 requests three jobs. --yes skips compute consent only; installation still asks. Ctrl+C stops; failed/expired work consumes its reserved uses. Updates await owner review.

The pretrained teacher runs on the OWNER'S machine, automatically downloaded by TEACHER.py. Volunteers only need the small student and its lessons. The website now uses labeled pretrained assistance until the owner switches to the student. This does not change contributor training.

Repair: the bootstrap checks client files against client_manifest.json. It can offer to restore missing/changed shared source files from your trusted backend ONLY when their SHA-256 matches the manifest shipped with your ZIP. Restoring overwrites local edits only after consent. If the launcher/bootstrap or manifest itself is broken, re-download the complete ZIP. Package import failures trigger one automatic reinstall attempt after installation consent. It cannot fix all Python, OS, hardware, antivirus, disk or network problems.

No personal files or chats are collected. Checkpoints and trained weights are uploaded to your chosen backend. This is early research code; a randomly initialized student is not immediately intelligent.
