# AXON Windows trainers

Both `CPU-AXON.exe` and `GPU-AXON.exe` start automatically after a volunteer launches them. On first run they install and cache Python, full-gradient PyTorch, the conversation dataset, and the local pretrained teacher used to propose and review additional examples.

The console shows `[=====               ] 25%  remaining: ~2m` for measured downloads and staged progress with approximate ETA for package setup. A first installation can download several gigabytes and needs additional disk space for the dataset index/cache.

The local dataset is the pinned, Apache-2.0 HuggingFaceTB/smol-smoltalk corpus. Its measured 1 GB cache has 213,571 deduplicated examples. Each participant receives a unique range from Neon. The local teacher generates variants from assigned records and must pass both a source-consistency check and a separately prompted factual/clarity check. The two checks may still agree on an incorrect answer.

The student uses a UTF-8 byte tokenizer and all model layers are updated through PyTorch backpropagation. No API-based teacher is called; the website chat uses the separately trained AXON model weights. The model is small and is not guaranteed to reach the quality of a model served by Ollama.

CPU uses CPU-only PyTorch. GPU uses CUDA PyTorch and a compatible NVIDIA driver. Trainer admission checks pause if predicted RAM usage would exceed 90%, but cannot guarantee against every transient spike. Run `CPU-AXON.exe` or `GPU-AXON.exe`; use Ctrl+C to stop. Reopening uses cached dependencies and dataset. No personal documents are accessed for training.
