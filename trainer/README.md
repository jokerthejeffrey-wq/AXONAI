# AXON conversational trainer

This trainer uses the website's existing Neon/Blob coordinator. It does not require the owner's PC to stay on. The old Windows executables are legacy arithmetic workers; they do not contain this new trainer.

Use Python 3.11 or 3.12. From the extracted project folder:

CPU: `python trainer/start.py`

NVIDIA GPU: `python trainer/start.py --gpu`

The launcher installs pinned CPU/CUDA PyTorch and teacher/data dependencies into a separate environment, then starts the console trainer. The console asks before corpus/model downloads and compute. Downloads can total several GB including dependencies, source dataset caches, the teacher, and checkpoints.

## Data and shared progress

The first run streams the pinned Apache-2.0 HuggingFaceTB/smol-smoltalk dataset into an indexed UTF-8 `training.txt` file under `~/.axonai/smoltalk-v1`. Each line is a real JSON conversation. The target is at most 1,000,000,000 bytes; the file stops at a record boundary or when the eligible dataset is exhausted. It is not padded. `corpus.json` reports actual bytes, lines, checksum, source and revision. This dataset does NOT contain 15 million independent eligible examples. A 15-million-line claim would be misleading. Do not infer intelligence from file size or number of lines.

Source: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/2ca82d2fe63c017af8d4cb30e81e95cb25516851
Dataset license: Apache-2.0, retained by the original authors. Teacher: Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0). Student initialization is unchanged; teacher weights are never copied into it.

Corpus construction deduplicates the normalized short conversation context used for training. All clients use the same revision, normalization and record order. A separate atomic database counter reserves non-overlapping record ranges across PCs. These ranges are never rewound, even after a failed/skipped job. This is at-most-once assignment, not guaranteed successful training. Do not edit the corpus/cache or reset coordinator counters. The cache/index requires extra disk space and rebuilding an incomplete download can repeat network traffic.

Each worker uses the pretrained local teacher to produce short responses for its assigned conversations. Multi-turn contexts are preserved when they fit the student's 512-byte context. Oversized/invalid targets are skipped rather than cut mid-answer. Teacher output can still be wrong; updates must pass the existing coordinator's validation before promotion. Current validation is lightweight and is not a comprehensive quality or poisoning defense.

The 90% RAM check pauses before loading/processing when reserve estimates do not fit. It is admission control, not a hard OS memory cap. CPU teacher generation can be slow. NVIDIA driver installation is not automated. Ctrl+C stops. A teacher of this size needs several GB of spare RAM; waiting cannot fix insufficient total RAM.

## Chat behavior

The model already has a UTF-8 byte tokenizer and causal next-token prediction. The update adds temperature/top-k/top-p sampling, recent-token repetition penalties, and actual NDJSON token streaming from the existing JavaScript inference implementation. UTF-8 bytes are decoded incrementally so multi-byte characters are not split across displayed fragments. No pretrained teacher is called by chat. Identity and calculator tools remain explicit deterministic app features. Version-zero fallback remains honest about the untrained model; promoted versions stream experimental student output. Streaming is not a demonstration of reasoning quality.

The student is only about 3.4 million parameters with short context. This change cannot make it ChatGPT-level or guarantee coherent conversation. Training quality, enough accepted optimization steps, and independent conversational evaluation are still needed. Existing daily job and queue limits remain to control hosted costs; more PCs do not remove those quotas.

## Checks

`npm test` covers sampling, actual checkpoint streaming, memory, API/security and previous regressions. `python -m unittest trainer.test_corpus` covers deterministic cache, deduplication, no padding, and multi-turn extraction. These are functionality tests, not proof of language quality.

Actual smoke verification: downloaded a 19-conversation sample from the pinned dataset; downloaded and ran the local Qwen teacher; trained the full student for one step and verified that its weights changed. GPU operation and Windows launchers have not been tested here.

Full corpus verification: 999,994,848 bytes (approximately 1 GB), 213,571 deduplicated conversation records. This is the measured dataset size, not 15 million examples. Data readers are pinned and iterators explicitly closed to avoid a native-library shutdown error observed with newer unpinned dependencies.

Live integration test: reserved records 0–7 from the deployed coordinator, generated teacher targets, trained all 8 examples, downloaded checkpoint v1, and uploaded the result successfully for validation. This submission was not manually forced into production weights. The production chat endpoint also returned HTTP 200 NDJSON with student-version metadata, text deltas and completion.
