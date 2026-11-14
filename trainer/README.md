# AXON Windows trainers

CPU-AXON.exe and GPU-AXON.exe install Python and full-gradient PyTorch once, then reuse the environment. GPU installation includes CUDA libraries and is larger. Progress bars show estimated remaining time; connection speed and installation work affect the estimate.

## Fast-start data

The release builder prepares the pinned Apache-2.0 HuggingFaceTB/smol-smoltalk corpus once. Each EXE embeds a manifest with SHA-256 hashes for independently compressed pages of up to 1,024 conversations. A volunteer downloads only pages overlapping their eight assigned records, not a 1 GB dataset. Pages are verified and cached. Corrupt or interrupted pages are retried; no unverified data enters training. CPU/GPU jobs use the existing coordinator's unique record reservation sequence, with the identical canonical record order as legacy trainers.

The corpus release is `corpus-smoltalk-v1`, separate from the latest trainer release. Its immutable dataset revision is `2ca82d2fe63c017af8d4cb30e81e95cb25516851`. Source shards are hash-checked before packaging; trainer manifests are built into the release EXEs. No pyarrow installation is needed on volunteer PCs.

## Optional teacher

By default the trainer uses the original conversational lessons immediately. Add `--teacher` to download and run the local pretrained reviewer/variant generator. This uses more disk, RAM, and time. No pretrained weights are copied into the student. The two teacher checks are not proof that an example is correct.

## Model and limitations

The active student remains the 3.4M-parameter causal transformer. It trains all layers through backpropagation with a UTF-8 byte tokenizer. Python generation now caches attention keys/values and can yield text incrementally through `student.generate_stream`; this does not alter the separately deployed JavaScript chat engine. An 11,240,448-parameter compact research configuration and the prior 50M experiment are available for offline evaluation. Neither has been silently switched into the production checkpoint protocol.

Small transformers are real neural language models but are not guaranteed GPT-3/GPT-4 capability. More contributors help only when updates pass evaluation and the data, compute, capacity, and coordinator throughput are sufficient. Current server quotas remain in place to control storage/validation cost.

The RAM admission check pauses before estimated usage exceeds 90%; it cannot guarantee against every transient spike. Ctrl+C stops the trainer. Reopening reuses installed dependencies and downloaded lesson pages. No personal documents are accessed.
