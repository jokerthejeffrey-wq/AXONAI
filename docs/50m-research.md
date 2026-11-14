# AXON AI 50M research track

This branch introduces and validates a second, separately versioned transformer
architecture. It **does not** automatically upgrade the deployed 3.4M model,
claim that five volunteers will produce natural conversation in days, or enroll
existing CPU/GPU EXEs into an incompatible job protocol.

| | Live AXON model | New research model |
|---|---:|---:|
| Parameters | ~3.4 million | **50,224,640** |
| Layers | 4 | 10 |
| Width | 256 | 640 |
| Attention heads | 8 | 8 |
| Context (UTF-8 byte tokens) | 512 | 1024 |
| Checkpoint format | safetensors/F32 | safetensors/F32 |
| Checkpoint size | 13.7 MB | ~201 MB |
| Status | Running on Vercel | Architecture and converter tested on Windows |

## What works

* `settings.LARGE_CONFIG` defines a 50M-parameter transformer.
* `trainer.expanded_model.expand_legacy` transfers compatible legacy
  embeddings, attention projections, and feed-forward parameters while
  initializing the newly added layers as residual identities.
* `python -m trainer.expanded_model --from-legacy OLD.safetensors --out EXPANDED.safetensors`
  exports a complete research checkpoint using the already-installed CPU
  PyTorch and safetensors packages. This performs only a local conversion.
* CI constructs the large transformer, takes a real full-model optimizer step,
  and verifies checkpoint serialization against the actual signed legacy v0.
* The original model, existing trainers, Vercel chat, and verified volunteer
  submissions remain unchanged and functional.

## What is still required before the expanded model can be public

* A separate `axon50m` checkpoint track with its own immutable version IDs,
  coordinator jobs, upload authorization, size limits, and PostgreSQL metadata.
* A secure one-time initialization of the ~201 MB baseline into durable Blob
  storage. It must not be included in the current Hobby static deployment
  source upload; that limit is 100 MB.
* Bounded-memory validation, loss-based acceptance on genuinely held-out
  language tasks, and safe aggregation of competing updates from multiple PCs.
* Incremental/delta checkpoint uploads, retention/garbage collection, and
  explicit storage/egress budgets. Five simultaneous full-checkpoint uploads
  are about a gigabyte before any merged model is stored.
* A performance-tested inference implementation for Vercel, or separate
  appropriate compute. Node's reference byte-token interpreter can be too
  slow for a 50M-parameter model, and 50M full-precision weights must fit in
  function memory.
* Tests on at least five actual volunteer PCs, including CPU-only machines,
  mixed hardware, interrupted uploads, teacher quality, RAM ceilings, and
  restart recovery.

Only after these checks should production switch models. Future claims about
ordinary conversation must be based on held-out conversation tests and human
evaluation, not parameter count or elapsed days.

## Why a staged upgrade

Five people can supply more CPU/GPU time, but training doesn't automatically
add parameters or guarantee understanding. A 50M model randomly initialized
or widened from a tiny model may initially reply worse than the current one.
Vercel's current code assumes the old checkpoint shape and chunk count; changing
the config alone would break every active CPU/GPU worker. Releasing the new
research track separately avoids discarding any approved volunteer work.
