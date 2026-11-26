# Real-Model GPU Memory Allocation/Free Sequences

**Purpose**: to facilitate independent verification of memory pool behavior
(memory utilization / performance evaluation). Every trace here is a real
malloc/free sequence captured from an actual training run, so a memory pool or
allocator can be evaluated by offline replay alone, without rerunning any model.

## Quick Start

```bash
cd <trace_dir>

# Usage
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python ../replay_malloc_free.py malloc_free_trace.csv
```

Options — the switches are independent and may be combined:

| Switch | Effect | Default |
|---|---|---|
| `--profile` | Enable `_record_memory_history("all")`, i.e. populate the snapshot's `device_traces` | off |
| `--snapshot [PATH]` | Dump a memory_viz-readable pickle at the end, **implies `--profile`**; with no path given, writes to `<trace>.snapshot.pickle` | off |
| `--last_step N` | Replay only rows with `step <= N`. The build phase is `-1` and is always included when `N >= -1`; `--last_step -1` replays the build phase alone | all rows |
| `--no-step-report` | Silence the per-step output line | per-step output is **on by default** |

## Catalog

| Trace directory | CSV rows | peak_allocated (GiB) | peak_reserved (GiB) | Description |
|---|---|---|---|---|
| `deepseek_moe` | 343105 | 30.0589 | 32.9746 | DeepSeek-V2-lite architecture (MLA + 64 routed experts top-6 + 2 shared, layers reduced from 27 to 12), 2-GPU FSDP2, bf16, AdamW, activation checkpointing, dynamic seq∈{512..1024}, mbs=1, 6 steps |
| `llama3b_dyn` | 142835 | 48.4807 | 52.4863 | Llama-3.2-3B, single GPU, bf16, AdamW, activation checkpointing, dynamic seq∈{1024..8192}, mbs=2, 10 steps |
| `llama3b_dyn_hi` | 143283 | 52.2601 | 56.5293 | Llama-3.2-3B, single GPU, bf16, AdamW, activation checkpointing, dynamic seq∈{1024..6144}, mbs=3 (high-pressure), 10 steps |
| `llama8b_dyn` | 177248 | 38.3906 | 42.6445 | Llama-3-8B, 2-GPU FSDP2, bf16, AdamW, activation checkpointing, dynamic seq∈{1024,1536,2048,2560,3072,3584,4096}, mbs=1, 10 steps |
| `llama8b_dyn_hi` | 144128 | 40.7826 | 50.5723 | Llama-3-8B, 2-GPU FSDP2, bf16, AdamW, activation checkpointing, dynamic seq, mbs=2 (high-pressure), 8 steps |
| `llama8b_static` | 178916 | 38.3906 | 42.6406 | Llama-3-8B, 2-GPU FSDP2 + per-layer fully_shard, bf16, AdamW, activation checkpointing, static seq=4096, mbs=1, 10 steps |
| `qwen1_5b_dyn` | 143283 | 42.7130 | 45.3262 | Qwen2.5-1.5B, single GPU, bf16, AdamW, activation checkpointing, dynamic seq∈{1024..8192}, mbs=2, 10 steps |
| `qwen1_5b_dyn_hi` | 142835 | 59.7542 | 60.3652 | Qwen2.5-1.5B, single GPU, bf16, AdamW, activation checkpointing, dynamic seq∈{1024..8192}, mbs=3 (high-pressure), 10 steps |
| `qwen7b_dyn` | 160242 | 36.6415 | 42.3320 | Qwen2.5-7B, 2-GPU FSDP2, bf16, AdamW, activation checkpointing, dynamic seq∈{512..4096}, mbs=1, 10 steps |
| `qwen7b_static` | 162096 | 36.6415 | 42.3477 | Qwen2.5-7B, 2-GPU FSDP2, bf16, AdamW, activation checkpointing, static seq=4096, mbs=1, 10 steps |
| `kimik3` | 1462551 | 42.7219 | 49.1445 | **kimiK3 real production run** (8 GPUs, rank0, all streams), 10 steps. **All sizes uniformly scaled by 0.75**: peak reserved drops from 60.80 GiB to 49.14 GiB, so a single-GPU replay will not OOM. The two columns are measured from replaying the scaled CSV. This directory contains only this one CSV |

Where the two columns come from: for the 10 locally-run models they are
measured values from the **original training run** (replay reproduces
`peak_allocated` bit-for-bit); for `kimik3` they are **replay measurements of
the scaled CSV** (the original-size peak reserved is 60.7969 GiB, which leaves
no headroom for a single-GPU replay, so this directory ships the 0.75-scaled
version).

## Contributing

**Contributions are very welcome!** To contribute a trace from any real
training run: extract the malloc/free sequence from memory profiling or from a
memory snapshot, verify it by replay, then open a PR.

### CSV format — 4 columns

| Column | Meaning |
|---|---|
| `op` | `malloc` or `free` |
| `size` | allocation size in bytes |
| `addr` | device address; the identity key that pairs each `free` with its `malloc` |
| `step` | training step the row belongs to.  |

Rows must be in chronological order.
