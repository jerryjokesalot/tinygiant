# TinyGiant hybrid (Metal attn+KV) vs cpu-only — Apple M1 16GB

**Date:** 2026-10-24 (America/New_York)  
**Commit:** `b317b01` (PR #7 merged)  
**Companion to:** [`benchmark_2026-10-24_m1_16gb.md`](benchmark_2026-10-24_m1_16gb.md) (TinyGiant vs llama.cpp CPU)

## Setup

| Item | Value |
|---|---|
| Hardware | Apple M1, 16 GiB unified memory, 8 cores |
| OS | macOS 26.7 (Build 25G229) |
| Model | `~/models/Qwen3-30B-A3B-Q4_K_M.gguf` → ollama blob (17.28 GiB, Q4_K_M) |
| TinyGiant | commit `b317b01`, fused Q4 NWS cache `/tmp/nws_q4_cache_v2` → `~/.tinygiant/cache` |
| Pins | non-uniform (14–92/layer, budget 2304) from concentration profile |
| Competing services | `com.tinygiant.server` unloaded; Ollama not running |

### Metric definitions

- **Decode tok/s (preferred):** `n_gen / decode_only_seconds` after prefill.
- **Wall tok/s:** `n_gen / (prefill + decode)`.

Harness: `experiments/bakeoff_hybrid_vs_cpu.py`  
Raw JSON: `benchmarks/bakeoff_hybrid_raw.json`  
`n_gen=128`, reps=3 per prompt × 5 tasks (coding / reasoning / creative / math / factual), calibrate_n=10, temperature 0.7 / top_p 0.9, seeds 42+rep.

**Metal verification:** every hybrid trial recorded `use_metal_attn=True` (Metal upload ~1728 MB FP16 attn; host attn+KV released). cpu-only trials all `use_metal_attn=False`.

## Primary results (sustained decode)

| Configuration | Decode tok/s | Wall tok/s | Metal used | Engine load |
|---|---:|---:|:---:|---:|
| TinyGiant `--profile cpu-only` + pin_nonuniform | **2.70 ± 0.81** | 2.27 ± 0.68 | no | 6.5 s |
| TinyGiant `--profile hybrid` (Metal attn+KV) + pin_nonuniform | **0.91 ± 0.24** | 0.80 ± 0.21 | yes (all 15) | 11.7 s |

**Speedup hybrid / cpu-only:** **0.34×** (hybrid is ~3× *slower*).

**Verdict:** On this M1 16GB machine, Slice A Metal attention+KV does **not** beat the cpu-only path. Do **not** flip `HYBRID_ATTN_READY`. Keep default `cpu-only`; hybrid remains opt-in / experimental.

## Per-task decode tok/s (mean ± std, n=3)

| Task | cpu-only | hybrid (Metal) |
|---|---:|---:|
| coding | 1.87 ± 0.12 | 0.65 ± 0.10 |
| reasoning | 3.01 ± 0.15 | 0.93 ± 0.02 |
| creative | 3.35 ± 0.32 | 1.10 ± 0.04 |
| math | 1.66 ± 0.12 | 0.67 ± 0.02 |
| factual | 3.59 ± 0.22 | 1.22 ± 0.05 |

Hybrid lost on **every** task.

## Interpretation (honest)

- Metal path is numerically close to CPU on synthetic tests (`make test-metal` PASS, max_abs ≈ 2e-6) and did run for all hybrid trials — this is a **performance** loss, not a silent CPU fallback.
- Likely causes on 16 GiB: FP16 attention weight expand (~1.7 GiB on Metal) competing with pinned experts (~6.1 GiB) + embeddings/output head; Metal matvecs for attention alone while MoE experts stay on CPU/SSD; possible sync overhead per token.
- Primary public claim remains **cpu-only TinyGiant vs llama.cpp CPU** (~2.5–2.6× from the earlier bake-off). Hybrid is not ready to advertise.

## Recommendation

| Question | Answer |
|---|---|
| Flip `HYBRID_ATTN_READY`? | **No** |
| Auto-select hybrid in `tinygiant-setup`? | **No** |
| Keep `--profile hybrid` flag? | Yes (opt-in, fail-open documented) |
| Next engineering if pursuing GPU | Profile where time goes (attn vs MoE); consider Q4 matvecs on Metal without FP16 expand; or hold Slice A until memory/layout improves. Slice B (experts-on-GPU) still held. |

## What was not claimed

- No tok/s claims for hybrid in README or setup auto-recommend.
- No comparison to ollama/MLX Metal full-offload (not a fair peer for this hybrid design).
