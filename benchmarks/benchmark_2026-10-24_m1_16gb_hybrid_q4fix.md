# TinyGiant hybrid Metal Q4 fix vs cpu-only — Apple M1 16GB

**Date:** 2026-10-24 (America/New_York)  
**Commit:** `6cb9a14` ([PR #8](https://github.com/jerryjokesalot/tinygiant/pull/8))  
**Companion:** [`benchmark_2026-10-24_m1_16gb_hybrid.md`](benchmark_2026-10-24_m1_16gb_hybrid.md) (prior FP16 Metal baseline)  
**Root-cause notes:** [`metal_hybrid_rootcause_2026-10-24.md`](metal_hybrid_rootcause_2026-10-24.md)

## Setup

Same as prior hybrid bake-off: Qwen3-30B-A3B Q4_K_M, pin_nonuniform budget 2304, `n_gen=128`, reps=3 × 5 tasks, calibrate_n=10. Metal used on **all** hybrid trials (`use_metal_attn=True`). Upload: **Q4_K Q/K/O + FP16 V** (~555 MB) — no FP16 expand of Q/K/O.

## Primary results (sustained decode)

| Configuration | Decode tok/s | Wall tok/s | Metal | vs prior hybrid |
|---|---:|---:|:---:|---:|
| cpu-only + pin_nonuniform | **3.01 ± 0.79** | 2.50 ± 0.63 | no | — |
| hybrid Q4 Metal (this fix) | **1.15 ± 0.23** | 1.04 ± 0.22 | yes | +26% abs vs FP16 hybrid 0.91 |
| *prior* hybrid FP16 (baseline) | 0.91 ± 0.24 | 0.80 ± 0.21 | yes | — |

**Speedup hybrid / cpu-only:** **0.38×** (still loses). Prior FP16 was 0.34×.

**Verdict:** Highest-leverage fix (drop FP16 expand, keep Q4 on Metal) improved hybrid absolute decode tok/s (~0.91 → ~1.15) but did **not** beat cpu-only. Do **not** flip `HYBRID_ATTN_READY`.

## Per-task decode tok/s (mean ± std, n=3)

| Task | cpu-only | hybrid Q4 Metal |
|---|---:|---:|
| coding | 2.45 ± 0.15 | 0.98 ± 0.06 |
| reasoning | 3.54 ± 0.13 | 1.12 ± 0.05 |
| creative | 3.50 ± 0.40 | 1.31 ± 0.02 |
| math | 1.79 ± 0.19 | 0.86 ± 0.04 |
| factual | 3.74 ± 0.19 | 1.48 ± 0.06 |

## Residual (why still slow)

1. **Per-layer GPU sync** — MoE on CPU forces `waitUntilCompleted` after each of 48 layers.
2. **Metal Q4×Q8 still ~3 ms/layer** (instrumented smoke) vs highly tuned NEON `vdotq` on CPU.
3. **gqa_attn** triple-pass O(seq) on GPU; grows with decode length (full 128-tok runs worse relative than short smoke).
4. **16 GB unified memory** — pins (~6.1 GiB) + embd/out + Metal buffers contend.

## Recommendation

| Question | Answer |
|---|---|
| Flip `HYBRID_ATTN_READY`? | **No** |
| Keep `--profile hybrid`? | Yes (opt-in / experimental) |
| Ship this PR? | Yes — correcter Metal path + honest numbers; no auto-select |

Raw JSON: `benchmarks/bakeoff_hybrid_raw_q4fix.json`  
Log: `benchmarks/bakeoff_hybrid_q4fix.log`
