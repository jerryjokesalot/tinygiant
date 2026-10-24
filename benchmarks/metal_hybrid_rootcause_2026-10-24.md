# Metal hybrid root-cause (M1 16GB) — 2026-10-24

**Branch:** `fix/metal-hybrid-perf` → [PR #8](https://github.com/jerryjokesalot/tinygiant/pull/8)  
**Baseline bake-off:** `benchmark_2026-10-24_m1_16gb_hybrid.md` (FP16 Metal, **0.91 vs 2.70 tok/s**, **0.34×**)  
**`HYBRID_ATTN_READY`:** remains **False**

## Ranked hypotheses

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | FP16 expand of Q/K/O (~1.7 GiB) vs CPU Q4 (~0.55 GiB) | **Primary — fixed** | Prior hybrid uploaded dequantized FP16; this branch uploads Q4_K Q/K/O + FP16 V (555 MB). Smoke decode rose ~0.91→~1.1 tok/s; greedy tokens match cpu-only 8/8. |
| 2 | Per-layer `waitUntilCompleted` (48×/token) | **Structural** | MoE experts stay on CPU between layers; residual needs attn out before expert FFN. Cannot remove without redesigning the forward pass. |
| 3 | Naive / scalar Metal matvecs | **Confirmed — mitigated** | Original `matvec_f16` was scalar half×float. Added SIMD half4 + Q4×Q8 integer Metal path matching CPU NEON algorithm. Still ~3.1–3.2 ms/layer Metal attn wall (instrumented). |
| 4 | Host re-touching weights/KV each step | **Ruled out** | After migrate, host attn+KV released; Metal status true; greedy match proves no silent CPU fallback. |
| 5 | Unified-memory contention with pin_nonuniform (~6.1 GiB) | **Plausible residual** | 16 GiB machine: embd+out (~2.4 GiB) + pins (6.1) + attn (0.55) + KV + OS leaves little headroom; GPU+CPU both hit the same DRAM. |

## Instrumentation

`tg_metal_timing` / `tg_metal_timing_reset`: cumulative wall ms inside `tg_metal_attention_decode` (includes GPU wait).

Smoke (16 decode toks, pin48 non-uniform, same prompt):

| Config | Decode tok/s | Metal attn ms/tok | Metal ms/call (÷48 layers) |
|---|---:|---:|---:|
| cpu-only | ~1.45 | — | — |
| hybrid Q4 (this PR) | ~1.08–1.14 | ~147–156 | ~3.1–3.2 |

→ Metal attention alone is ~150 ms/token; cpu-only total is ~690 ms/token, so Metal attn is a large but not sole share — residual is sync + weaker GPU Q4 vs NEON vdot + memory pressure.

## What changed in code

- `tg_metal_upload_layer_q4` + engine prefers it when `attn_q4`
- Metal shaders: `matvec_q4k`, `quantize_q8`, `matvec_q4k_q8`, SIMD `matvec_f16`
- Timing API for bake-offs / profiling
- cpu-only path unchanged; no flip of `HYBRID_ATTN_READY`

## Verdict (pending full bake-off JSON)

Expect hybrid to remain **slower than cpu-only** but **much closer than 0.34×**. Do **not** auto-select hybrid. Optional: further SIMD/threadgroup Q4×Q8, or park Slice A as experimental.

## Full bake-off (final)

See [`benchmark_2026-10-24_m1_16gb_hybrid_q4fix.md`](benchmark_2026-10-24_m1_16gb_hybrid_q4fix.md).

| Config | Decode tok/s | Speedup vs cpu |
|---|---:|---:|
| cpu-only (this run) | **3.01 ± 0.79** | 1.00× |
| hybrid Q4 Metal (this PR, `6cb9a14`) | **1.15 ± 0.23** | **0.38×** |
| *prior* hybrid FP16 (`b317b01`) | 0.91 ± 0.24 | 0.34× |

Absolute hybrid gain vs FP16 baseline: **~+26%** (0.91 → 1.15). Still loses to cpu-only on every task. `HYBRID_ATTN_READY` stays false.

## Product decision (2026-10-24)

Jerry: **cpu-only remains the ship default**; Metal/hybrid stays **optional / experimental**.
`HYBRID_ATTN_READY` stays **false**. Slice B (experts on GPU) still held. Merge this branch
as experimental Metal path + honest numbers — not as a recommended profile.

