# TinyGiant vs llama.cpp CPU Bake-off — Apple M1 16GB

**Date:** 2026-10-24 (America/New_York)
**Supersedes method of** [`benchmark_2026-08-19_m3_16gb.md`](benchmark_2026-08-19_m3_16gb.md) for *this machine* (M1). That file remains as the prior M3 note.

## Setup

| Item | Value |
|---|---|
| Hardware | Apple M1, 16 GiB unified memory, 8 cores (physical=logical=8) |
| OS | macOS 26.7 (Build 25G229) |
| Model | `~/models/Qwen3-30B-A3B-Q4_K_M.gguf` → ollama blob (17.28 GiB, Q4_K_M, GGUF) |
| TinyGiant | commit `2d0a62e`, fused Q4 NWS cache `~/.tinygiant/cache` (q4_k v2, 48/48 layers) |
| llama.cpp | commit `73adf89`, `~/src/llama.cpp/build/bin/llama-{cli,bench}`, CPU build (Metal OFF, Accelerate BLAS ON) |
| Competing services | `com.tinygiant.server` unloaded; Ollama quit for the run |

### Metric definitions (labeled)

- **Decode tok/s (preferred):** `n_gen / decode_only_seconds` after prefill completes. This is the sustained generation rate.
- **Wall generate() tok/s:** `n_gen / (prefill + decode)` — same shape as `experiments/pinning_ab_test.py`.
- **llama-bench tg128:** built-in generation benchmark, `n_gen=128`, reported mean±stddev over repetitions.

## Primary results (sustained decode)

| Configuration | Decode tok/s | vs llama.cpp tg128 | Wall tok/s | Notes |
|---|---:|---:|---:|---|
| llama.cpp CPU (`-ngl 0 -t 8`, llama-bench tg128, `-p 1`) | **1.06 ± 0.05** | baseline | n/a | primary peer |
| llama.cpp CPU (same, `-p 32`) | 1.04 ± 0.11 | 0.98x | n/a | prompt-length check |
| TinyGiant `pin_nonuniform` | **2.68 ± 0.62** | **2.53x** | 2.25 ± 0.45 | entropy pins from concentration profile |
| TinyGiant `pin48` (uniform) | 2.78 ± 0.64 | 2.62x | 2.35 ± 0.48 | reference |

**Verdict:** On this M1 16GB machine, TinyGiant sustained decode (~2.68–2.78 tok/s) still beats llama.cpp CPU tg128 (~1.06 tok/s) by about **2.5–2.6×**. In *this* 128-token bake-off, uniform `pin48` edged `pin_nonuniform` slightly on mean decode; earlier 60-token A/B wall metric had non-uniform ahead — treat pin choice as within noise here.

## TinyGiant method

- Harness: `experiments/bakeoff_fair_toks.py`
- Raw JSON: `benchmarks/bakeoff_tg_raw_2026-10-24.json`
- `n_gen=128`, reps=3 per prompt, calibrate_n=10
- Prompts: same 5 as `experiments/pinning_ab_test.py` (coding / reasoning / creative / math / factual)
- Engine load wall: 8.7s
- Pin budget: 2304 experts; non-uniform from `experiments/layer_concentration_results.json` (min 14 / max 92)
- Temperature 0.7, top_p 0.9; seeds 42+rep

### Per-task decode tok/s (mean ± std, n=3)

| Task | pin_nonuniform | pin48 |
|---|---:|---:|
| coding | 2.00 ± 0.07 | 2.06 ± 0.10 |
| reasoning | 3.23 ± 0.19 | 3.33 ± 0.15 |
| creative | 3.02 ± 0.08 | 3.18 ± 0.04 |
| math | 1.93 ± 0.14 | 1.99 ± 0.04 |
| factual | 3.22 ± 0.13 | 3.34 ± 0.03 |

## llama.cpp method

```text
/Users/jerrybecker/src/llama.cpp/build/bin/llama-bench \
  -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 0 -t 8 -n 128 -r 3 -o md --progress \
  -p 1    # and separately -p 32
```

| Run | Prompt len | pp t/s | tg128 t/s | wall clock (time -l) | max RSS | peak footprint |
|---|---:|---:|---:|---:|---:|---:|
| p1 | 1 | 8.11 ± 2.71 | 1.06 ± 0.05 | 437s real | 10.56 GiB | 17.15 GiB |
| p32 | 32 | 2.23 ± 0.18 | 1.04 ± 0.11 | 512s real | 11.02 GiB | 17.16 GiB |

Backend column in llama-bench output: **BLAS** (Accelerate). GPU offload not used (`-ngl 0`). Homebrew `brew install llama.cpp` failed on this machine (`ggml: undefined method 'new'`); binary built from source instead.

## What was not run (optional peers)

- **ollama Metal / MLX:** not timed in this pass (would not be a fair CPU peer). Ollama was quit during the run to free RAM; MLX weights exist on disk but Python `mlx`/`mlx_lm` were not installed.
- **llama-cli prompt-matched chat:** not required for primary peer; llama-bench tg128 is the apples-to-apples decode metric vs prior M3 note.

## Cold/warm notes

- `purge` was attempted before llama-bench but requires root — skipped (no privilege).
- TinyGiant runs were warm-ish after engine load (mmap expert cache already present from Aug).
- llama.cpp CPU on 16 GiB with a 17.28 GiB weights file is memory-pressure heavy (max RSS ~10.6–11.0 GiB); machine briefly became unreachable under swap during first load.

## Artifacts

- `benchmarks/bakeoff_tg_raw_2026-10-24.json`
- `/tmp/tinygiant_runs/llama_bench_p1.md` / `llama_bench_p32.md` (+ `.err` with `time -l`)
- llama.cpp tree: `~/src/llama.cpp` @ `73adf89`

