# TinyGiant NWS vs llama.cpp Benchmark

## Setup

- **Model:** Qwen3-30B-A3B-Q4_K_M (17.28 GiB GGUF)
- **Hardware:** Apple M3, 16GB unified memory, 8 cores (4P+4E)
- **Date:** 2026-08-19
- **llama.cpp version:** b10470-34af94cd9 (Homebrew)
- **TinyGiant version:** NWS with mmap expert cache, Q4_K attention, Q6_K fused kernel, mlock pinning

## Results

### Decode Speed (tok/s)

| Configuration | tok/s | vs llama.cpp |
|---------------|-------|--------------|
| llama.cpp CPU (ngl=0, 8 threads) | 1.37 ± 0.31 | baseline |
| llama.cpp GPU (ngl=10+) | FAILED | model exceeds Metal working set |
| TinyGiant (no pinning) | 2.03 | 1.48x |
| TinyGiant (mlock pin32) | 3.62 | 2.64x |
| TinyGiant (mlock pin48) | 3.70-4.54 | 2.70-3.31x |
| TinyGiant (mlock pin64) | 4.83-5.08 | **3.53-3.71x** |
| TinyGiant (warm, all in RAM) | 7.0-8.4 | 5.1-6.1x |

### Methodology

- llama-bench: 3 repetitions, 128 token decode (tg128), prompt length 1
- TinyGiant: 128 token decode from 11-token prompt, temperature=0.7
- Each test run after `purge` to clear file cache (cold start)
- llama.cpp tested with ngl=0 (CPU only) — GPU offload fails on this model
- Run-to-run variance exists due to SSD performance variability on cold reads

### Optimizations Applied (this session)

1. **mlock pinning** — replaced `np.sum()` page-cache-only pinning with
   `mlock()` which wires pages into physical memory, preventing OS eviction.
   This made performance consistent (attention always 40ms, was 40-300ms).

2. **Attention weight mlock** — lock attention weights (555 MB) in RAM so they
   are never evicted by expert pinning. Guarantees 40ms attention latency.

3. **Combined pin strategy** — uses calibration data first, fills remaining
   slots from static activation profile. Gets full pin64 coverage from only
   10 tokens of calibration.

### Memory Budget

| Component | Size | mlock'd |
|-----------|------|---------|
| Attention weights | 555 MB | Yes |
| Embeddings | 1,187 MB | No (accessed once/token) |
| Output head | 1,187 MB | No (accessed once/token) |
| Expert cache (pin48) | 6.1 GB | Yes |
| Expert cache (pin64) | 6.9 GB | Yes |
| **Total (pin48)** | **9.0 GB** | 6.7 GB |
| **Total (pin64)** | **9.8 GB** | 7.5 GB |

Sweet spot: pin48 (9 GB total) leaves 7 GB for OS. Pin64 (9.8 GB) is
higher performance but more sensitive to system memory pressure.

### Startup Time

| Phase | llama.cpp | TinyGiant |
|-------|-----------|-----------|
| Model load | ~8-10 min | 5-7s |
| Calibration | N/A | ~12s (10 tokens) |
| mlock pinning | N/A | 1.6-2.3s |
| First token | ~10 min | ~20s |

### Why TinyGiant is Faster

1. **Expert-contiguous layout**: Pre-processed GGUF into expert-contiguous
   binary files. Accessing one expert page-faults ~3MB, not the full tensor.

2. **Selective loading**: Only 3 GB loaded at startup. 16 GB of expert data
   demand-paged via mmap, hot experts locked with mlock.

3. **mlock pinning**: Guarantees hot experts stay in physical memory. OS
   page cache eviction can't steal them. 87-93% hit rate.

4. **Fused quantized kernels**: Q4_K x Q8 and Q6_K x Q8 NEON kernels
   operate directly on quantized data — no dequantization step.

### Compute Ceiling Analysis

Warm speed (all experts in RAM): 7.0-8.4 tok/s. This is the true compute
ceiling on M3 — limited by NEON matmul throughput.

With mlock pin64: 4.8-5.1 tok/s sustained = 60-72% of compute ceiling.
Remaining gap is SSD reads for ~7% cold expert accesses (~30 per token,
~90 MB/token from SSD).

Further improvements would require either more RAM (32 GB), faster SSD
(M3 Pro/Max), or reduced expert size via more aggressive quantization.
