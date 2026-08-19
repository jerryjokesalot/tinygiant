# TinyGiant

**Tiny footprint. Giant model.** Run 30B+ parameter AI models on any laptop with 8-16GB RAM and a standard SSD. No GPU required.

Large AI models are locked behind expensive hardware. A 30B model needs 18GB+ RAM just to load — more than most laptops have. TinyGiant changes that by streaming only the weights the model actually uses from your SSD, keeping everything else on disk.

Modern Mixture-of-Experts (MoE) models like Qwen3-30B activate only **3 billion** of their 30 billion parameters per token. That's 128 experts per layer, but only 8 fire at any moment. Current inference engines load all 128 into RAM anyway. We don't.

## Results

Measured on an **M1 MacBook Air with 16GB RAM** and its stock NVMe SSD.

### Fused Q4 compute ceiling (25.9 tok/s)

The breakthrough: **don't dequantize weights to RAM.** Read Q4 blocks, dequantize in CPU registers, multiply-accumulate, discard. This reduces memory traffic from 38 MB to 2.53 MB per expert (15x reduction). Combined with Q8 input quantization and ARM's `vdotq_s32` integer dot product instruction, the speedup is massive:

| Kernel | tok/s | vs Scalar | Technique |
|--------|-------|-----------|-----------|
| Scalar C | 1.7 | baseline | Fused Q4, but no SIMD |
| Float NEON | 5.5 | 3.2x | NEON intrinsics, algebraic trick |
| **Q8 + vdotq_s32** | **25.9** | **15.2x** | Quantize input to int8, integer dot product |

The Q8+vdot kernel processes 16 multiply-accumulates in a single ARM instruction. Per-layer compute drops to **0.80 ms**.

### Three-tier memory with Q8+vdot (pipelined SSD)

| Pins/layer | Hit% | tok/s | I/O Wait | Warm RAM |
|------------|------|-------|----------|----------|
| 0 | 0% | 2.6 | 350 ms | 0 MB |
| 3 | 38% | 3.9 | 207 ms | 364 MB |
| 5 | 62% | 6.7 | 110 ms | 608 MB |
| 7 | 88% | 20.6 | 11 ms | 850 MB |
| 8 | 100% | 25.9 | 0 ms | 972 MB |

### RAM budget (all experts pinned, 16 GB machine)

| Component | RAM |
|-----------|-----|
| OS + background | 5,000 MB |
| Attention weights (Q4) | 1,500 MB |
| KV cache | 500 MB |
| Warm tier (8 experts/layer) | 972 MB |
| Working buffers | 50 MB |
| **Total** | **~8,022 MB** |

Fits comfortably in 16 GB with headroom.

### Previous results (dequant + Accelerate BLAS)

Before fused Q4, we measured the dequant-to-f32 + BLAS approach:

| Cache Hit Rate | Sequential | Pipelined | Speedup |
|----------------|-----------|-----------|---------|
| 0% | 1.53 tok/s | 1.43 tok/s | -- |
| 50% | 2.20 tok/s | 2.40 tok/s | 1.09x |
| 62% | 2.50 tok/s | 2.99 tok/s | 1.20x |
| 75% | 2.92 tok/s | 3.81 tok/s | 1.30x |
| 88% | 3.47 tok/s | 4.10 tok/s | 1.18x |
| 100% | 4.46 tok/s | 4.45 tok/s | -- |

The old approach dequantized Q4 to f32 in RAM (18 MB write), then ran Accelerate sgemv on the f32 data (18 MB read). That's 38 MB of memory traffic per expert vs 2.53 MB with fused Q4. The BLAS kernel was fast, but memory bandwidth was the bottleneck.

### Earlier prototype results (simulated compute)

| Approach | Speed | RAM Used |
|----------|-------|----------|
| Standard loading (dense 32B) | <1 tok/s | 20 GB+ (swap thrashing) |
| Standard loading (MoE 30B) | 1.7 tok/s | 11.3 GB |
| Expert Cascade (simulated compute) | 4.7 tok/s | ~2 GB |

## How It Works

### 1. Fused Q4 Matmul (the core innovation)

Standard inference dequantizes Q4 weights to f32 in RAM, then runs a BLAS matmul on the f32 data. This means every expert writes 18 MB of f32 to RAM and reads it back — 38 MB of memory traffic for 2.53 MB of actual weight data.

Fused Q4 matmul reads the Q4 block directly, dequantizes in CPU registers, and multiplies immediately. The f32 intermediate never exists in RAM. Memory traffic drops from 38 MB to 2.53 MB per expert.

Combined with Q8 input quantization (quantize the input vector to int8, then do integer dot products with ARM's `vdotq_s32`), this achieves 25.9 tok/s on the MoE experts alone — a 15.2x speedup over scalar C.

```
Old:  [SSD] → Q4 → [RAM: f32 weights (18 MB)] → sgemv → result
                     ↑ write 18 MB    ↑ read 18 MB = 38 MB total

New:  [SSD] → Q4 → [CPU register: dequant + multiply] → result
                     ↑ read 2.53 MB only. 15x less traffic.
```

### 2. Three-Tier Memory Hierarchy

Experts live across three tiers based on access frequency:

- **Hot** — CPU registers. Weights being dequantized and multiplied right now. Zero latency.
- **Warm** — Q4 experts pinned in CPU RAM. The most frequently activated experts per layer stay here (~972 MB for all 8 active per layer).
- **Cold** — Q4 experts on SSD. Loaded via async `pread` when needed. At 2.53 MB per expert and 5+ GB/s SSD, each cold load takes <1 ms.

A pipelined I/O thread pre-fetches cold experts for the next layer while the current layer computes. When compute is slower than I/O (the scalar/NEON cases), the pipeline hides all SSD reads completely.

### 3. Expert-Contiguous Re-Layout

GGUF files store all 128 experts interleaved in one tensor. Loading 1 expert page-faults all 6,912 pages of the tensor. That's 111x amplification.

The re-layout tool (`nws_expert_relayout.py`) dequantizes the expert tensors and repacks them so each expert's gate/up/down weights sit contiguously. Same weights, different byte ordering, zero quality loss.

After re-layout, loading 1 expert touches 192 pages instead of 6,912. Each expert sits at a known offset: `pread(fd, buf, 9MB, expert_id * 9MB)`.

```
GGUF (interleaved):   [e0_row0][e1_row0][e2_row0]...[e0_row1][e1_row1]...
                       Loading 1 expert touches ALL pages (6,912)

Contiguous:           [expert 0: gate|up|down][expert 1: gate|up|down]...
                       Loading 1 expert touches 192 pages
```

### 4. Pipelined I/O

A dedicated I/O thread loads the next layer's cache misses from SSD while the main thread computes the current layer's hits using Accelerate BLAS. By the time compute finishes, the next batch of experts is ready.

### 5. LRU Cache with Text-Calibrated Pinning

Experts stay in an LRU cache between tokens. Token-to-token expert overlap is ~43% (measured), so the cache helps.

For pinning, random activation profiles are useless (6% hit = random chance). A short calibration pass over real text produces 42% hit rate, matching oracle top-8 performance (45%). Calibrated pinning + LRU together hit 56%.

### 6. Confidence Routing (prototype)

A small 1.5B draft model runs entirely in RAM at 38 tok/s. High-confidence predictions (~55% of tokens) skip the big model entirely. Medium-confidence tokens (~30%) get a partial check. Only low-confidence tokens (~15%) get full verification.

## Try It Yourself

Requires Python 3.9+ and a GGUF model file. The C benchmarks need macOS with the Accelerate framework.

### Build the expert-contiguous cache

```bash
# Install dependencies
pip install gguf numpy

# Build contiguous cache from GGUF model
# Processes all 48 layers, takes ~2.5 minutes, outputs ~54 GB (float16)
python3 tools/nws_expert_relayout.py ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

### Run end-to-end inference

```bash
# Also needs llama-cpp-python for tokenization
pip install llama-cpp-python

# Generate text using the contiguous cache
python3 tools/nws_e2e_inference.py --tokens 10

# With text-calibrated pinning (run calibration pass, then pin hot experts)
python3 tools/nws_e2e_inference.py --calibrate 10 --pin 8 --tokens 10
```

### Run the C benchmarks

```bash
# Fused Q4 with Q8+vdot (the flagship benchmark — requires Apple Silicon)
clang -O3 -mcpu=apple-m1 -lpthread tools/nws_q8dot_bench.c -o nws_q8dot_bench
./nws_q8dot_bench 3

# Float NEON benchmark (intermediate step)
clang -O3 -mcpu=apple-m1 -framework Accelerate -lpthread tools/nws_neon_bench.c -o nws_neon_bench
./nws_neon_bench 3

# Fused Q4 scalar (concept proof, no SIMD)
clang -O2 -framework Accelerate -lpthread tools/nws_fused_bench.c -o nws_fused_bench
./nws_fused_bench 3

# Original benchmarks (dequant + BLAS approach)
clang -O2 -framework Accelerate tools/nws_moe_bench.c -o nws_moe_bench
clang -O2 -framework Accelerate -lpthread tools/nws_pipeline_bench.c -o nws_pipeline_bench
./nws_moe_bench nws_cache 20
./nws_pipeline_bench nws_cache 5
```

### Map expert layout (original tool)

```bash
# Map expert tensor byte ranges in a GGUF file
python3 tools/gguf_expert_mapper.py ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

## Repository Structure

```
tinygiant/
├── tools/
│   ├── nws_q8dot_bench.c        # Q8+vdot benchmark — 25.9 tok/s (flagship)
│   ├── nws_neon_bench.c         # Float NEON benchmark — 5.5 tok/s
│   ├── nws_fused_bench.c        # Fused Q4 scalar benchmark — 1.7 tok/s
│   ├── nws_pipeline_bench.c     # Pipelined I/O benchmark (C/Accelerate)
│   ├── nws_moe_bench.c          # Component-level microbenchmark (C/Accelerate)
│   ├── nws_expert_relayout.py   # Build expert-contiguous cache from GGUF
│   ├── nws_e2e_inference.py     # End-to-end inference engine (Python/numpy)
│   ├── gguf_expert_mapper.py    # Map expert tensor byte ranges in GGUF
│   ├── layer_streaming_poc.py   # SSD streaming benchmark
│   └── cascade_prototype.py     # Cascade loop with real SSD I/O
├── docs/
│   ├── results.md               # Full measurements and analysis
│   ├── proposal.md              # Technical proposal with literature survey
│   └── llama-cpp-rfc.md         # RFC for llama.cpp integration
├── simulations/
│   ├── simulation.py            # v1 feasibility simulation
│   └── simulation_v2.py         # v2 with MoE projections
├── LICENSE
└── README.md
```

## Roadmap

- [x] Feasibility simulation
- [x] Real SSD I/O validation (6.06 GB/s double-buffered)
- [x] GGUF expert tensor mapping
- [x] MoE baseline benchmark (Ollama, 1.7 tok/s)
- [x] Expert Cascade prototype (4.7 tok/s, 100 tokens, simulated compute)
- [x] Expert-contiguous re-layout (36x page fault reduction)
- [x] End-to-end inference with real matmul (verified correct text output)
- [x] Compiled C benchmark with Accelerate BLAS (4.4 tok/s compute ceiling)
- [x] Pipelined I/O benchmark (3.81 tok/s at 75% hit, 4.10 at 88%)
- [x] Text-calibrated pinning (42% hit rate vs 6% random)
- [x] File llama.cpp RFC ([Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149))
- [x] Post benchmark data to existing MoE offload discussion ([Discussion #23324](https://github.com/ggml-org/llama.cpp/discussions/23324#discussioncomment-18076154))
- [x] Fused Q4 matmul (eliminate f32 intermediate, 15x memory traffic reduction)
- [x] ARM NEON intrinsics (3.2x over scalar C, 5.5 tok/s)
- [x] Q8 input quantization + vdotq_s32 (15.2x over scalar, 25.9 tok/s compute ceiling)
- [x] Three-tier memory benchmark with Q8+vdot (2.6-25.9 tok/s across hit rates)
- [ ] End-to-end inference with fused Q4 kernel (integrate into nws_e2e_inference.py)
- [ ] Multi-platform testing (Linux, Windows, different SSDs)
- [ ] llama.cpp integration (CPU offload path with fused Q4×Q8)

## Related Discussions

- [llama.cpp #23324: MoE Expert Offload to Disk with On-Demand Paging](https://github.com/ggml-org/llama.cpp/discussions/23324) — independent PoC achieving 13 tok/s on M1 Pro 16GB with Metal. Our contiguous layout and calibrated pinning are complementary.
- [llama.cpp #27149: Expert-Aware SSD Streaming](https://github.com/ggml-org/llama.cpp/discussions/27149) — our original RFC and prototype measurements.
- [llama.cpp #20757: MoE Expert Cache Feature Request](https://github.com/ggml-org/llama.cpp/issues/20757)

## Why "TinyGiant"?

Tiny footprint (~2 GB RAM). Giant model (30B+ parameters). You shouldn't need a giant machine to run a giant model.

## License

MIT
