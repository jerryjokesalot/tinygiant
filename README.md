# TinyGiant

**Tiny footprint. Giant model.** Run 30B+ parameter AI models on any laptop with 8-16GB RAM and a standard SSD. No GPU required.

Large AI models are locked behind expensive hardware. A 30B model needs 18GB+ RAM just to load — more than most laptops have. TinyGiant changes that by streaming only the weights the model actually uses from your SSD, keeping everything else on disk.

Modern Mixture-of-Experts (MoE) models like Qwen3-30B activate only **3 billion** of their 30 billion parameters per token. That's 128 experts per layer, but only 8 fire at any moment. Current inference engines load all 128 into RAM anyway. We don't.

## Results

Measured on an **M1 MacBook Air with 16GB RAM** and its stock NVMe SSD.

### Pipelined I/O benchmark (compiled C, Apple Accelerate BLAS)

| Cache Hit Rate | Sequential | Pipelined | Speedup |
|----------------|-----------|-----------|---------|
| 0% | 1.53 tok/s | 1.43 tok/s | -- |
| 50% | 2.20 tok/s | 2.40 tok/s | 1.09x |
| 62% | 2.50 tok/s | 2.99 tok/s | 1.20x |
| 75% | 2.92 tok/s | 3.81 tok/s | 1.30x |
| 88% | 3.47 tok/s | 4.10 tok/s | 1.18x |
| 100% | 4.46 tok/s | 4.45 tok/s | -- |

At 75% cache hit with pipelining, inference exceeds 3.8 tok/s. At 88%, the SSD reads are completely hidden behind compute.

### Measured hardware costs

| Component | Time | Notes |
|-----------|------|-------|
| Compute ceiling (CPU BLAS) | 4.4 tok/s | f16→f32 conversion + Accelerate sgemv |
| SSD sequential read | 3.0+ GB/s | From expert-contiguous cache |
| Per-layer MoE compute | 4.7 ms | 8 experts, gate/up/down matmuls + SiLU |
| Per-expert I/O (9 MB, f16) | 1.1 ms warm, 2.9 ms cold | pread from contiguous layout |
| Pipelining capacity | 4 misses/layer hidden | Compute-bound at 50% cache hit |

### Q4 projection

With Q4_K_M quantized experts (~2.2 MB each), per-expert I/O drops to ~0.27 ms vs 4.7 ms compute/layer. Pipelining hides all misses at any cache hit rate. 32 pinned experts/layer = 3.4 GB RAM for 88% oracle hit.

This is projected from measured SSD throughput, not directly benchmarked at Q4.

### Earlier prototype results (simulated compute)

| Approach | Speed | RAM Used |
|----------|-------|----------|
| Standard loading (dense 32B) | <1 tok/s | 20 GB+ (swap thrashing) |
| Standard loading (MoE 30B) | 1.7 tok/s | 11.3 GB |
| Expert Cascade (simulated compute) | 4.7 tok/s | ~2 GB |

## How It Works

### 1. Expert-Contiguous Re-Layout

GGUF files store all 128 experts interleaved in one tensor. Loading 1 expert page-faults all 6,912 pages of the tensor. That's 111x amplification.

The re-layout tool (`nws_expert_relayout.py`) dequantizes the expert tensors and repacks them so each expert's gate/up/down weights sit contiguously. Same weights, different byte ordering, zero quality loss.

After re-layout, loading 1 expert touches 192 pages instead of 6,912. Each expert sits at a known offset: `pread(fd, buf, 9MB, expert_id * 9MB)`.

```
GGUF (interleaved):   [e0_row0][e1_row0][e2_row0]...[e0_row1][e1_row1]...
                       Loading 1 expert touches ALL pages (6,912)

Contiguous:           [expert 0: gate|up|down][expert 1: gate|up|down]...
                       Loading 1 expert touches 192 pages
```

### 2. Pipelined I/O

A dedicated I/O thread loads the next layer's cache misses from SSD while the main thread computes the current layer's hits using Accelerate BLAS. By the time compute finishes, the next batch of experts is ready.

### 3. LRU Cache with Text-Calibrated Pinning

Experts stay in an LRU cache between tokens. Token-to-token expert overlap is ~43% (measured), so the cache helps.

For pinning, random activation profiles are useless (6% hit = random chance). A short calibration pass over real text produces 42% hit rate, matching oracle top-8 performance (45%). Calibrated pinning + LRU together hit 56%.

### 4. Confidence Routing (prototype)

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
# Compile (macOS with Accelerate)
clang -O2 -framework Accelerate tools/nws_moe_bench.c -o nws_moe_bench
clang -O2 -framework Accelerate -lpthread tools/nws_pipeline_bench.c -o nws_pipeline_bench

# Component-level benchmark (compute ceiling, SSD speed, attention)
./nws_moe_bench nws_cache 20

# Pipelined I/O benchmark (hit-rate sweep, sequential vs pipelined)
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
│   ├── nws_expert_relayout.py   # Build expert-contiguous cache from GGUF
│   ├── nws_e2e_inference.py     # End-to-end inference engine (Python/numpy)
│   ├── nws_pipeline_bench.c     # Pipelined I/O benchmark (C/Accelerate)
│   ├── nws_moe_bench.c          # Component-level microbenchmark (C/Accelerate)
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
- [ ] Q4 expert cache (validate the 0.27 ms/expert projection)
- [ ] Multi-platform testing (Linux, Windows, different SSDs)
- [ ] llama.cpp integration

## Related Discussions

- [llama.cpp #23324: MoE Expert Offload to Disk with On-Demand Paging](https://github.com/ggml-org/llama.cpp/discussions/23324) — independent PoC achieving 13 tok/s on M1 Pro 16GB with Metal. Our contiguous layout and calibrated pinning are complementary.
- [llama.cpp #27149: Expert-Aware SSD Streaming](https://github.com/ggml-org/llama.cpp/discussions/27149) — our original RFC and prototype measurements.
- [llama.cpp #20757: MoE Expert Cache Feature Request](https://github.com/ggml-org/llama.cpp/issues/20757)

## Why "TinyGiant"?

Tiny footprint (~2 GB RAM). Giant model (30B+ parameters). You shouldn't need a giant machine to run a giant model.

## License

MIT
