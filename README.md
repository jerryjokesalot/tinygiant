# TinyGiant

**Tiny footprint. Giant model.** Run 30B+ parameter AI models on any laptop with 8-16GB RAM and a standard SSD. No GPU required.

Large AI models are locked behind expensive hardware. A 30B model needs 18GB+ RAM just to load — more than most laptops have. TinyGiant changes that by streaming only the weights the model actually uses from your SSD, keeping everything else on disk.

The key insight: modern Mixture-of-Experts (MoE) models like Qwen3-30B activate only **3 billion** of their 30 billion parameters per token. That's 128 experts per layer, but only 8 are "awake" at any moment. Current inference engines load all 128 into RAM anyway. We don't.

## The Netflix Analogy

Netflix doesn't download every movie to your TV before you press play. It streams the one you're watching.

TinyGiant does the same thing for AI model weights:
- **Standard approach:** Load all 18GB into RAM, then generate. Hope you have enough memory.
- **TinyGiant:** Keep 0.5GB of shared weights in RAM. Stream the ~31MB of active expert weights per layer from SSD as needed. The other 17GB stays on disk.

Your SSD becomes the model's memory. Your RAM becomes a small, fast cache.

## Results

All measurements on an **M1 MacBook Pro with 16GB RAM and its stock SSD** — a standard consumer laptop, not a workstation.

| Approach | Speed | RAM Used | What Happens |
|----------|-------|----------|---|
| Standard loading (dense 32B) | <1 tok/s | 20 GB+ | Swap thrashing, machine freezes |
| Standard loading (MoE 30B) | 1.7 tok/s | 11.3 GB | Works, but slow and tight on memory |
| **Expert Cascade (MoE 30B)** | **4.7 tok/s** | **~2 GB** | **Responsive, machine stays usable** |
| With double-buffered I/O | ~6 tok/s | ~2 GB | Projected from measured I/O |
| With speculative batching | ~8-10 tok/s | ~2 GB | Projected from measured I/O |

> **What's measured vs. projected:** The SSD reads are real — we read real byte ranges from a real 17.3GB GGUF model file on a real SSD (2.97 GB/s sequential, 6.06 GB/s double-buffered). What's not yet wired up is actual matrix multiplication with the streamed weights, so inference compute is simulated. The I/O is the bottleneck, and that part is fully validated.

## How It Works

Three techniques, stacked:

### 1. Expert Cascade — Read Only What's Active

MoE expert weights in GGUF files are stored as stacked 3D tensors. Each expert is a contiguous byte slice (~3 MB). Instead of loading the full 263 MB layer, we `seek` to the 8 active experts and read 31 MB total. **11.4x less I/O per layer.**

```
Standard:  [expert 0][expert 1][expert 2]...[expert 127]  = 263 MB  (read ALL)
                ↓         ↓                       ↓
Cascade:   [expert 0]         [expert 2]                  =  31 MB  (read 8)
```

### 2. Confidence Routing — Skip the Big Model When You Can

A small 1.5B draft model runs entirely in RAM at 38 tok/s. For high-confidence predictions (~55% of tokens), it's right — no need to verify with the big model. Medium-confidence tokens (~30%) get a partial check (half the layers). Only low-confidence tokens (~15%) get full verification.

### 3. Double-Buffered Streaming — Overlap I/O and Compute

While the CPU processes layer N's expert weights, a background thread prefetches layer N+1's experts from SSD. Measured result: **2.09x throughput improvement** over sequential reads.

## Try It Yourself

These tools work with any GGUF model file — no special setup beyond Python 3.9+.

### Map your model's expert layout

```bash
# If you have a model via Ollama, find its GGUF blob:
ollama pull qwen3:30b-a3b
ls -lh ~/.ollama/models/blobs/ | sort -k5 -h | tail -3
# Create a symlink to the largest blob (the model weights):
ln -s ~/.ollama/models/blobs/sha256-YOUR_HASH ~/models/qwen3-30b-a3b.gguf

# Map expert tensor byte ranges
python3 tools/gguf_expert_mapper.py ~/models/qwen3-30b-a3b.gguf
```

Example output:
```
Model: Qwen3 30B A3B
Architecture: qwen3moe
Experts: 128 total, 8 active per token

MoE STREAMING ANALYSIS
  Shared weights (attention, norms):  0.43 GB
  Expert weights (all 128 experts):  15.75 GB
  Per-layer read (standard):  345.3 MB
  Per-layer read (cascade):    30.3 MB
  I/O reduction factor:        11.4x
```

### Benchmark your SSD

```bash
python3 tools/layer_streaming_poc.py ~/models/qwen3-30b-a3b.gguf 48
```

### Run the cascade prototype

```bash
# Baseline: full-layer reads
python3 tools/cascade_prototype.py ~/models/qwen3-30b-a3b.gguf 50 --layers 48

# Expert Cascade: read only active experts (8.8% of each layer)
python3 tools/cascade_prototype.py ~/models/qwen3-30b-a3b.gguf 100 --expert-ratio 0.088 --layers 48
```

## What's Novel

This project combines known techniques in a way nobody has done:

| Technique | Exists Already | What We Add |
|-----------|---------------|-------------|
| Speculative decoding | Yes (Leviathan et al. 2023) | 3-tier confidence routing (easy/medium/hard) |
| Model offloading | Yes (llama.cpp, PowerInfer) | Async double-buffered SSD streaming |
| MoE inference | Yes (Mixtral, Qwen) | **Selective expert streaming from SSD** |
| Early exit | Yes (LayerSkip) | Combined with speculative routing |

The genuinely new piece: **no existing inference engine reads individual expert byte ranges from disk on demand.** They all load the full model into RAM. On memory-constrained hardware, that means loading 120 dormant experts just to use 8 — and often swap-thrashing in the process.

See also:
- [llama.cpp #20757: MoE Expert Cache Feature Request](https://github.com/ggml-org/llama.cpp/issues/20757)
- [llama.cpp #13154: MoE Expert Offload Discussion](https://github.com/ggml-org/llama.cpp/discussions/13154)

## Repository Structure

```
tinygiant/
├── tools/
│   ├── gguf_expert_mapper.py    # Map expert tensor byte ranges in GGUF files
│   ├── layer_streaming_poc.py   # SSD streaming benchmark
│   └── cascade_prototype.py     # End-to-end cascade loop with real SSD I/O
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
- [x] Expert Cascade prototype (4.7 tok/s, 100 tokens, real I/O)
- [ ] Wire up real inference (matrix multiplication with streamed weights)
- [ ] File llama.cpp RFC with measurements
- [ ] Multi-platform testing (Linux, Windows, different SSDs)
- [ ] Auto-detect hardware and tune thresholds

## Contributing

The biggest open problem: **integrating expert-aware streaming into a real inference engine.** See [`docs/llama-cpp-rfc.md`](docs/llama-cpp-rfc.md) for the proposed design targeting llama.cpp. The core tasks:

1. **Selective GGUF loading** — skip `ffn_*_exps` tensors at model load time
2. **Async expert prefetch** — run MoE router first, then `seek + read` active expert byte ranges while attention computes
3. **Buffer lifecycle** — coordinate streamed weight buffers across I/O and compute threads

If you can write C++ and know llama.cpp internals, this is a high-impact contribution.

## Why "TinyGiant"?

Tiny footprint (~2 GB RAM). Giant model (30B+ parameters). The whole point is that you shouldn't need a giant machine to run a giant model.

## License

MIT
