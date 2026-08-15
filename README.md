# TinyGiant

Run 30B+ AI models on consumer laptops with 8-16GB RAM. No GPU required.

## The Problem

Large language models need RAM. A 30B model quantized to Q4 takes ~18GB — more than most laptops have. Current approaches:

- **Standard loading** — loads the whole model into RAM/swap, causes thrashing (<1 tok/s on 16GB)
- **Cloud APIs** — work but defeat the purpose of local AI
- **Quantization** — helps but 30B+ still overflows consumer RAM

## The Idea: Expert Cascade

Mixture-of-Experts (MoE) models activate only a fraction of their parameters per token. Qwen3-30B-A3B has 30B total parameters but only uses **3B per token** (8 of 128 experts). Current inference engines ignore this — they load all 128 experts into RAM even though 120 are idle.

**Expert Cascade** streams only the active expert weights from SSD on demand:

| What changes | Standard | Expert Cascade |
|---|---|---|
| Data read per layer | 263 MB (all experts) | 31 MB (8 active experts) |
| I/O reduction | — | **11.4x** |
| RAM needed | 18+ GB | ~2 GB |

Combined with confidence-gated routing (a small draft model handles easy tokens without touching the big model at all), this produces interactive-speed inference on consumer hardware.

## Measured Results

All measurements on an **M1 MacBook Pro, 16GB RAM, stock NVMe SSD**.

| Approach | Speed | RAM | Status |
|----------|-------|-----|--------|
| Dense 32B, standard loading | <1 tok/s | 20 GB+ (swap thrash) | Measured |
| MoE 30B, standard loading (Ollama) | 1.7 tok/s | 11.3 GB | Measured |
| Cognitive Cascade (dense, SSD streaming) | 0.9 tok/s | ~2 GB | Measured I/O |
| **Expert Cascade (MoE, selective streaming)** | **4.7 tok/s** | **~2 GB** | **Measured I/O** |
| Expert Cascade + double-buffered I/O | ~6.3 tok/s | ~2 GB | Projected |
| Expert Cascade + speculative batching | ~8-10 tok/s | ~2 GB | Projected |
| Draft model only (1.5B) | 38.3 tok/s | 1.5 GB | Measured |

"Measured I/O" means real SSD reads of real model files with simulated compute — the I/O bottleneck is faithfully measured, but actual matrix multiplication is not yet wired up.

### Key I/O Measurements

| Metric | Value |
|--------|-------|
| Sequential SSD read | 2.97 GB/s |
| Double-buffered pipeline | 6.06 GB/s (2.09x speedup) |
| MoE layer (all experts) | 263 MB |
| MoE layer (8 active experts) | 31 MB |
| Expert slice size | ~3 MB per expert |

## How It Works

```
User prompt
    |
    v
+-------------+
| Draft Model | <-- 1.5B, always in RAM (38 tok/s)
+------+------+
       |
       v
+--------------+
|  Confidence  |
|   Router     |
+--+---+---+--+
   |   |   |
   v   v   v
 Easy  Med  Hard     <-- 55% / 30% / 15% of tokens
  |    |    |
  |    |    v
  |    |  Stream ALL layers from SSD (full verification)
  |    v
  |  Stream HALF layers (early exit)
  |
  v
 Accept draft token directly (no big-model I/O)
```

For MoE models, each layer read streams only the **active expert weights** (~31 MB instead of 263 MB).

## Try It Yourself

### 1. Map your model's expert layout

```bash
# Download a MoE model via Ollama
ollama pull qwen3:30b-a3b

# Find the GGUF blob
ls -la ~/.ollama/models/blobs/ | sort -k5 -n | tail -3

# Create a symlink (replace with your blob hash)
ln -s ~/.ollama/models/blobs/sha256-YOUR_HASH ~/models/qwen3-30b-a3b.gguf

# Map expert tensor byte ranges
python3 tools/gguf_expert_mapper.py ~/models/qwen3-30b-a3b.gguf
```

### 2. Benchmark SSD layer streaming

```bash
python3 tools/layer_streaming_poc.py ~/models/qwen3-30b-a3b.gguf 48
```

### 3. Run the cascade prototype

```bash
# Full-layer reads (baseline)
python3 tools/cascade_prototype.py ~/models/qwen3-30b-a3b.gguf 50 --layers 48

# Expert Cascade mode (8.8% of each layer = active experts only)
python3 tools/cascade_prototype.py ~/models/qwen3-30b-a3b.gguf 100 --expert-ratio 0.088 --layers 48
```

## GGUF Expert Layout

The expert mapper reveals how MoE weights are stored in GGUF files:

```
Qwen3 30B-A3B (qwen3moe architecture):
  - 48 layers, 128 experts per layer, 8 active per token
  - Shared weights (attention + norms): 0.43 GB  --> always in RAM
  - Expert weights (all 128 experts):  15.75 GB  --> stream from SSD
  - Router weights:                     0.05 GB  --> always in RAM

  Expert tensors are stacked 3D: blk.N.ffn_gate_exps.weight
  Shape: [128, hidden_dim, expert_dim]
  Each expert is a contiguous ~3 MB byte slice
  --> Selective reading = seek + 3 MB read per active expert
```

## Research Context

| Technique | Prior Art | What's New Here |
|-----------|-----------|-----------------|
| Speculative decoding | Leviathan et al. 2023 | Confidence-gated 3-tier routing (easy/medium/hard) |
| Model offloading | llama.cpp, PowerInfer | Async double-buffered layer streaming from consumer SSD |
| MoE inference | Switch Transformer, Mixtral | Expert-aware selective SSD streaming (read only active experts) |
| Early exit | LayerSkip, LITE | Combined with speculative decoding for medium-confidence tokens |

**The novel combination:** Expert-aware SSD streaming + confidence routing + double-buffered I/O as a CPU-only, zero-config approach. No existing inference engine reads individual expert weights from disk on demand.

Related llama.cpp discussions:
- [#20757: MoE Expert Cache Feature Request](https://github.com/ggml-org/llama.cpp/issues/20757)
- [#13154: MoE Expert Offload Discussion](https://github.com/ggml-org/llama.cpp/discussions/13154)

## Repository Structure

```
tinygiant/
├── tools/
│   ├── gguf_expert_mapper.py    # Map expert tensor byte ranges in GGUF files
│   ├── layer_streaming_poc.py   # SSD streaming benchmark (sequential, double-buffered, mmap)
│   └── cascade_prototype.py     # End-to-end cascade loop with real SSD I/O
├── docs/
│   ├── results.md               # Detailed measurements and analysis
│   ├── proposal.md              # Full technical proposal with literature survey
│   └── llama-cpp-rfc.md         # RFC for llama.cpp integration
├── simulations/
│   ├── simulation.py            # v1 feasibility simulation
│   └── simulation_v2.py         # v2 with 3-tier routing and MoE projections
└── README.md
```

## Status

- [x] Mathematical feasibility simulation (two versions)
- [x] Real hardware I/O validation (double-buffered pipeline: 6.06 GB/s)
- [x] GGUF expert tensor mapping (byte-range manifest for selective reads)
- [x] MoE model benchmarked via Ollama (1.7 tok/s baseline)
- [x] Expert Cascade prototype with real I/O (4.7 tok/s over 100 tokens)
- [ ] Real inference integration (wire up actual matrix multiplication)
- [ ] llama.cpp RFC filed
- [ ] Hardware auto-detection and tuning
- [ ] Multi-platform testing (Linux, Windows)

## Contributing

The biggest open problem is integrating expert-aware streaming into a real inference engine. See `docs/llama-cpp-rfc.md` for the proposed design. Key implementation tasks:

1. **GGUF selective tensor loading** — modify model loading to skip expert tensors
2. **Async expert prefetch** — run MoE router, then prefetch active expert weights from SSD while attention computes
3. **Double-buffered layer pipeline** — overlap I/O for layer N+1 with compute on layer N
4. **Confidence router** — calibrate draft model confidence thresholds per task type

## License

MIT
