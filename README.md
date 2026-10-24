# TinyGiant

**Tiny footprint. Giant model.** Run 30B+ parameter MoE models on a laptop with 8–16GB RAM and a standard SSD. No GPU required for the CPU path.

Large AI models are locked behind expensive hardware. A 30B model needs 18GB+ RAM just to load — more than most laptops have. TinyGiant streams only the experts the model actually uses from SSD, keeping dormant weights on disk.

Modern Mixture-of-Experts (MoE) models like Qwen3-30B activate only **3 billion** of their 30 billion parameters per token. That's 128 experts per layer, but only 8 fire at any moment. Many inference stacks still keep all 128 resident. TinyGiant doesn't.

## What is actually novel here

TinyGiant's packaging contribution is **contiguous expert re-layout** plus **text-calibrated mlock pinning** — not new math. Fused Q4×Q8 kernels (including ARM `vdotq_s32`) are the same family of quantized matmul techniques used elsewhere; we don't claim them as a TinyGiant invention. What we ship is a downloadable path that:

1. Repacks GGUF expert tensors so each expert is a contiguous SSD read (instead of page-faulting an interleaved tensor).
2. Calibrates which experts to pin from a short real-text pass, then `mlock`s them so the OS can't steal the hot set.
3. Runs end-to-end chat through a small Python package + `libtinygiant` C library.

Closest peer work: [llama.cpp MoE SSD streaming discussion / PRs](https://github.com/ggml-org/llama.cpp/discussions/23324). Contiguous layout and calibrated pinning are complementary to that line of work.

## Results (honest)

### Sustained chat (what you should expect)

Prior bake-off on **Apple M3, 16GB**, same model (`Qwen3-30B-A3B-Q4_K_M`), cold file cache (`purge`), 128-token decode — see [`benchmarks/benchmark_2026-08-19_m3_16gb.md`](benchmarks/benchmark_2026-08-19_m3_16gb.md) for method and raw numbers:

| Configuration | tok/s | vs llama.cpp CPU |
|---------------|-------|------------------|
| llama.cpp CPU (`ngl=0`, 8 threads) | 1.37 ± 0.31 | baseline |
| TinyGiant (mlock pin48) | 3.70–4.54 | ~2.7–3.3× |
| TinyGiant (mlock pin64) | 4.83–5.08 | ~3.5–3.7× |

Those are **sustained generation** numbers, not a warm microbenchmark. A fresh fair bake-off (same machine, same model, same prompts) is planned for the MVP; do not treat the table above as a new claim.

### Pinning A/B (M1 16GB, 2026-10-24)

Same machine / model / prompts; wall-clock `NWSEngine.generate()` tok/s including prefill (60 decode tokens). Method + raw numbers: [`experiments/pinning_ab_results.json`](experiments/pinning_ab_results.json).

| Task | Uniform pin48 | Non-uniform | Delta |
|------|--------------:|------------:|------:|
| coding | 1.70 | 1.89 | +11.4% |
| reasoning | 1.37 | 1.44 | +5.5% |
| creative | 2.19 | 2.39 | +9.0% |
| math | 1.44 | 1.51 | +5.4% |
| factual | 2.80 | 3.01 | +7.6% |
| **average** | **1.90** | **2.05** | **+8.0%** |

Shipped default is non-uniform when the profile exists. These are **not** the warm ~25.9 tok/s compute ceiling.

### Warm compute ceiling (not chat)

Microbenchmarks of the fused Q4×Q8 / Q8+vdot kernel on Apple Silicon can hit **~25.9 tok/s** when experts are already warm in RAM and I/O is zero. That is a **compute ceiling**, not sustained chat speed. Sustained chat is I/O- and pin-hit-rate limited; see the M3 table above.

Earlier dequant+BLAS and prototype cascade numbers live in [`docs/results.md`](docs/results.md). Cascade / Learning Cache remain deferred.

### RAM budget (pinning hot experts, 16 GB machine)

With attention mlock'd and a pin48–pin64 warm expert set, the working set fits in 16 GB with OS headroom. Exact budgets for the M3 run are in the [benchmark note](benchmarks/benchmark_2026-08-19_m3_16gb.md).

## How It Works

### 1. Expert-contiguous re-layout

GGUF files store all 128 experts interleaved in one tensor. Loading 1 expert page-faults thousands of pages of the tensor.

`tinygiant-setup` (package entry → `tinygiant.relayout`) repacks expert tensors so each expert's gate/up/down weights sit contiguously. Same weights, different byte ordering, zero quality loss. After re-layout, one expert is a known-offset `pread`.

```
GGUF (interleaved):   [e0_row0][e1_row0][e2_row0]...[e0_row1][e1_row1]...
                       Loading 1 expert touches essentially the whole tensor

Contiguous:           [expert 0: gate|up|down][expert 1: gate|up|down]...
                       Loading 1 expert touches that expert's pages only
```

### 2. Text-calibrated mlock pinning

Random activation profiles are near chance. A short calibration pass over real text finds the hot experts per layer; TinyGiant then `mlock`s them so the OS cannot reclaim the warm set under memory pressure.

**Default pin policy:** when a concentration profile is available (`tinygiant/suggested_pins.json`, or `experiments/layer_concentration_results.json`), TinyGiant uses **non-uniform** (entropy-weighted) pinning via `ExpertCache.pin_nonuniform` — same total budget as uniform `pin48` (2304 experts), more pins on high-entropy layers. Without a profile, it falls back to uniform pin48. On Apple M1 16GB (2026-10-24), non-uniform averaged **+8.0%** wall-clock `generate()` tok/s vs uniform across 5 tasks (includes prefill; see [`experiments/pinning_ab_results.json`](experiments/pinning_ab_results.json)).

### 3. Fused Q4 matmul (implementation detail, not the novelty claim)

Standard paths often dequantize Q4 weights to f32 in RAM, then BLAS. Fused Q4×Q8 reads quantized blocks, dequantizes in registers / uses integer dot products, and never materializes a full f32 weight matrix in RAM. On Apple Silicon the Q8+vdot path is what produces the ~25.9 tok/s **warm** ceiling above. End-to-end inference already uses this via `NWSEngine._forward_c` → `csrc/libtinygiant.c`.

### 4. Three-tier memory + pipelined I/O

Experts live across hot (in-flight compute), warm (mlock'd RAM), and cold (SSD via async `pread`). An I/O thread can prefetch the next layer's misses while the current layer computes.

## Try It Yourself

Requires **Python 3.9+**, a C compiler (`clang`/`gcc`), and ~20+ GB free disk for the model + Q4 expert cache. Apple Silicon is the best-tested path; Linux x86_64/ARM builds via the same `Makefile`.

### 1. Download the model (~17 GiB)

Official Qwen GGUF (filename TinyGiant expects):

```bash
mkdir -p ~/models
# Needs: pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF \
  Qwen3-30B-A3B-Q4_K_M.gguf \
  --local-dir ~/models
# → ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

Model card / file: https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/blob/main/Qwen3-30B-A3B-Q4_K_M.gguf

### 2. Install TinyGiant and build `libtinygiant`

```bash
git clone https://github.com/jerryjokesalot/tinygiant.git
cd tinygiant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[tokenizer]"   # pulls numpy, gguf, llama-cpp-python
make all                        # builds tinygiant/libtinygiant.{dylib,so}
make test                       # optional: C unit tests
```

### 3. Build the expert-contiguous cache (once per model)

```bash
# Q4 cache — processes all layers; disk use is large; takes a few minutes
tinygiant-setup \
  --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --cache ~/models/qwen3-30b-nws-cache \
  --format q4
```

### 4. First chat (CLI)

```bash
tinygiant \
  --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --cache ~/models/qwen3-30b-nws-cache \
  --pin 48 --calibrate 10 --tokens 64 \
  --prompt "The key insight about mixture-of-experts models is that"
```

### 5. Optional: OpenAI-compatible local server

```bash
tinygiant-server \
  --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --cache ~/models/qwen3-30b-nws-cache \
  --pin 48 --port 8000
# POST http://127.0.0.1:8000/v1/chat/completions
```

### Package entry points

| Command | Role |
|---------|------|
| `tinygiant-setup` | One-shot relayout → expert cache |
| `tinygiant-relayout` | Same relayout, lower-level CLI |
| `tinygiant` | Generate text (NWSEngine + libtinygiant) |
| `tinygiant-server` | OpenAI-compatible HTTP server |

### C microbenchmarks (optional)

```bash
make nws_q8dot_bench && ./nws_q8dot_bench 3   # warm Q8+vdot ceiling (Apple Silicon)
make nws_attention_bench && ./nws_attention_bench
```

## Repository Structure

```
tinygiant/
├── tinygiant/                 # Installable package
│   ├── cli.py                 # tinygiant / tinygiant-setup / tinygiant-relayout
│   ├── engine.py              # NWSEngine (Python + _forward_c → libtinygiant)
│   ├── relayout.py            # Contiguous expert cache builder
│   ├── server.py              # tinygiant-server
│   └── cache.py               # Expert cache + pinning helpers
├── csrc/
│   ├── libtinygiant.c         # Fused Q4×Q8 / attention kernels
│   └── test_libtinygiant.c
├── tools/                     # Standalone benches & prototypes (not the install path)
│   ├── nws_q8dot_bench.c
│   ├── nws_neon_bench.c
│   ├── nws_fused_bench.c
│   ├── nws_pipeline_bench.c
│   ├── nws_moe_bench.c
│   ├── nws_attention_bench.c
│   ├── gguf_expert_mapper.py
│   ├── layer_streaming_poc.py
│   └── cascade_prototype.py   # Cascade — deferred
├── experiments/               # Pinning A/B (layer_concentration, pinning_ab_test)
├── benchmarks/                # Dated bake-off notes (method + numbers)
├── docs/
│   ├── results.md
│   ├── proposal.md
│   └── llama-cpp-rfc.md
├── service/                   # Optional launchd / install helpers
├── Makefile                   # make all | test | help | clean
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] Feasibility simulation
- [x] Real SSD I/O validation
- [x] GGUF expert tensor mapping
- [x] MoE baseline (Ollama / standard load)
- [x] Expert-contiguous re-layout
- [x] End-to-end inference with real matmul (verified text output)
- [x] Text-calibrated pinning + mlock
- [x] Fused Q4 matmul + ARM NEON / Q8+vdot kernels
- [x] **End-to-end fused Q4 path** (`NWSEngine._forward_c` + `csrc/libtinygiant.c`)
- [x] Packaged install (`pip install -e .`, `tinygiant` / `tinygiant-setup` / `tinygiant-server`)
- [x] File llama.cpp RFC ([Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149))
- [x] Pinning A/B shipped with result JSON; non-uniform default when profile exists (+8% avg wall tok/s on M1 16GB)
- [ ] Fair bake-off refresh vs llama.cpp CPU (same machine / model / prompts)
- [ ] Multi-platform hardening (Linux, Windows, varied SSDs)
- [ ] llama.cpp integration (CPU offload path)
- [ ] Cascade / Learning Cache — **deferred**

## Related Discussions

- [llama.cpp #23324: MoE Expert Offload to Disk](https://github.com/ggml-org/llama.cpp/discussions/23324)
- [llama.cpp #27149: Expert-Aware SSD Streaming](https://github.com/ggml-org/llama.cpp/discussions/27149)
- [llama.cpp #20757: MoE Expert Cache Feature Request](https://github.com/ggml-org/llama.cpp/issues/20757)

## Why "TinyGiant"?

Tiny footprint. Giant model. You shouldn't need a giant machine to run a giant model.

## License

MIT
