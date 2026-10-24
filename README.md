# TinyGiant

**Tiny footprint. Giant model.** Run 30B+ parameter MoE models on a laptop with 8–16GB RAM and a standard SSD. No GPU required for the CPU path — **cpu-only is the ship default**. A **hybrid** profile (Metal attention+KV; experts stay CPU+SSD+pin) is available as an **optional / experimental** opt-in on Apple Silicon when libtinygiant is built on Darwin. On M1 16GB after the Q4 Metal fix, hybrid decode was ~**1.15** vs cpu-only ~**3.01** tok/s (**≈0.38×**); see [`benchmarks/benchmark_2026-10-24_m1_16gb_hybrid_q4fix.md`](benchmarks/benchmark_2026-10-24_m1_16gb_hybrid_q4fix.md). `HYBRID_ATTN_READY` stays **false** — do not treat hybrid as faster or recommended.

Large AI models are locked behind expensive hardware. A 30B model needs 18GB+ RAM just to load — more than most laptops have. TinyGiant streams only the experts the model actually uses from SSD, keeping dormant weights on disk.

Modern Mixture-of-Experts (MoE) models like Qwen3-30B activate only **3 billion** of their 30 billion parameters per token. That's 128 experts per layer, but only 8 fire at any moment. Many inference stacks still keep all 128 resident. TinyGiant doesn't.

## What is actually novel here

TinyGiant's packaging contribution is **contiguous expert re-layout** plus **text-calibrated mlock pinning** — not new math. Fused Q4×Q8 kernels (including ARM `vdotq_s32`) are the same family of quantized matmul techniques used elsewhere; we don't claim them as a TinyGiant invention. What we ship is a downloadable path that:

1. Repacks GGUF expert tensors so each expert is a contiguous SSD read (instead of page-faulting an interleaved tensor).
2. Calibrates which experts to pin from a short real-text pass, then `mlock`s them so the OS can't steal the hot set.
3. Runs end-to-end chat through a small Python package + `libtinygiant` C library.

Closest peer work: [llama.cpp MoE SSD streaming discussion / PRs](https://github.com/ggml-org/llama.cpp/discussions/23324). Contiguous layout and calibrated pinning are complementary to that line of work.

## Results (honest)

### Sustained chat — M1 16GB bake-off (2026-10-24)

Fair peer on **Apple M1, 16GB**, same model (`Qwen3-30B-A3B-Q4_K_M`), same prompts, measured **decode tok/s** (preferred) after prefill. Method + raw JSON: [`benchmarks/benchmark_2026-10-24_m1_16gb.md`](benchmarks/benchmark_2026-10-24_m1_16gb.md).

| Configuration | Decode tok/s | vs llama.cpp CPU |
|---------------|-------------:|-----------------:|
| llama.cpp CPU (`-ngl 0 -t 8`, llama-bench tg128) | **1.06 ± 0.05** | baseline |
| TinyGiant `pin_nonuniform` (shipped default) | **2.68 ± 0.62** | **~2.5×** |
| TinyGiant `pin48` (uniform) | **2.78 ± 0.64** | **~2.6×** |

**Verdict:** TinyGiant beats llama.cpp CPU by about **2.5–2.6×** on this machine. In *this* 128-token bake-off, uniform `pin48` slightly edged `pin_nonuniform` on mean decode; the earlier 60-token A/B wall metric had non-uniform ahead (+8%) and remains the ship basis — treat pin choice as within noise on the bake-off decode metric. These are **not** the warm ~25.9 tok/s compute ceiling.

### Prior M3 note (2026-08-19)

Earlier bake-off on **Apple M3, 16GB**, cold file cache (`purge`), 128-token decode — see [`benchmarks/benchmark_2026-08-19_m3_16gb.md`](benchmarks/benchmark_2026-08-19_m3_16gb.md):

| Configuration | tok/s | vs llama.cpp CPU |
|---------------|-------|------------------|
| llama.cpp CPU (`ngl=0`, 8 threads) | 1.37 ± 0.31 | baseline |
| TinyGiant (mlock pin48) | 3.70–4.54 | ~2.7–3.3× |
| TinyGiant (mlock pin64) | 4.83–5.08 | ~3.5–3.7× |

Different machine / day; do not average with the M1 table.

### Pinning A/B (M1 16GB, 2026-10-24) — ship basis for non-uniform default

Wall-clock `NWSEngine.generate()` tok/s including prefill (60 decode tokens). Raw: [`experiments/pinning_ab_results.json`](experiments/pinning_ab_results.json).

| Task | Uniform pin48 | Non-uniform | Delta |
|------|--------------:|------------:|------:|
| coding | 1.70 | 1.89 | +11.4% |
| reasoning | 1.37 | 1.44 | +5.5% |
| creative | 2.19 | 2.39 | +9.0% |
| math | 1.44 | 1.51 | +5.4% |
| factual | 2.80 | 3.01 | +7.6% |
| **average** | **1.90** | **2.05** | **+8.0%** |

Shipped default is non-uniform when the concentration profile exists; uniform pin48 otherwise.

### Warm compute ceiling (not chat)

Microbenchmarks of the fused Q4×Q8 / Q8+vdot kernel on Apple Silicon can hit **~25.9 tok/s** when experts are already warm in RAM and I/O is zero. That is a **compute ceiling**, not sustained chat speed. Sustained chat is I/O- and pin-hit-rate limited; see the M1 / M3 tables above.

Earlier dequant+BLAS and prototype cascade numbers live in [`docs/results.md`](docs/results.md). Cascade / Learning Cache remain deferred.

### RAM budget (pinning hot experts, 16 GB machine)

With attention mlock'd and a pin48–pin64 warm expert set, the working set fits in 16 GB with OS headroom. Exact budgets: [M1 bake-off](benchmarks/benchmark_2026-10-24_m1_16gb.md), [M3 note](benchmarks/benchmark_2026-08-19_m3_16gb.md).

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

Setup also probes hardware and writes `profile.json` next to the cache
(`cpu-only` / `hybrid` / `gpu-heavy`). **Selected defaults to `cpu-only`**
(`HYBRID_ATTN_READY` is false). On Apple Silicon the probe may *recommend*
`hybrid` while still selecting `cpu-only`. Hybrid/Metal is **optional and
experimental** — slower than cpu-only on the M1 16GB bake-off (~0.38× decode);
not recommended for normal use. Never auto-selects `gpu-heavy` on ≤16 GiB RAM.
Override anytime:

```bash
tinygiant --model … --cache … --profile hybrid   # experimental Metal attn; slower today; fail-opens to CPU
tinygiant-server --model … --cache … --profile cpu-only
```

### 4. First chat (CLI)

`--pin 48` is the total expert budget (48 × 48 layers = 2304). With the shipped concentration profile, TinyGiant allocates that budget **non-uniformly** (entropy-weighted); without a profile it falls back to uniform pin48.

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
| `tinygiant-setup` | One-shot relayout → expert cache + hardware `profile.json` |
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
│   ├── probe.py               # Hardware probe (Metal/RAM/CUDA/SSD class)
│   ├── profile.py             # cpu-only / hybrid / gpu-heavy selection
│   ├── attention.py           # AttentionBackend protocol (CPU + Metal)
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
- [x] Fair bake-off vs llama.cpp CPU on M1 16GB (2026-10-24; ~2.5–2.6× decode; see `benchmarks/benchmark_2026-10-24_m1_16gb.md`)
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
