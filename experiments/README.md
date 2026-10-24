# Experiments

Pinning studies for TinyGiant’s expert `mlock` budget.

## Scripts

| Script | Role |
|--------|------|
| `layer_concentration.py` | Per-layer routing entropy → suggested non-uniform pin map |
| `pinning_ab_test.py` | Uniform pin48 vs `pin_nonuniform` on the same prompts |

Both expect a Q4 GGUF + NWS cache (see repo README). Paths default to `~/models/Qwen3-30B-A3B-Q4_K_M.gguf` and `/tmp/nws_q4_cache_v2`.

## Results (tracked)

| File | Contents |
|------|----------|
| `layer_concentration_results.json` | Entropy stats + `suggested_pins` (budget 2304) |
| `pinning_ab_results.json` | Per-task wall tok/s, hit rates, averages |

**2026-10-24 / Apple M1 16GB:** non-uniform averaged **2.05 tok/s** vs uniform **1.90** (**+7.95%**). Rates are wall-clock `generate()` including prefill — not the warm fused-kernel ceiling (~25.9 tok/s).

The runtime default loads `suggested_pins` from `tinygiant/suggested_pins.json` (packaged copy of the concentration map) or this directory’s concentration JSON. See `NWSEngine.pin_experts`.
