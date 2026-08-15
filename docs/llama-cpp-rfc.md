# RFC: Expert-Aware SSD Streaming for MoE Models on Memory-Constrained Hardware

## Summary

Enable interactive-speed inference for large MoE models (30B+) on machines with 8-16GB RAM by streaming only **active expert weights** from NVMe SSD per token, rather than loading the entire model into memory.

Current llama.cpp behavior: loads the entire GGUF file into RAM (or swap). For MoE models like Qwen3-30B-A3B (18GB), this causes swap thrashing on 16GB machines (<1 tok/s, unresponsive system).

Proposed behavior: keep only shared weights (attention + norms + router) in RAM, stream per-expert FFN tensors from SSD on demand. For Qwen3-30B-A3B (128 experts, 8 active per token), this reduces per-layer I/O from 263 MB to 31 MB — an **11.4x reduction**.

## Motivation

Real measurements on an M1 MacBook Pro (16GB):

| Scenario | Speed | RAM | System State |
|----------|-------|-----|-------------|
| qwen3:32b (dense) standard loading | <1 tok/s | 20 GB+ (swap) | Unresponsive |
| qwen3:30b-a3b (MoE) standard loading | 1.7 tok/s | 11.3 GB | Functional |
| Sequential layer read from SSD | 2.97 GB/s | — | — |
| Double-buffered pipeline | 6.06 GB/s | — | — |
| Expert Cascade prototype (real I/O) | 4.7 tok/s | ~2 GB | Responsive |

The SSD throughput is already fast enough. The bottleneck is that llama.cpp loads dormant experts into RAM where they compete for the same scarce memory. We built a prototype that reads only active expert byte ranges from disk and measured **4.7 tok/s over 100 tokens** — see [TinyGiant](https://github.com/jerryjokesalot/tinygiant) for the full measurements and tools.

## Design

### Phase 1: Expert Weight Map

Parse GGUF tensor metadata to build a byte-range map for each expert in each layer.

MoE models store expert FFN weights as stacked 3D tensors:
- `blk.N.ffn_gate_exps.weight` — shape `[n_experts, hidden_dim, expert_dim]`
- `blk.N.ffn_up_exps.weight` — same
- `blk.N.ffn_down_exps.weight` — same

Each expert's slice is contiguous: `offset + expert_idx * (tensor_bytes / n_experts)`.

Shared weights (loaded once, kept in RAM):
- `blk.N.attn_*.weight` — attention
- `blk.N.ffn_norm.weight` — layer norms
- `blk.N.ffn_gate_inp.weight` — MoE router
- `token_embd.weight`, `output.weight` — embeddings

### Phase 2: Async Expert Streaming

For each layer during forward pass:
1. Run attention computation on shared weights (in RAM)
2. Run MoE router to determine which experts are active
3. **Issue async reads** for active expert tensors from SSD
4. Complete attention computation while reads land
5. Execute expert FFN using streamed weights
6. Release expert weight buffers

Key optimization: the router computation (step 2) is a small matmul that completes before attention (step 4), giving us a **prefetch window** for expert I/O.

### Phase 3: Double-Buffered Pipeline

Overlap I/O for layer N+1's experts with compute on layer N's experts:

```
Time →
Layer 0: [attn+router] [read experts] [expert FFN]
Layer 1:                [attn+router]  [read experts] [expert FFN]
Layer 2:                               [attn+router]  [read experts] ...
```

Measured double-buffered throughput: 6.06 GB/s (2.09x over sequential).

## Implementation Notes

### Memory Budget

For Qwen3-30B-A3B (48 layers, 128 experts, 8 active per token):

| Component | Size | Lifetime |
|-----------|------|----------|
| Embeddings | ~500 MB | Persistent |
| Shared weights (attn + router + norms) per layer | ~10 MB | 2 layers buffered |
| Active expert weights per layer (8 of 128) | ~24 MB | 1 layer |
| KV cache | Variable | Persistent |
| **Total** | **~1-2 GB** | |

### GGUF Integration Points

- `llama_model_load()` — add expert-streaming mode that skips loading `ffn_*_exps` tensors
- `llama_decode_internal()` — before each MoE layer, trigger async expert reads
- New: `ggml_backend_buffer_ssd` — SSD-backed buffer type that reads on demand
- New: expert prefetch scheduler tied to the MoE router output

### Compatibility

- Falls back to standard loading when sufficient RAM is available
- Works with existing GGUF files (no format changes)
- Compatible with existing quantization types
- Can coexist with GPU offloading (stream CPU layers from SSD, GPU layers stay in VRAM)

## Related Work

- PowerInfer: GPU-CPU hybrid with activation-aware loading, but requires precomputation of activation patterns
- llama.cpp `--n-gpu-layers`: partial offloading, but still loads all layers into some memory
- [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757): MoE Expert Cache feature request (no implementation)
- [Discussion #13154](https://github.com/ggml-org/llama.cpp/discussions/13154): MoE expert offload discussion

## Open Questions

1. **Thread safety**: SSD reads on a background thread while compute runs on another — need to coordinate buffer lifecycle
2. **Expert prediction**: Can we predict experts for token N+1 based on token N's routing to prefetch even earlier?
3. **Batched verification**: When combined with speculative decoding, multiple tokens can share expert reads (read once, verify batch)
4. **GGUF alignment**: Expert slices within stacked tensors may not be 32-byte aligned — need to handle padding
