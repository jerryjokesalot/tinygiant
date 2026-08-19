"""
NWS Expert Re-Layout Tool

Transforms MoE GGUF models from interleaved expert layout to expert-contiguous
layout, eliminating the ~100x page fault amplification that makes MoE models
unusable on consumer hardware with limited RAM.

Problem: GGUF stores all 128 experts in one interleaved tensor [dim, intermediate, 128].
Accessing one expert page-faults every page of the tensor. On a 16GB laptop, this
causes the 30B model to run at 0.11 tok/s.

Solution: Re-lay expert data so each expert is contiguous. Page faults load only
the needed expert. With MBOM-guided pinning of hot experts, most accesses hit RAM.

Input: Standard GGUF MoE model (e.g., Qwen3-30B-A3B-Q4_K_M.gguf)
Output: Directory with expert-contiguous binary data + index

Usage:
  python nws_expert_relayout.py ~/models/Qwen3-30B-A3B-Q4_K_M.gguf ./nws_cache/
  python nws_expert_relayout.py ~/models/Qwen3-30B-A3B-Q4_K_M.gguf ./nws_cache/ --layers 0,24,47
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

EXPERT_TENSORS = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
EXPERT_KEYS = ["gate", "up", "down"]

MAGIC = b"NWSMOE01"


def analyze_model(reader):
    """Extract MoE architecture info from GGUF metadata."""
    fields = {}
    for name in reader.fields:
        f = reader.fields[name]
        try:
            fields[name] = f.parts[f.data[0]]
        except (IndexError, TypeError):
            pass

    n_layers = None
    n_experts = None
    n_experts_used = None
    embed_dim = None

    for key, val in fields.items():
        v = int(val[0]) if hasattr(val, '__len__') else int(val)
        if "block_count" in key:
            n_layers = v
        elif "expert_count" in key and "used" not in key:
            n_experts = v
        elif "expert_used_count" in key:
            n_experts_used = v
        elif "embedding_length" in key:
            embed_dim = v

    return {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_experts_used": n_experts_used,
        "embed_dim": embed_dim,
    }


def get_expert_shapes(tensor_map, layer_idx):
    """Get the shapes of expert tensors for a given layer."""
    shapes = {}
    for key, suffix in zip(EXPERT_KEYS, EXPERT_TENSORS):
        name = f"blk.{layer_idx}.{suffix}"
        if name in tensor_map:
            t = tensor_map[name]
            shapes[key] = {
                "shape": [int(x) for x in t.shape],
                "type": int(t.tensor_type),
                "n_bytes": int(t.n_bytes),
            }
    return shapes


def dequantize_expert_tensor(tensor):
    """Dequantize a GGUF expert tensor. gguf_dequantize returns HF convention."""
    data = gguf_dequantize(tensor.data, tensor.tensor_type)
    return data.astype(np.float32)


def extract_experts(tensor_map, layer_idx, n_experts):
    """Extract individual experts from interleaved tensors.

    Returns dict mapping expert_id -> {gate: array, up: array, down: array}
    where each array is float16 with shape [out_dim, in_dim].
    """
    deq = {}
    for key, suffix in zip(EXPERT_KEYS, EXPERT_TENSORS):
        name = f"blk.{layer_idx}.{suffix}"
        if name not in tensor_map:
            raise ValueError(f"Missing tensor: {name}")
        t0 = time.perf_counter()
        data = dequantize_expert_tensor(tensor_map[name])
        elapsed = time.perf_counter() - t0
        deq[key] = data
        print(f"    {key}: dequantized {tensor_map[name].n_bytes/1024**2:.0f} MB"
              f" -> {data.shape} in {elapsed:.1f}s")

    experts = {}
    for exp_id in range(n_experts):
        expert = {}
        for key in EXPERT_KEYS:
            expert[key] = deq[key][exp_id].astype(np.float16)
        experts[exp_id] = expert

    return experts


def write_expert_cache(output_dir, layer_idx, experts, shapes):
    """Write expert-contiguous binary data for one layer.

    File layout: expert 0 (gate|up|down), expert 1 (gate|up|down), ...
    Each weight is contiguous float16.
    """
    layer_file = output_dir / f"layer_{layer_idx:03d}.bin"

    offsets = {}
    current_offset = 0

    with open(layer_file, "wb") as f:
        for exp_id in sorted(experts.keys()):
            expert = experts[exp_id]
            exp_offset = current_offset
            exp_sizes = {}

            for key in EXPERT_KEYS:
                data = expert[key]
                raw = data.tobytes()
                f.write(raw)
                exp_sizes[key] = len(raw)
                current_offset += len(raw)

            offsets[str(exp_id)] = {
                "offset": exp_offset,
                "sizes": exp_sizes,
            }

    return {
        "file": layer_file.name,
        "total_bytes": current_offset,
        "expert_offsets": offsets,
    }


def verify_expert(tensor_map, layer_idx, exp_id, cache_dir, layer_index):
    """Verify that cached expert data matches original."""
    layer_file = cache_dir / layer_index["file"]
    offsets = layer_index["expert_offsets"][str(exp_id)]

    with open(layer_file, "rb") as f:
        f.seek(offsets["offset"])
        errors = []

        for key, suffix in zip(EXPERT_KEYS, EXPERT_TENSORS):
            cached_bytes = f.read(offsets["sizes"][key])
            cached = np.frombuffer(cached_bytes, dtype=np.float16)

            name = f"blk.{layer_idx}.{suffix}"
            original = dequantize_expert_tensor(tensor_map[name])
            expected = original[exp_id].astype(np.float16).flatten()

            if not np.array_equal(cached, expected):
                max_diff = np.max(np.abs(cached.astype(np.float32) - expected.astype(np.float32)))
                errors.append(f"{key}: max_diff={max_diff:.6f}")

        return errors


def profile_activation(tensor_map, n_layers, n_experts, top_k=8, n_samples=1000):
    """Profile expert activation patterns to identify hot experts."""
    np.random.seed(42)

    embed_dim = None
    for name in tensor_map:
        if "ffn_gate_inp" in name:
            t = tensor_map[name]
            shape = [int(x) for x in t.shape]
            embed_dim = shape[0]
            break

    if embed_dim is None:
        return None

    inputs = np.random.randn(n_samples, embed_dim).astype(np.float32)
    layer_stats = {}

    for layer_idx in range(n_layers):
        name = f"blk.{layer_idx}.ffn_gate_inp.weight"
        if name not in tensor_map:
            continue

        t = tensor_map[name]
        router = np.frombuffer(t.data, dtype=np.float32).reshape([int(x) for x in t.shape])

        logits = inputs @ router
        top_indices = np.argsort(logits, axis=1)[:, -top_k:]

        counts = np.zeros(n_experts, dtype=int)
        for i in range(n_samples):
            for j in top_indices[i]:
                counts[j] += 1

        sorted_idx = np.argsort(counts)[::-1]
        layer_stats[layer_idx] = {
            "expert_ranking": [int(x) for x in sorted_idx],
            "activation_counts": [int(counts[x]) for x in sorted_idx],
        }

    return layer_stats


def main():
    parser = argparse.ArgumentParser(description="NWS Expert Re-Layout Tool")
    parser.add_argument("model", help="Path to GGUF MoE model")
    parser.add_argument("output", help="Output directory for expert cache")
    parser.add_argument("--layers", help="Comma-separated layer indices (default: all)")
    parser.add_argument("--verify", action="store_true", help="Verify output matches original")
    parser.add_argument("--profile", action="store_true", help="Profile expert activations")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("NWS Expert Re-Layout Tool")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"File size: {os.path.getsize(model_path) / 1024**3:.2f} GB")

    print("\nLoading GGUF reader...")
    t0 = time.perf_counter()
    reader = GGUFReader(model_path)
    tensor_map = {t.name: t for t in reader.tensors}
    print(f"Loaded in {time.perf_counter()-t0:.1f}s ({len(reader.tensors)} tensors)")

    arch = analyze_model(reader)
    print(f"\nArchitecture:")
    print(f"  Layers: {arch['n_layers']}")
    print(f"  Experts: {arch['n_experts']} ({arch['n_experts_used']} active per token)")
    print(f"  Embedding dim: {arch['embed_dim']}")

    n_layers = arch["n_layers"]
    n_experts = arch["n_experts"]

    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        layer_indices = list(range(n_layers))

    shapes = get_expert_shapes(tensor_map, layer_indices[0])
    per_expert_bytes = sum(
        np.prod([s for s in info["shape"][:2]]) * 2  # float16
        for info in shapes.values()
    )
    print(f"\n  Per expert (float16): {per_expert_bytes / 1024:.0f} KB")
    print(f"  Per layer (all experts): {per_expert_bytes * n_experts / 1024**2:.0f} MB")
    print(f"  Selected layers: {len(layer_indices)}")
    print(f"  Estimated output: {per_expert_bytes * n_experts * len(layer_indices) / 1024**3:.1f} GB")

    # Profile expert activations
    activation_profile = None
    if args.profile or True:  # always profile for now
        print("\nProfiling expert activations...")
        t0 = time.perf_counter()
        activation_profile = profile_activation(tensor_map, n_layers, n_experts)
        print(f"  Profiled in {time.perf_counter()-t0:.1f}s")

    # Re-layout
    print(f"\n{'='*60}")
    print(f"RE-LAYOUT: Processing {len(layer_indices)} layers")
    print(f"{'='*60}")

    index = {
        "magic": MAGIC.decode(),
        "version": 1,
        "model": os.path.basename(model_path),
        "model_path": model_path,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_experts_used": arch["n_experts_used"],
        "embed_dim": arch["embed_dim"],
        "dtype": "float16",
        "expert_shapes": shapes,
        "layers": {},
        "activation_profile": activation_profile,
    }

    total_bytes = 0
    total_time = 0

    for i, layer_idx in enumerate(layer_indices):
        print(f"\n--- Layer {layer_idx} ({i+1}/{len(layer_indices)}) ---")
        t0 = time.perf_counter()

        print("  Extracting experts...")
        experts = extract_experts(tensor_map, layer_idx, n_experts)

        print("  Writing expert-contiguous cache...")
        layer_index = write_expert_cache(output_dir, layer_idx, experts, shapes)
        index["layers"][layer_idx] = layer_index
        total_bytes += layer_index["total_bytes"]

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        print(f"  Done: {layer_index['total_bytes']/1024**2:.0f} MB in {elapsed:.1f}s")

        # Optional verification
        if args.verify:
            print("  Verifying...")
            for exp_id in [0, 63, 127]:
                errors = verify_expert(tensor_map, layer_idx, exp_id, output_dir, layer_index)
                status = "OK" if not errors else f"ERRORS: {errors}"
                print(f"    Expert {exp_id}: {status}")

        del experts

    # Write index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Layers processed: {len(layer_indices)}")
    print(f"  Total cache size: {total_bytes / 1024**3:.2f} GB")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Index: {index_path}")

    if activation_profile:
        # Show hot expert summary
        print(f"\n--- HOT EXPERT SUMMARY ---")
        for n_pin in [8, 16, 32, 64]:
            pinned_mem = n_pin * per_expert_bytes * n_layers / 1024**3
            print(f"  Pin top-{n_pin:>3}/layer: {pinned_mem:.1f} GB RAM")

    # Access pattern comparison
    print(f"\n--- ACCESS PATTERN COMPARISON ---")
    original_tensor = tensor_map[f"blk.{layer_indices[0]}.ffn_gate_exps.weight"]
    orig_bytes = original_tensor.n_bytes
    page_size = 16384  # Apple Silicon
    orig_pages = orig_bytes / page_size
    expert_pages = per_expert_bytes / 3 / page_size  # one tensor

    print(f"  Original (interleaved):")
    print(f"    Loading 1 expert touches: {orig_pages:.0f} pages ({orig_bytes/1024**2:.0f} MB)")
    print(f"    Loading 8 experts touches: {orig_pages:.0f} pages (same — ALL pages)")
    print(f"  Re-laid (contiguous):")
    print(f"    Loading 1 expert touches: {expert_pages:.0f} pages ({per_expert_bytes/3/1024:.0f} KB)")
    print(f"    Loading 8 experts touches: {expert_pages*8:.0f} pages ({per_expert_bytes*8/3/1024:.0f} KB)")
    print(f"  Page fault reduction: {orig_pages / max(1, expert_pages*8):.0f}x")


if __name__ == "__main__":
    main()
