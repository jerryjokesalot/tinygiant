"""Expert re-layout: transforms MoE GGUF from interleaved to expert-contiguous layout."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

EXPERT_TENSORS = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
EXPERT_KEYS = ["gate", "up", "down"]
MAGIC = b"NWSMOE01"
Q4K_BSIZE = 144
Q6K_BSIZE = 210
QK_K = 256
BLOCK_SIZES = {12: Q4K_BSIZE, 14: Q6K_BSIZE}
GGML_Q4_K = 12
GGML_Q6_K = 14


def analyze_model(reader):
    fields = {}
    for name in reader.fields:
        f = reader.fields[name]
        try:
            fields[name] = f.parts[f.data[0]]
        except (IndexError, TypeError):
            pass

    n_layers = n_experts = n_experts_used = embed_dim = None
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

    return {"n_layers": n_layers, "n_experts": n_experts,
            "n_experts_used": n_experts_used, "embed_dim": embed_dim}


def get_expert_shapes(tensor_map, layer_idx):
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
    return gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def extract_experts(tensor_map, layer_idx, n_experts):
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


def compute_q4_expert_sizes(tensor_map, layer_idx):
    sizes = {}
    for key, suffix in zip(EXPERT_KEYS, EXPERT_TENSORS):
        name = f"blk.{layer_idx}.{suffix}"
        t = tensor_map[name]
        qtype = int(t.tensor_type)
        bsize = BLOCK_SIZES.get(qtype, Q4K_BSIZE)
        data_shape = list(t.data.shape)
        n_experts = data_shape[0]
        out_dim = data_shape[1]
        bytes_per_row = data_shape[2]
        expert_bytes = out_dim * bytes_per_row
        bpr = bytes_per_row // bsize
        in_dim = bpr * QK_K
        sizes[key] = {
            "n_experts": n_experts, "out_dim": out_dim, "in_dim": in_dim,
            "bpr": bpr, "expert_bytes": expert_bytes,
            "total_bytes": int(t.n_bytes), "quant_type": qtype,
        }
    return sizes


def extract_experts_q4(tensor_map, layer_idx, n_experts):
    sizes = compute_q4_expert_sizes(tensor_map, layer_idx)
    experts = {}
    formats = {}

    for key, suffix in zip(EXPERT_KEYS, EXPERT_TENSORS):
        name = f"blk.{layer_idx}.{suffix}"
        t = tensor_map[name]
        qtype = int(t.tensor_type)
        expert_bytes = sizes[key]["expert_bytes"]

        if qtype == GGML_Q4_K:
            print(f"    {key}: Q4_K raw, {t.n_bytes/1024**2:.0f} MB"
                  f" -> {n_experts} x {expert_bytes/1024:.0f} KB")
            formats[key] = "q4_k"
        elif qtype == GGML_Q6_K:
            print(f"    {key}: Q6_K raw, {t.n_bytes/1024**2:.0f} MB"
                  f" -> {n_experts} x {expert_bytes/1024:.0f} KB")
            formats[key] = "q6_k"
        else:
            raise ValueError(f"Unsupported quant type {qtype} for {name}")

        for exp_id in range(n_experts):
            if exp_id not in experts:
                experts[exp_id] = {}
            experts[exp_id][key] = bytes(t.data[exp_id])

    sizes["_formats"] = formats
    return experts, sizes


def write_expert_cache_q4(output_dir, layer_idx, experts, q4_sizes):
    layer_file = output_dir / f"layer_{layer_idx:03d}.q4.bin"
    offsets = {}
    current_offset = 0
    with open(layer_file, "wb") as f:
        for exp_id in sorted(experts.keys()):
            expert = experts[exp_id]
            exp_offset = current_offset
            exp_sizes = {}
            for key in EXPERT_KEYS:
                data = expert[key]
                f.write(data)
                exp_sizes[key] = len(data)
                current_offset += len(data)
            offsets[str(exp_id)] = {"offset": exp_offset, "sizes": exp_sizes}
    return {"file": layer_file.name, "total_bytes": current_offset, "expert_offsets": offsets}


def write_expert_cache(output_dir, layer_idx, experts, shapes):
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
            offsets[str(exp_id)] = {"offset": exp_offset, "sizes": exp_sizes}
    return {"file": layer_file.name, "total_bytes": current_offset, "expert_offsets": offsets}


def profile_activation(tensor_map, n_layers, n_experts, top_k=8, n_samples=1000):
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
    parser = argparse.ArgumentParser(
        prog="tinygiant-relayout",
        description="TinyGiant — transform MoE GGUF to expert-contiguous layout")
    parser.add_argument("model", help="Path to GGUF MoE model")
    parser.add_argument("output", help="Output directory for expert cache")
    parser.add_argument("--layers", help="Comma-separated layer indices (default: all)")
    parser.add_argument("--verify", action="store_true", help="Verify output matches original")
    parser.add_argument("--format", choices=["f16", "q4"], default="f16",
                        help="Output format: f16 (dequantized) or q4 (raw Q4 blocks)")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TinyGiant Expert Re-Layout")
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
        np.prod([s for s in info["shape"][:2]]) * 2
        for info in shapes.values()
    )
    print(f"\n  Per expert (float16): {per_expert_bytes / 1024:.0f} KB")
    print(f"  Per layer (all experts): {per_expert_bytes * n_experts / 1024**2:.0f} MB")
    print(f"  Selected layers: {len(layer_indices)}")
    print(f"  Estimated output: {per_expert_bytes * n_experts * len(layer_indices) / 1024**3:.1f} GB")

    print("\nProfiling expert activations...")
    t0 = time.perf_counter()
    activation_profile = profile_activation(tensor_map, n_layers, n_experts)
    print(f"  Profiled in {time.perf_counter()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"RE-LAYOUT: Processing {len(layer_indices)} layers")
    print(f"{'='*60}")

    use_q4 = args.format == "q4"

    index = {
        "magic": MAGIC.decode(),
        "version": 2 if use_q4 else 1,
        "model": os.path.basename(model_path),
        "model_path": model_path,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_experts_used": arch["n_experts_used"],
        "embed_dim": arch["embed_dim"],
        "dtype": "q4_k" if use_q4 else "float16",
        "expert_shapes": shapes,
        "layers": {},
        "activation_profile": activation_profile,
    }

    if use_q4:
        print(f"\n  Format: Mixed Q4_K/Q6_K raw storage")
        q4_sizes = compute_q4_expert_sizes(tensor_map, layer_indices[0])
        index["tensor_formats"] = {}
        for k, v in q4_sizes.items():
            if k.startswith("_"):
                continue
            index["q4_sizes"] = index.get("q4_sizes", {})
            index["q4_sizes"][k] = {"out_dim": v["out_dim"], "in_dim": v["in_dim"]}

    total_bytes = 0
    total_time = 0

    for i, layer_idx in enumerate(layer_indices):
        print(f"\n--- Layer {layer_idx} ({i+1}/{len(layer_indices)}) ---")
        t0 = time.perf_counter()

        if use_q4:
            print("  Extracting expert blocks (raw quantized)...")
            experts, q4s = extract_experts_q4(tensor_map, layer_idx, n_experts)
            print("  Writing expert-contiguous cache...")
            layer_index = write_expert_cache_q4(output_dir, layer_idx, experts, q4s)
            layer_index["tensor_formats"] = q4s.get("_formats", {})
        else:
            print("  Extracting experts...")
            experts = extract_experts(tensor_map, layer_idx, n_experts)
            print("  Writing expert-contiguous cache...")
            layer_index = write_expert_cache(output_dir, layer_idx, experts, shapes)

        index["layers"][layer_idx] = layer_index
        total_bytes += layer_index["total_bytes"]

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        print(f"  Done: {layer_index['total_bytes']/1024**2:.0f} MB in {elapsed:.1f}s")
        del experts

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
        print(f"\n--- HOT EXPERT SUMMARY ---")
        for n_pin in [8, 16, 32, 64]:
            pinned_mem = n_pin * per_expert_bytes * n_layers / 1024**3
            print(f"  Pin top-{n_pin:>3}/layer: {pinned_mem:.1f} GB RAM")
