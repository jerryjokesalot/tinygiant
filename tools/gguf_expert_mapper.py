#!/usr/bin/env python3
"""
GGUF Expert Mapper
==================
Parses a GGUF model file to map tensor locations, with special focus
on MoE expert weights. Outputs a JSON manifest that tells the Expert
Cascade engine exactly which byte ranges to read for each expert in
each layer.

This is the foundation for selective expert streaming from SSD.
"""

import json
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import BinaryIO


# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32
GGML_TYPE_SIZES = {
    0: 4,    # F32
    1: 2,    # F16
    2: 0.5,  # Q4_0 (block of 32 = 18 bytes → 0.5625 per element, approx 0.5)
    3: 0.5,  # Q4_1
    6: 0.5,  # Q5_0
    7: 0.5,  # Q5_1
    8: 1,    # Q8_0
    9: 1,    # Q8_1
    10: 0.5, # Q2_K
    11: 0.5, # Q3_K
    12: 0.5, # Q4_K
    13: 0.5, # Q5_K
    14: 1,   # Q6_K
    15: 0.5, # IQ2_XXS
    16: 0.5, # IQ2_XS
    17: 0.5, # IQ3_XXS
    18: 0.5, # IQ1_S
    19: 0.5, # IQ4_NL
    20: 0.5, # IQ3_S
    21: 0.5, # IQ2_S
    22: 0.5, # IQ4_XS
}

GGUF_TYPE_READERS = {
    0: ('B', 1),    # UINT8
    1: ('b', 1),    # INT8
    2: ('<H', 2),   # UINT16
    3: ('<h', 2),   # INT16
    4: ('<I', 4),   # UINT32
    5: ('<i', 4),   # INT32
    6: ('<f', 4),   # FLOAT32
    7: ('?', 1),    # BOOL
    8: None,        # STRING (special)
    9: None,        # ARRAY (special)
    10: ('<Q', 8),  # UINT64
    11: ('<q', 8),  # INT64
    12: ('<d', 8),  # FLOAT64
}


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    shape: list
    dtype: int
    offset: int  # relative to data section start
    size_bytes: int
    abs_offset: int  # absolute file offset
    layer: int = -1
    expert: int = -1
    tensor_type: str = ""  # "attention", "ffn", "expert", "router", "norm", "embed"


def read_string(f: BinaryIO) -> str:
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_value(f: BinaryIO, vtype: int):
    if vtype == 8:  # STRING
        return read_string(f)
    elif vtype == 9:  # ARRAY
        atype = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, atype) for _ in range(count)]
    else:
        fmt, size = GGUF_TYPE_READERS[vtype]
        return struct.unpack(fmt, f.read(size))[0]


def classify_tensor(name: str) -> tuple:
    """Classify a tensor by layer, expert index, and type."""
    layer = -1
    expert = -1
    tensor_type = "other"

    # Extract layer number: blk.N.xxx
    if "blk." in name:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "blk" and i + 1 < len(parts):
                try:
                    layer = int(parts[i + 1])
                except ValueError:
                    pass

    # Classify tensor type
    if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
        tensor_type = "expert_stacked"
    elif "ffn_gate_inp" in name:
        tensor_type = "router"
    elif ".ffn_gate." in name or ".ffn_up." in name or ".ffn_down." in name:
        # Could be per-expert (blk.N.ffn_gate.E.weight) or dense (blk.N.ffn_gate.weight)
        found_expert = False
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p in ("ffn_gate", "ffn_up", "ffn_down") and i + 1 < len(parts):
                try:
                    expert = int(parts[i + 1])
                    found_expert = True
                except ValueError:
                    pass
        tensor_type = "expert" if found_expert else "ffn"
    elif "ffn" in name:
        tensor_type = "ffn"
    elif "attn" in name:
        tensor_type = "attention"
    elif "norm" in name:
        tensor_type = "norm"
    elif "token_embd" in name or "output" in name:
        tensor_type = "embedding"

    return layer, expert, tensor_type


def estimate_tensor_size(shape: list, dtype: int) -> int:
    """Estimate tensor size in bytes from shape and dtype."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    bytes_per_element = GGML_TYPE_SIZES.get(dtype, 1)
    return int(n_elements * bytes_per_element)


def parse_gguf(filepath: str) -> dict:
    """Parse a GGUF file and return tensor metadata."""
    tensors = []
    metadata = {}

    with open(filepath, 'rb') as f:
        # Header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: 0x{magic:08X})")

        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        metadata['version'] = version
        metadata['n_tensors'] = n_tensors
        metadata['n_kv'] = n_kv

        # Read KV pairs
        kv = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            kv[key] = value

        # Extract useful metadata
        metadata['architecture'] = kv.get('general.architecture', 'unknown')
        metadata['name'] = kv.get('general.name', 'unknown')
        metadata['n_layers'] = kv.get(f'{metadata["architecture"]}.block_count', 0)
        metadata['n_experts'] = kv.get(f'{metadata["architecture"]}.expert_count', 0)
        metadata['n_experts_used'] = kv.get(f'{metadata["architecture"]}.expert_used_count', 0)

        # Read tensor infos
        tensor_infos_raw = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos_raw.append((name, n_dims, shape, dtype, offset))

        # Data section starts at next 32-byte aligned position
        current_pos = f.tell()
        data_start = (current_pos + 31) & ~31  # align to 32 bytes

        # Build tensor info objects
        for name, n_dims, shape, dtype, offset in tensor_infos_raw:
            size = estimate_tensor_size(shape, dtype)
            layer, expert, tensor_type = classify_tensor(name)
            abs_offset = data_start + offset

            tensors.append(TensorInfo(
                name=name,
                n_dims=n_dims,
                shape=shape,
                dtype=dtype,
                offset=offset,
                size_bytes=size,
                abs_offset=abs_offset,
                layer=layer,
                expert=expert,
                tensor_type=tensor_type,
            ))

    return {
        'metadata': metadata,
        'data_start': data_start,
        'tensors': tensors,
        'file_size': os.path.getsize(filepath),
    }


def build_expert_manifest(parsed: dict) -> dict:
    """Build a manifest of expert byte ranges for selective streaming."""
    metadata = parsed['metadata']
    tensors = parsed['tensors']

    manifest = {
        'model': metadata.get('name', 'unknown'),
        'architecture': metadata.get('architecture', 'unknown'),
        'n_layers': metadata.get('n_layers', 0),
        'n_experts': metadata.get('n_experts', 0),
        'n_experts_used': metadata.get('n_experts_used', 0),
        'is_moe': metadata.get('n_experts', 0) > 0,
        'file_size_gb': parsed['file_size'] / (1024**3),
        'layers': {},
        'summary': {},
    }

    # Group tensors by layer
    layers = defaultdict(lambda: {
        'shared': [],  # attention, norms — always loaded
        'router': [],  # MoE router weights
        'experts': defaultdict(list),  # per-expert FFN weights
        'ffn': [],     # dense FFN (non-MoE)
        'other': [],
    })

    total_shared = 0
    total_expert = 0
    total_router = 0

    for t in tensors:
        entry = {
            'name': t.name,
            'offset': t.abs_offset,
            'size': t.size_bytes,
            'shape': t.shape,
        }

        if t.layer < 0:
            layers[-1]['other'].append(entry)
            continue

        if t.tensor_type in ('attention', 'norm'):
            layers[t.layer]['shared'].append(entry)
            total_shared += t.size_bytes
        elif t.tensor_type == 'router':
            layers[t.layer]['router'].append(entry)
            total_router += t.size_bytes
        elif t.tensor_type == 'expert':
            layers[t.layer]['experts'][t.expert].append(entry)
            total_expert += t.size_bytes
        elif t.tensor_type == 'expert_stacked':
            # Fused expert tensors contain ALL experts in one tensor
            # We'd need to compute sub-ranges for individual experts
            layers[t.layer]['experts'][-1].append(entry)
            total_expert += t.size_bytes
        elif t.tensor_type == 'ffn':
            layers[t.layer]['ffn'].append(entry)
            total_shared += t.size_bytes
        else:
            layers[t.layer]['other'].append(entry)

    # Convert to serializable format
    for layer_idx in sorted(layers.keys()):
        layer_data = layers[layer_idx]
        layer_key = str(layer_idx)

        shared_size = sum(t['size'] for t in layer_data['shared'])
        expert_sizes = {}
        for exp_idx, exp_tensors in layer_data['experts'].items():
            expert_sizes[str(exp_idx)] = sum(t['size'] for t in exp_tensors)

        manifest['layers'][layer_key] = {
            'shared_bytes': shared_size,
            'shared_mb': round(shared_size / (1024**2), 1),
            'n_shared_tensors': len(layer_data['shared']),
            'router_bytes': sum(t['size'] for t in layer_data['router']),
            'expert_sizes': expert_sizes,
            'n_experts': len(layer_data['experts']),
            'ffn_bytes': sum(t['size'] for t in layer_data['ffn']),
        }

    # Summary
    n_layers = metadata.get('n_layers', 0) or len([k for k in layers if k >= 0])
    n_experts = metadata.get('n_experts', 0)
    n_experts_used = metadata.get('n_experts_used', 0)

    total_model = parsed['file_size']
    if n_experts > 0 and n_experts_used > 0:
        # Per-token read = shared + router + active experts only
        per_layer_shared = total_shared / max(n_layers, 1)
        per_layer_all_experts = total_expert / max(n_layers, 1)
        per_layer_active_experts = per_layer_all_experts * n_experts_used / max(n_experts, 1)
        per_layer_full = per_layer_shared + per_layer_all_experts
        per_layer_cascade = per_layer_shared + per_layer_active_experts

        manifest['summary'] = {
            'total_model_gb': round(total_model / (1024**3), 1),
            'total_shared_gb': round(total_shared / (1024**3), 2),
            'total_expert_gb': round(total_expert / (1024**3), 2),
            'total_router_gb': round(total_router / (1024**3), 4),
            'per_layer_full_mb': round(per_layer_full / (1024**2), 1),
            'per_layer_cascade_mb': round(per_layer_cascade / (1024**2), 1),
            'io_reduction_factor': round(per_layer_full / per_layer_cascade, 1) if per_layer_cascade > 0 else 0,
            'experts_total': n_experts,
            'experts_active_per_token': n_experts_used,
        }
    else:
        manifest['summary'] = {
            'total_model_gb': round(total_model / (1024**3), 1),
            'is_dense': True,
            'per_layer_mb': round(total_model / max(n_layers, 1) / (1024**2), 1),
        }

    return manifest


def print_report(manifest: dict):
    """Print a human-readable report of the expert mapping."""
    print("=" * 70)
    print("  GGUF EXPERT MAPPER — TENSOR MANIFEST")
    print("=" * 70)
    print(f"\n  Model: {manifest['model']}")
    print(f"  Architecture: {manifest['architecture']}")
    print(f"  File size: {manifest['file_size_gb']:.1f} GB")
    print(f"  Layers: {manifest['n_layers']}")
    print(f"  MoE: {'Yes' if manifest['is_moe'] else 'No'}")

    if manifest['is_moe']:
        print(f"  Experts: {manifest['n_experts']} total, {manifest['n_experts_used']} active per token")

    summary = manifest['summary']
    print(f"\n{'─' * 70}")

    if manifest['is_moe'] and not summary.get('is_dense'):
        print("  MoE STREAMING ANALYSIS")
        print(f"{'─' * 70}")
        print(f"  Shared weights (attention, norms): {summary['total_shared_gb']:.2f} GB")
        print(f"  Expert weights (all):              {summary['total_expert_gb']:.2f} GB")
        print(f"  Router weights:                    {summary['total_router_gb']:.4f} GB")
        print(f"\n  Per-layer read (standard):  {summary['per_layer_full_mb']:.1f} MB")
        print(f"  Per-layer read (cascade):   {summary['per_layer_cascade_mb']:.1f} MB")
        print(f"  I/O reduction factor:       {summary['io_reduction_factor']:.1f}x")

        # Project speeds using measured SSD throughput
        ssd_gbps = 2.97  # measured on M1 MacBook Pro
        n_layers = manifest['n_layers']

        full_time = (summary['per_layer_full_mb'] / 1024) / ssd_gbps * n_layers
        cascade_time = (summary['per_layer_cascade_mb'] / 1024) / ssd_gbps * n_layers

        print(f"\n  Projected I/O time per token (at {ssd_gbps:.2f} GB/s):")
        print(f"    Full model stream:     {full_time*1000:.0f} ms ({1/full_time:.1f} tok/s)")
        print(f"    Expert Cascade stream: {cascade_time*1000:.0f} ms ({1/cascade_time:.1f} tok/s)")
    else:
        print("  DENSE MODEL ANALYSIS")
        print(f"{'─' * 70}")
        print(f"  Per-layer size: {summary.get('per_layer_mb', 0):.1f} MB")

    # Show first few layers
    print(f"\n{'─' * 70}")
    print("  LAYER DETAILS (first 3)")
    print(f"{'─' * 70}")

    shown = 0
    for layer_key in sorted(manifest['layers'].keys(), key=lambda x: int(x)):
        if int(layer_key) < 0:
            continue
        if shown >= 3:
            print(f"  ... ({manifest['n_layers'] - 3} more layers)")
            break

        layer = manifest['layers'][layer_key]
        print(f"\n  Layer {layer_key}:")
        print(f"    Shared: {layer['shared_mb']:.1f} MB ({layer['n_shared_tensors']} tensors)")
        if layer['n_experts'] > 0:
            print(f"    Experts: {layer['n_experts']}")
            for exp_key, exp_size in sorted(layer['expert_sizes'].items(), key=lambda x: int(x[0])):
                print(f"      Expert {exp_key}: {exp_size/(1024**2):.1f} MB")
        if layer['ffn_bytes'] > 0:
            print(f"    FFN (dense): {layer['ffn_bytes']/(1024**2):.1f} MB")
        if layer['router_bytes'] > 0:
            print(f"    Router: {layer['router_bytes']/(1024**2):.3f} MB")

        shown += 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gguf_expert_mapper.py <model.gguf> [--json output.json]")
        sys.exit(1)

    model_path = os.path.expanduser(sys.argv[1])
    json_output = None
    if '--json' in sys.argv:
        json_idx = sys.argv.index('--json')
        if json_idx + 1 < len(sys.argv):
            json_output = sys.argv[json_idx + 1]

    if not os.path.exists(model_path):
        real_path = os.path.realpath(model_path)
        if os.path.exists(real_path):
            model_path = real_path
        else:
            print(f"Error: {model_path} not found")
            sys.exit(1)

    print(f"Parsing {model_path}...")
    parsed = parse_gguf(model_path)
    manifest = build_expert_manifest(parsed)
    print_report(manifest)

    if json_output:
        with open(json_output, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to {json_output}")
