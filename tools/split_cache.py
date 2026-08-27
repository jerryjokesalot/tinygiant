"""Split a full NWS expert cache into coordinator and worker halves.

Usage:
    python tools/split_cache.py <cache_dir> <split_layer> <output_dir>

Produces:
    <output_dir>/coordinator/  — layers 0 to split_layer-1
    <output_dir>/worker/       — layers split_layer to 47

Each half gets its own index.json with only its layers.
Layer binary files are symlinked, not copied.
"""
import json
import os
import sys
from pathlib import Path


def split_cache(cache_dir, split_layer, output_dir):
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)

    with open(cache_dir / "index.json") as f:
        index = json.load(f)

    for role, start, end in [("coordinator", 0, split_layer),
                              ("worker", split_layer, 48)]:
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)

        role_index = dict(index)
        role_index["layers"] = {}

        for layer_key, layer_info in index["layers"].items():
            layer_idx = int(layer_key)
            if start <= layer_idx < end:
                role_index["layers"][layer_key] = layer_info
                # Symlink the layer binary file
                src = cache_dir / layer_info["file"]
                dst = role_dir / layer_info["file"]
                if not dst.exists():
                    dst.symlink_to(src.resolve())

        with open(role_dir / "index.json", "w") as f:
            json.dump(role_index, f, indent=2)

        print(f"{role}: {len(role_index['layers'])} layers -> {role_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <cache_dir> <split_layer> <output_dir>")
        sys.exit(1)
    split_cache(sys.argv[1], int(sys.argv[2]), sys.argv[3])
