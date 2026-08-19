import ctypes
import json
import mmap
from pathlib import Path

import numpy as np

from ._constants import EXPERT_KEYS_ORDER


class ExpertCache:
    """mmap-based expert cache. OS handles page caching at kernel speed."""

    def __init__(self, cache_dir, index, max_experts=256):
        self.cache_dir = Path(cache_dir)
        self.index = index
        self.max_experts = max_experts
        self.hits = 0
        self.misses = 0
        self.pinned_hits = 0
        self.pinned = set()
        self.access_counts = {}
        self.is_q4 = index.get("dtype") == "q4_k"
        self._lib = None

        self._mmaps = {}
        self._fds = {}
        self._expert_views = {}

        self._init_mmaps()

    def _init_mmaps(self):
        for layer_key, layer_info in self.index["layers"].items():
            layer_idx = int(layer_key)
            layer_file = self.cache_dir / layer_info["file"]
            fd = open(layer_file, "rb")
            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
            self._fds[layer_idx] = fd
            self._mmaps[layer_idx] = mm

            formats = layer_info.get("tensor_formats", {})
            for exp_key, exp_info in layer_info["expert_offsets"].items():
                exp_id = int(exp_key)
                offset = exp_info["offset"]
                views = {"_formats": formats}
                pos = offset
                for k in EXPERT_KEYS_ORDER:
                    size = exp_info["sizes"][k]
                    fmt = formats.get(k, "q4_k")
                    if fmt == "float16":
                        views[k] = np.frombuffer(mm, dtype=np.float16,
                                                 count=size // 2, offset=pos)
                    else:
                        views[k] = np.frombuffer(mm, dtype=np.uint8,
                                                 count=size, offset=pos)
                    pos += size
                self._expert_views[(layer_idx, exp_id)] = views

    def _lock_expert(self, layer, exp_id):
        views = self._expert_views[(layer, exp_id)]
        for k, v in views.items():
            if k != "_formats" and isinstance(v, np.ndarray):
                if self._lib:
                    self._lib.tg_mlock(v.ctypes.data, ctypes.c_size_t(v.nbytes))
                else:
                    np.sum(v)
        self.pinned.add((layer, exp_id))

    def set_lib(self, lib):
        self._lib = lib

    def pin_hot_experts(self, n_per_layer):
        profile = self.index.get("activation_profile")
        if not profile:
            return 0
        count = 0
        for layer_key in sorted(profile.keys(), key=int):
            for exp_id in profile[layer_key]["expert_ranking"][:n_per_layer]:
                self._lock_expert(int(layer_key), exp_id)
                count += 1
        self.hits = self.misses = self.pinned_hits = 0
        return count

    def pin_from_usage(self, n_per_layer, n_layers):
        per_layer = {}
        for (layer, expert), cnt in self.access_counts.items():
            per_layer.setdefault(layer, []).append((-cnt, expert))
        profile = self.index.get("activation_profile", {})

        count = 0
        for layer in range(n_layers):
            pinned_this_layer = set()

            entries = sorted(per_layer.get(layer, []))
            for _, exp_id in entries[:n_per_layer]:
                self._lock_expert(layer, exp_id)
                pinned_this_layer.add(exp_id)
                count += 1

            remaining = n_per_layer - len(pinned_this_layer)
            if remaining > 0 and str(layer) in profile:
                for exp_id in profile[str(layer)]["expert_ranking"]:
                    if exp_id not in pinned_this_layer:
                        self._lock_expert(layer, exp_id)
                        pinned_this_layer.add(exp_id)
                        count += 1
                        remaining -= 1
                        if remaining <= 0:
                            break

        self.hits = self.misses = self.pinned_hits = 0
        self.access_counts = {}
        return count

    def pin_nonuniform(self, pins_per_layer, n_layers):
        """Pin a variable number of experts per layer.

        pins_per_layer: dict mapping layer_idx -> n_experts to pin
        Uses calibration data first, fills from static profile.
        """
        per_layer = {}
        for (layer, expert), cnt in self.access_counts.items():
            per_layer.setdefault(layer, []).append((-cnt, expert))
        profile = self.index.get("activation_profile", {})

        count = 0
        for layer in range(n_layers):
            n_pin = pins_per_layer.get(layer, 0)
            if n_pin <= 0:
                continue
            pinned_this_layer = set()

            entries = sorted(per_layer.get(layer, []))
            for _, exp_id in entries[:n_pin]:
                self._lock_expert(layer, exp_id)
                pinned_this_layer.add(exp_id)
                count += 1

            remaining = n_pin - len(pinned_this_layer)
            if remaining > 0 and str(layer) in profile:
                for exp_id in profile[str(layer)]["expert_ranking"]:
                    if exp_id not in pinned_this_layer:
                        self._lock_expert(layer, exp_id)
                        pinned_this_layer.add(exp_id)
                        count += 1
                        remaining -= 1
                        if remaining <= 0:
                            break

        self.hits = self.misses = self.pinned_hits = 0
        self.access_counts = {}
        return count

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        if key in self.pinned:
            self.pinned_hits += 1
        self.hits += 1
        return self._expert_views[key]
