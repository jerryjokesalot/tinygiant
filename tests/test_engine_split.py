"""Tests for split-layer engine loading.

These tests verify constructor parameters and layer filtering without
needing a real GGUF model. We mock the heavy loading and test the
range logic.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tinygiant._constants import N_LAYERS, EMBED_DIM


def test_cache_layer_filter():
    """ExpertCache should only mmap layers in [layer_start, layer_end)."""
    from tinygiant.cache import ExpertCache

    # Build a minimal fake index with layers 0-47
    index = {"dtype": "q4_k", "layers": {}}
    for i in range(N_LAYERS):
        index["layers"][str(i)] = {
            "file": f"layer_{i}.bin",
            "expert_offsets": {},
            "tensor_formats": {},
        }

    # Patch _init_mmaps to track which layers it tries to open
    opened_layers = []
    original_init = ExpertCache._init_mmaps

    def mock_init(self):
        for layer_key in self.index["layers"]:
            opened_layers.append(int(layer_key))

    with patch.object(ExpertCache, '_init_mmaps', mock_init):
        ec = ExpertCache("/fake", index, layer_start=24, layer_end=48)

    assert sorted(opened_layers) == list(range(24, 48))


def test_cache_default_range_opens_all():
    """Default layer_start=0, layer_end=48 should open all layers."""
    from tinygiant.cache import ExpertCache

    index = {"dtype": "q4_k", "layers": {}}
    for i in range(N_LAYERS):
        index["layers"][str(i)] = {
            "file": f"layer_{i}.bin",
            "expert_offsets": {},
            "tensor_formats": {},
        }

    opened_layers = []

    def mock_init(self):
        for layer_key in self.index["layers"]:
            opened_layers.append(int(layer_key))

    with patch.object(ExpertCache, '_init_mmaps', mock_init):
        ec = ExpertCache("/fake", index)

    assert sorted(opened_layers) == list(range(48))


def test_cache_activation_profile_filtered():
    """activation_profile should only include in-range layers."""
    from tinygiant.cache import ExpertCache

    index = {"dtype": "q4_k", "layers": {}, "activation_profile": {}}
    for i in range(N_LAYERS):
        index["layers"][str(i)] = {
            "file": f"layer_{i}.bin",
            "expert_offsets": {},
            "tensor_formats": {},
        }
        index["activation_profile"][str(i)] = {"expert_ranking": [0, 1, 2]}

    def mock_init(self):
        pass

    with patch.object(ExpertCache, '_init_mmaps', mock_init):
        ec = ExpertCache("/fake", index, layer_start=24, layer_end=48)

    profile_keys = [int(k) for k in ec.index["activation_profile"].keys()]
    assert sorted(profile_keys) == list(range(24, 48))


def test_cache_stores_layer_range():
    """ExpertCache should store layer_start and layer_end."""
    from tinygiant.cache import ExpertCache

    index = {"dtype": "q4_k", "layers": {}}

    def mock_init(self):
        pass

    with patch.object(ExpertCache, '_init_mmaps', mock_init):
        ec = ExpertCache("/fake", index, layer_start=12, layer_end=36)

    assert ec.layer_start == 12
    assert ec.layer_end == 36


def test_cache_first_half_filter():
    """layer_start=0, layer_end=24 should only open layers 0-23."""
    from tinygiant.cache import ExpertCache

    index = {"dtype": "q4_k", "layers": {}}
    for i in range(N_LAYERS):
        index["layers"][str(i)] = {
            "file": f"layer_{i}.bin",
            "expert_offsets": {},
            "tensor_formats": {},
        }

    opened_layers = []

    def mock_init(self):
        for layer_key in self.index["layers"]:
            opened_layers.append(int(layer_key))

    with patch.object(ExpertCache, '_init_mmaps', mock_init):
        ec = ExpertCache("/fake", index, layer_start=0, layer_end=24)

    assert sorted(opened_layers) == list(range(0, 24))
