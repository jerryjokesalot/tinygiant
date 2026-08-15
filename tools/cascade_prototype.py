#!/usr/bin/env python3
"""
TinyGiant Cascade Prototype
============================
End-to-end proof-of-concept that demonstrates the cascade inference loop
with real SSD I/O on real GGUF model files.

1. Confidence router classifies tokens by difficulty (easy/medium/hard)
2. Easy tokens: accepted from draft model (no big-model I/O)
3. Medium tokens: stream half the layers from SSD (early exit)
4. Hard tokens: stream all layers from SSD (full verification)

With --expert-ratio, simulates Expert Cascade (reading only active expert
weights instead of full layers). No dependencies beyond Python 3.9+.
"""

import json
import math
import os
import struct
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CascadeConfig:
    draft_model_path: str
    big_model_path: str
    easy_threshold: float = 0.85     # above = accept draft directly
    medium_threshold: float = 0.60   # above = early exit (half layers)
    draft_batch_size: int = 5        # tokens to draft before verification
    max_tokens: int = 128
    n_threads: int = 8
    expert_ratio: float = 1.0       # 1.0 = full layer, <1.0 = expert-aware
    n_layers: int = 64


@dataclass
class CascadeStats:
    total_tokens: int = 0
    easy_tokens: int = 0
    medium_tokens: int = 0
    hard_tokens: int = 0
    draft_time_ms: float = 0
    verify_time_ms: float = 0
    io_time_ms: float = 0
    total_time_ms: float = 0

    @property
    def tokens_per_sec(self):
        if self.total_time_ms == 0:
            return 0
        return self.total_tokens / (self.total_time_ms / 1000)

    @property
    def easy_pct(self):
        return self.easy_tokens / max(self.total_tokens, 1) * 100

    @property
    def medium_pct(self):
        return self.medium_tokens / max(self.total_tokens, 1) * 100

    @property
    def hard_pct(self):
        return self.hard_tokens / max(self.total_tokens, 1) * 100


class LayerHotel:
    """Async double-buffered layer streaming from SSD."""

    def __init__(self, model_path: str, n_layers: int, expert_ratio: float = 1.0):
        self.model_path = model_path
        self.n_layers = n_layers
        self.file_size = os.path.getsize(model_path)
        self.tensor_data_size = int(self.file_size * 0.95)
        full_layer_size = self.tensor_data_size // n_layers
        # expert_ratio < 1.0 simulates reading only active expert weights
        self.layer_size = int(full_layer_size * expert_ratio)
        self.full_layer_size = full_layer_size

        self.buffers = [None, None]
        self.current_buf = 0
        self._prefetch_thread = None
        # Separate file handles for main and prefetch threads (avoid race)
        self._file_main = open(model_path, 'rb')
        self._file_prefetch = open(model_path, 'rb')

        # Stats
        self.total_reads = 0
        self.total_read_time_ms = 0
        self._lock = threading.Lock()

    def _read_from(self, f, layer_idx: int) -> bytes:
        """Read a single layer's data from a specific file handle."""
        offset = layer_idx * self.layer_size
        start = time.perf_counter()
        f.seek(offset)
        data = f.read(self.layer_size)
        elapsed = (time.perf_counter() - start) * 1000
        with self._lock:
            self.total_reads += 1
            self.total_read_time_ms += elapsed
        return data

    def read_layer(self, layer_idx: int) -> bytes:
        """Read a single layer synchronously on main thread."""
        return self._read_from(self._file_main, layer_idx)

    def prefetch_layer(self, layer_idx: int):
        """Start prefetching a layer in background thread."""
        buf_idx = (self.current_buf + 1) % 2

        def _read():
            self.buffers[buf_idx] = self._read_from(self._file_prefetch, layer_idx)

        self._prefetch_thread = threading.Thread(target=_read)
        self._prefetch_thread.start()

    def get_prefetched(self) -> bytes:
        """Get the prefetched layer data (blocks until ready)."""
        if self._prefetch_thread:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        buf_idx = (self.current_buf + 1) % 2
        self.current_buf = buf_idx
        return self.buffers[buf_idx]

    def stream_layers(self, start_layer: int, end_layer: int):
        """Stream a range of layers with double buffering."""
        if end_layer <= start_layer:
            return []

        # Read first layer on main thread
        self.buffers[0] = self.read_layer(start_layer)
        self.current_buf = 0

        for i in range(start_layer + 1, end_layer):
            self.prefetch_layer(i)
            # Simulate 10ms compute on current layer while I/O happens
            time.sleep(0.01)
            self.get_prefetched()

        return None  # we don't need to accumulate layer data

    def close(self):
        self._file_main.close()
        self._file_prefetch.close()


class ConfidenceRouter:
    """Routes tokens based on draft model confidence."""

    def __init__(self, easy_threshold: float, medium_threshold: float):
        self.easy_threshold = easy_threshold
        self.medium_threshold = medium_threshold

    def classify(self, top_prob: float) -> str:
        """Classify a token based on its top probability."""
        if top_prob >= self.easy_threshold:
            return "easy"
        elif top_prob >= self.medium_threshold:
            return "medium"
        else:
            return "hard"


def simulate_cascade_inference(config: CascadeConfig):
    """
    Simulate the full cascade inference loop.

    This uses real SSD I/O but simulated model inference
    (since wiring llama-cpp-python layer-by-layer is a separate effort).
    The purpose is to measure the REALISTIC I/O + routing overhead.
    """
    stats = CascadeStats()

    # Initialize components
    n_layers = config.n_layers
    hotel = LayerHotel(config.big_model_path, n_layers, expert_ratio=config.expert_ratio)
    router = ConfidenceRouter(config.easy_threshold, config.medium_threshold)

    print(f"Layer size: {hotel.layer_size / (1024**2):.0f} MB")
    print(f"Generating {config.max_tokens} tokens...\n")

    total_start = time.perf_counter()

    for token_idx in range(config.max_tokens):
        # Simulate draft model generating a token with confidence
        draft_start = time.perf_counter()
        # Simulate ~26ms draft latency (measured 38 tok/s)
        time.sleep(0.001)  # minimal sleep — we're measuring I/O, not draft

        # Simulate varying confidence levels
        import random
        r = random.random()
        if r < 0.55:
            top_prob = random.uniform(0.85, 0.99)  # easy
        elif r < 0.85:
            top_prob = random.uniform(0.60, 0.85)  # medium
        else:
            top_prob = random.uniform(0.10, 0.60)  # hard

        draft_elapsed = (time.perf_counter() - draft_start) * 1000
        stats.draft_time_ms += draft_elapsed

        # Route the token
        difficulty = router.classify(top_prob)

        # Execute based on difficulty
        verify_start = time.perf_counter()

        if difficulty == "easy":
            stats.easy_tokens += 1

        elif difficulty == "medium":
            stats.medium_tokens += 1
            hotel.stream_layers(0, n_layers // 2)

        else:
            stats.hard_tokens += 1
            hotel.stream_layers(0, n_layers)

        verify_elapsed = (time.perf_counter() - verify_start) * 1000
        stats.verify_time_ms += verify_elapsed
        stats.total_tokens += 1

        # Progress
        if (token_idx + 1) % 10 == 0:
            elapsed_so_far = (time.perf_counter() - total_start) * 1000
            rate = stats.total_tokens / (elapsed_so_far / 1000)
            print(f"  Token {token_idx + 1}/{config.max_tokens} "
                  f"({difficulty}, p={top_prob:.2f}) — "
                  f"{rate:.1f} tok/s overall")

    stats.total_time_ms = (time.perf_counter() - total_start) * 1000
    stats.io_time_ms = hotel.total_read_time_ms

    hotel.close()
    return stats


def print_stats(stats: CascadeStats):
    """Print cascade inference statistics."""
    print(f"\n{'=' * 60}")
    print(f"  CASCADE INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total tokens:   {stats.total_tokens}")
    print(f"  Total time:     {stats.total_time_ms:.0f} ms")
    print(f"  Speed:          {stats.tokens_per_sec:.1f} tok/s")
    print(f"\n  Token routing:")
    print(f"    Easy (draft only):  {stats.easy_tokens} ({stats.easy_pct:.0f}%)")
    print(f"    Medium (half layers): {stats.medium_tokens} ({stats.medium_pct:.0f}%)")
    print(f"    Hard (full layers):   {stats.hard_tokens} ({stats.hard_pct:.0f}%)")
    print(f"\n  Time breakdown:")
    print(f"    Draft model: {stats.draft_time_ms:.0f} ms ({stats.draft_time_ms/stats.total_time_ms*100:.0f}%)")
    print(f"    I/O reads:   {stats.io_time_ms:.0f} ms ({stats.io_time_ms/stats.total_time_ms*100:.0f}%)")
    print(f"    Verify:      {stats.verify_time_ms:.0f} ms ({stats.verify_time_ms/stats.total_time_ms*100:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cascade_prototype.py <big_model.gguf> [max_tokens] [--expert-ratio R] [--layers N]")
        print("\nRuns the cascade inference loop with real SSD I/O.")
        print("  --expert-ratio 0.088  Simulate Expert Cascade (read 8.8% of each layer)")
        print("  --layers 48           Number of model layers (48 for MoE, 64 for dense)")
        sys.exit(1)

    big_model = os.path.expanduser(sys.argv[1])
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 50

    expert_ratio = 1.0
    n_layers = 64
    for i, arg in enumerate(sys.argv):
        if arg == '--expert-ratio' and i + 1 < len(sys.argv):
            expert_ratio = float(sys.argv[i + 1])
        if arg == '--layers' and i + 1 < len(sys.argv):
            n_layers = int(sys.argv[i + 1])

    if not os.path.exists(big_model):
        real_path = os.path.realpath(big_model)
        if os.path.exists(real_path):
            big_model = real_path
        else:
            print(f"Error: {big_model} not found")
            sys.exit(1)

    config = CascadeConfig(
        draft_model_path="",
        big_model_path=big_model,
        max_tokens=max_tokens,
        expert_ratio=expert_ratio,
        n_layers=n_layers,
    )

    mode = "Expert Cascade" if expert_ratio < 1.0 else "Cognitive Cascade"
    print(f"{mode} Prototype")
    print(f"Big model: {big_model}")
    print(f"File size: {os.path.getsize(big_model) / (1024**3):.1f} GB")
    print(f"Layers: {n_layers}, Expert ratio: {expert_ratio}")
    print()

    stats = simulate_cascade_inference(config)
    print_stats(stats)
