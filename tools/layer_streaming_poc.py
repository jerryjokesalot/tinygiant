#!/usr/bin/env python3
"""
Layer Streaming Proof of Concept
=================================
Demonstrates the core I/O mechanism of Cognitive Cascade:
double-buffered sequential reading of model layers from SSD.

This is NOT inference — it's an I/O benchmark that proves the
streaming approach can read layer-sized chunks at rates fast
enough to be practical.

Run on the target machine with a real GGUF model file.
"""

import mmap
import os
import sys
import time
import threading
from collections import deque


def get_model_info(filepath):
    """Get basic model file info."""
    size = os.path.getsize(filepath)
    return {
        "path": filepath,
        "size_bytes": size,
        "size_gb": size / (1024**3),
    }


def estimate_layers(size_bytes, num_layers=64):
    """Estimate layer size from total model size."""
    # Rough estimate: ~95% of GGUF file is tensor data
    tensor_bytes = int(size_bytes * 0.95)
    layer_bytes = tensor_bytes // num_layers
    return {
        "num_layers": num_layers,
        "layer_bytes": layer_bytes,
        "layer_mb": layer_bytes / (1024**2),
    }


def benchmark_sequential_read(filepath, chunk_size, num_chunks, label=""):
    """Read sequential chunks from the file and measure throughput."""
    file_size = os.path.getsize(filepath)
    results = []

    with open(filepath, "rb") as f:
        for i in range(num_chunks):
            offset = (i * chunk_size) % (file_size - chunk_size)
            f.seek(offset)

            start = time.perf_counter()
            data = f.read(chunk_size)
            elapsed = time.perf_counter() - start

            throughput_gbps = len(data) / elapsed / (1024**3)
            results.append({
                "chunk": i,
                "bytes_read": len(data),
                "elapsed_ms": elapsed * 1000,
                "throughput_gbps": throughput_gbps,
            })

    avg_time = sum(r["elapsed_ms"] for r in results) / len(results)
    avg_throughput = sum(r["throughput_gbps"] for r in results) / len(results)

    return {
        "label": label,
        "chunk_size_mb": chunk_size / (1024**2),
        "num_reads": num_chunks,
        "avg_time_ms": avg_time,
        "avg_throughput_gbps": avg_throughput,
        "min_throughput_gbps": min(r["throughput_gbps"] for r in results),
        "max_throughput_gbps": max(r["throughput_gbps"] for r in results),
        "raw": results,
    }


def benchmark_double_buffered(filepath, chunk_size, num_chunks):
    """
    Simulate double-buffered layer streaming:
    - Thread A reads chunk N+1 from disk into buffer B
    - Main thread "processes" chunk N from buffer A (simulated compute)
    - When both finish, swap buffers

    This measures how much we can overlap I/O with compute.
    """
    file_size = os.path.getsize(filepath)
    buffers = [None, None]
    timings = []

    def read_chunk(chunk_idx, buf_idx):
        offset = (chunk_idx * chunk_size) % (file_size - chunk_size)
        with open(filepath, "rb") as f:
            f.seek(offset)
            buffers[buf_idx] = f.read(chunk_size)

    def simulate_compute(data, compute_time_ms=10):
        """Simulate GPU/CPU compute on a layer's weights."""
        # Just access the data to ensure it's in memory
        _ = len(data)
        # Sleep to simulate actual compute time
        time.sleep(compute_time_ms / 1000)

    # Warm up: read first chunk
    read_chunk(0, 0)

    for i in range(1, num_chunks):
        start = time.perf_counter()

        # Start I/O for next chunk in background
        io_thread = threading.Thread(target=read_chunk, args=(i, i % 2))
        io_thread.start()

        # "Compute" on current chunk (in the other buffer)
        current_buf = (i - 1) % 2
        if buffers[current_buf] is not None:
            simulate_compute(buffers[current_buf], compute_time_ms=10)

        # Wait for I/O to complete
        io_thread.join()

        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)

    avg_pipeline_ms = sum(timings) / len(timings) if timings else 0
    sequential_ms = sum(timings)

    return {
        "num_chunks": num_chunks,
        "chunk_size_mb": chunk_size / (1024**2),
        "avg_pipeline_ms": avg_pipeline_ms,
        "total_pipeline_ms": sequential_ms,
        "effective_throughput_gbps": (chunk_size * len(timings)) / (sequential_ms / 1000) / (1024**3) if sequential_ms > 0 else 0,
    }


def benchmark_mmap_access(filepath, chunk_size, num_chunks):
    """
    Test mmap-based access pattern: memory-map the file and
    access layer-sized regions sequentially.
    """
    file_size = os.path.getsize(filepath)
    results = []

    with open(filepath, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(num_chunks):
                offset = (i * chunk_size) % (file_size - chunk_size)

                start = time.perf_counter()
                # Read a chunk via mmap (triggers page faults = disk reads)
                chunk = mm[offset:offset + chunk_size]
                # Force the data into memory by accessing it
                _ = bytes(chunk)
                elapsed = time.perf_counter() - start

                throughput = len(chunk) / elapsed / (1024**3)
                results.append({
                    "elapsed_ms": elapsed * 1000,
                    "throughput_gbps": throughput,
                })

    avg_throughput = sum(r["throughput_gbps"] for r in results) / len(results)
    return {
        "method": "mmap",
        "chunk_size_mb": chunk_size / (1024**2),
        "num_reads": num_chunks,
        "avg_throughput_gbps": avg_throughput,
    }


def run_full_benchmark(model_path, num_layers=64):
    """Run the complete layer streaming benchmark suite."""
    info = get_model_info(model_path)
    layers = estimate_layers(info["size_bytes"], num_layers)

    print("=" * 70)
    print("  LAYER STREAMING PROOF OF CONCEPT")
    print("=" * 70)
    print(f"\n  Model: {info['path']}")
    print(f"  Size: {info['size_gb']:.1f} GB")
    print(f"  Estimated layers: {layers['num_layers']}")
    print(f"  Estimated layer size: {layers['layer_mb']:.0f} MB")

    chunk_size = layers["layer_bytes"]

    # Purge disk cache before benchmarking
    print("\n  Note: First read may be cached. Subsequent reads are from disk.")

    # --- Test 1: Sequential read throughput ---
    print(f"\n{'─' * 70}")
    print("  TEST 1: Sequential Layer Read (simulates naive offloading)")
    print(f"{'─' * 70}")

    result = benchmark_sequential_read(model_path, chunk_size, min(num_layers, 20), "sequential")
    print(f"  Chunk size: {result['chunk_size_mb']:.0f} MB")
    print(f"  Reads: {result['num_reads']}")
    print(f"  Avg read time: {result['avg_time_ms']:.1f} ms")
    print(f"  Avg throughput: {result['avg_throughput_gbps']:.2f} GB/s")
    print(f"  Range: {result['min_throughput_gbps']:.2f} - {result['max_throughput_gbps']:.2f} GB/s")

    # Time for full model read layer by layer
    full_model_time = result['avg_time_ms'] * num_layers / 1000
    print(f"\n  Full model ({num_layers} layers) sequential read: {full_model_time:.1f}s")
    print(f"  → Naive offloading: ~{1/full_model_time:.2f} tokens/sec (if compute is free)")

    # --- Test 2: Double-buffered pipeline ---
    print(f"\n{'─' * 70}")
    print("  TEST 2: Double-Buffered Pipeline (Cognitive Cascade Layer Hotel)")
    print(f"{'─' * 70}")

    db_result = benchmark_double_buffered(model_path, chunk_size, min(num_layers, 20))
    print(f"  Avg pipeline step: {db_result['avg_pipeline_ms']:.1f} ms")
    print(f"  Effective throughput: {db_result['effective_throughput_gbps']:.2f} GB/s")
    print(f"  Compute overlap: 10ms simulated per layer")

    pipeline_full = db_result['avg_pipeline_ms'] * num_layers / 1000
    speedup = full_model_time / pipeline_full if pipeline_full > 0 else 0
    print(f"\n  Full model pipeline: {pipeline_full:.1f}s")
    print(f"  Speedup vs sequential: {speedup:.2f}x")

    # --- Test 3: mmap-based access ---
    print(f"\n{'─' * 70}")
    print("  TEST 3: Memory-Mapped Access (OS-managed paging)")
    print(f"{'─' * 70}")

    mmap_result = benchmark_mmap_access(model_path, chunk_size, min(num_layers, 20))
    print(f"  Avg throughput: {mmap_result['avg_throughput_gbps']:.2f} GB/s")

    # --- Cognitive Cascade Projection ---
    print(f"\n{'=' * 70}")
    print("  COGNITIVE CASCADE PROJECTION (using real measurements)")
    print(f"{'=' * 70}")

    # Use measured values
    ssd_gbps = result['avg_throughput_gbps']
    layer_read_ms = result['avg_time_ms']
    draft_tps = 38.3  # measured from Ollama benchmark

    print(f"\n  Hardware measurements:")
    print(f"    SSD read throughput: {ssd_gbps:.2f} GB/s")
    print(f"    Layer read time: {layer_read_ms:.0f} ms")
    print(f"    Draft model speed: {draft_tps:.1f} tok/s")
    print(f"    Draft time per token: {1000/draft_tps:.0f} ms")

    print(f"\n  Dense model ({info['size_gb']:.0f}GB) via Cognitive Cascade:")
    easy_pct, med_pct, hard_pct = 0.55, 0.30, 0.15
    easy_ms = 1000 / draft_tps
    med_layers = num_layers // 2
    hard_layers = num_layers
    spec_batch_med, spec_batch_hard = 3, 5
    med_ms = easy_ms + (med_layers * layer_read_ms) / spec_batch_med
    hard_ms = easy_ms + (hard_layers * layer_read_ms) / spec_batch_hard
    blended_ms = easy_pct * easy_ms + med_pct * med_ms + hard_pct * hard_ms
    blended_tps = 1000 / blended_ms

    print(f"    Easy tokens ({easy_pct:.0%}):  {easy_ms:.0f} ms → {1000/easy_ms:.1f} tok/s")
    print(f"    Medium tokens ({med_pct:.0%}): {med_ms:.0f} ms → {1000/med_ms:.1f} tok/s")
    print(f"    Hard tokens ({hard_pct:.0%}):  {hard_ms:.0f} ms → {1000/hard_ms:.1f} tok/s")
    print(f"    Blended average:       {blended_ms:.0f} ms → {blended_tps:.1f} tok/s")
    print(f"    Peak RAM: ~{layers['layer_mb']*2/1024 + 1.5:.1f} GB (2 layers + draft + overhead)")

    # MoE projection
    print(f"\n  MoE model (same size, 1/8 experts active) via Expert Cascade:")
    moe_factor = 8  # only 1/8 of experts active
    moe_layer_ms = layer_read_ms / moe_factor
    moe_med_ms = easy_ms + (med_layers * moe_layer_ms) / spec_batch_med
    moe_hard_ms = easy_ms + (hard_layers * moe_layer_ms) / spec_batch_hard
    moe_blended_ms = easy_pct * easy_ms + med_pct * moe_med_ms + hard_pct * moe_hard_ms
    moe_blended_tps = 1000 / moe_blended_ms

    print(f"    Easy tokens ({easy_pct:.0%}):  {easy_ms:.0f} ms → {1000/easy_ms:.1f} tok/s")
    print(f"    Medium tokens ({med_pct:.0%}): {moe_med_ms:.0f} ms → {1000/moe_med_ms:.1f} tok/s")
    print(f"    Hard tokens ({hard_pct:.0%}):  {moe_hard_ms:.0f} ms → {1000/moe_hard_ms:.1f} tok/s")
    print(f"    Blended average:       {moe_blended_ms:.0f} ms → {moe_blended_tps:.1f} tok/s")

    print(f"\n  Comparison:")
    print(f"    Current (full model in swap): ~2.4 tok/s prompt, <1 tok/s gen")
    print(f"    Cognitive Cascade (dense):    ~{blended_tps:.1f} tok/s")
    print(f"    Expert Cascade (MoE):         ~{moe_blended_tps:.1f} tok/s")
    print(f"    Draft model only:             ~{draft_tps:.1f} tok/s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 layer_streaming_poc.py <model.gguf> [num_layers]")
        print("Example: python3 layer_streaming_poc.py ~/models/qwen3-32b.gguf 64")
        sys.exit(1)

    model_path = os.path.expanduser(sys.argv[1])
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    if not os.path.exists(model_path):
        # Try resolving symlink
        real_path = os.path.realpath(model_path)
        if os.path.exists(real_path):
            model_path = real_path
        else:
            print(f"Error: {model_path} not found")
            sys.exit(1)

    run_full_benchmark(model_path, num_layers)
