"""A/B test: uniform vs non-uniform expert pinning.

Same total memory budget (2304 experts), different allocation across layers.
Measures tok/s and pinned hit rate to determine if non-uniform pinning
produces faster inference.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinygiant._constants import N_LAYERS
from tinygiant.engine import NWSEngine

MODEL_PATH = os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf")
CACHE_DIR = "/tmp/nws_q4_cache_v2"

EVAL_PROMPTS = {
    "coding": "Write a Python function that implements binary search on a sorted list and returns the index of the target element.",
    "reasoning": "There are three boxes. One contains only apples, one contains only oranges, and one contains both. The labels on all three boxes are wrong. You can pick one fruit from one box. How do you determine what's in each box?",
    "creative": "Write a short story about an astronaut who discovers music coming from an empty planet.",
    "math": "What is the sum of the first 100 positive integers? Show your reasoning step by step.",
    "factual": "Describe the key differences between TCP and UDP protocols, including when you would choose each one.",
}

N_TOKENS = 60


def load_concentration_results():
    path = Path(__file__).parent / "layer_concentration_results.json"
    with open(path) as f:
        return json.load(f)


def tokenize(text):
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=32, n_gpu_layers=0,
                vocab_only=True, verbose=False)
    return llm.tokenize(text.encode(), add_bos=True), llm


def detokenize(llm, tokens):
    return llm.detokenize(tokens).decode("utf-8", errors="replace")


def run_trial(engine, tokens, llm, n_tokens, label):
    """Run generation and return stats."""
    engine.reset_kv()
    engine.expert_cache.hits = 0
    engine.expert_cache.pinned_hits = 0

    np.random.seed(42)
    t0 = time.perf_counter()
    generated = engine.generate(tokens, n_tokens=n_tokens, temperature=0.7)
    elapsed = time.perf_counter() - t0

    ec = engine.expert_cache
    hit_rate = ec.pinned_hits / ec.hits if ec.hits > 0 else 0
    tok_s = n_tokens / elapsed if elapsed > 0 else 0

    text = detokenize(llm, generated)

    return {
        "label": label,
        "tok_s": tok_s,
        "total_time": elapsed,
        "hits": ec.hits,
        "pinned_hits": ec.pinned_hits,
        "hit_rate": hit_rate,
        "pinned_count": len(ec.pinned),
        "text": text,
    }


def main():
    print("Pinning A/B Test: Uniform vs Non-Uniform")
    print("=" * 70)

    # Load concentration results
    conc = load_concentration_results()
    suggested = {int(k): v for k, v in conc["suggested_pins"].items()}
    total_budget = sum(suggested.values())
    print(f"Total pin budget: {total_budget} experts")
    print(f"Uniform: 48 per layer")
    print(f"Non-uniform: min={min(suggested.values())}, max={max(suggested.values())}")

    # Build engine
    engine = NWSEngine(MODEL_PATH, CACHE_DIR)

    # Tokenize all prompts
    token_data = {}
    for task, prompt in EVAL_PROMPTS.items():
        tokens, llm = tokenize(prompt)
        token_data[task] = (tokens, llm)

    results = {"uniform": {}, "nonuniform": {}}

    # --- Trial A: Uniform pin48 ---
    print(f"\n{'='*70}")
    print("TRIAL A: Uniform pin48")
    print(f"{'='*70}")

    # Calibrate with all prompts combined
    for task, (tokens, _) in token_data.items():
        engine.reset_kv()
        engine.expert_cache.access_counts = {}
        engine.calibrate(tokens, n_tokens=10)

    # Pin uniform
    engine.expert_cache.pinned = set()
    count = engine.expert_cache.pin_from_usage(48, N_LAYERS)
    print(f"Pinned {count} experts (uniform 48/layer)")

    for task, (tokens, llm) in token_data.items():
        print(f"\n--- {task} ---")
        r = run_trial(engine, tokens, llm, N_TOKENS, f"uniform-{task}")
        results["uniform"][task] = r
        print(f"  {r['tok_s']:.2f} tok/s | hit rate: {r['hit_rate']:.1%} | "
              f"pinned hits: {r['pinned_hits']}/{r['hits']}")
        print(f"  Output: {r['text'][:100]}...")

    # --- Trial B: Non-uniform pinning ---
    print(f"\n{'='*70}")
    print("TRIAL B: Non-Uniform (entropy-weighted)")
    print(f"{'='*70}")

    # Re-calibrate (need fresh access_counts)
    engine.expert_cache.pinned = set()
    for task, (tokens, _) in token_data.items():
        engine.reset_kv()
        engine.expert_cache.access_counts = {}
        engine.calibrate(tokens, n_tokens=10)

    # Pin non-uniform
    count = engine.expert_cache.pin_nonuniform(suggested, N_LAYERS)
    print(f"Pinned {count} experts (non-uniform {min(suggested.values())}-{max(suggested.values())}/layer)")

    for task, (tokens, llm) in token_data.items():
        print(f"\n--- {task} ---")
        r = run_trial(engine, tokens, llm, N_TOKENS, f"nonuniform-{task}")
        results["nonuniform"][task] = r
        print(f"  {r['tok_s']:.2f} tok/s | hit rate: {r['hit_rate']:.1%} | "
              f"pinned hits: {r['pinned_hits']}/{r['hits']}")
        print(f"  Output: {r['text'][:100]}...")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Task':<12} {'Uniform':>10} {'Non-Uni':>10} {'Delta':>8} "
          f"{'U-Hit%':>8} {'NU-Hit%':>8} {'Hit-Delta':>10}")
    print("-" * 70)

    uniform_speeds = []
    nonuniform_speeds = []

    for task in EVAL_PROMPTS:
        u = results["uniform"][task]
        n = results["nonuniform"][task]
        delta_s = n["tok_s"] - u["tok_s"]
        delta_pct = delta_s / u["tok_s"] * 100 if u["tok_s"] > 0 else 0
        hit_delta = n["hit_rate"] - u["hit_rate"]

        uniform_speeds.append(u["tok_s"])
        nonuniform_speeds.append(n["tok_s"])

        sign = "+" if delta_s >= 0 else ""
        print(f"{task:<12} {u['tok_s']:>8.2f}  {n['tok_s']:>8.2f}  "
              f"{sign}{delta_pct:>6.1f}%  {u['hit_rate']:>7.1%}  "
              f"{n['hit_rate']:>7.1%}  {hit_delta:>+9.1%}")

    avg_u = np.mean(uniform_speeds)
    avg_n = np.mean(nonuniform_speeds)
    avg_delta = (avg_n - avg_u) / avg_u * 100

    print("-" * 70)
    sign = "+" if avg_delta >= 0 else ""
    print(f"{'AVERAGE':<12} {avg_u:>8.2f}  {avg_n:>8.2f}  {sign}{avg_delta:>6.1f}%")

    # Verdict
    print(f"\n{'='*70}")
    if avg_delta > 5:
        print(f"VERDICT: Non-uniform pinning is {avg_delta:.1f}% faster — worth integrating")
    elif avg_delta > 1:
        print(f"VERDICT: Non-uniform pinning is {avg_delta:.1f}% faster — marginal but real")
    elif avg_delta > -1:
        print(f"VERDICT: No meaningful difference ({avg_delta:+.1f}%) — uniform pinning is fine")
    else:
        print(f"VERDICT: Non-uniform pinning is {avg_delta:.1f}% slower — stick with uniform")

    # Save results
    save_results = {
        "uniform": {k: {kk: vv for kk, vv in v.items() if kk != "text"}
                    for k, v in results["uniform"].items()},
        "nonuniform": {k: {kk: vv for kk, vv in v.items() if kk != "text"}
                       for k, v in results["nonuniform"].items()},
        "suggested_pins": suggested,
        "avg_uniform_toks": avg_u,
        "avg_nonuniform_toks": avg_n,
        "delta_pct": avg_delta,
    }
    out_path = Path(__file__).parent / "pinning_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
