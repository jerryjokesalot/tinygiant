"""Layer concentration analysis — is there signal for non-uniform pinning?

Runs calibration on diverse prompts, then analyzes per-layer expert routing
to see if some layers need more pinned experts than others.

Metrics per layer:
  - unique_experts: how many distinct experts were routed to
  - entropy: Shannon entropy of access distribution (higher = more spread)
  - top8_coverage: % of accesses going to the top 8 experts
  - top32_coverage: % of accesses going to the top 32 experts
  - gini: Gini coefficient (0 = perfectly uniform, 1 = one expert gets all)

If there's a big spread in these metrics across layers, non-uniform pinning
could help. If they're all similar, uniform pinning is fine.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinygiant._constants import N_EXPERTS_USED, N_LAYERS
from tinygiant.engine import NWSEngine

MODEL_PATH = os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf")
CACHE_DIR = "/tmp/nws_q4_cache_v2"

PROMPTS = {
    "coding": "Write a Python function that finds the longest common subsequence of two strings using dynamic programming.",
    "reasoning": "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item. How does he get everything across?",
    "creative": "Write a short poem about the feeling of discovering something beautiful in an unexpected place.",
    "math": "Prove that the square root of 2 is irrational using proof by contradiction.",
    "factual": "Explain how mRNA vaccines work, including the role of spike proteins and the immune response.",
}


def tokenize(text):
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=32, n_gpu_layers=0,
                vocab_only=True, verbose=False)
    return llm.tokenize(text.encode(), add_bos=True), llm


def compute_layer_stats(access_counts, n_layers):
    per_layer = defaultdict(lambda: defaultdict(int))
    for (layer, expert), cnt in access_counts.items():
        per_layer[layer][expert] = cnt

    stats = {}
    for layer in range(n_layers):
        counts = per_layer[layer]
        if not counts:
            stats[layer] = {"unique": 0, "entropy": 0, "top8_cov": 0,
                            "top32_cov": 0, "gini": 0, "total": 0}
            continue

        total = sum(counts.values())
        sorted_counts = sorted(counts.values(), reverse=True)
        n_unique = len(counts)

        # Shannon entropy
        probs = np.array(sorted_counts, dtype=np.float64) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # Top-K coverage
        top8_cov = sum(sorted_counts[:8]) / total
        top32_cov = sum(sorted_counts[:32]) / total

        # Gini coefficient
        n = len(sorted_counts)
        vals = np.array(sorted(sorted_counts), dtype=np.float64)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * vals) - (n + 1) * np.sum(vals)) / (n * np.sum(vals)) if np.sum(vals) > 0 else 0

        stats[layer] = {
            "unique": n_unique,
            "entropy": float(entropy),
            "top8_cov": float(top8_cov),
            "top32_cov": float(top32_cov),
            "gini": float(gini),
            "total": total,
        }

    return stats


def run_analysis():
    print("Layer Concentration Analysis")
    print("=" * 70)

    # Build engine once
    engine = NWSEngine(MODEL_PATH, CACHE_DIR)

    all_stats = {}

    for task_name, prompt in PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"Task: {task_name}")
        print(f"Prompt: {prompt[:80]}...")

        tokens, _ = tokenize(prompt)
        print(f"Tokens: {len(tokens)}")

        # Reset state
        engine.reset_kv()
        engine.expert_cache.access_counts = {}
        engine.expert_cache.hits = 0
        engine.expert_cache.pinned_hits = 0

        # Run calibration (prefill + 20 decode tokens)
        engine.calibrate(tokens, n_tokens=20)

        # Analyze routing
        stats = compute_layer_stats(engine.expert_cache.access_counts, N_LAYERS)
        all_stats[task_name] = stats

        # Print summary
        entropies = [stats[l]["entropy"] for l in range(N_LAYERS)]
        top32s = [stats[l]["top32_cov"] for l in range(N_LAYERS)]
        uniques = [stats[l]["unique"] for l in range(N_LAYERS)]

        print(f"\n  Entropy:     min={min(entropies):.2f}  max={max(entropies):.2f}  "
              f"spread={max(entropies)-min(entropies):.2f}")
        print(f"  Top-32 cov:  min={min(top32s):.1%}  max={max(top32s):.1%}  "
              f"spread={max(top32s)-min(top32s):.1%}")
        print(f"  Unique:      min={min(uniques)}  max={max(uniques)}  "
              f"spread={max(uniques)-min(uniques)}")

    # Cross-task analysis
    print(f"\n{'='*70}")
    print("CROSS-TASK LAYER ANALYSIS")
    print(f"{'='*70}")

    print(f"\n{'Layer':>5} ", end="")
    for task in PROMPTS:
        print(f"  {task[:6]:>6}H", end="")
    print(f"  {'avg_H':>6}  {'spread':>6}  {'pin_rec':>7}")

    layer_avg_entropy = {}
    layer_entropy_spread = {}

    for layer in range(N_LAYERS):
        entropies = [all_stats[task][layer]["entropy"] for task in PROMPTS]
        avg_e = np.mean(entropies)
        spread_e = max(entropies) - min(entropies)
        layer_avg_entropy[layer] = avg_e
        layer_entropy_spread[layer] = spread_e

        # Recommend pin count based on entropy
        # Higher entropy = needs more pins (experts spread out)
        # Lower entropy = needs fewer pins (concentrated on few experts)
        if avg_e > 5.5:
            rec = "pin64"
        elif avg_e > 5.0:
            rec = "pin48"
        elif avg_e > 4.5:
            rec = "pin32"
        else:
            rec = "pin16"

        print(f"{layer:>5} ", end="")
        for task in PROMPTS:
            print(f"  {all_stats[task][layer]['entropy']:>7.2f}", end="")
        print(f"  {avg_e:>6.2f}  {spread_e:>6.2f}  {rec:>7}")

    # Summary statistics
    all_avg = list(layer_avg_entropy.values())
    all_spread = list(layer_entropy_spread.values())
    print(f"\nEntropy range across layers: {min(all_avg):.2f} - {max(all_avg):.2f} "
          f"(spread: {max(all_avg)-min(all_avg):.2f})")
    print(f"Cross-task spread per layer: {min(all_spread):.2f} - {max(all_spread):.2f}")

    # Verdict
    entropy_range = max(all_avg) - min(all_avg)
    if entropy_range > 1.5:
        verdict = "STRONG signal for non-uniform pinning"
    elif entropy_range > 0.8:
        verdict = "MODERATE signal — worth testing non-uniform pinning"
    elif entropy_range > 0.3:
        verdict = "WEAK signal — marginal benefit from non-uniform pinning"
    else:
        verdict = "NO signal — uniform pinning is fine"

    print(f"\nVerdict: {verdict}")

    task_spread = np.mean(all_spread)
    if task_spread > 0.5:
        print(f"Task sensitivity: HIGH ({task_spread:.2f}) — task-aware pinning would help")
    elif task_spread > 0.2:
        print(f"Task sensitivity: MODERATE ({task_spread:.2f}) — task-aware pinning may help")
    else:
        print(f"Task sensitivity: LOW ({task_spread:.2f}) — same pin profile works for all tasks")

    # Compute suggested pin allocation for pin48-equivalent budget
    total_budget = 48 * N_LAYERS  # 2304 total experts
    min_per_layer = 8
    remaining_budget = total_budget - (min_per_layer * N_LAYERS)

    # Allocate remaining proportional to entropy
    e_vals = np.array([layer_avg_entropy[l] for l in range(N_LAYERS)])
    e_shifted = e_vals - e_vals.min() + 0.1
    weights = e_shifted / e_shifted.sum()
    extra = (weights * remaining_budget).astype(int)
    # Fix rounding
    diff = remaining_budget - extra.sum()
    for i in np.argsort(-e_vals)[:diff]:
        extra[i] += 1

    suggested = min_per_layer + extra

    print(f"\nSuggested non-uniform allocation (same {total_budget} total budget):")
    print(f"  Uniform:     {48} per layer")
    print(f"  Non-uniform: min={suggested.min()}, max={suggested.max()}, "
          f"mean={suggested.mean():.1f}, std={suggested.std():.1f}")
    print(f"\n  Top 5 layers (most pins):")
    for l in np.argsort(-suggested)[:5]:
        print(f"    Layer {l:>2}: {suggested[l]} pins (entropy={layer_avg_entropy[l]:.2f})")
    print(f"  Bottom 5 layers (fewest pins):")
    for l in np.argsort(suggested)[:5]:
        print(f"    Layer {l:>2}: {suggested[l]} pins (entropy={layer_avg_entropy[l]:.2f})")

    # Save results
    results = {
        "per_task": {task: {str(l): s for l, s in stats.items()}
                     for task, stats in all_stats.items()},
        "layer_avg_entropy": {str(l): v for l, v in layer_avg_entropy.items()},
        "layer_entropy_spread": {str(l): v for l, v in layer_entropy_spread.items()},
        "suggested_pins": {str(l): int(suggested[l]) for l in range(N_LAYERS)},
        "verdict": verdict,
    }
    out_path = Path(__file__).parent / "layer_concentration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_analysis()
