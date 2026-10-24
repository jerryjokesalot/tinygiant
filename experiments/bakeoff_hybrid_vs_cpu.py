#!/usr/bin/env python3
"""Fair bake-off: TinyGiant cpu-only vs hybrid (Metal attn+KV).

Same model, same pins (non-uniform when concentration profile exists),
same prompts as pinning_ab_test / prior M1 bake-off. Preferred metric:
decode tok/s = n_gen / decode_only_seconds after prefill.

Does NOT flip HYBRID_ATTN_READY — report numbers only.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinygiant._constants import N_LAYERS  # noqa: E402
from tinygiant.engine import NWSEngine  # noqa: E402

MODEL_PATH = os.path.expanduser(
    os.environ.get("TG_MODEL", "~/models/Qwen3-30B-A3B-Q4_K_M.gguf")
)
CACHE_DIR = os.environ.get("TG_CACHE", "/tmp/nws_q4_cache_v2")
N_TOKENS = int(os.environ.get("TG_N_GEN", "128"))
N_REPS = int(os.environ.get("TG_N_REPS", "3"))
CALIBRATE_N = int(os.environ.get("TG_CALIBRATE_N", "10"))
PIN_N = int(os.environ.get("TG_PIN", "48"))
OUT_JSON = Path(
    os.environ.get(
        "TG_OUT_JSON",
        str(ROOT / "benchmarks" / "bakeoff_hybrid_raw.json"),
    )
)

EVAL_PROMPTS = {
    "coding": (
        "Write a Python function that implements binary search on a sorted "
        "list and returns the index of the target element."
    ),
    "reasoning": (
        "There are three boxes. One contains only apples, one contains only "
        "oranges, and one contains both. The labels on all three boxes are "
        "wrong. You can pick one fruit from one box. How do you determine "
        "what's in each box?"
    ),
    "creative": (
        "Write a short story about an astronaut who discovers music coming "
        "from an empty planet."
    ),
    "math": (
        "What is the sum of the first 100 positive integers? Show your "
        "reasoning step by step."
    ),
    "factual": (
        "Describe the key differences between TCP and UDP protocols, "
        "including when you would choose each one."
    ),
}


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def tokenize(text: str):
    from llama_cpp import Llama

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32,
        n_gpu_layers=0,
        vocab_only=True,
        verbose=False,
    )
    return llm.tokenize(text.encode(), add_bos=True), llm


def timed_generate(engine: NWSEngine, prompt_tokens, n_tokens: int, seed: int):
    """Prefill then decode with separate wall clocks (matches prior bake-off)."""
    engine.reset_kv()
    ec = engine.expert_cache
    ec.hits = 0
    ec.pinned_hits = 0

    np.random.seed(seed)

    t0 = time.perf_counter()
    logits = None
    for i, tok in enumerate(prompt_tokens):
        if i < len(prompt_tokens) - 1:
            engine.forward_one_token(tok, i)
        else:
            logits = engine.forward_one_token(tok, i)
    t_prefill = time.perf_counter() - t0

    generated = []
    t1 = time.perf_counter()
    next_token = None
    for step in range(n_tokens):
        pos = len(prompt_tokens) + step
        if step == 0:
            next_logits = logits
        else:
            next_logits = engine.forward_one_token(next_token, pos - 1)

        probs = np.exp((next_logits - np.max(next_logits)) / 0.7)
        probs /= np.sum(probs)
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumsum, 0.9)) + 1
        candidates = sorted_idx[:cutoff]
        candidate_probs = probs[candidates]
        candidate_probs /= np.sum(candidate_probs)
        next_token = int(np.random.choice(candidates, p=candidate_probs))
        generated.append(next_token)
    t_decode = time.perf_counter() - t1
    wall = t_prefill + t_decode

    n_prompt = len(prompt_tokens)
    hit_rate = (ec.pinned_hits / ec.hits) if ec.hits else 0.0
    return {
        "n_prompt": n_prompt,
        "n_gen": n_tokens,
        "prefill_s": t_prefill,
        "decode_s": t_decode,
        "wall_s": wall,
        "prefill_tok_s": (n_prompt / t_prefill) if t_prefill > 0 else 0.0,
        "decode_tok_s": (n_tokens / t_decode) if t_decode > 0 else 0.0,
        "wall_tok_s": (n_tokens / wall) if wall > 0 else 0.0,
        "hit_rate": hit_rate,
        "hits": int(ec.hits),
        "pinned_hits": int(ec.pinned_hits),
        "pinned_count": len(ec.pinned),
        "use_metal_attn": bool(getattr(engine, "use_metal_attn", False)),
        "attn_backend": engine.attn_backend.describe(),
        "profile": engine.profile,
    }


def summarize(trials, config: str):
    decode = [t["decode_tok_s"] for t in trials]
    wall = [t["wall_tok_s"] for t in trials]
    per_task = {}
    for task in EVAL_PROMPTS:
        td = [t["decode_tok_s"] for t in trials if t["task"] == task]
        tw = [t["wall_tok_s"] for t in trials if t["task"] == task]
        if not td:
            continue
        per_task[task] = {
            "decode_mean": float(statistics.fmean(td)),
            "decode_std": float(statistics.pstdev(td)) if len(td) > 1 else 0.0,
            "wall_mean": float(statistics.fmean(tw)),
            "wall_std": float(statistics.pstdev(tw)) if len(tw) > 1 else 0.0,
            "n": len(td),
        }
    return {
        "config": config,
        "n": len(trials),
        "decode_tok_s_mean": float(statistics.fmean(decode)) if decode else 0.0,
        "decode_tok_s_std": float(statistics.pstdev(decode)) if len(decode) > 1 else 0.0,
        "wall_tok_s_mean": float(statistics.fmean(wall)) if wall else 0.0,
        "wall_tok_s_std": float(statistics.pstdev(wall)) if len(wall) > 1 else 0.0,
        "metal_used_all": all(t.get("use_metal_attn") for t in trials) if trials else False,
        "per_task": per_task,
    }


def run_config(profile: str, token_data, require_metal: bool):
    print(f"\n{'=' * 70}")
    print(f"CONFIG: profile={profile}")
    print(f"{'=' * 70}")
    t_load = time.perf_counter()
    engine = NWSEngine(MODEL_PATH, CACHE_DIR, profile=profile)
    engine_load_s = time.perf_counter() - t_load
    print(f"  use_metal_attn={engine.use_metal_attn} "
          f"attn={engine.attn_backend.describe()}")

    if require_metal and not engine.use_metal_attn:
        raise RuntimeError(
            f"profile={profile} did not enable Metal attn "
            f"(backend={engine.attn_backend.describe()}). Aborting fair bake-off."
        )
    if not require_metal and engine.use_metal_attn:
        raise RuntimeError(
            f"profile={profile} unexpectedly enabled Metal attn. Aborting."
        )

    # Calibrate once across prompts, then pin_nonuniform (engine default).
    for task, (tokens, _) in token_data.items():
        engine.reset_kv()
        engine.expert_cache.access_counts = {}
        engine.calibrate(tokens, n_tokens=CALIBRATE_N)

    # Prefer pin_nonuniform via pin_experts (uses concentration profile).
    engine.expert_cache.pinned = set()
    # Clear access so pin uses static suggested pins (calibrate already warmed caches).
    count = engine.pin_experts(PIN_N)
    print(f"Pinned {count} experts for profile={profile}")

    trials = []
    for task, (tokens, _) in token_data.items():
        for rep in range(1, N_REPS + 1):
            print(f"\n--- {profile} / {task} / rep {rep} ---")
            stats = timed_generate(
                engine, tokens, N_TOKENS, seed=42 + rep
            )
            stats["config"] = profile
            stats["task"] = task
            stats["rep"] = rep
            print(
                f"  decode={stats['decode_tok_s']:.3f} tok/s  "
                f"wall={stats['wall_tok_s']:.3f}  "
                f"metal={stats['use_metal_attn']}  "
                f"hit_rate={stats['hit_rate']:.3f}"
            )
            if require_metal and not stats["use_metal_attn"]:
                raise RuntimeError(
                    f"Metal attn dropped mid-run at {task} rep {rep}"
                )
            trials.append(stats)
            # Persist incrementally so a kill still leaves partial data.
            _write_partial(trials, engine_load_s)
    return trials, engine_load_s, engine.use_metal_attn


_PARTIAL = {"trials": [], "engine_load_s": {}}


def _write_partial(trials, engine_load_s=None):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Merge into partial blob keyed later in main.
    blob = {
        "partial": True,
        "trials_so_far": trials,
        "n": len(trials),
        "updated": time.time(),
    }
    OUT_JSON.with_suffix(".partial.json").write_text(json.dumps(blob, indent=2))


def main():
    print("TinyGiant hybrid vs cpu-only bake-off")
    print(f"model={MODEL_PATH}")
    print(f"cache={CACHE_DIR}")
    print(f"n_gen={N_TOKENS} reps={N_REPS} pin={PIN_N}")
    print(f"commit={git_head()}")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"missing model: {MODEL_PATH}")
    if not os.path.isdir(CACHE_DIR):
        raise SystemExit(f"missing cache: {CACHE_DIR}")

    token_data = {}
    for task, prompt in EVAL_PROMPTS.items():
        tokens, llm = tokenize(prompt)
        token_data[task] = (tokens, llm)
        print(f"  tokenize {task}: {len(tokens)} tokens")

    all_trials = []
    loads = {}
    metal_flags = {}

    for profile, require_metal in (("cpu-only", False), ("hybrid", True)):
        trials, load_s, metal = run_config(profile, token_data, require_metal)
        all_trials.extend(trials)
        loads[profile] = load_s
        metal_flags[profile] = metal

    summaries = [
        summarize([t for t in all_trials if t["config"] == "cpu-only"], "cpu-only"),
        summarize([t for t in all_trials if t["config"] == "hybrid"], "hybrid"),
    ]

    cpu = summaries[0]["decode_tok_s_mean"]
    hyb = summaries[1]["decode_tok_s_mean"]
    speedup = (hyb / cpu) if cpu > 0 else 0.0
    hybrid_wins = hyb > cpu * 1.05  # clear win >5%

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": "MacBook-Pro Apple M1 16GB",
        "model": MODEL_PATH,
        "cache": CACHE_DIR,
        "tinygiant_commit": git_head(),
        "n_tokens": N_TOKENS,
        "n_reps": N_REPS,
        "calibrate_n": CALIBRATE_N,
        "pin": PIN_N,
        "pin_mode": "nonuniform_when_available",
        "engine_load_s": loads,
        "metal_enabled": metal_flags,
        "metric_notes": ["decode_tok_s", "wall_tok_s"],
        "summaries": summaries,
        "speedup_hybrid_vs_cpu": speedup,
        "hybrid_clear_win_gt_5pct": hybrid_wins,
        "recommend_flip_HYBRID_ATTN_READY": hybrid_wins,
        "trials": all_trials,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    partial = OUT_JSON.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()

    print("\n" + "=" * 70)
    print("SUMMARY")
    for s in summaries:
        print(
            f"  {s['config']}: decode {s['decode_tok_s_mean']:.3f} ± "
            f"{s['decode_tok_s_std']:.3f} tok/s  "
            f"wall {s['wall_tok_s_mean']:.3f} ± {s['wall_tok_s_std']:.3f}  "
            f"metal_all={s['metal_used_all']}"
        )
    print(f"  hybrid/cpu speedup: {speedup:.3f}x")
    print(f"  clear win (>5%): {hybrid_wins}")
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
