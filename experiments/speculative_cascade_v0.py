"""Speculative cascade Phase 1 v0 — draft proposes, TinyGiant verifies serial.

Classic speculative decoding with greedy (temp=0) draft + verifier.
v0 uses token-by-token verifier forwards (serial) to measure accept_rate.
Effective tok/s is expected to fail Gate A; teacher-forced batch verify is v1.

Does not flip HYBRID_ATTN_READY or start Slice B.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant._constants import N_LAYERS
from tinygiant.engine import NWSEngine

CODING_PROMPT = (
    "Write a Python function that implements binary search on a sorted list "
    "and returns the index of the target element."
)

DEFAULT_MODEL = os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf")
DEFAULT_CACHE = os.path.expanduser("~/.tinygiant/cache")
DEFAULT_DRAFT = os.path.expanduser("~/models/qwen3-1.7b.gguf")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "speculative_cascade_phase1.json"


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    if len(a) == 0:
        return 0.0, 0.0
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def quit_ollama():
    """Free RAM before loading Tiny; restart later if draft needs it."""
    try:
        subprocess.run(["ollama", "stop", "qwen3:1.7b"], capture_output=True, timeout=30)
    except Exception:
        pass
    # Stop server if running as app/service — best-effort
    try:
        subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True, timeout=15)
    except Exception:
        pass
    time.sleep(1)


def load_draft(path, n_gpu_layers, n_ctx=2048):
    from llama_cpp import Llama

    print(f"Loading draft {path} n_gpu_layers={n_gpu_layers}...", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        verbose=False,
    )
    print(f"  draft load {time.perf_counter() - t0:.1f}s", flush=True)
    return llm


def tokenize_prompt(draft_llm, text):
    # Same family tokenizer as verifier (Qwen3); add_bos for bake-off parity.
    return draft_llm.tokenize(text.encode("utf-8"), add_bos=True)


def draft_propose(draft_llm, context_tokens, gamma):
    """Greedy draft: propose up to gamma tokens given full context. Re-prefills each call."""
    t0 = time.perf_counter()
    proposed = []
    # temp=0 + top_k=1 → greedy; reset=True re-evals context each round (v0 simple)
    for tok in draft_llm.generate(
        context_tokens,
        top_k=1,
        top_p=1.0,
        temp=0.0,
        repeat_penalty=1.0,
        reset=True,
    ):
        proposed.append(int(tok))
        if len(proposed) >= gamma:
            break
    dt = time.perf_counter() - t0
    return proposed, dt


def prefill_verifier(engine, prompt_tokens):
    """Prefill prompt; return logits after last prompt token."""
    logits = None
    t0 = time.perf_counter()
    for i, tok in enumerate(prompt_tokens):
        logits = engine.forward_one_token(tok, i)
    return logits, time.perf_counter() - t0


def speculative_generate(
    engine,
    draft_llm,
    prompt_tokens,
    n_gen,
    gamma,
):
    """Serial speculative decode. Returns metrics + generated token ids."""
    engine.reset_kv()
    engine.expert_cache.hits = 0
    engine.expert_cache.pinned_hits = 0

    logits, prefill_s = prefill_verifier(engine, prompt_tokens)
    pos = len(prompt_tokens)

    generated = []
    accepted_draft = 0
    proposed_draft = 0
    verify_forwards = 0  # decode-phase verifier forwards only
    bonus_tokens = 0
    full_gamma_accepts = 0
    rounds = 0
    draft_time = 0.0
    verify_time = 0.0
    mismatch_rounds = 0

    t_decode0 = time.perf_counter()

    while len(generated) < n_gen:
        rounds += 1
        context = prompt_tokens + generated
        need = n_gen - len(generated)
        g = min(gamma, need)

        proposed, dt_draft = draft_propose(draft_llm, context, g)
        draft_time += dt_draft
        if not proposed:
            # Draft produced nothing — fall back to verifier greedy
            t1 = time.perf_counter()
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            bonus_tokens += 1
            if len(generated) < n_gen:
                logits = engine.forward_one_token(next_tok, pos)
                verify_forwards += 1
                pos += 1
            verify_time += time.perf_counter() - t1
            continue

        proposed_draft += len(proposed)
        accepted_this_round = 0

        for j, dtok in enumerate(proposed):
            if len(generated) >= n_gen:
                break
            t1 = time.perf_counter()
            vtok = int(np.argmax(logits))
            # Compare draft proposal to verifier greedy choice at this position
            if dtok == vtok:
                generated.append(dtok)
                accepted_draft += 1
                accepted_this_round += 1
                # Advance verifier KV with accepted token (unless last needed)
                if len(generated) < n_gen:
                    logits = engine.forward_one_token(dtok, pos)
                    verify_forwards += 1
                    pos += 1
                verify_time += time.perf_counter() - t1
            else:
                # Reject: emit verifier token instead
                generated.append(vtok)
                bonus_tokens += 1
                mismatch_rounds += 1
                if len(generated) < n_gen:
                    logits = engine.forward_one_token(vtok, pos)
                    verify_forwards += 1
                    pos += 1
                verify_time += time.perf_counter() - t1
                break
        else:
            # All gamma draft tokens accepted — classic speculative bonus token
            if accepted_this_round == len(proposed) and len(generated) < n_gen:
                full_gamma_accepts += 1
                t1 = time.perf_counter()
                vtok = int(np.argmax(logits))
                generated.append(vtok)
                bonus_tokens += 1
                if len(generated) < n_gen:
                    logits = engine.forward_one_token(vtok, pos)
                    verify_forwards += 1
                    pos += 1
                verify_time += time.perf_counter() - t1

    decode_s = time.perf_counter() - t_decode0
    wall_s = prefill_s + decode_s
    n = len(generated)
    accept_rate = (accepted_draft / proposed_draft) if proposed_draft else 0.0
    # Mean accepted draft tokens per proposal round (excluding pure-bonus fallback)
    mean_accept_len = (accepted_draft / rounds) if rounds else 0.0

    ec = engine.expert_cache
    hit_rate = ec.pinned_hits / ec.hits if ec.hits else 0.0

    return {
        "n_prompt": len(prompt_tokens),
        "n_gen": n,
        "gamma": gamma,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "wall_s": wall_s,
        "draft_time_s": draft_time,
        "verify_time_s": verify_time,
        "effective_tok_s": n / decode_s if decode_s > 0 else 0.0,
        "wall_tok_s": n / wall_s if wall_s > 0 else 0.0,
        "draft_tok_s": proposed_draft / draft_time if draft_time > 0 else 0.0,
        "verify_forwards": verify_forwards,
        "verify_forwards_per_gen_tok": verify_forwards / n if n else 0.0,
        "proposed_draft": proposed_draft,
        "accepted_draft": accepted_draft,
        "bonus_tokens": bonus_tokens,
        "accept_rate": accept_rate,
        "rounds": rounds,
        "full_gamma_accepts": full_gamma_accepts,
        "mismatch_rounds": mismatch_rounds,
        "mean_accept_len": mean_accept_len,
        "hit_rate": hit_rate,
        "generated_tokens": generated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--draft", default=DEFAULT_DRAFT)
    ap.add_argument("--n-gen", type=int, default=int(os.environ.get("SPEC_N_GEN", "64")))
    ap.add_argument("--n-reps", type=int, default=int(os.environ.get("SPEC_N_REPS", "2")))
    ap.add_argument("--gammas", default=os.environ.get("SPEC_GAMMAS", "2,4,8,16"))
    ap.add_argument("--draft-ngl", type=int, default=int(os.environ.get("SPEC_DRAFT_NGL", "0")),
                    help="llama.cpp n_gpu_layers for draft; 0=CPU (safer w/ Tiny pins), -1=Metal")
    ap.add_argument("--calibrate-n", type=int, default=10)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-quit-ollama", action="store_true")
    args = ap.parse_args()

    gammas = [int(x) for x in args.gammas.split(",") if x.strip()]
    repo = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
    ).strip()

    print("=" * 70)
    print("Speculative cascade Phase 1 v0")
    print(f"verifier={args.model}")
    print(f"cache={args.cache}")
    print(f"draft={args.draft} ngl={args.draft_ngl}")
    print(f"n_gen={args.n_gen} n_reps={args.n_reps} gammas={gammas}")
    print(f"commit={commit}")
    print("=" * 70)

    if not args.skip_quit_ollama:
        print("Quitting Ollama to free RAM...", flush=True)
        quit_ollama()

    # Tokenize with a vocab-only draft handle first (tiny RAM), then load Tiny+pins,
    # then full draft. Keeps pin budget from fighting Metal draft heaps on 16GB.
    from llama_cpp import Llama
    print("Loading draft vocab-only for tokenize...", flush=True)
    vocab_llm = Llama(
        model_path=args.draft, n_ctx=32, n_gpu_layers=0,
        vocab_only=True, verbose=False,
    )
    prompt_tokens = tokenize_prompt(vocab_llm, CODING_PROMPT)
    print(f"prompt tokens: {len(prompt_tokens)} -> {prompt_tokens[:12]}...", flush=True)
    del vocab_llm

    t_load0 = time.perf_counter()
    engine = NWSEngine(args.model, args.cache, profile="cpu-only")
    engine_load_s = time.perf_counter() - t_load0
    print(f"Engine load wall: {engine_load_s:.1f}s", flush=True)

    # pin48 uniform after calibrate on coding prompt (Phase 0 parity)
    engine.expert_cache.pinned = set()
    engine.expert_cache.access_counts = {}
    engine.reset_kv()
    print(f"Calibrating {args.calibrate_n} tokens...", flush=True)
    engine.calibrate(prompt_tokens, n_tokens=args.calibrate_n)
    engine.expert_cache.pin_from_usage(48, N_LAYERS)
    print(f"Pinned {len(engine.expert_cache.pinned)} experts (pin48)", flush=True)

    draft_llm = load_draft(args.draft, n_gpu_layers=args.draft_ngl)

    baseline_coding = 2.236  # Phase 0 mean
    gate_a_threshold = 3.0 * baseline_coding

    trials = []
    for gamma in gammas:
        for rep in range(args.n_reps):
            print(f"\n--- gamma={gamma} rep={rep+1}/{args.n_reps} ---", flush=True)
            r = speculative_generate(
                engine, draft_llm, prompt_tokens, args.n_gen, gamma
            )
            r["rep"] = rep + 1
            trials.append(r)
            print(
                f"  accept_rate={r['accept_rate']:.3f}  "
                f"effective={r['effective_tok_s']:.3f} tok/s  "
                f"draft={r['draft_tok_s']:.1f} tok/s  "
                f"verify_fwd={r['verify_forwards']}  "
                f"mean_accept_len={r['mean_accept_len']:.2f}  "
                f"({r['decode_s']:.1f}s decode)",
                flush=True,
            )
            # Drop heavy token list from printed path but keep in JSON optionally trimmed
            # Keep generated_tokens in trial for debugging quality later

    by_gamma = {}
    for gamma in gammas:
        subset = [t for t in trials if t["gamma"] == gamma]
        ar, ar_s = mean_std([t["accept_rate"] for t in subset])
        ef, ef_s = mean_std([t["effective_tok_s"] for t in subset])
        dr, dr_s = mean_std([t["draft_tok_s"] for t in subset])
        vf, vf_s = mean_std([t["verify_forwards"] for t in subset])
        mal, mal_s = mean_std([t["mean_accept_len"] for t in subset])
        by_gamma[str(gamma)] = {
            "n": len(subset),
            "accept_rate_mean": ar,
            "accept_rate_std": ar_s,
            "effective_tok_s_mean": ef,
            "effective_tok_s_std": ef_s,
            "draft_tok_s_mean": dr,
            "draft_tok_s_std": dr_s,
            "verify_forwards_mean": vf,
            "verify_forwards_std": vf_s,
            "mean_accept_len_mean": mal,
            "mean_accept_len_std": mal_s,
            "gate_a_pass": ef >= gate_a_threshold,
        }

    best_gamma = max(gammas, key=lambda g: by_gamma[str(g)]["effective_tok_s_mean"])
    best = by_gamma[str(best_gamma)]
    any_gate_a = any(by_gamma[str(g)]["gate_a_pass"] for g in gammas)
    # Gate B: accept_rate >= 0.55 on coding
    best_accept = max(by_gamma[str(g)]["accept_rate_mean"] for g in gammas)
    gate_b_pass = best_accept >= 0.55

    # Strip full token lists from summary trials copy? Keep them — useful, small for n=64.
    out = {
        "date": "2026-10-25",
        "timezone": "America/New_York",
        "phase": 1,
        "variant": "v0_serial_verify",
        "machine": {
            "machineId": "c8ccd663-f16c-4443-a6ca-d6f91982e0aa",
            "label": "MacBook-Pro",
            "chip": "Apple M1",
            "ram_gib": 16,
        },
        "tinygiant_commit": commit,
        "HYBRID_ATTN_READY": False,
        "profile": "cpu-only",
        "pin": "pin48 uniform after calibrate_n on coding",
        "calibrate_n": args.calibrate_n,
        "pinned_experts": len(engine.expert_cache.pinned),
        "model": args.model,
        "cache": args.cache,
        "draft": {
            "path": args.draft,
            "family": "Qwen3-1.7B",
            "n_gpu_layers": args.draft_ngl,
            "backend": "metal" if args.draft_ngl != 0 else "cpu",
        },
        "prompt_task": "coding",
        "prompt": CODING_PROMPT,
        "n_prompt_tokens": len(prompt_tokens),
        "n_gen": args.n_gen,
        "n_reps": args.n_reps,
        "gammas": gammas,
        "engine_load_s": engine_load_s,
        "phase0_baseline_coding_decode_tok_s": baseline_coding,
        "gate_a_threshold_tok_s": gate_a_threshold,
        "gate_a_rule": "effective_tok_s >= 3× Phase0 Tiny coding (~2.236 → ≥6.708)",
        "gate_b_rule": "accept_rate >= 0.55 on coding",
        "gate_a_pass_any": any_gate_a,
        "gate_b_pass": gate_b_pass,
        "best_gamma_by_effective": best_gamma,
        "best_effective_tok_s": best["effective_tok_s_mean"],
        "best_accept_rate": best_accept,
        "by_gamma": by_gamma,
        "trials": [
            {k: v for k, v in t.items() if k != "generated_tokens"}
            for t in trials
        ],
        "trials_with_tokens": trials,
        "algorithm_notes": [
            "Greedy draft (temp=0, top_k=1) proposes γ tokens via llama-cpp generate(reset=True).",
            "Verifier (NWSEngine) accepts longest matching greedy prefix token-by-token (serial).",
            "On mismatch: emit verifier argmax and end round. On full-γ accept: emit bonus verifier token.",
            "Serial verify ⇒ ~1 Tiny forward per output token ⇒ effective ≈ baseline; Gate A expected FAIL.",
            "accept_rate is the Phase-1 signal; teacher-forced batch verify is the speed path (v1).",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Gate A any pass: {any_gate_a} (threshold {gate_a_threshold:.3f} tok/s)", flush=True)
    print(f"Gate B accept>=0.55: {gate_b_pass} (best accept {best_accept:.3f})", flush=True)
    for g in gammas:
        s = by_gamma[str(g)]
        print(
            f"  γ={g}: accept={s['accept_rate_mean']:.3f}±{s['accept_rate_std']:.3f}  "
            f"effective={s['effective_tok_s_mean']:.3f}±{s['effective_tok_s_std']:.3f}  "
            f"GateA={'PASS' if s['gate_a_pass'] else 'FAIL'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
