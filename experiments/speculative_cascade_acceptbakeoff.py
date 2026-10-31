"""Speculative cascade accept-rate bake-off (2026-10-31).

Changes vs v0: both draft and TinyGiant see the *same* token ids built from the
Qwen3 chat template (no-think: enable_thinking=False -> empty <think></think>),
so formatting/thinking mode match by construction. Compares draft GGUFs
(Qwen3-0.6B vs 1.7B) over coding + chat/prose prompts and gammas 2/4/8.
Serial verify (v0 algorithm). Does not touch HYBRID_ATTN_READY.

theoretical_speedup_perfect_batch = n_gen / rounds: tokens emitted per Tiny
forward if each round's gamma tokens were verified in one batched forward,
draft cost ignored. Also reports analytic (1-a^(g+1))/(1-a).
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speculative_cascade_v0 import (  # noqa: E402
    CODING_PROMPT, DEFAULT_CACHE, DEFAULT_MODEL, load_draft, mean_std,
    quit_ollama, speculative_generate,
)
from tinygiant._constants import N_LAYERS  # noqa: E402
from tinygiant.engine import NWSEngine  # noqa: E402

PROMPTS = {
    "coding": [CODING_PROMPT],
    "chat": [
        "Explain in a few sentences why the sky is blue.",
        "Write a short, friendly email inviting a coworker to lunch on Friday.",
    ],
}


def chat_template(user_msg: str, thinking: bool = False) -> str:
    """Qwen3 chat template, generation prompt, enable_thinking flag."""
    s = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    if not thinking:
        s += "<think>\n\n</think>\n\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--drafts", default=os.path.expanduser(
        "0.6b=~/models/qwen3-0.6b.gguf,1.7b=~/models/qwen3-1.7b.gguf"))
    ap.add_argument("--gammas", default="2,4,8")
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--draft-ngl", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--calibrate-n", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "benchmarks" / "speculative_cascade_acceptbakeoff_2026-10-31.json"))
    args = ap.parse_args()

    drafts = {}
    for kv in args.drafts.split(","):
        k, v = kv.split("=", 1)
        drafts[k] = os.path.expanduser(v)
    gammas = [int(x) for x in args.gammas.split(",")]
    repo = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()

    quit_ollama()
    from llama_cpp import Llama
    vocab = Llama(model_path=next(iter(drafts.values())), n_ctx=32, n_gpu_layers=0, vocab_only=True, verbose=False)
    tok = {}
    for task, ps in PROMPTS.items():
        for i, p in enumerate(ps):
            tok[f"{task}{i}"] = (task, p, vocab.tokenize(chat_template(p, args.thinking).encode(), add_bos=False, special=True))
    # Sanity: tokenizers must agree across drafts (same Qwen3 vocab)
    for name, path in drafts.items():
        v2 = Llama(model_path=path, n_ctx=32, n_gpu_layers=0, vocab_only=True, verbose=False)
        for key, (_, p, ids) in tok.items():
            assert v2.tokenize(chat_template(p, args.thinking).encode(), add_bos=False, special=True) == ids, (name, key)
        del v2
    del vocab

    engine = NWSEngine(args.model, args.cache, profile="cpu-only")
    engine.expert_cache.pinned = set(); engine.expert_cache.access_counts = {}
    engine.reset_kv()
    engine.calibrate(tok["coding0"][2], n_tokens=args.calibrate_n)
    engine.expert_cache.pin_from_usage(48, N_LAYERS)

    trials = []
    for dname, dpath in drafts.items():
        llm = load_draft(dpath, n_gpu_layers=args.draft_ngl)
        for key, (task, p, ids) in tok.items():
            for g in gammas:
                r = speculative_generate(engine, llm, ids, args.n_gen, g)
                r.update(draft=dname, prompt_key=key, task=task,
                         theoretical_speedup_perfect_batch=r["n_gen"] / r["rounds"] if r["rounds"] else 0.0)
                a = r["accept_rate"]
                r["analytic_speedup"] = (g + 1) if a >= 1 else (1 - a ** (g + 1)) / (1 - a)
                trials.append(r)
                print(f"{dname} {key} g={g} accept={a:.3f} mal={r['mean_accept_len']:.2f} "
                      f"draft={r['draft_tok_s']:.1f}t/s spd={r['theoretical_speedup_perfect_batch']:.2f}", flush=True)
        del llm

    summary = {}
    for dname in drafts:
        for task in list(PROMPTS) + ["all"]:
            for g in gammas:
                s = [t for t in trials if t["draft"] == dname and t["gamma"] == g and (task == "all" or t["task"] == task)]
                acc = sum(t["accepted_draft"] for t in s); prop = sum(t["proposed_draft"] for t in s)
                summary[f"{dname}|{task}|{g}"] = {
                    "n_trials": len(s),
                    "accept_rate_pooled": acc / prop if prop else 0.0,
                    "mean_accept_len": mean_std([t["mean_accept_len"] for t in s])[0],
                    "draft_tok_s": mean_std([t["draft_tok_s"] for t in s])[0],
                    "theoretical_speedup_perfect_batch": sum(t["n_gen"] for t in s) / sum(t["rounds"] for t in s),
                    "effective_tok_s_serial": mean_std([t["effective_tok_s"] for t in s])[0],
                }
    best = max(v["accept_rate_pooled"] for k, v in summary.items() if "|all|" in k)
    out = dict(date="2026-10-31", timezone="America/New_York", machine="MacBook-Pro M1 16GB c8ccd663",
               tinygiant_commit=commit, HYBRID_ATTN_READY=False, template="qwen3 chat, enable_thinking="
               + str(args.thinking), draft_ngl=args.draft_ngl, n_gen=args.n_gen, gammas=gammas,
               drafts=drafts, prompts={k: v[1] for k, v in tok.items()},
               n_prompt_tokens={k: len(v[2]) for k, v in tok.items()},
               best_accept_rate_all=best, go_bar_0p8=best >= 0.8, gate_b_0p55=best >= 0.55,
               summary=summary, trials=trials)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out, "best", best)


if __name__ == "__main__":
    main()
