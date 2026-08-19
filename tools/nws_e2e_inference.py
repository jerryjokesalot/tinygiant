"""
NWS End-to-End Inference Engine

Generates text from Qwen3-30B-A3B using expert-contiguous layout.
Attention + embeddings loaded from GGUF (small, load fine).
MoE experts loaded from NWS contiguous cache (563x faster than interleaved).

This is a PoC to prove the approach works end-to-end and measure real tok/s.
"""

import json
import mmap
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

MODEL_PATH = os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf")
CACHE_DIR = Path(__file__).parent / "nws_cache"

# Architecture constants (from GGUF metadata)
N_LAYERS = 48
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 128
EMBED_DIM = 2048
N_EXPERTS = 128
N_EXPERTS_USED = 8
EXPERT_INTERMEDIATE = 768
VOCAB_SIZE = 151936
ROPE_BASE = 1_000_000.0
RMS_EPS = 1e-6
GQA_RATIO = N_HEADS // N_KV_HEADS  # 8


def dequant(tensor):
    """Dequantize a GGUF tensor to float32. gguf_dequantize returns HF convention."""
    return gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def rms_norm(x, weight):
    rr = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + RMS_EPS)
    return (x / rr) * weight


def build_rope_cache(max_pos, dim=HEAD_DIM, base=ROPE_BASE):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin


def apply_rope(x, cos, sin, pos):
    """Apply rotary position encoding. x shape: (n_heads, head_dim)"""
    c = cos[pos]  # (head_dim//2,)
    s = sin[pos]
    x1 = x[:, :HEAD_DIM // 2]
    x2 = x[:, HEAD_DIM // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class ExpertCache:
    """Expert cache with persistent file handles and heap-copied data."""

    def __init__(self, cache_dir, index, max_experts=256):
        self.cache_dir = Path(cache_dir)
        self.index = index
        self.max_experts = max_experts
        self.hits = 0
        self.misses = 0
        self.pinned_hits = 0
        self.access_counts = {}
        self._files = {}
        self.cache = OrderedDict()
        self.pinned = set()

    def _get_file(self, layer_idx):
        if layer_idx not in self._files:
            layer_key = str(layer_idx)
            layer_info = self.index["layers"][layer_key]
            layer_file = self.cache_dir / layer_info["file"]
            self._files[layer_idx] = open(layer_file, "rb")
        return self._files[layer_idx]

    def pin_hot_experts(self, n_per_layer):
        """Pre-load top-N most activated experts per layer."""
        profile = self.index.get("activation_profile")
        if not profile:
            return 0
        count = 0
        for layer_key in sorted(profile.keys(), key=int):
            for exp_id in profile[layer_key]["expert_ranking"][:n_per_layer]:
                self.get(int(layer_key), exp_id)
                self.pinned.add((int(layer_key), exp_id))
                count += 1
        self.hits = self.misses = self.pinned_hits = 0
        return count

    def pin_from_usage(self, n_per_layer, n_layers):
        """Pin top-N experts per layer based on observed access patterns."""
        per_layer = {}
        for (layer, expert), cnt in self.access_counts.items():
            per_layer.setdefault(layer, []).append((-cnt, expert))
        count = 0
        for layer in range(n_layers):
            entries = sorted(per_layer.get(layer, []))
            for _, exp_id in entries[:n_per_layer]:
                if (layer, exp_id) not in self.cache:
                    self.get(layer, exp_id)
                self.pinned.add((layer, exp_id))
                count += 1
        self.hits = self.misses = self.pinned_hits = 0
        self.access_counts = {}
        return count

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            if key in self.pinned:
                self.pinned_hits += 1
            return self.cache[key]

        self.misses += 1
        f = self._get_file(layer_idx)
        layer_key = str(layer_idx)
        offsets = self.index["layers"][layer_key]["expert_offsets"][str(expert_id)]

        expert = {}
        f.seek(offsets["offset"])
        for k, shape in [("gate", (EXPERT_INTERMEDIATE, EMBED_DIM)),
                         ("up", (EXPERT_INTERMEDIATE, EMBED_DIM)),
                         ("down", (EMBED_DIM, EXPERT_INTERMEDIATE))]:
            raw = f.read(offsets["sizes"][k])
            expert[k] = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()

        self.cache[key] = expert
        while len(self.cache) > self.max_experts:
            evicted = False
            for k in self.cache:
                if k not in self.pinned:
                    del self.cache[k]
                    evicted = True
                    break
            if not evicted:
                break

        return expert


class NWSEngine:

    def __init__(self):
        print("NWS End-to-End Inference Engine")
        print("=" * 60)

        t_start = time.perf_counter()

        # Load GGUF
        print("Loading GGUF reader...", end="", flush=True)
        t0 = time.perf_counter()
        self.reader = GGUFReader(MODEL_PATH)
        self.tensor_map = {t.name: t for t in self.reader.tensors}
        print(f" {time.perf_counter()-t0:.1f}s")

        # Load cache index
        with open(CACHE_DIR / "index.json") as f:
            self.cache_index = json.load(f)
        n_cached = len(self.cache_index["layers"])
        print(f"NWS cache: {n_cached} layers")

        # Expert LRU cache
        self.expert_cache = ExpertCache(CACHE_DIR, self.cache_index, max_experts=512)

        # Dequantize embeddings
        print("Loading embeddings...", end="", flush=True)
        t0 = time.perf_counter()
        self.token_embd = dequant(self.tensor_map["token_embd.weight"])
        print(f" {time.perf_counter()-t0:.1f}s ({self.token_embd.nbytes/1024**2:.0f} MB)")

        # Dequantize output head
        print("Loading output head...", end="", flush=True)
        t0 = time.perf_counter()
        self.output_weight = dequant(self.tensor_map["output.weight"])
        self.output_norm = dequant(self.tensor_map["output_norm.weight"])
        print(f" {time.perf_counter()-t0:.1f}s ({self.output_weight.nbytes/1024**2:.0f} MB)")

        # Dequantize all layer norms + routers (tiny, always needed)
        print("Loading norms and routers...", end="", flush=True)
        t0 = time.perf_counter()
        self.attn_norms = []
        self.ffn_norms = []
        self.routers = []
        self.qk_norms = []
        for i in range(N_LAYERS):
            self.attn_norms.append(dequant(self.tensor_map[f"blk.{i}.attn_norm.weight"]))
            self.ffn_norms.append(dequant(self.tensor_map[f"blk.{i}.ffn_norm.weight"]))
            self.routers.append(dequant(self.tensor_map[f"blk.{i}.ffn_gate_inp.weight"]))
            q_norm = dequant(self.tensor_map[f"blk.{i}.attn_q_norm.weight"])
            k_norm = dequant(self.tensor_map[f"blk.{i}.attn_k_norm.weight"])
            self.qk_norms.append((q_norm, k_norm))
        print(f" {time.perf_counter()-t0:.1f}s")

        # Preload attention weights (small enough to keep in RAM: ~3.4 GB)
        print("Loading attention weights...", end="", flush=True)
        t0 = time.perf_counter()
        self.attn_weights = []
        for i in range(N_LAYERS):
            self.attn_weights.append({
                "q": dequant(self.tensor_map[f"blk.{i}.attn_q.weight"]),
                "k": dequant(self.tensor_map[f"blk.{i}.attn_k.weight"]),
                "v": dequant(self.tensor_map[f"blk.{i}.attn_v.weight"]),
                "o": dequant(self.tensor_map[f"blk.{i}.attn_output.weight"]),
            })
        attn_mb = sum(sum(w.nbytes for w in l.values()) for l in self.attn_weights) / 1024**2
        print(f" {time.perf_counter()-t0:.1f}s ({attn_mb:.0f} MB)")

        # RoPE cache
        self.rope_cos, self.rope_sin = build_rope_cache(4096)

        # KV cache
        self.kv_cache = [
            {
                "k": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
                "v": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
            }
            for _ in range(N_LAYERS)
        ]

        elapsed = time.perf_counter() - t_start
        print(f"\nEngine ready in {elapsed:.1f}s")

    def calibrate(self, prompt_tokens, n_tokens=10):
        """Run a calibration pass to observe expert access patterns, then reset state."""
        for i, tok in enumerate(prompt_tokens):
            logits = self.forward_one_token(tok, i)
        next_token = int(np.argmax(logits))
        for step in range(n_tokens):
            logits = self.forward_one_token(next_token, len(prompt_tokens) + step)
            next_token = int(np.argmax(logits))
        self.kv_cache = [
            {"k": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
             "v": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)}
            for _ in range(N_LAYERS)
        ]

    def attention(self, layer_idx, x, pos):
        """GQA attention with KV cache. x shape: (embed_dim,)"""
        aw = self.attn_weights[layer_idx]
        q = (aw["q"] @ x).reshape(N_HEADS, HEAD_DIM)      # (32, 128)
        k = (aw["k"] @ x).reshape(N_KV_HEADS, HEAD_DIM)   # (4, 128)
        v = (aw["v"] @ x).reshape(N_KV_HEADS, HEAD_DIM)   # (4, 128)

        # QK normalization
        q_norm_w, k_norm_w = self.qk_norms[layer_idx]
        q = rms_norm(q, q_norm_w)
        k = rms_norm(k, k_norm_w)

        # RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin, pos)
        k = apply_rope(k, self.rope_cos, self.rope_sin, pos)

        # Update KV cache
        kv = self.kv_cache[layer_idx]
        kv["k"] = np.concatenate([kv["k"], k[:, None, :]], axis=1)
        kv["v"] = np.concatenate([kv["v"], v[:, None, :]], axis=1)

        seq_len = kv["k"].shape[1]

        # GQA: each group of 8 query heads shares 1 KV head
        output = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)

        for h in range(N_HEADS):
            kv_h = h // GQA_RATIO
            q_h = q[h]  # (128,)
            k_all = kv["k"][kv_h]  # (seq_len, 128)
            v_all = kv["v"][kv_h]  # (seq_len, 128)

            scores = (k_all @ q_h) / np.sqrt(HEAD_DIM)  # (seq_len,)

            # Causal mask (all positions visible in autoregressive decode)
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / np.sum(exp_scores)

            out_h = attn_weights @ v_all  # (128,)
            output[h * HEAD_DIM:(h + 1) * HEAD_DIM] = out_h

        result = aw["o"] @ output  # (2048,)
        return result

    def moe_layer(self, layer_idx, x):
        """MoE forward pass with expert loading from contiguous cache."""
        logits = self.routers[layer_idx] @ x  # (128,)
        top_k_idx = np.argsort(logits)[-N_EXPERTS_USED:]
        top_k_logits = logits[top_k_idx]

        # Softmax
        exp_l = np.exp(top_k_logits - np.max(top_k_logits))
        weights = exp_l / np.sum(exp_l)

        # Forward through selected experts
        output = np.zeros(EMBED_DIM, dtype=np.float32)
        for i, exp_id in enumerate(top_k_idx):
            expert = self.expert_cache.get(layer_idx, int(exp_id))
            gate_out = expert["gate"].astype(np.float32) @ x   # (768,)
            up_out = expert["up"].astype(np.float32) @ x       # (768,)
            silu = gate_out / (1 + np.exp(-np.clip(gate_out, -20, 20)))
            hidden = silu * up_out
            exp_out = expert["down"].astype(np.float32) @ hidden  # (2048,)
            output += weights[i] * exp_out

        return output

    def forward_one_token(self, token_id, pos):
        """Full forward pass for one token. Returns logits."""
        x = self.token_embd[token_id]  # (2048,)

        # Transformer layers
        for i in range(N_LAYERS):
            # Attention block
            normed = rms_norm(x, self.attn_norms[i])
            attn_out = self.attention(i, normed, pos)
            x = x + attn_out

            # MoE block
            normed = rms_norm(x, self.ffn_norms[i])
            moe_out = self.moe_layer(i, normed)
            x = x + moe_out

        # Final norm + LM head
        x = rms_norm(x, self.output_norm)
        logits = self.output_weight @ x  # (vocab_size,)
        return logits

    def generate(self, prompt_tokens, n_tokens=10, temperature=0.7, top_p=0.9):
        """Generate tokens autoregressively."""
        print(f"\nGenerating {n_tokens} tokens from {len(prompt_tokens)}-token prompt")
        print("-" * 60)

        generated = []
        total_positions = len(prompt_tokens) + n_tokens

        # Prefill: process prompt tokens
        print("Prefill...", end="", flush=True)
        t_prefill_start = time.perf_counter()
        for i, tok in enumerate(prompt_tokens):
            if i < len(prompt_tokens) - 1:
                self.forward_one_token(tok, i)
            else:
                logits = self.forward_one_token(tok, i)
            if (i + 1) % 5 == 0 or i == len(prompt_tokens) - 1:
                print(f" {i+1}", end="", flush=True)
        t_prefill = time.perf_counter() - t_prefill_start
        print(f"\nPrefill: {len(prompt_tokens)} tokens in {t_prefill:.1f}s "
              f"({len(prompt_tokens)/t_prefill:.1f} tok/s)")

        # Decode: generate new tokens
        print("\nDecoding:", end="", flush=True)
        t_decode_start = time.perf_counter()
        layer_times = {"attn": 0, "moe": 0, "other": 0}

        for step in range(n_tokens):
            pos = len(prompt_tokens) + step

            if step == 0:
                next_logits = logits
            else:
                next_logits = self.forward_one_token(next_token, pos - 1)

            # Sample
            if temperature > 0:
                probs = np.exp((next_logits - np.max(next_logits)) / temperature)
                probs /= np.sum(probs)

                # Top-p
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, top_p) + 1
                candidates = sorted_idx[:cutoff]
                candidate_probs = probs[candidates]
                candidate_probs /= np.sum(candidate_probs)
                next_token = int(np.random.choice(candidates, p=candidate_probs))
            else:
                next_token = int(np.argmax(next_logits))

            generated.append(next_token)

            elapsed = time.perf_counter() - t_decode_start
            tok_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f" [{next_token}]", end="", flush=True)

        t_decode = time.perf_counter() - t_decode_start
        print(f"\n\nDecode: {n_tokens} tokens in {t_decode:.1f}s "
              f"({n_tokens/t_decode:.2f} tok/s)")

        # Expert cache stats
        ec = self.expert_cache
        total = ec.hits + ec.misses
        print(f"Expert cache: {ec.hits}/{total} hits ({ec.hits/total*100:.0f}%)")
        if ec.pinned:
            lru_only = ec.hits - ec.pinned_hits
            print(f"  Pinned hits: {ec.pinned_hits} ({ec.pinned_hits/total*100:.0f}%)")
            print(f"  LRU hits: {lru_only} ({lru_only/total*100:.0f}%)")
            print(f"  SSD reads: {ec.misses} ({ec.misses/total*100:.0f}%)")
        print(f"Experts in cache: {len(ec.cache)} ({len(ec.pinned)} pinned)")

        return generated


def tokenize_simple(text):
    """Use llama-cpp-python for tokenization."""
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=32, n_gpu_layers=0, vocab_only=True,
                verbose=False)
    tokens = llm.tokenize(text.encode(), add_bos=True)
    return tokens, llm


def detokenize(llm, tokens):
    """Detokenize a list of token IDs."""
    return llm.detokenize(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", type=int, default=0, help="Pin top-N experts per layer")
    parser.add_argument("--tokens", type=int, default=10, help="Tokens to generate")
    parser.add_argument("--calibrate", type=int, default=0, help="Calibration tokens (pins from actual usage)")
    cli_args = parser.parse_args()

    # Check cache completeness
    with open(CACHE_DIR / "index.json") as f:
        idx = json.load(f)
    cached_layers = sorted(int(k) for k in idx["layers"].keys())
    missing = [i for i in range(N_LAYERS) if i not in cached_layers]
    if missing:
        print(f"ERROR: Missing layers in cache: {missing}")
        print(f"Run nws_expert_relayout.py for all layers first.")
        sys.exit(1)

    # Tokenize prompt
    prompt = "The key insight about mixture-of-experts models is that"
    print(f"Prompt: {prompt!r}")
    tokens, llm = tokenize_simple(prompt)
    print(f"Tokens: {tokens} ({len(tokens)} tokens)")

    # Build engine
    engine = NWSEngine()

    # Calibrate and/or pin hot experts
    if cli_args.calibrate:
        n_pin = cli_args.pin or 8
        cal_tokens = cli_args.calibrate
        print(f"\nCalibrating ({cal_tokens} tokens)...", end="", flush=True)
        t0 = time.perf_counter()
        engine.calibrate(tokens, n_tokens=cal_tokens)
        print(f" {time.perf_counter()-t0:.1f}s")

        print(f"Pinning top-{n_pin} experts/layer from usage...", end="", flush=True)
        t0 = time.perf_counter()
        engine.expert_cache.max_experts = n_pin * N_LAYERS + 256
        count = engine.expert_cache.pin_from_usage(n_pin, N_LAYERS)
        pin_gb = count * 9.0 / 1024
        print(f" {count} experts ({pin_gb:.1f} GB) in {time.perf_counter()-t0:.1f}s")
    elif cli_args.pin:
        n_pin = cli_args.pin
        print(f"\nPinning top-{n_pin} experts/layer (random profile)...", end="", flush=True)
        t0 = time.perf_counter()
        engine.expert_cache.max_experts = n_pin * N_LAYERS + 256
        count = engine.expert_cache.pin_hot_experts(n_pin)
        pin_gb = count * 9.0 / 1024
        print(f" {count} experts ({pin_gb:.1f} GB) in {time.perf_counter()-t0:.1f}s")

    # Generate
    np.random.seed(42)
    generated = engine.generate(tokens, n_tokens=cli_args.tokens, temperature=0.7)

    # Decode output
    full_tokens = list(tokens) + generated
    output_text = detokenize(llm, full_tokens)
    gen_text = detokenize(llm, generated)
    print(f"\nFull output: {output_text}")
    print(f"Generated: {gen_text}")
