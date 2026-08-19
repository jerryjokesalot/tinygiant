import ctypes
import json
import os
import time

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

from ._constants import (
    EMBED_DIM, EXPERT_INTERMEDIATE, GQA_RATIO, HEAD_DIM,
    N_EXPERTS_USED, N_HEADS, N_KV_HEADS, N_LAYERS, RMS_EPS,
)
from ._lib import load_tinygiant_lib
from ._math import apply_rope, build_rope_cache, rms_norm
from .cache import ExpertCache


def _dequant(tensor):
    return gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


class NWSEngine:

    def __init__(self, model_path, cache_dir, lib_path=None):
        print("TinyGiant Inference Engine")
        print("=" * 60)

        t_start = time.perf_counter()

        # Load GGUF
        print("Loading GGUF reader...", end="", flush=True)
        t0 = time.perf_counter()
        self.reader = GGUFReader(model_path)
        self.tensor_map = {t.name: t for t in self.reader.tensors}
        print(f" {time.perf_counter()-t0:.1f}s")

        # Load cache index
        with open(os.path.join(cache_dir, "index.json")) as f:
            self.cache_index = json.load(f)
        n_cached = len(self.cache_index["layers"])
        self.is_q4 = self.cache_index.get("dtype") == "q4_k"
        fmt_label = "Q4 (fused)" if self.is_q4 else "f16 (numpy)"
        print(f"NWS cache: {n_cached} layers [{fmt_label}]")

        # Load C library
        self.tg_lib = None
        if self.is_q4:
            self.tg_lib = load_tinygiant_lib(lib_path)
            if self.tg_lib:
                print("  libtinygiant: loaded (fused Q4xQ8 kernel)")
            else:
                print("  WARNING: libtinygiant not found, falling back to numpy")
                self.is_q4 = False

        # Expert cache
        self.expert_cache = ExpertCache(cache_dir, self.cache_index, max_experts=512)
        if self.tg_lib:
            self.expert_cache.set_lib(self.tg_lib)

        # Embeddings
        print("Loading embeddings...", end="", flush=True)
        t0 = time.perf_counter()
        self.token_embd = _dequant(self.tensor_map["token_embd.weight"])
        print(f" {time.perf_counter()-t0:.1f}s ({self.token_embd.nbytes/1024**2:.0f} MB)")

        # Output head
        print("Loading output head...", end="", flush=True)
        t0 = time.perf_counter()
        self.output_weight = _dequant(self.tensor_map["output.weight"])
        self.output_norm = _dequant(self.tensor_map["output_norm.weight"])
        print(f" {time.perf_counter()-t0:.1f}s ({self.output_weight.nbytes/1024**2:.0f} MB)")

        # Norms + routers
        print("Loading norms and routers...", end="", flush=True)
        t0 = time.perf_counter()
        self.attn_norms = []
        self.ffn_norms = []
        self.routers = []
        self.qk_norms = []
        for i in range(N_LAYERS):
            self.attn_norms.append(_dequant(self.tensor_map[f"blk.{i}.attn_norm.weight"]))
            self.ffn_norms.append(_dequant(self.tensor_map[f"blk.{i}.ffn_norm.weight"]))
            self.routers.append(_dequant(self.tensor_map[f"blk.{i}.ffn_gate_inp.weight"]))
            q_norm = _dequant(self.tensor_map[f"blk.{i}.attn_q_norm.weight"])
            k_norm = _dequant(self.tensor_map[f"blk.{i}.attn_k_norm.weight"])
            self.qk_norms.append((q_norm, k_norm))
        print(f" {time.perf_counter()-t0:.1f}s")

        # Attention weights
        print("Loading attention weights...", end="", flush=True)
        t0 = time.perf_counter()
        self.attn_weights = []
        self.attn_q4 = self.is_q4 and self.tg_lib
        for i in range(N_LAYERS):
            if self.attn_q4:
                tq = self.tensor_map[f"blk.{i}.attn_q.weight"]
                tk = self.tensor_map[f"blk.{i}.attn_k.weight"]
                tv = self.tensor_map[f"blk.{i}.attn_v.weight"]
                to = self.tensor_map[f"blk.{i}.attn_output.weight"]
                wq_raw = tq.data.reshape(-1).view(np.uint8).copy()
                wk_raw = tk.data.reshape(-1).view(np.uint8).copy()
                wv_f16 = gguf_dequantize(tv.data, tv.tensor_type).astype(np.float16).copy()
                wo_raw = to.data.reshape(-1).view(np.uint8).copy()
                self.attn_weights.append({
                    "q": wq_raw, "k": wk_raw, "v": wv_f16, "o": wo_raw,
                })
            else:
                self.attn_weights.append({
                    "q": _dequant(self.tensor_map[f"blk.{i}.attn_q.weight"]),
                    "k": _dequant(self.tensor_map[f"blk.{i}.attn_k.weight"]),
                    "v": _dequant(self.tensor_map[f"blk.{i}.attn_v.weight"]),
                    "o": _dequant(self.tensor_map[f"blk.{i}.attn_output.weight"]),
                })
        attn_mb = sum(sum(w.nbytes for w in l.values()) for l in self.attn_weights) / 1024**2
        print(f" {time.perf_counter()-t0:.1f}s ({attn_mb:.0f} MB)")

        # RoPE cache
        self.rope_cos, self.rope_sin = build_rope_cache(4096)

        # KV cache
        self.kv_max = 4096
        self.kv_len = 0
        self.kv_k = [np.zeros((N_KV_HEADS, self.kv_max, HEAD_DIM), dtype=np.float32)
                      for _ in range(N_LAYERS)]
        self.kv_v = [np.zeros((N_KV_HEADS, self.kv_max, HEAD_DIM), dtype=np.float32)
                      for _ in range(N_LAYERS)]
        self.kv_cache = [
            {"k": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
             "v": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)}
            for _ in range(N_LAYERS)
        ]

        # mlock attention weights
        if self.tg_lib:
            for i in range(N_LAYERS):
                for k, v in self.attn_weights[i].items():
                    self.tg_lib.tg_mlock(v.ctypes.data, ctypes.c_size_t(v.nbytes))
            print(f"  Attention weights: mlock'd ({attn_mb:.0f} MB)")

        elapsed = time.perf_counter() - t_start
        print(f"\nEngine ready in {elapsed:.1f}s")

    def reset_kv(self):
        self.kv_len = 0
        for i in range(N_LAYERS):
            self.kv_k[i][:] = 0
            self.kv_v[i][:] = 0
            self.kv_cache[i] = {
                "k": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
                "v": np.zeros((N_KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
            }

    def calibrate(self, prompt_tokens, n_tokens=10):
        for i, tok in enumerate(prompt_tokens):
            logits = self.forward_one_token(tok, i)
        next_token = int(np.argmax(logits))
        for step in range(n_tokens):
            logits = self.forward_one_token(next_token, len(prompt_tokens) + step)
            next_token = int(np.argmax(logits))
        self.reset_kv()

    def pin_experts(self, n_per_layer, calibrate_tokens=None, prompt_tokens=None):
        """Pin hot experts into physical memory.

        If calibrate_tokens and prompt_tokens are given, runs calibration first
        and uses observed access patterns + static profile backfill.
        Otherwise uses static activation profile only.
        """
        if calibrate_tokens and prompt_tokens:
            print(f"Calibrating ({calibrate_tokens} tokens)...", end="", flush=True)
            t0 = time.perf_counter()
            self.calibrate(prompt_tokens, n_tokens=calibrate_tokens)
            print(f" {time.perf_counter()-t0:.1f}s")

            print(f"Pinning top-{n_per_layer} experts/layer from usage...", end="", flush=True)
            t0 = time.perf_counter()
            count = self.expert_cache.pin_from_usage(n_per_layer, N_LAYERS)
        else:
            print(f"Pinning top-{n_per_layer} experts/layer (static profile)...", end="", flush=True)
            t0 = time.perf_counter()
            count = self.expert_cache.pin_hot_experts(n_per_layer)

        pin_gb = count * 2.7 / 1024
        print(f" {count} experts ({pin_gb:.1f} GB) in {time.perf_counter()-t0:.1f}s")
        return count

    def forward_one_token(self, token_id, pos):
        x = np.ascontiguousarray(self.token_embd[token_id].copy(), dtype=np.float32)
        if self.is_q4 and self.tg_lib:
            return self._forward_c(x, pos)
        else:
            return self._forward_numpy(x, pos)

    def _forward_c(self, x, pos):
        lib = self.tg_lib
        ec = self.expert_cache

        for i in range(N_LAYERS):
            aw = self.attn_weights[i]
            q_norm_w, k_norm_w = self.qk_norms[i]

            attn_out = np.empty(EMBED_DIM, dtype=np.float32)
            normed = np.empty(EMBED_DIM, dtype=np.float32)
            lib.tg_rms_norm(x.ctypes.data, self.attn_norms[i].ctypes.data,
                            normed.ctypes.data, EMBED_DIM, ctypes.c_float(RMS_EPS))
            if self.attn_q4:
                lib.tg_attention_decode_q4(
                    aw["q"].ctypes.data, aw["k"].ctypes.data,
                    aw["v"].ctypes.data, aw["o"].ctypes.data,
                    q_norm_w.ctypes.data, k_norm_w.ctypes.data,
                    self.kv_k[i].ctypes.data, self.kv_v[i].ctypes.data,
                    pos, self.kv_max,
                    self.rope_cos.ctypes.data, self.rope_sin.ctypes.data,
                    normed.ctypes.data, attn_out.ctypes.data,
                    EMBED_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, pos)
            else:
                lib.tg_attention_decode(
                    aw["q"].ctypes.data, aw["k"].ctypes.data,
                    aw["v"].ctypes.data, aw["o"].ctypes.data,
                    q_norm_w.ctypes.data, k_norm_w.ctypes.data,
                    self.kv_k[i].ctypes.data, self.kv_v[i].ctypes.data,
                    pos, self.kv_max,
                    self.rope_cos.ctypes.data, self.rope_sin.ctypes.data,
                    normed.ctypes.data, attn_out.ctypes.data,
                    EMBED_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, pos)
            x += attn_out

            lib.tg_rms_norm(x.ctypes.data, self.ffn_norms[i].ctypes.data,
                            normed.ctypes.data, EMBED_DIM, ctypes.c_float(RMS_EPS))
            logits = self.routers[i] @ normed
            top_k_idx = np.argsort(logits)[-N_EXPERTS_USED:]
            top_k_logits = logits[top_k_idx]
            exp_l = np.exp(top_k_logits - np.max(top_k_logits))
            weights = np.ascontiguousarray(
                (exp_l / np.sum(exp_l)).astype(np.float32))

            experts = [ec.get(i, int(eid)) for eid in top_k_idx]

            moe_out = np.zeros(EMBED_DIM, dtype=np.float32)
            for j, expert in enumerate(experts):
                fmt = expert.get("_formats", {}).get("down", "q4_k")
                if fmt == "float16":
                    down_fmt = 1
                elif fmt == "q6_k":
                    down_fmt = 2
                else:
                    down_fmt = 0
                lib.tg_expert_forward_mixed(
                    expert["gate"].ctypes.data,
                    expert["up"].ctypes.data,
                    expert["down"].ctypes.data,
                    down_fmt,
                    normed.ctypes.data,
                    moe_out.ctypes.data,
                    ctypes.c_float(float(weights[j])),
                    EMBED_DIM, EXPERT_INTERMEDIATE)
            x += moe_out

        self.kv_len = pos + 1

        out = np.empty(EMBED_DIM, dtype=np.float32)
        lib.tg_rms_norm(x.ctypes.data, self.output_norm.ctypes.data,
                        out.ctypes.data, EMBED_DIM, ctypes.c_float(RMS_EPS))
        logits = self.output_weight @ out
        return logits

    def _forward_numpy(self, x, pos):
        for i in range(N_LAYERS):
            aw = self.attn_weights[i]

            normed = rms_norm(x, self.attn_norms[i])
            q = (aw["q"] @ normed).reshape(N_HEADS, HEAD_DIM)
            k = (aw["k"] @ normed).reshape(N_KV_HEADS, HEAD_DIM)
            v = (aw["v"] @ normed).reshape(N_KV_HEADS, HEAD_DIM)
            q_norm_w, k_norm_w = self.qk_norms[i]
            q = rms_norm(q, q_norm_w)
            k = rms_norm(k, k_norm_w)
            q = apply_rope(q, self.rope_cos, self.rope_sin, pos)
            k = apply_rope(k, self.rope_cos, self.rope_sin, pos)
            kv = self.kv_cache[i]
            kv["k"] = np.concatenate([kv["k"], k[:, None, :]], axis=1)
            kv["v"] = np.concatenate([kv["v"], v[:, None, :]], axis=1)
            output = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)
            for h in range(N_HEADS):
                kv_h = h // GQA_RATIO
                q_h = q[h]
                k_all = kv["k"][kv_h]
                v_all = kv["v"][kv_h]
                scores = (k_all @ q_h) / np.sqrt(HEAD_DIM)
                scores_max = np.max(scores)
                exp_scores = np.exp(scores - scores_max)
                attn_w = exp_scores / np.sum(exp_scores)
                output[h * HEAD_DIM:(h + 1) * HEAD_DIM] = attn_w @ v_all
            x = x + aw["o"] @ output

            normed = rms_norm(x, self.ffn_norms[i])
            logits = self.routers[i] @ normed
            top_k_idx = np.argsort(logits)[-N_EXPERTS_USED:]
            top_k_logits = logits[top_k_idx]
            exp_l = np.exp(top_k_logits - np.max(top_k_logits))
            weights = exp_l / np.sum(exp_l)
            moe_out = np.zeros(EMBED_DIM, dtype=np.float32)
            for j, exp_id in enumerate(top_k_idx):
                expert = self.expert_cache.get(i, int(exp_id))
                gate_out = expert["gate"].astype(np.float32) @ normed
                up_out = expert["up"].astype(np.float32) @ normed
                silu = gate_out / (1 + np.exp(-np.clip(gate_out, -20, 20)))
                hidden = silu * up_out
                exp_out = expert["down"].astype(np.float32) @ hidden
                moe_out += weights[j] * exp_out
            x = x + moe_out

        x = rms_norm(x, self.output_norm)
        return self.output_weight @ x

    def generate(self, prompt_tokens, n_tokens=10, temperature=0.7, top_p=0.9):
        print(f"\nGenerating {n_tokens} tokens from {len(prompt_tokens)}-token prompt")
        print("-" * 60)

        generated = []

        # Prefill
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

        # Decode
        print("\nDecoding:", end="", flush=True)
        t_decode_start = time.perf_counter()

        for step in range(n_tokens):
            pos = len(prompt_tokens) + step

            if step == 0:
                next_logits = logits
            else:
                next_logits = self.forward_one_token(next_token, pos - 1)

            if temperature > 0:
                probs = np.exp((next_logits - np.max(next_logits)) / temperature)
                probs /= np.sum(probs)
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
            print(f" [{next_token}]", end="", flush=True)

        t_decode = time.perf_counter() - t_decode_start
        print(f"\n\nDecode: {n_tokens} tokens in {t_decode:.1f}s "
              f"({n_tokens/t_decode:.2f} tok/s)")

        ec = self.expert_cache
        print(f"Expert cache: mmap-backed, {ec.hits} accesses, "
              f"{len(ec._mmaps)} layer files mapped")
        if ec.pinned:
            print(f"  Pinned experts: {len(ec.pinned)} "
                  f"({ec.pinned_hits} pinned-region accesses)")

        return generated

    def generate_stream(self, prompt_tokens, n_tokens=128, temperature=0.7, top_p=0.9):
        """Generate tokens one at a time, yielding each as it's produced."""
        # Prefill
        for i, tok in enumerate(prompt_tokens):
            if i < len(prompt_tokens) - 1:
                self.forward_one_token(tok, i)
            else:
                logits = self.forward_one_token(tok, i)

        # Decode
        for step in range(n_tokens):
            pos = len(prompt_tokens) + step

            if step == 0:
                next_logits = logits
            else:
                next_logits = self.forward_one_token(next_token, pos - 1)

            if temperature > 0:
                probs = np.exp((next_logits - np.max(next_logits)) / temperature)
                probs /= np.sum(probs)
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, top_p) + 1
                candidates = sorted_idx[:cutoff]
                candidate_probs = probs[candidates]
                candidate_probs /= np.sum(candidate_probs)
                next_token = int(np.random.choice(candidates, p=candidate_probs))
            else:
                next_token = int(np.argmax(next_logits))

            yield next_token
