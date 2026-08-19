import numpy as np

from ._constants import HEAD_DIM, RMS_EPS, ROPE_BASE


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
    c = cos[pos]
    s = sin[pos]
    x1 = x[:, :HEAD_DIM // 2]
    x2 = x[:, HEAD_DIM // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
