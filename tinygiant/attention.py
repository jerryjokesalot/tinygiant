"""Attention backend interface for Slice A (Metal attn+KV).

v1 ships:
  - ``CpuAttentionBackend`` — today's libtinygiant / NumPy path (always works)
  - ``MetalAttentionBackend`` — feature-flagged stub; ``available`` is False until
    real Metal kernels are linked. Selecting ``hybrid`` logs once and fail-opens
    to CPU so chat never crashes.

Experts-on-GPU (Slice B) is intentionally out of scope.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger("tinygiant.attention")

# Flip when a real Metal attention decode is linked into the build.
METAL_ATTN_LINKED = False


@runtime_checkable
class AttentionBackend(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def describe(self) -> str: ...


class CpuAttentionBackend:
    name = "cpu"

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "CPU (libtinygiant NEON/AVX / NumPy fallback)"


class MetalAttentionBackend:
    """Stub until Metal shaders / FFI land.

    ``available`` stays False while METAL_ATTN_LINKED is False. Callers must
    fall back to CPU; do not claim tok/s gains from this stub.
    """

    name = "metal"

    def __init__(self, linked: bool = METAL_ATTN_LINKED):
        self._linked = bool(linked)

    @property
    def available(self) -> bool:
        return self._linked

    def describe(self) -> str:
        if self._linked:
            return "Metal attention+KV (experimental)"
        return (
            "Metal attention not yet linked; falling back to CPU "
            "(see tinygiant/attention.py TODO)"
        )


def resolve_attention_backend(
    profile: str,
    *,
    metal_linked: bool = METAL_ATTN_LINKED,
    log: bool = True,
) -> AttentionBackend:
    """Pick an attention backend for a profile. Always returns a usable backend.

    ``hybrid`` / ``gpu-heavy`` request Metal; if the stub is not linked, log and
    return CPU (fail-open).
    """
    cpu = CpuAttentionBackend()
    p = (profile or "cpu-only").strip().lower().replace("_", "-")
    if p in ("hybrid", "gpu-heavy"):
        metal = MetalAttentionBackend(linked=metal_linked)
        if metal.available:
            if log:
                logger.info("Attention backend: %s (profile=%s)", metal.describe(), p)
            return metal
        msg = (
            f"profile={p}: Metal attn not yet linked; falling back to CPU. "
            "Experts remain on CPU+SSD+pin. No tok/s claim for hybrid until "
            "kernels land and a bake-off wins."
        )
        if log:
            logger.warning(msg)
            # Also print so CLI users see it without configuring logging.
            print(f"[tinygiant] {msg}")
        return cpu

    if log:
        logger.info("Attention backend: %s (profile=%s)", cpu.describe(), p)
    return cpu


def metal_attn_status() -> dict:
    """Introspection helper for tests / doctor."""
    return {
        "metal_attn_linked": METAL_ATTN_LINKED,
        "todo": (
            "Implement Metal attention decode (Q/K/V/O + KV) via thin Metal "
            "shader / MPS / ggml-metal-style FFI; migrate attn off host mlock "
            "into Metal buffers (no double residency); bake-off vs cpu-only "
            "on same Mac/model/prompts before flipping HYBRID_ATTN_READY."
        ),
    }
