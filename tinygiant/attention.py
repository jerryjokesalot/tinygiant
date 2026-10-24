"""Attention backend interface for Slice A (Metal attn+KV).

v1:
  - ``CpuAttentionBackend`` — libtinygiant / NumPy (always works; default)
  - ``MetalAttentionBackend`` — real Metal path when ``tg_metal_*`` is linked
    and a device+shaders initialize. Selecting ``hybrid`` fail-opens to CPU
    on any failure so chat never crashes.

``METAL_ATTN_LINKED`` is True only when the native library exposes a working
Metal attention compute path (probed at import). ``HYBRID_ATTN_READY`` in
profile.py stays False until a dated bake-off wins — do not invent tok/s.

Experts-on-GPU (Slice B) is intentionally out of scope.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger("tinygiant.attention")


def _probe_metal_linked() -> bool:
    """True only if libtinygiant exports Metal attn and the device works."""
    try:
        from ._lib import load_tinygiant_lib

        lib = load_tinygiant_lib()
        if lib is None or not hasattr(lib, "tg_metal_available"):
            return False
        return bool(lib.tg_metal_available())
    except Exception:
        return False


# Flip only when a real Metal attention decode is linked AND probes clean.
METAL_ATTN_LINKED = _probe_metal_linked()


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
    """Metal attention+KV when native path computes; else unavailable."""

    name = "metal"

    def __init__(self, linked: Optional[bool] = None):
        self._linked = METAL_ATTN_LINKED if linked is None else bool(linked)

    @property
    def available(self) -> bool:
        return self._linked

    def describe(self) -> str:
        if self._linked:
            detail = ""
            try:
                import ctypes
                from ._lib import load_tinygiant_lib

                lib = load_tinygiant_lib()
                if lib and hasattr(lib, "tg_metal_status"):
                    lib.tg_metal_status.restype = ctypes.c_char_p
                    s = lib.tg_metal_status()
                    if s:
                        detail = f" — {s.decode() if isinstance(s, bytes) else s}"
            except Exception:
                detail = ""
            return f"Metal attention+KV (experimental){detail}"
        return (
            "Metal attention not available; falling back to CPU "
            "(build on Darwin with Metal, or device/shader probe failed)"
        )


def resolve_attention_backend(
    profile: str,
    *,
    metal_linked: Optional[bool] = None,
    log: bool = True,
) -> AttentionBackend:
    """Pick an attention backend for a profile. Always returns a usable backend.

    ``hybrid`` / ``gpu-heavy`` request Metal; if not linked, log and return CPU.
    """
    linked = METAL_ATTN_LINKED if metal_linked is None else bool(metal_linked)
    cpu = CpuAttentionBackend()
    p = (profile or "cpu-only").strip().lower().replace("_", "-")
    if p in ("hybrid", "gpu-heavy"):
        metal = MetalAttentionBackend(linked=linked)
        if metal.available:
            if log:
                logger.info("Attention backend: %s (profile=%s)", metal.describe(), p)
            return metal
        msg = (
            f"profile={p}: Metal attn not available; falling back to CPU. "
            "Experts remain on CPU+SSD+pin. No tok/s claim for hybrid until "
            "a dated bake-off wins."
        )
        if log:
            logger.warning(msg)
            print(f"[tinygiant] {msg}")
        return cpu

    if log:
        logger.info("Attention backend: %s (profile=%s)", cpu.describe(), p)
    return cpu


def metal_attn_status() -> dict:
    """Introspection helper for tests / doctor."""
    status = "n/a"
    try:
        from ._lib import load_tinygiant_lib
        import ctypes

        lib = load_tinygiant_lib()
        if lib and hasattr(lib, "tg_metal_status"):
            lib.tg_metal_status.restype = ctypes.c_char_p
            raw = lib.tg_metal_status()
            status = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception as exc:
        status = f"probe error: {exc}"
    return {
        "metal_attn_linked": METAL_ATTN_LINKED,
        "native_status": status,
        "todo": (
            "Bake-off hybrid vs cpu-only on same Mac/model/prompts before "
            "flipping HYBRID_ATTN_READY; optional: Q4 matvecs on Metal to "
            "avoid FP16 weight expand; Slice B (experts-on-GPU) still held."
        ),
    }
