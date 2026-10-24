"""Compute / device profile selection for TinyGiant.

Profiles:
  - cpu-only: today's path (default until hybrid Metal attn wins a bake-off)
  - hybrid:   attention+KV on Metal; MoE experts stay CPU+SSD+pin
  - gpu-heavy: attention + hot experts on GPU (never auto on ≤16 GiB)

Until Metal attention kernels are linked AND proven faster than CPU TinyGiant,
``selected`` stays ``cpu-only`` even when Metal is present. The probe may still
``recommend`` ``hybrid`` so setup UX can show intent. Fail-open always.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .probe import PROFILES, HardwareFacts, probe_hardware

PROFILE_FILENAME = "profile.json"

# Hybrid attention must be proven before auto-selecting hybrid.
# Flip this (or pass hybrid_attn_ready=True) only after a dated bake-off wins.
HYBRID_ATTN_READY = False

# Never auto-pick gpu-heavy at or below this unified/system RAM.
GPU_HEAVY_MIN_RAM_GIB = 24.0
# Soft floor where hybrid is even *recommended* (not selected yet).
HYBRID_MIN_RAM_GIB = 12.0


@dataclass
class ProfileDecision:
    selected: str
    recommended: str
    reason: str
    override: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "recommended": self.recommended,
            "reason": self.reason,
            "override": self.override,
        }


def _normalize_profile(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    n = str(name).strip().lower().replace("_", "-")
    if n in PROFILES:
        return n
    return None


def recommend_profile(
    facts: HardwareFacts,
    *,
    hybrid_attn_ready: bool = HYBRID_ATTN_READY,
) -> tuple[str, str]:
    """Return (recommended, reason) from hardware facts alone (no override)."""
    ram = float(facts.ram_gib or 0.0)

    # Discrete GPU with headroom → may recommend gpu-heavy
    if (facts.cuda_available or facts.rocm_available) and ram >= GPU_HEAVY_MIN_RAM_GIB:
        return (
            "gpu-heavy",
            "CUDA/ROCm present and RAM≥24 GiB; gpu-heavy recommended "
            "(not yet implemented — treat as future)",
        )

    # Apple Metal path
    if facts.metal_available and (ram >= HYBRID_MIN_RAM_GIB or ram <= 0):
        if ram > 0 and ram <= 16.0:
            # Explicit honesty for Jerry's 16 GiB class
            if hybrid_attn_ready:
                return (
                    "hybrid",
                    "Metal present, RAM≤16 GiB: hybrid (attn+KV Metal; experts CPU) "
                    "recommended once kernels proven",
                )
            return (
                "hybrid",
                "Metal present on ≤16 GiB machine: recommend hybrid when Metal "
                "attn ships; selected stays cpu-only until then",
            )
        if hybrid_attn_ready:
            return "hybrid", "Metal present and RAM≥12 GiB; hybrid recommended"
        return (
            "hybrid",
            "Metal present: recommend hybrid when Metal attn ships; "
            "selected stays cpu-only until proven",
        )

    if ram > 0 and ram < HYBRID_MIN_RAM_GIB:
        return "cpu-only", f"RAM {ram:.1f} GiB < {HYBRID_MIN_RAM_GIB:.0f} GiB; cpu-only"

    return "cpu-only", "No usable GPU detected; cpu-only"


def select_profile(
    facts: HardwareFacts,
    *,
    override: Optional[str] = None,
    hybrid_attn_ready: bool = HYBRID_ATTN_READY,
) -> ProfileDecision:
    """Choose selected profile. Override wins (validated); else cpu-only until ready.

    Rules:
      - Never *auto*-select gpu-heavy when RAM ≤ 16 GiB.
      - Never *auto*-select hybrid until hybrid_attn_ready.
      - Override may force hybrid/gpu-heavy (CLI); runtime must fail-open to CPU
        if the backend is missing.
    """
    recommended, rec_reason = recommend_profile(
        facts, hybrid_attn_ready=hybrid_attn_ready
    )
    ov = _normalize_profile(override)

    if ov:
        # Still warn/annotate unsafe overrides but honor them for experimentation.
        reason = f"CLI/config override → {ov} (recommended was {recommended}: {rec_reason})"
        if ov == "gpu-heavy" and facts.ram_gib and facts.ram_gib <= 16.0:
            reason += "; WARNING: gpu-heavy on ≤16 GiB is unsupported / thrash risk"
        return ProfileDecision(
            selected=ov, recommended=recommended, reason=reason, override=ov
        )

    # Auto path: selected is cpu-only until hybrid is ready AND recommended hybrid.
    if hybrid_attn_ready and recommended == "hybrid":
        # Still never auto gpu-heavy on ≤16
        return ProfileDecision(
            selected="hybrid",
            recommended=recommended,
            reason=rec_reason + "; hybrid_attn_ready → selected hybrid",
        )

    if recommended == "gpu-heavy":
        # Auto gpu-heavy only when RAM clearly above 16 and flag ready (future).
        # Today: never auto — fall to cpu-only.
        return ProfileDecision(
            selected="cpu-only",
            recommended=recommended,
            reason=rec_reason + "; auto gpu-heavy deferred → selected cpu-only",
        )

    return ProfileDecision(
        selected="cpu-only",
        recommended=recommended,
        reason=rec_reason + "; selected cpu-only (default until hybrid wins bake-off)",
    )


def write_profile_json(
    cache_dir: Union[str, Path],
    facts: HardwareFacts,
    decision: ProfileDecision,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``profile.json`` next to the expert cache. Returns path written."""
    cache = Path(os.path.expanduser(str(cache_dir)))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / PROFILE_FILENAME
    payload: dict[str, Any] = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": decision.selected,
        "recommended": decision.recommended,
        "reason": decision.reason,
        "override": decision.override,
        "hybrid_attn_ready": HYBRID_ATTN_READY,
        "probe": facts.to_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def load_profile_json(cache_dir: Union[str, Path]) -> Optional[dict]:
    """Load cache ``profile.json`` or return None. Never raises."""
    path = Path(os.path.expanduser(str(cache_dir))) / PROFILE_FILENAME
    try:
        if not path.is_file():
            return None
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def resolve_runtime_profile(
    cache_dir: Union[str, Path],
    *,
    cli_profile: Optional[str] = None,
) -> ProfileDecision:
    """Resolve profile for tinygiant / tinygiant-server.

    Priority: CLI ``--profile`` > cache profile.json selected > probe fresh → cpu-only.
    """
    ov = _normalize_profile(cli_profile)
    cached = load_profile_json(cache_dir)
    if ov:
        # Re-probe lightly for reason text; keep cheap (no SSD).
        facts = probe_hardware(cache_dir=str(cache_dir), include_ssd=False)
        return select_profile(facts, override=ov)

    if cached and cached.get("profile"):
        selected = _normalize_profile(cached.get("profile")) or "cpu-only"
        recommended = _normalize_profile(cached.get("recommended")) or selected
        return ProfileDecision(
            selected=selected,
            recommended=recommended,
            reason=str(cached.get("reason") or "from cache profile.json"),
            override=None,
        )

    facts = probe_hardware(cache_dir=str(cache_dir), include_ssd=False)
    return select_profile(facts)
