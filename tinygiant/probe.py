"""Hardware probe for TinyGiant setup profiles.

Fail-open: any detection error yields conservative defaults (no Metal/CUDA,
unknown SSD). Pure selection logic lives in ``profile.py`` so unit tests can
inject a synthetic probe without touching sysctl / nvidia-smi.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


PROFILES = ("cpu-only", "hybrid", "gpu-heavy")


@dataclass
class HardwareFacts:
    """Facts gathered about the machine; all fields are optional/safe defaults."""

    os: str = "unknown"
    arch: str = "unknown"
    ram_gib: float = 0.0
    metal_available: bool = False
    metal_detail: str = ""
    cuda_available: bool = False
    cuda_detail: str = ""
    rocm_available: bool = False
    ssd_class: str = "unknown"  # fast | medium | slow | unknown
    ssd_read_gbs: Optional[float] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""


def probe_ram_gib() -> float:
    """Return total physical RAM in GiB, or 0.0 if unknown."""
    system = platform.system()
    try:
        if system == "Darwin":
            rc, out = _run(["sysctl", "-n", "hw.memsize"])
            if rc == 0 and out.strip().isdigit():
                return int(out.strip()) / (1024 ** 3)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # kB
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        elif system == "Windows":
            rc, out = _run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                timeout=8.0,
            )
            if rc == 0:
                nums = re.findall(r"\d+", out)
                if nums:
                    return int(nums[-1]) / (1024 ** 3)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def probe_metal() -> tuple[bool, str]:
    """Best-effort Apple Metal detection. Never raises."""
    if platform.system() != "Darwin":
        return False, ""
    # Prefer system_profiler (no compile needed).
    rc, out = _run(
        ["system_profiler", "SPDisplaysDataType"],
        timeout=20.0,
    )
    if rc != 0 or not out:
        # Fallback: presence of Metal.framework
        if os.path.isdir("/System/Library/Frameworks/Metal.framework"):
            return True, "Metal.framework present (profiler unavailable)"
        return False, ""
    metalish = bool(
        re.search(r"Metal|Chipset Model|Apple M\d", out, re.IGNORECASE)
    )
    # Extract a short detail line if possible
    detail = ""
    for key in ("Chipset Model", "Total Number of Cores", "Vendor"):
        m = re.search(rf"{key}:\s*(.+)", out)
        if m:
            detail = (detail + "; " if detail else "") + f"{key}: {m.group(1).strip()}"
    if not detail:
        detail = "system_profiler SPDisplaysDataType OK"
    return metalish, detail[:240]


def probe_cuda() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, ""
    rc, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=8.0)
    if rc != 0:
        return False, "nvidia-smi present but query failed"
    line = out.strip().splitlines()[0] if out.strip() else "nvidia-smi OK"
    return True, line[:240]


def probe_rocm() -> bool:
    return shutil.which("rocminfo") is not None


def probe_ssd_class(cache_dir: Optional[str] = None, sample_bytes: int = 16 * 1024 * 1024) -> tuple[str, Optional[float]]:
    """Crude sequential-read class against cache_dir (or /tmp). Optional.

    Returns (class, gib_per_s). Skips quickly if the path is missing or
    sample would be expensive; never raises.
    """
    root = cache_dir or os.environ.get("TMPDIR") or "/tmp"
    try:
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, ".tinygiant_ssd_probe.bin")
        # Write once
        chunk = b"\0" * (1024 * 1024)
        with open(path, "wb") as f:
            written = 0
            while written < sample_bytes:
                f.write(chunk)
                written += len(chunk)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        # Sequential read
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        elapsed = max(time.perf_counter() - t0, 1e-6)
        gbs = (sample_bytes / (1024 ** 3)) / elapsed
        try:
            os.remove(path)
        except OSError:
            pass
        if gbs >= 1.5:
            return "fast", round(gbs, 2)
        if gbs >= 0.5:
            return "medium", round(gbs, 2)
        return "slow", round(gbs, 2)
    except OSError:
        return "unknown", None


def probe_hardware(
    *,
    cache_dir: Optional[str] = None,
    include_ssd: bool = False,
    facts: Optional[Mapping[str, Any]] = None,
) -> HardwareFacts:
    """Gather hardware facts. Pass ``facts`` to inject values for tests."""
    if facts is not None:
        base = HardwareFacts()
        for k, v in facts.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    h = HardwareFacts(
        os=platform.system() or "unknown",
        arch=platform.machine() or "unknown",
        ram_gib=round(probe_ram_gib(), 2),
    )
    metal_ok, metal_detail = probe_metal()
    h.metal_available = metal_ok
    h.metal_detail = metal_detail
    cuda_ok, cuda_detail = probe_cuda()
    h.cuda_available = cuda_ok
    h.cuda_detail = cuda_detail
    h.rocm_available = probe_rocm()
    if include_ssd:
        h.ssd_class, h.ssd_read_gbs = probe_ssd_class(cache_dir)
    if h.ram_gib <= 0:
        h.notes.append("RAM unknown; treating conservatively")
    return h
