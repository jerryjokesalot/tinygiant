"""Unit tests for profile selection rules — no GPU required."""

import json
from pathlib import Path

from tinygiant.probe import HardwareFacts
from tinygiant.profile import (
    HYBRID_ATTN_READY,
    load_profile_json,
    recommend_profile,
    resolve_runtime_profile,
    select_profile,
    write_profile_json,
)


def _m1_16() -> HardwareFacts:
    return HardwareFacts(
        os="Darwin",
        arch="arm64",
        ram_gib=16.0,
        metal_available=True,
        metal_detail="Apple M1",
    )


def _linux_cpu_8() -> HardwareFacts:
    return HardwareFacts(os="Linux", arch="x86_64", ram_gib=8.0)


def _cuda_32() -> HardwareFacts:
    return HardwareFacts(
        os="Linux",
        arch="x86_64",
        ram_gib=32.0,
        cuda_available=True,
        cuda_detail="RTX 4090",
    )


def test_m1_16_recommends_hybrid_selects_cpu_only():
    # Until hybrid attn is proven, selected must stay cpu-only.
    assert HYBRID_ATTN_READY is False
    d = select_profile(_m1_16())
    assert d.recommended == "hybrid"
    assert d.selected == "cpu-only"
    assert "cpu-only" in d.reason


def test_never_auto_gpu_heavy_on_16gib():
    facts = HardwareFacts(
        os="Darwin",
        arch="arm64",
        ram_gib=16.0,
        metal_available=True,
        cuda_available=True,  # even if somehow true
    )
    d = select_profile(facts)
    assert d.selected != "gpu-heavy"
    assert d.selected == "cpu-only"


def test_override_hybrid_honored():
    d = select_profile(_m1_16(), override="hybrid")
    assert d.selected == "hybrid"
    assert d.override == "hybrid"
    assert d.recommended == "hybrid"


def test_override_gpu_heavy_warns_on_16():
    d = select_profile(_m1_16(), override="gpu-heavy")
    assert d.selected == "gpu-heavy"
    assert "WARNING" in d.reason


def test_low_ram_cpu_only():
    rec, reason = recommend_profile(_linux_cpu_8())
    assert rec == "cpu-only"
    d = select_profile(_linux_cpu_8())
    assert d.selected == "cpu-only"


def test_cuda_32_recommends_gpu_heavy_but_auto_cpu():
    d = select_profile(_cuda_32())
    assert d.recommended == "gpu-heavy"
    assert d.selected == "cpu-only"  # auto deferred today


def test_hybrid_ready_selects_hybrid(monkeypatch):
    import tinygiant.profile as profile_mod

    monkeypatch.setattr(profile_mod, "HYBRID_ATTN_READY", True)
    d = select_profile(_m1_16(), hybrid_attn_ready=True)
    assert d.selected == "hybrid"
    assert d.recommended == "hybrid"


def test_write_and_load_profile_json(tmp_path: Path):
    facts = _m1_16()
    decision = select_profile(facts)
    path = write_profile_json(tmp_path, facts, decision)
    assert path.name == "profile.json"
    loaded = load_profile_json(tmp_path)
    assert loaded is not None
    assert loaded["profile"] == "cpu-only"
    assert loaded["recommended"] == "hybrid"
    assert loaded["probe"]["metal_available"] is True
    # Round-trip JSON is valid
    raw = json.loads(path.read_text())
    assert raw["version"] == 1


def test_resolve_runtime_prefers_cli(tmp_path: Path):
    facts = _m1_16()
    write_profile_json(tmp_path, facts, select_profile(facts))
    d = resolve_runtime_profile(tmp_path, cli_profile="hybrid")
    assert d.selected == "hybrid"
    assert d.override == "hybrid"


def test_resolve_runtime_reads_cache(tmp_path: Path):
    facts = _m1_16()
    write_profile_json(tmp_path, facts, select_profile(facts))
    d = resolve_runtime_profile(tmp_path, cli_profile=None)
    assert d.selected == "cpu-only"
