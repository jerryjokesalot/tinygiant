"""Unit tests for hardware probe — no GPU required."""

from tinygiant.probe import HardwareFacts, probe_hardware


def test_probe_inject_facts():
    facts = probe_hardware(
        facts={
            "os": "Darwin",
            "arch": "arm64",
            "ram_gib": 16.0,
            "metal_available": True,
            "metal_detail": "Chipset Model: Apple M1",
            "cuda_available": False,
            "ssd_class": "fast",
            "ssd_read_gbs": 2.5,
        }
    )
    assert isinstance(facts, HardwareFacts)
    assert facts.os == "Darwin"
    assert facts.arch == "arm64"
    assert facts.ram_gib == 16.0
    assert facts.metal_available is True
    assert facts.cuda_available is False
    assert facts.ssd_class == "fast"
    d = facts.to_dict()
    assert d["metal_available"] is True
    assert "notes" in d


def test_probe_live_does_not_raise():
    # Live probe on whatever CI runner; must never raise.
    facts = probe_hardware(include_ssd=False)
    assert facts.os
    assert facts.arch
    assert facts.ram_gib >= 0.0
