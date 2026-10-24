"""Attention backend scaffolding tests — no Metal required."""

from tinygiant.attention import (
    METAL_ATTN_LINKED,
    CpuAttentionBackend,
    MetalAttentionBackend,
    metal_attn_status,
    resolve_attention_backend,
)


def test_metal_not_linked_by_default():
    assert METAL_ATTN_LINKED is False
    stub = MetalAttentionBackend()
    assert stub.available is False
    assert "not yet linked" in stub.describe()


def test_cpu_always_available():
    cpu = CpuAttentionBackend()
    assert cpu.available is True
    assert "CPU" in cpu.describe()


def test_hybrid_fail_opens_to_cpu(capsys):
    backend = resolve_attention_backend("hybrid", metal_linked=False, log=True)
    assert backend.name == "cpu"
    out = capsys.readouterr().out
    assert "falling back to CPU" in out


def test_hybrid_uses_metal_when_linked():
    backend = resolve_attention_backend("hybrid", metal_linked=True, log=False)
    assert backend.name == "metal"
    assert backend.available is True


def test_cpu_only_stays_cpu():
    backend = resolve_attention_backend("cpu-only", metal_linked=True, log=False)
    assert backend.name == "cpu"


def test_metal_attn_status_has_todo():
    st = metal_attn_status()
    assert st["metal_attn_linked"] is False
    assert "TODO" in st["todo"] or "Implement" in st["todo"]
