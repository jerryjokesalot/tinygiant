"""Attention backend tests — Metal optional; CPU fail-open always."""

from tinygiant.attention import (
    METAL_ATTN_LINKED,
    CpuAttentionBackend,
    MetalAttentionBackend,
    metal_attn_status,
    resolve_attention_backend,
)


def test_cpu_always_available():
    cpu = CpuAttentionBackend()
    assert cpu.available is True
    assert "CPU" in cpu.describe()


def test_metal_backend_respects_linked_flag():
    stub = MetalAttentionBackend(linked=False)
    assert stub.available is False
    assert "not available" in stub.describe() or "falling back" in stub.describe()

    linked = MetalAttentionBackend(linked=True)
    assert linked.available is True
    assert "Metal" in linked.describe()


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


def test_metal_attn_status_shape():
    st = metal_attn_status()
    assert "metal_attn_linked" in st
    assert st["metal_attn_linked"] is METAL_ATTN_LINKED
    assert "todo" in st
    assert "Bake-off" in st["todo"] or "bake-off" in st["todo"].lower()


def test_metal_linked_is_bool():
    assert isinstance(METAL_ATTN_LINKED, bool)
