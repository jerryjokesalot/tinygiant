"""Tests for dynamic context window allocation and server context limit enforcement.

Zero hardcoded paths:
  Model discovery uses:
    1. TINYGIANT_MODEL env var (direct path to .gguf)
    2. TINYGIANT_MODELS_DIR env var (directory containing .gguf)
    3. Sibling models directory relative to workspace root (e.g. repo.parent / "models")
"""

import json
import os
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path


def discover_model_and_cache():
    """Discover GGUF model and its NWS expert cache without hardcoded paths."""
    explicit_model = os.environ.get("TINYGIANT_MODEL")
    if explicit_model and Path(explicit_model).is_file():
        m = Path(explicit_model)
        c = m.parent / "cache"
        return m, (c if c.is_dir() else None)

    search_dirs = []
    models_dir = os.environ.get("TINYGIANT_MODELS_DIR")
    if models_dir and Path(models_dir).is_dir():
        search_dirs.append(Path(models_dir))

    # Repo-relative discovery: sibling models/ folder
    repo_root = Path(__file__).resolve().parent.parent
    sibling_models = repo_root.parent / "models"
    if sibling_models.is_dir():
        search_dirs.append(sibling_models)

    for d in search_dirs:
        for f in sorted(d.glob("*.gguf")):
            c = d / "cache"
            return f, (c if c.is_dir() else None)

    return None, None


MODEL_PATH, CACHE_DIR = discover_model_and_cache()


class TestServerContextEnforcement(unittest.TestCase):
    """Test HTTP API server bounds checking on context length."""

    def setUp(self):
        import tinygiant.server as srv_mod

        # Minimal stub engine with configurable max_context
        class MockEngine:
            max_context = 512

            def reset_kv(self):
                pass

            def generate_stream(self, tokens, n_tokens=128, **kw):
                yield 42

        self.srv_mod = srv_mod
        self.orig_engine = srv_mod._engine
        self.orig_tokenizer = srv_mod._tokenizer
        self.orig_stop = srv_mod.STOP_TOKENS

        srv_mod._engine = MockEngine()
        # Mock tokenizer that returns 10 tokens for any prompt
        srv_mod._tokenizer = type("TokenizerStub", (), {
            "tokenize": staticmethod(lambda text, add_bos=True: [1] * 10),
            "detokenize": staticmethod(lambda ids: b"test response"),
            "token_eos": staticmethod(lambda: 2),
        })()
        srv_mod.STOP_TOKENS = {2}

        self.server = HTTPServer(("127.0.0.1", 0), srv_mod.Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.srv_mod._engine = self.orig_engine
        self.srv_mod._tokenizer = self.orig_tokenizer
        self.srv_mod.STOP_TOKENS = self.orig_stop

    def _post(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, data

    def test_chat_within_limit_succeeds(self):
        # 10 prompt tokens + 100 max_tokens = 110 <= 512
        status, data = self._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        })
        self.assertEqual(status, 200)
        self.assertIn("choices", data)

    def test_chat_exceeding_limit_returns_400(self):
        # 10 prompt tokens + 505 max_tokens = 515 > 512
        status, data = self._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 505,
        })
        self.assertEqual(status, 400)
        self.assertIn("error", data)
        self.assertEqual(data["error"].get("code"), "context_length_exceeded")
        self.assertIn("512", data["error"].get("message", ""))

    def test_paperclip_exceeding_limit_returns_400(self):
        # 10 prompt tokens + 600 max_tokens = 610 > 512
        status, data = self._post("/v1/paperclip", {
            "prompt": "write a long story",
            "max_tokens": 600,
        })
        self.assertEqual(status, 400)
        self.assertEqual(data.get("status"), "error")
        self.assertIn("context window is 512", data.get("result", ""))


class TestModelContextDynamicAllocation(unittest.TestCase):
    """Test model-driven context window detection and allocation."""

    @unittest.skipIf(MODEL_PATH is None, "No GGUF model found in environment or search paths")
    def test_detect_native_context_length_from_gguf(self):
        from gguf import GGUFReader
        reader = GGUFReader(str(MODEL_PATH))
        native_ctx = None
        for key in reader.fields:
            if "context_length" in key:
                val = reader.fields[key]
                raw = val.parts[val.data[0]]
                native_ctx = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
                break

        self.assertIsNotNone(native_ctx, "GGUF model should define context_length")
        self.assertGreater(native_ctx, 0)
        # Should NOT be arbitrarily hardcoded
        print(f"\n[Test] Dynamically detected model native context length: {native_ctx}")

    @unittest.skipIf(MODEL_PATH is None or CACHE_DIR is None,
                     "Model or NWS cache directory not found")
    def test_engine_allocates_requested_context_window(self):
        from tinygiant.engine import NWSEngine
        from tinygiant._constants import N_LAYERS, N_KV_HEADS, HEAD_DIM

        test_context = 2048
        engine = NWSEngine(str(MODEL_PATH), str(CACHE_DIR), max_context=test_context)

        # 1. Context window field
        self.assertEqual(engine.max_context, test_context)
        self.assertEqual(engine.kv_max, test_context)

        # 2. RoPE table sized to test_context
        self.assertEqual(engine.rope_cos.shape[0], test_context)
        self.assertEqual(engine.rope_sin.shape[0], test_context)

        # 3. KV cache buffers sized to test_context
        for i in range(N_LAYERS):
            if engine.kv_k[i] is not None:
                self.assertEqual(engine.kv_k[i].shape, (N_KV_HEADS, test_context, HEAD_DIM))
                self.assertEqual(engine.kv_v[i].shape, (N_KV_HEADS, test_context, HEAD_DIM))

    @unittest.skipIf(MODEL_PATH is None or CACHE_DIR is None,
                     "Model or NWS cache directory not found")
    def test_engine_rejects_exceeding_native_model_context(self):
        from tinygiant.engine import NWSEngine
        # Discover model's native context length first
        from gguf import GGUFReader
        reader = GGUFReader(str(MODEL_PATH))
        native_ctx = None
        for key in reader.fields:
            if "context_length" in key:
                val = reader.fields[key]
                raw = val.parts[val.data[0]]
                native_ctx = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
                break

        if native_ctx is not None:
            with self.assertRaises(ValueError) as ctx:
                NWSEngine(str(MODEL_PATH), str(CACHE_DIR), max_context=native_ctx + 1)
            self.assertIn("exceeds model supported context length", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
