"""OpenAI-compatible API server for TinyGiant.

Implements /v1/chat/completions with SSE streaming.
Zero external dependencies — uses only Python stdlib.

Usage:
    tinygiant-server --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
                     --cache /path/to/nws_cache \
                     --port 8000
"""

import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

from ._constants import N_LAYERS
from .engine import NWSEngine

# Qwen3 chat template tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Stop tokens for Qwen3
STOP_TOKENS = set()

_engine = None
_tokenizer = None
_model_name = "tinygiant-qwen3-30b"


def _init_tokenizer(model_path):
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=32, n_gpu_layers=0,
                vocab_only=True, verbose=False)
    STOP_TOKENS.add(llm.token_eos())
    return llm


def _format_chat(messages):
    """Format messages into Qwen3 chat template."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}")
    parts.append(f"{IM_START}assistant\n")
    return "\n".join(parts)


def _tokenize(text):
    return _tokenizer.tokenize(text.encode(), add_bos=True)


def _detokenize(token_ids):
    return _tokenizer.detokenize(token_ids).decode("utf-8", errors="replace")


import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TAG_RE = re.compile(r"<\|im_(?:start|end)\|>")


def _clean_output(text):
    """Strip thinking blocks and chat template tags from model output."""
    text = _THINK_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    # Clean up residual role headers from spurious turns
    text = re.sub(r"\n*assistant\n*", "", text)
    return text.strip()


class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length > 0 else {}

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": _model_name,
                    "object": "model",
                    "owned_by": "tinygiant",
                }]
            })
        elif self.path == "/health":
            self._send_json(200, {"status": "ok", "model": _model_name})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat(self):
        body = self._read_body()

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 128)
        top_p = body.get("top_p", 0.9)

        if not messages:
            self._send_json(400, {"error": {"message": "messages is required"}})
            return

        prompt = _format_chat(messages)
        tokens = _tokenize(prompt)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        _engine.reset_kv()

        if stream:
            self._stream_response(tokens, request_id, created,
                                  temperature, max_tokens, top_p)
        else:
            self._full_response(tokens, request_id, created,
                                temperature, max_tokens, top_p)

    def _stream_response(self, tokens, request_id, created,
                         temperature, max_tokens, top_p):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Initial chunk with role
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

        finish_reason = "length"
        generated = []
        emitting = False

        for token_id in _engine.generate_stream(tokens, n_tokens=max_tokens,
                                                 temperature=temperature, top_p=top_p):
            if token_id in STOP_TOKENS:
                finish_reason = "stop"
                break

            generated.append(token_id)

            if not emitting:
                # Buffer until thinking is done
                buf = _detokenize(generated)
                if "</think>" in buf:
                    emitting = True
                    text = _clean_output(buf)
                    if text:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": _model_name,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                continue

            text = _detokenize([token_id])
            # Skip stray template tags
            if "<|im_" in text:
                continue

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": _model_name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # If model never entered thinking mode, flush buffered content
        if not emitting and generated:
            text = _clean_output(_detokenize(generated))
            if text:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_name,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

        # Final chunk
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        n = len(generated)
        print(f"[stream] {n} tokens, reason={finish_reason}")

    def _full_response(self, tokens, request_id, created,
                       temperature, max_tokens, top_p):
        generated = []
        finish_reason = "length"

        for token_id in _engine.generate_stream(tokens, n_tokens=max_tokens,
                                                 temperature=temperature, top_p=top_p):
            if token_id in STOP_TOKENS:
                finish_reason = "stop"
                break
            generated.append(token_id)

        raw_text = _detokenize(generated)
        text = _clean_output(raw_text)

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": _model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            },
        }

        self._send_json(200, response)
        print(f"[full] {len(generated)} tokens, reason={finish_reason}")


def cmd_serve():
    import argparse
    global _engine, _tokenizer, _model_name

    parser = argparse.ArgumentParser(
        prog="tinygiant-server",
        description="TinyGiant OpenAI-compatible API server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--cache", required=True, help="Path to NWS expert cache")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--pin", type=int, default=48, help="Pin experts/layer (default: 48)")
    parser.add_argument("--calibrate", type=int, default=10, help="Calibration tokens")
    parser.add_argument("--model-name", default="tinygiant-qwen3-30b",
                        help="Model name in API responses")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    cache_dir = os.path.expanduser(args.cache)
    _model_name = args.model_name

    # Validate cache
    with open(os.path.join(cache_dir, "index.json")) as f:
        idx = json.load(f)
    cached_layers = sorted(int(k) for k in idx["layers"].keys())
    missing = [i for i in range(N_LAYERS) if i not in cached_layers]
    if missing:
        print(f"ERROR: Missing layers in cache: {missing}")
        sys.exit(1)

    # Init tokenizer
    print("Loading tokenizer...", end="", flush=True)
    _tokenizer = _init_tokenizer(model_path)
    print(" done")

    # Init engine
    _engine = NWSEngine(model_path, cache_dir)

    # Calibrate and pin
    cal_prompt = "The key insight about mixture-of-experts models is that"
    cal_tokens = _tokenize(cal_prompt)
    _engine.pin_experts(args.pin, calibrate_tokens=args.calibrate,
                        prompt_tokens=cal_tokens)

    # Start server
    server = HTTPServer((args.host, args.port), Handler)
    print(f"\nTinyGiant API server running on http://{args.host}:{args.port}")
    print(f"  Model: {_model_name}")
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions")
    print(f"    GET  /v1/models")
    print(f"    GET  /health")
    print(f"\nTest with:")
    print(f'  curl http://localhost:{args.port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"messages": [{{"role": "user", "content": "Hello!"}}], "stream": true}}\'')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
