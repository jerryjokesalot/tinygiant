import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _tokenize(model_path, text):
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=32, n_gpu_layers=0,
                vocab_only=True, verbose=False)
    tokens = llm.tokenize(text.encode(), add_bos=True)
    return tokens, llm


def _detokenize(llm, tokens):
    return llm.detokenize(tokens).decode("utf-8", errors="replace")


def cmd_generate():
    from ._constants import N_LAYERS
    from .engine import NWSEngine

    parser = argparse.ArgumentParser(
        prog="tinygiant",
        description="TinyGiant — MoE inference on consumer hardware")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--cache", required=True, help="Path to NWS expert cache")
    parser.add_argument("--pin", type=int, default=48, help="Pin top-N experts/layer (default: 48)")
    parser.add_argument("--calibrate", type=int, default=10, help="Calibration tokens (default: 10)")
    parser.add_argument("--tokens", type=int, default=128, help="Tokens to generate (default: 128)")
    parser.add_argument("--prompt", type=str,
                        default="The key insight about mixture-of-experts models is that",
                        help="Prompt text")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    cache_dir = os.path.expanduser(args.cache)

    # Validate cache
    with open(os.path.join(cache_dir, "index.json")) as f:
        idx = json.load(f)
    cached_layers = sorted(int(k) for k in idx["layers"].keys())
    missing = [i for i in range(N_LAYERS) if i not in cached_layers]
    if missing:
        print(f"ERROR: Missing layers in cache: {missing}")
        print("Run 'tinygiant-relayout' first.")
        sys.exit(1)

    # Tokenize
    print(f"Prompt: {args.prompt!r}")
    tokens, llm = _tokenize(model_path, args.prompt)
    print(f"Tokens: {tokens} ({len(tokens)} tokens)")

    # Build engine
    engine = NWSEngine(model_path, cache_dir)

    # Pin experts
    engine.pin_experts(args.pin, calibrate_tokens=args.calibrate, prompt_tokens=tokens)

    # Generate
    np.random.seed(args.seed)
    generated = engine.generate(tokens, n_tokens=args.tokens,
                                temperature=args.temperature, top_p=args.top_p)

    # Decode output
    full_tokens = list(tokens) + generated
    output_text = _detokenize(llm, full_tokens)
    gen_text = _detokenize(llm, generated)
    print(f"\nFull output: {output_text}")
    print(f"Generated: {gen_text}")


def cmd_relayout():
    from .relayout import main as relayout_main
    relayout_main()


def cmd_setup():
    parser = argparse.ArgumentParser(
        prog="tinygiant-setup",
        description="TinyGiant setup — prepare model for inference")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--cache", required=True, help="Output cache directory")
    parser.add_argument("--format", choices=["f16", "q4"], default="q4",
                        help="Cache format (default: q4)")
    parser.add_argument("--verify", action="store_true", help="Verify output")
    args = parser.parse_args()

    print("TinyGiant Setup")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Cache: {args.cache}")
    print(f"Format: {args.format}")
    print()

    # Run relayout
    sys.argv = [
        "tinygiant-relayout",
        args.model,
        args.cache,
        "--format", args.format,
    ]
    if args.verify:
        sys.argv.append("--verify")
    cmd_relayout()

    print("\nSetup complete! Run inference with:")
    print(f"  tinygiant --model {args.model} --cache {args.cache}")
