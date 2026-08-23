"""Minimal tok/s benchmark — no GTK, no Box code, just the SDK.

If this hits ~7 tok/s on the user's machine but the Box app gets <3 tok/s
with the same model + settings, the slowdown is something *inside Box's
process* (state, threads, lock contention). If both numbers match, the
cause is system-level — not anything in Box's code.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import litert_lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .litertlm file")
    ap.add_argument("--cache", default="~/.cache/box/litert-lm")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--mtp", action="store_true", help="Enable speculative decoding")
    ap.add_argument(
        "--prompt",
        default="What is the capital of France? Reply in two sentences.",
    )
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-k", type=int, default=39)
    ap.add_argument("--top-p", type=float, default=0.97)
    args = ap.parse_args()

    cache = str(Path(args.cache).expanduser())
    print(f"Model: {args.model}")
    print(f"Cache: {cache}")
    print(f"Context: {args.ctx}    MTP/spec-decoding: {args.mtp}")

    t0 = time.monotonic()
    eng = litert_lm.Engine(
        args.model,
        backend=litert_lm.Backend.CPU,
        cache_dir=cache,
        max_num_tokens=args.ctx,
        enable_speculative_decoding=args.mtp or None,
    )
    eng.__enter__()
    print(f"Engine open: {time.monotonic() - t0:.2f}s")

    sampler = litert_lm.SamplerConfig(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )
    conv = eng.create_conversation(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful, concise assistant.",
                    }
                ],
            }
        ],
        sampler_config=sampler,
    )
    conv.__enter__()
    print(f"Conversation ready, sending prompt…")
    print(f"User: {args.prompt}\nAssistant: ", end="", flush=True)

    t_first = None
    t_start = time.monotonic()
    n_tok = 0
    full = []
    for chunk in conv.send_message_async(args.prompt):
        for item in chunk.get("content", []) or []:
            if item.get("type") == "text":
                t = item.get("text", "")
                if t:
                    if t_first is None:
                        t_first = time.monotonic()
                    print(t, end="", flush=True)
                    full.append(t)
                    n_tok += 1
    t_end = time.monotonic()
    print()
    if t_first is None:
        print("No tokens emitted.")
        return
    prefill = t_first - t_start
    decode = t_end - t_first
    print(
        f"\nPrefill: {prefill:.2f}s  ·  "
        f"Decode: {decode:.2f}s for {n_tok} chunks  ·  "
        f"{n_tok / decode:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
