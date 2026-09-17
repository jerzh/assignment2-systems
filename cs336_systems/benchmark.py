from __future__ import annotations

import argparse
import contextlib
import logging
import timeit

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")

    # ---- model architecture ----
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--model-size", choices=["small", "medium", "large", "xl", "10B"],
                   default="small")
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # ---- optimizer (AdamW) ----
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # ---- benchmark parameters ----
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measurement-steps", type=int, default=10)
    p.add_argument("--mode", choices=["forward", "backward", "optimizer"], default="forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--profile-memory", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_size == "small":
        d_model = 768
        d_ff = 3072
        num_layers = 12
        num_heads = 12
    elif args.model_size == "medium":
        d_model = 1024
        d_ff = 4096
        num_layers = 24
        num_heads = 16
    elif args.model_size == "large":
        d_model = 1280
        d_ff = 5120
        num_layers = 36
        num_heads = 20
    elif args.model_size == "xl":
        d_model = 2560
        d_ff = 10240
        num_layers = 32
        num_heads = 32
    else:
        d_model = 4608
        d_ff = 12288
        num_layers = 50
        num_heads = 36

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
    ).to(device=args.device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    if args.mixed_precision:
        loop_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        loop_context = contextlib.nullcontext()
    if args.mode == "forward":
        def step():
            with loop_context, torch.no_grad():
                model(inputs)
            if args.device == "cuda":
                torch.cuda.synchronize()
    elif args.mode == "backward":
        def step():
            with loop_context:
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
    else:
        def step():
            with loop_context:
                optimizer.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()

    timeit.timeit("step()", number=args.warmup_steps, globals=globals())
    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000, stacks="python")
    times = timeit.repeat("step()", number=1, repeat=args.measurement_steps, globals=globals())
    if args.profile_memory:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    logging.info(f"mode: {args.mode}")
    logging.info("times:")
    for t in times:
        logging.info(t)
