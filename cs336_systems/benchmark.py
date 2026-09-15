from __future__ import annotations

import argparse
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
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)   # ~ (8/3) * d_model, rounded to multiple of 64
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # ---- optimizer (AdamW) ----
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # ---- benchmark parameters ----
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measurement-steps", type=int, default=10)
    p.add_argument("--mode", choices=["forward", "backward", "optimizer"], default="forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
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

    if args.mode == "forward":
        def step():
            model(inputs)
            if args.device == "cuda":
                torch.cuda.synchronize()
    elif args.mode == "backward":
        def step():
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
    else:
        def step():
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()

    timeit.timeit("step()", number=args.warmup_steps, globals=globals())
    times = timeit.repeat("step()", number=1, repeat=args.measurement_steps, globals=globals())
    logging.info(f"mode: {args.mode}")
    logging.info("times:")
    for t in times:
        logging.info(t)
