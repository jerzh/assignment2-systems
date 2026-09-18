from __future__ import annotations

import argparse
import gc
import logging
import timeit

import numpy as np
import torch

from cs336_basics.model import scaled_dot_product_attention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")

    # ---- model architecture ----
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # ---- benchmark parameters ----
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measurement-steps", type=int, default=100)
    p.add_argument("--use-compiled", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def run_test(args: argparse.Namespace, context_length: int, d_model: int):
    logging.info(f"context length: {context_length}  d_model: {d_model}")
    this_size = (args.batch_size, context_length, d_model)
    Q = torch.randn(this_size, device=args.device, requires_grad=True)
    K = torch.randn(this_size, device=args.device, requires_grad=True)
    V = torch.randn(this_size, device=args.device, requires_grad=True)

    if args.use_compiled:
        attention = torch.compile(scaled_dot_product_attention)
    else:
        attention = scaled_dot_product_attention

    def forward():
        with torch.no_grad():
            attention(Q, K, V)
        if args.device == "cuda":
            torch.cuda.synchronize()

    def test_memory():
        out = attention(Q, K, V)
        return torch.cuda.memory_allocated()

    def backward():
        Q.grad = K.grad = V.grad = None
        out = attention(Q, K, V)
        out.sum().backward()
        if args.device == "cuda":
            torch.cuda.synchronize()

    # forward
    timeit.timeit(forward, number=args.warmup_steps, globals=globals())
    time_forward = timeit.timeit(forward, number=args.measurement_steps, globals=globals())
    logging.info(f"forward time ({args.measurement_steps} iterations): {time_forward}")

    # test memory
    current_allocated = test_memory()
    logging.info(f"memory allocated for graph: {current_allocated}")

    # backward
    timeit.timeit(backward, number=args.warmup_steps, globals=globals())
    time_backward = timeit.timeit(backward, number=args.measurement_steps, globals=globals())
    logging.info(f"backward time ({args.measurement_steps} iterations): {time_backward}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)

    for d_model in [16, 32, 64, 128]:
        for context_length in [256, 1024, 4096, 8192, 16384]:
            try:
                run_test(args, context_length, d_model)
            except torch.cuda.OutOfMemoryError as e:
                logging.info("CUDA OOM caught! Cleaning up memory...")
                del e
                gc.collect()
                torch.cuda.empty_cache()
