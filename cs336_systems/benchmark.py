from __future__ import annotations

import argparse
import logging
import math
import timeit

import torch
import torch.cuda.nvtx as nvtx

from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, softmax
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measurement-steps", type=int, default=10)
    p.add_argument("--mode", choices=["forward", "backward", "optimizer"], default="forward")
    p.add_argument("--no-sync", action="store_false", dest="sync")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("compute attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("attention softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("compute attention output"):
        attention_output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return attention_output


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


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
            with torch.autograd.profiler.emit_nvtx():
                with nvtx.range("forward"), torch.no_grad():
                    model(inputs)
            if args.device == "cuda" and args.sync:
                torch.cuda.synchronize()
    elif args.mode == "backward":
        def step():
            with torch.autograd.profiler.emit_nvtx():
                with nvtx.range("forward"):
                    logits = model(inputs)
                with nvtx.range("backward"):
                    loss = cross_entropy(logits, targets)
                    loss.backward()
            if args.device == "cuda" and args.sync:
                torch.cuda.synchronize()
    else:
        def step():
            with torch.autograd.profiler.emit_nvtx():
                optimizer.zero_grad()
                with nvtx.range("forward"):
                    logits = model(inputs)
                with nvtx.range("backward"):
                    loss = cross_entropy(logits, targets)
                    loss.backward()
                with nvtx.range("optimizer"):
                    optimizer.step()
            if args.device == "cuda" and args.sync:
                torch.cuda.synchronize()

    timeit.timeit("step()", number=args.warmup_steps, globals=globals())
    with nvtx.range("measure"):
        times = timeit.repeat("step()", number=1, repeat=args.measurement_steps, globals=globals())
    logging.info(f"mode: {args.mode}")
    logging.info("times:")
    for t in times:
        logging.info(t)
