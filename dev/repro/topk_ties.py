#!/usr/bin/env python3
"""top_k_per_row_prefill keeps a run-dependent subset of pools tied at the k-th score.

No checkpoint and no server: one CUDA device in a container that has vLLM
importable.

    python topk_ties.py

Each row gets random scores plus two extra copies of its k-th largest score,
so the cut falls inside an exact tie. The same input then goes through the
kernel 200 times, and the distinct outputs are counted (each row's ids sorted,
so only the chosen set counts). One row alone comes back the same every time.
A launch with as many rows as a real prefill chunk comes back different on
almost every call.
"""
import torch

import vllm._custom_ops as ops

K = 512
CALLS = 200


def distinct_outputs(rows, cols, extra_ties):
    torch.manual_seed(0)
    scores = torch.randn(rows, cols, device="cuda")
    kth = scores.topk(K, dim=1).values[:, -1:]
    # Put the copies on entries below the cut, chosen at random per row.
    below = torch.rand(rows, cols, device="cuda") + (scores >= kth).float() * 2
    scores.scatter_(1, below.argsort(dim=1)[:, :extra_ties], kth.expand(-1, extra_ties))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    seen = set()
    for _ in range(CALLS):
        out = torch.full((rows, K), -1, dtype=torch.int32, device="cuda")
        ops.top_k_per_row_prefill(scores, starts, ends, out, rows, scores.stride(0), scores.stride(1), K)
        seen.add(out.sort(dim=1).values.cpu().numpy().tobytes())
    return len(seen)


for rows, cols in [(1, 1300), (1, 8000), (64, 1300), (16128, 1300), (16128, 8000)]:
    print(f"{rows:5d} rows x {cols} pools, 2 extra tied at the cut: "
          f"{distinct_outputs(rows, cols, 2)} distinct outputs in {CALLS} calls")
