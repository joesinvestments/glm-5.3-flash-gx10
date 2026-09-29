#!/usr/bin/env python3
"""KDA RecoverSSM align-mode commit with a block table filled only as far as
align mode allocates it.

Align mode allocates cdiv(num_computed + query_len, block_size) KDA blocks per
request. The runner's gathered block table keeps whatever an earlier request
left past that. A commit whose accepted tokens end exactly on a block boundary
used to store the request's state in column n // block_size, one past the
allocation, and the state was lost (2026-09-29).

One commit over rows that each end in a different place relative to a
boundary (block_size 16, up to 7 drafts):

  exact, all accepted      nc 24 + 8 of 8   = 32   (the bug)
  exact, all accepted, k=2 nc 29 + 3 of 3   = 32   (the bug)
  exact, partial           nc 26 + 6 of 8   = 32
  exact, partial, k=2      nc 30 + 2 of 3   = 32
  past a boundary          nc 28 + 8 of 8   = 36
  mid-block                nc 17 + 3 of 8   = 20
  from a boundary          nc 32 + 8 of 8   = 40

Each row's columns past cdiv(nc + q, 16) point at unallocated blocks, or at
the null block (0) in the second pass. Checks, bit for bit against the stock
fused_recurrent_kda:
  - the state after the last accepted token is in the allocated block holding
    that token, with the conv window at column 0
  - a row that reaches a boundary has the state after the boundary token in
    the boundary block
  - no other block changed
  - record_state_blocks points the running state column at that block

Fails on the old column math (n // block_size) and passes on the new one.

Needs a CUDA device and the glm53 image's vLLM, no weights. The module is
found as in _glm53_recoverssm_test.py. On a GB10 that is not serving:

    mkdir -p /tmp/rs && cp dev/patch-tests/_glm53_recoverssm_boundary_test.py \\
        experimental/fixes/recoverssm.py /tmp/rs/
    sudo docker run --rm --gpus all --ipc=host -v /tmp/rs:/rs -w / \\
        --entrypoint python3 <glm53 image> /rs/_glm53_recoverssm_boundary_test.py

It allocates about 250 MB of GPU memory.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMPostprocessMetadata,
)


def _load_module():
    name = "vllm.models.glm5next.common.recoverssm"
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    candidates = [
        os.environ.get("RECOVERSSM_PY"),
        here / "recoverssm.py",
        here.parent.parent / "experimental" / "fixes" / "recoverssm.py",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            print(f"recoverssm module: {path}")
            return module
    raise SystemExit("recoverssm.py not found; set RECOVERSSM_PY")


rs = _load_module()

dev = torch.device("cuda")
H, D = 16, 128  # per rank at TP=4
PROJ = H * D
CONV_DIM = 3 * PROJ
WIDTH = 4
HIST = WIDTH - 1
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16
K = 7
Q = K + 1
BS = 16  # mamba block size
W = 6  # block table width
NUM_LAYERS = 2
DS = is_conv_state_dim_first()

# (name, num_computed, query_len, num_sampled)
CASES = [
    ("exact, all accepted", 24, 8, 8),
    ("exact, all accepted, k=2", 29, 3, 3),
    ("exact, partial", 26, 8, 6),
    ("exact, partial, k=2", 30, 3, 2),
    ("past a boundary", 28, 8, 8),
    ("mid-block", 17, 8, 3),
    ("from a boundary", 32, 8, 8),
]
ROWS = len(CASES)
LINES = 1 + ROWS * W


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def block_id(row: int, col: int) -> int:
    return 1 + row * W + col


def conv_pool(lines: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw kv_cache[0], its (lines, dim, state_len) view)."""
    state_len = HIST + K
    if DS:
        raw = torch.zeros(lines, CONV_DIM, state_len, device=dev, dtype=DTYPE)
        return raw, raw
    raw = torch.zeros(lines, state_len, CONV_DIM, device=dev, dtype=DTYPE)
    return raw, raw.transpose(-1, -2)


def make_layer(g: torch.Generator):
    layer = SimpleNamespace()
    layer.A_log = torch.log(
        torch.rand(1, 1, H, 1, device=dev, generator=g) * 8 + 1
    ).float()
    layer.dt_bias = (torch.randn(PROJ, device=dev, generator=g) * 0.5).float()
    layer.kda_lower_bound = LOWER_BOUND
    conv_raw, layer.conv = conv_pool(LINES)
    layer.conv.copy_(torch.randn(layer.conv.shape, device=dev, generator=g).to(DTYPE))
    layer.rec = torch.randn(LINES, H, D, D, device=dev, generator=g) * 0.1
    layer.kv_cache = (conv_raw, layer.rec)
    layer.recoverssm_records = rs.allocate_records(ROWS, H, D, Q, DTYPE, dev)
    return layer


def run(unallocated_fill: str, seed: int) -> list[str]:
    g = torch.Generator(device=dev).manual_seed(seed)
    layers = [make_layer(g) for _ in range(NUM_LAYERS)]
    ctx = rs.Glm5NextRecoverSSMCommitContext.create(
        layers, spec_query_len=Q, max_num_reqs=ROWS
    )

    block_table = torch.zeros(ROWS, W, dtype=torch.int32)
    source = []
    for i, (_, nc, q, _) in enumerate(CASES):
        allocated = cdiv(nc + q, BS)
        assert allocated < W
        for c in range(W):
            if c < allocated or unallocated_fill == "stale":
                block_table[i, c] = block_id(i, c)
        # The verify reads the block the runner's pre-step copy moved the
        # state to: (seq_len - 1) // block_size.
        source.append(block_id(i, (nc + q - 1) // BS))
    block_table = block_table.to(dev)
    idx = torch.tensor(source, dtype=torch.int32, device=dev)
    qlens = [q for _, _, q, _ in CASES]
    qsl_cpu = torch.zeros(ROWS + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(torch.tensor(qlens), 0)
    qsl = qsl_cpu.to(dev)
    T = int(qsl_cpu[-1])
    num_computed = torch.tensor([nc for _, nc, _, _ in CASES], dtype=torch.int32, device=dev)
    num_sampled = torch.tensor([n for _, _, _, n in CASES], dtype=torch.int32, device=dev)

    # Stock reference: row i's state after token t goes to scratch slot
    # 1 + i * Q + t, starting from its checkpoint in slot 1 + i * Q.
    ref_cols = (
        1 + torch.arange(ROWS * Q, dtype=torch.int32, device=dev)
    ).view(ROWS, Q)
    ones = torch.ones(ROWS, dtype=torch.int32, device=dev)

    refs, before = [], []
    for layer in layers:
        qkv = [torch.randn(1, T, H, D, device=dev, generator=g).to(DTYPE) for _ in range(3)]
        g1 = torch.randn(1, T, H, D, device=dev, generator=g).to(DTYPE)
        beta = torch.randn(1, T, H, device=dev, generator=g).to(DTYPE)
        scratch = torch.zeros(1 + ROWS * Q, H, D, D, device=dev, dtype=torch.float32)
        scratch[ref_cols[:, 0].long()] = layer.rec[idx.long()]
        fused_recurrent_kda(
            q=qkv[0], k=qkv[1], v=qkv[2], g=g1, beta=beta, initial_state=scratch,
            use_qk_l2norm_in_kernel=True, cu_seqlens=qsl, ssm_state_indices=ref_cols,
            num_accepted_tokens=ones, sigmoid_beta=True, a_log=layer.A_log,
            g_bias=layer.dt_bias, compute_gate=True, lower_bound=LOWER_BOUND,
        )
        refs.append(scratch)
        rs.kda_recoverssm_verify(
            q=qkv[0], k=qkv[1], v=qkv[2], g=g1, beta=beta, a_log=layer.A_log,
            g_bias=layer.dt_bias, lower_bound=LOWER_BOUND,
            checkpoint_state=layer.rec, records=layer.recoverssm_records,
            query_start_loc=qsl, state_indices=idx,
        )
        # The conv state as the verify's conv update leaves it is random here:
        # the commit only moves windows, and the checks read them from this copy.
        before.append((layer.rec.clone(), layer.conv.clone()))

    ctx.commit(
        num_sampled, idx, qsl,
        block_table=block_table, num_computed_tokens=num_computed, mamba_block_size=BS,
    )

    state_idx = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    num_accepted = torch.full((ROWS,), 9, dtype=torch.int32, device=dev)
    rs.record_state_blocks(
        RecoverSSMPostprocessMetadata(
            num_spec_decodes=ROWS,
            request_indices=None,
            block_table=block_table,
            num_computed_tokens=num_computed,
            block_size=BS,
        ),
        torch.arange(ROWS, dtype=torch.int32, device=dev),
        num_sampled,
        state_idx,
        num_accepted,
    )
    torch.cuda.synchronize()

    failures = []

    def fail(msg: str) -> None:
        failures.append(f"[{unallocated_fill} seed={seed}] {msg}")

    written = set()
    for i, (name, nc, q, n) in enumerate(CASES):
        final_col = (nc + n - 1) // BS
        if final_col >= cdiv(nc + q, BS):
            fail(f"{name}: final column {final_col} is not allocated")
            continue
        final_b = block_id(i, final_col)
        written.add(final_b)
        next_boundary = (nc // BS + 1) * BS
        boundary_b = None
        if nc + n >= next_boundary:
            boundary_b = block_id(i, next_boundary // BS - 1)
            written.add(boundary_b)
        src = source[i]
        for li, layer in enumerate(layers):
            ref = refs[li]
            _, conv0 = before[li]
            if not torch.equal(layer.rec[final_b], ref[int(ref_cols[i, n - 1])]):
                fail(f"{name} layer {li}: state after token {nc + n} not in block "
                     f"{final_b} (column {final_col})")
            if not torch.equal(layer.conv[final_b, :, :HIST], conv0[src, :, n - 1 : n - 1 + HIST]):
                fail(f"{name} layer {li}: conv window not in block {final_b}")
            if boundary_b is not None:
                blen = next_boundary - nc
                if not torch.equal(layer.rec[boundary_b], ref[int(ref_cols[i, blen - 1])]):
                    fail(f"{name} layer {li}: boundary state after token "
                         f"{next_boundary} not in block {boundary_b}")
                if not torch.equal(
                    layer.conv[boundary_b, :, :HIST], conv0[src, :, blen - 1 : blen - 1 + HIST]
                ):
                    fail(f"{name} layer {li}: boundary conv window not in block {boundary_b}")
        if int(state_idx[i]) != final_col:
            fail(f"{name}: recorded state column {int(state_idx[i])}, want {final_col}")
        if int(num_accepted[i]) != 1:
            fail(f"{name}: num_accepted {int(num_accepted[i])}, want 1")

    for li, layer in enumerate(layers):
        rec0, conv0 = before[li]
        for b in range(1, LINES):
            if b in written:
                continue
            if not torch.equal(layer.rec[b], rec0[b]):
                fail(f"layer {li}: recurrent state of block {b} changed")
            if not torch.equal(layer.conv[b], conv0[b]):
                fail(f"layer {li}: conv state of block {b} changed")
    return failures


failures = []
for fill in ("stale", "null"):
    for seed in (0, 1):
        failures += run(fill, seed)
for msg in failures[:20]:
    print("  FAIL", msg)
if failures:
    print(f"RECOVERSSM BOUNDARY: FAIL ({len(failures)} failures)")
    sys.exit(1)
print(f"RECOVERSSM BOUNDARY: PASS ({len(CASES)} cases x 2 fills x 2 seeds)")
