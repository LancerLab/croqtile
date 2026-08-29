#!/usr/bin/env python3
"""Benchmark Tawa TMA-based Fused Attention with and without SALA."""

import os
import sys
import torch
import triton

sys.path.insert(0, "/home/wsj/dev/triton-aref/python/test/unit/auto_ws")
from fmha_common import init_tensors, run_attention

sys.path.insert(0, "/home/wsj/dev/triton-aref/python/test/unit/auto_ws")
from test_tma_fused_attention import _attn_fwd

DEVICE = "cuda"
BATCH, H, HEAD_DIM = 4, 32, 128
dtype = torch.float16

configs = [
    (2048,),
    (4096,),
    (8192,),
    (16384,),
]

def run_bench(N_CTX, sala_on):
    os.environ["SALA_ENABLE"] = "1" if sala_on else "0"
    os.environ["TRITON_CACHE_MANAGER"] = "triton.runtime.cache:FileCacheManager"
    tag = "SALA" if sala_on else "BASE"

    # Clear Triton cache to force recompilation
    import importlib
    import triton.runtime.cache
    importlib.reload(triton.runtime.cache)

    q, k, v = init_tensors(BATCH, H, N_CTX, HEAD_DIM, dtype)
    sm_scale = 1.3
    causal = True

    NUM_WARPS = 8
    WG_SPEC = "mma_first"
    MATH_WG_PIPE = True

    def fn():
        return run_attention(
            _attn_fwd, q, k, v, causal, sm_scale,
            BLOCK_M=128, BLOCK_N=128, NUM_STAGES=2,
            NUM_WARPS=NUM_WARPS, USE_TTG_WS=False,
            WG_SPEC=WG_SPEC, MATH_WG_PIPE=MATH_WG_PIPE,
            FORCE_MEMBAR=False,
        )

    # Warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    ms = triton.testing.do_bench(fn, warmup=25, rep=500)
    flops = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops * 0.5  # causal
    tflops = total_flops / ms * 1e-9

    return ms, tflops

print(f"{'N_CTX':>8}  {'BASE_ms':>10}  {'BASE_TFLOPS':>12}  {'SALA_ms':>10}  {'SALA_TFLOPS':>12}  {'delta%':>8}")
print("-" * 75)

for (N_CTX,) in configs:
    # Force fresh compilation for each mode
    triton.runtime.cache.default_cache_dir = lambda: f"/tmp/triton_cache_base_{N_CTX}"
    ms_base, tf_base = run_bench(N_CTX, False)

    triton.runtime.cache.default_cache_dir = lambda: f"/tmp/triton_cache_sala_{N_CTX}"
    ms_sala, tf_sala = run_bench(N_CTX, True)

    delta = (tf_sala - tf_base) / tf_base * 100
    print(f"{N_CTX:>8}  {ms_base:>10.3f}  {tf_base:>12.1f}  {ms_sala:>10.3f}  {tf_sala:>12.1f}  {delta:>+8.2f}%")
