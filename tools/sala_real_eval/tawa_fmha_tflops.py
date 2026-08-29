#!/usr/bin/env python3
"""Benchmark Tawa TMA FMHA: 2-stage baseline vs 3-stage SALA-enabled.

Usage:
  rm -rf ~/.triton/cache/
  PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_fmha_tflops.py [--gpu 1]
"""
import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Mock pytest before any test imports
sys.modules['pytest'] = type(sys)('pytest')
sys.modules['pytest'].mark = type(sys)('mark')
sys.modules['pytest'].mark.parametrize = lambda *a, **kw: (lambda f: f)
sys.modules['pytest'].skip = lambda msg='': None

import torch, triton

sys.path.insert(0, "/home/wsj/dev/triton-aref/python/test/unit/auto_ws")
from fmha_common import init_tensors, run_attention
from test_tma_fused_attention import _attn_fwd

BATCH, H, HEAD_DIM = 4, 32, 128
dtype = torch.float16
causal = True
sm_scale = 1.3
WARMUP, REP = 50, 500

seq_lens = [1024, 2048, 4096, 8192, 16384]

def bench(N_CTX, num_stages, sala_on):
    os.environ["SALA_ENABLE"] = "1" if sala_on else "0"
    # Clear triton JIT cache to force recompilation
    triton.runtime.cache.default_cache_dir = lambda: f"/tmp/triton_fmha_s{num_stages}_sala{int(sala_on)}_{N_CTX}"

    q, k, v = init_tensors(BATCH, H, N_CTX, HEAD_DIM, dtype)
    def fn():
        return run_attention(
            _attn_fwd, q, k, v, causal, sm_scale,
            BLOCK_M=128, BLOCK_N=128, NUM_STAGES=num_stages,
            NUM_WARPS=8, USE_TTG_WS=False,
            WG_SPEC="mma_first", MATH_WG_PIPE=True,
            FORCE_MEMBAR=False,
        )
    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)
    flops = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops * (0.5 if causal else 1.0)
    tflops = total_flops / ms * 1e-9
    return ms, tflops

print("Tawa TMA FMHA: 2-stage baseline vs 3-stage SALA-enabled")
print(f"Config: BATCH={BATCH}, H={H}, D={HEAD_DIM}, BM=128, BN=128, causal={causal}")
print(f"Warmup={WARMUP}, Rep={REP}")
print()
print(f"{'SEQ':>8}  {'2s_BASE_ms':>12}  {'2s_TFLOPS':>10}  {'3s_SALA_ms':>12}  {'3s_TFLOPS':>10}  {'delta%':>8}")
print("-" * 72)

for N_CTX in seq_lens:
    ms_2s, tf_2s = bench(N_CTX, 2, False)
    try:
        ms_3s, tf_3s = bench(N_CTX, 3, True)
        delta = (tf_3s - tf_2s) / tf_2s * 100
        print(f"{N_CTX:>8}  {ms_2s:>12.3f}  {tf_2s:>10.1f}  {ms_3s:>12.3f}  {tf_3s:>10.1f}  {delta:>+8.2f}%")
    except Exception as e:
        print(f"{N_CTX:>8}  {ms_2s:>12.3f}  {tf_2s:>10.1f}  {'FAIL':>12}  {'---':>10}  {'---':>8}  ({e})")

print()
print("Also testing: 2-stage with SALA (for reference)")
print(f"{'SEQ':>8}  {'2s_noSALA':>12}  {'2s+SALA':>12}  {'delta%':>8}")
print("-" * 50)
for N_CTX in seq_lens:
    ms_base, tf_base = bench(N_CTX, 2, False)
    ms_sala, tf_sala = bench(N_CTX, 2, True)
    delta = (tf_sala - tf_base) / tf_base * 100
    print(f"{N_CTX:>8}  {tf_base:>12.1f}  {tf_sala:>12.1f}  {delta:>+8.2f}%")
