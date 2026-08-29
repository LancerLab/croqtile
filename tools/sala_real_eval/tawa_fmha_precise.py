#!/usr/bin/env python3
"""Precise TFLOPS measurement for Tawa TMA FMHA: 2-stage vs 3-stage.

Usage:
  rm -rf ~/.triton/cache/
  PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_fmha_precise.py --gpu 1
"""
import sys, os, argparse, statistics

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--trials", type=int, default=5)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
WARMUP, REP = 100, 1000
TRIALS = args.trials

target_seqs = [4096, 8192, 16384]

def bench_one(N_CTX, num_stages, sala_on):
    os.environ["SALA_ENABLE"] = "1" if sala_on else "0"
    triton.runtime.cache.default_cache_dir = lambda: f"/tmp/triton_fmha_precise_s{num_stages}_sala{int(sala_on)}_{N_CTX}"
    q, k, v = init_tensors(BATCH, H, N_CTX, HEAD_DIM, dtype)
    def fn():
        return run_attention(
            _attn_fwd, q, k, v, causal, sm_scale,
            BLOCK_M=128, BLOCK_N=128, NUM_STAGES=num_stages,
            NUM_WARPS=8, USE_TTG_WS=False,
            WG_SPEC="mma_first", MATH_WG_PIPE=True,
            FORCE_MEMBAR=False,
        )
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)
    flops = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops * 0.5
    tflops = total_flops / ms * 1e-9
    return tflops

print(f"Precise TFLOPS: Tawa FMHA 2-stage baseline vs 3-stage SALA-enabled")
print(f"Config: BATCH={BATCH}, H={H}, D={HEAD_DIM}, BM=128, BN=128, causal")
print(f"Warmup={WARMUP}, Rep={REP}, Trials={TRIALS}")
print()

for N_CTX in target_seqs:
    tf_2s_list = []
    tf_3s_list = []
    for t in range(TRIALS):
        tf_2s = bench_one(N_CTX, 2, False)
        tf_3s = bench_one(N_CTX, 3, True)
        tf_2s_list.append(tf_2s)
        tf_3s_list.append(tf_3s)
        print(f"  SEQ={N_CTX} trial {t+1}: 2s={tf_2s:.1f}  3s+SALA={tf_3s:.1f}  delta={((tf_3s-tf_2s)/tf_2s*100):+.2f}%")

    mean_2s = statistics.mean(tf_2s_list)
    mean_3s = statistics.mean(tf_3s_list)
    std_2s = statistics.stdev(tf_2s_list) if len(tf_2s_list) > 1 else 0
    std_3s = statistics.stdev(tf_3s_list) if len(tf_3s_list) > 1 else 0
    delta = (mean_3s - mean_2s) / mean_2s * 100

    print(f"  SEQ={N_CTX} SUMMARY: 2s={mean_2s:.1f}+/-{std_2s:.1f}  3s+SALA={mean_3s:.1f}+/-{std_3s:.1f}  delta={delta:+.2f}%")
    print()
