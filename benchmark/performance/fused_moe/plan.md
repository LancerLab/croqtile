# Fused MoE Basic Plan

## Goal

- Build the first workspace-local scaffold for a fused MoE kernel.
- Keep the GPU side minimal: only a `__co__` kernel skeleton, no inner logic, and host-side verify skipped.
- Lock down the router/top-k softmax contract first with a C++ reference checked against the Python reference in `fused_moe_topk_softmax_cpu.py`.

## What We Are Implementing First

- `fused_moe_topk_softmax_cpu.py` defines the routing semantics we need to preserve:
  - upcast logits to float32
  - optional tanh softcapping
  - optional per-expert correction bias
  - softmax across all experts
  - stable top-k selection with lower-index tie breaking
  - optional renormalization over the selected top-k weights
- `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90.co` gives the SM90 host/device harness pattern we can reuse for the initial `.co` file.
- `study.md` reinforces that the longer-term GPU path should follow SM90-style staged/TMA/WGMMA structure, but this first step only sets up the workspace and reference behavior.

## Deliverables In This Workspace

- `fused_moe_topk_softmax_cpu_verify.cpp`
  - standalone C++ reference for top-k softmax
  - deterministic cases compared numerically against Python-generated golden outputs
- `fused_moe_basic_sm90.co`
  - minimal SM90 fused-MoE test harness
  - empty `__co__` kernel body for now
  - host-side allocation and launch only
  - verification intentionally skipped

## Next Implementation Steps

1. Extend the `.co` skeleton with routing buffers and per-expert work decomposition.
2. Decide whether the fused kernel should target a single expert GEMM stage first or the full MLP path.
3. Reuse the SM90 matmul structure for the expert compute stage once routing is wired in.
4. Replace skipped verification with a host reference that combines routing and expert accumulation.
