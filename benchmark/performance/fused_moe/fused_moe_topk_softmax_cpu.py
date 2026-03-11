from __future__ import annotations

from typing import Optional, Tuple

import torch


def topk_softmax_cpu(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch CPU reference for ``sgl_kernel.topk_softmax``.

    This mirrors the public kernel behavior:
    1. Upcast logits to float32.
    2. Apply optional tanh softcapping.
    3. Apply optional per-expert correction bias.
    4. Compute softmax over all experts.
    5. Select top-k experts with lower-index tie breaking.
    6. Optionally renormalize the selected top-k weights.
    """

    if gating_output.dim() != 2:
        raise ValueError("gating_output must be 2D [num_tokens, num_experts]")

    if topk < 1 or topk > gating_output.shape[1]:
        raise ValueError("topk must be in [1, num_experts]")

    logits = gating_output.to(device="cpu", dtype=torch.float32)

    if moe_softcapping != 0.0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping

    if correction_bias is not None:
        bias = correction_bias.to(device="cpu", dtype=torch.float32)
        if bias.dim() != 1 or bias.shape[0] != logits.shape[1]:
            raise ValueError("correction_bias must be [num_experts]")
        logits = logits + bias

    probs = torch.softmax(logits, dim=-1)

    sorted_indices = torch.argsort(probs, dim=-1, descending=True, stable=True)
    topk_indices = sorted_indices[:, :topk].to(torch.int32)
    topk_weights = torch.gather(probs, dim=1, index=topk_indices.to(torch.int64))

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_indices
