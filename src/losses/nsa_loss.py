"""Standard null-space absorbing loss for non-skip layers."""

from __future__ import annotations

import torch


def null_space_loss(
    w_eff: torch.Tensor,
    teacher_act: torch.Tensor,
    student_act: torch.Tensor,
) -> torch.Tensor:
    """Compute ||W_eff @ e||^2 where e = teacher_act - student_act.

    Args:
        w_eff: (C_out, C_in) effective weight matrix
        teacher_act: (B, C_in, H, W) teacher activation (detached)
        student_act: (B, C_in, H, W) student activation

    Returns:
        Scalar loss
    """
    # Nested autocast(enabled=False) — prevents AMP from overriding .float()
    with torch.amp.autocast("cuda", enabled=False):
        w_eff = w_eff.float()
        # Row-normalize W_eff so loss is scale-invariant
        w_eff = w_eff / (w_eff.norm(dim=1, keepdim=True) + 1e-8)
        error = (teacher_act - student_act).float()  # (B, C_in, H, W)
        b, c_in, h, w = error.shape

        # Reshape to (B*H*W, C_in) for matrix multiply
        error_flat = error.permute(0, 2, 3, 1).reshape(-1, c_in)  # (N, C_in)

        projected = error_flat @ w_eff.t()  # (N, C_out)

        return (projected ** 2).mean()
