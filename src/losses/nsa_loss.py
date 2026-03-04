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
    error = teacher_act - student_act  # (B, C_in, H, W)
    b, c_in, h, w = error.shape

    # Reshape to (B*H*W, C_in) for matrix multiply
    error_flat = error.permute(0, 2, 3, 1).reshape(-1, c_in)  # (N, C_in)

    # W_eff @ e^T -> (C_out, N), but compute as (N, C_in) @ (C_in, C_out) = (N, C_out)
    projected = error_flat @ w_eff.t()  # (N, C_out)

    return (projected ** 2).mean()
