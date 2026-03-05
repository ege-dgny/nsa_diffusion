"""Orthogonality regularization for CP decomposition factors."""

from __future__ import annotations

import torch
import torch.nn as nn


def orthogonality_loss(student: nn.Module) -> torch.Tensor:
    """Compute ||U^T U - I||_F^2 on pw_out weights of all CP sequences.

    Finds all Sequential modules that look like CP decompositions
    (4 layers, last is 1x1 conv) and regularizes the output factor.
    """
    total = torch.tensor(0.0)
    count = 0

    for module in student.modules():
        if not isinstance(module, nn.Sequential):
            continue
        if len(module) != 4:
            continue
        pw_out = module[-1]
        if not isinstance(pw_out, nn.Conv2d):
            continue
        if pw_out.kernel_size != (1, 1):
            continue

        # pw_out.weight: (C_out, R, 1, 1)
        u = pw_out.weight.squeeze(-1).squeeze(-1)  # (C_out, R)
        if total.device != u.device:
            total = total.to(u.device)

        # Normalize columns to unit norm before computing U^T U - I
        u_norm = u / (u.norm(dim=0, keepdim=True) + 1e-8)

        # U^T U should be identity: (R, R)
        r = u_norm.shape[1]
        utu = u_norm.t() @ u_norm  # (R, R)
        eye = torch.eye(r, device=u.device, dtype=u.dtype)
        total = total + ((utu - eye) ** 2).sum()
        count += 1

    if count > 0:
        total = total / count
    return total
