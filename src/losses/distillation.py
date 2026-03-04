"""Knowledge distillation losses: KD, FitNets, Gramian."""

from __future__ import annotations

import torch


def kd_loss(
    teacher_pred: torch.Tensor,
    student_pred: torch.Tensor,
) -> torch.Tensor:
    """Output-level KD: MSE on noise predictions.

    Args:
        teacher_pred: (B, C, H, W) teacher noise prediction
        student_pred: (B, C, H, W) student noise prediction
    """
    return torch.nn.functional.mse_loss(student_pred, teacher_pred)


def fitnets_loss(
    teacher_acts: dict[str, torch.Tensor],
    student_acts: dict[str, torch.Tensor],
) -> torch.Tensor:
    """FitNets: MSE on intermediate activations (hint-based).

    Computes MSE between matched teacher/student activation pairs.
    """
    total = torch.tensor(0.0)
    count = 0
    for name in teacher_acts:
        if name not in student_acts:
            continue
        t_act = teacher_acts[name]
        s_act = student_acts[name]
        if t_act.shape == s_act.shape:
            if total.device != t_act.device:
                total = total.to(t_act.device)
            total = total + torch.nn.functional.mse_loss(s_act, t_act)
            count += 1
    if count == 0:
        return total
    return total / count


def gramian_loss(
    teacher_acts: dict[str, torch.Tensor],
    student_acts: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Gramian matching: ||G_teacher - G_student||_F^2 per layer.

    G = F @ F^T where F is (C, H*W) per sample, averaged over batch.
    """
    total = torch.tensor(0.0)
    count = 0
    for name in teacher_acts:
        if name not in student_acts:
            continue
        t_act = teacher_acts[name]  # (B, C, H, W)
        s_act = student_acts[name]
        if t_act.shape != s_act.shape:
            continue

        if total.device != t_act.device:
            total = total.to(t_act.device)

        b, c, h, w = t_act.shape
        t_flat = t_act.reshape(b, c, h * w)  # (B, C, HW)
        s_flat = s_act.reshape(b, c, h * w)

        g_t = torch.bmm(t_flat, t_flat.transpose(1, 2)) / (h * w)  # (B, C, C)
        g_s = torch.bmm(s_flat, s_flat.transpose(1, 2)) / (h * w)

        total = total + ((g_t - g_s) ** 2).mean()
        count += 1

    if count == 0:
        return total
    return total / count
