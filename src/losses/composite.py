"""Per-method loss aggregator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from configs import ExperimentConfig
from src.decomposition.cp_decompose import get_effective_weight
from src.decomposition.student_builder import SkipLayerInfo
from src.losses.nsa_loss import null_space_loss
from src.losses.conditional_nsa import conditional_null_space_loss
from src.losses.distillation import kd_loss, fitnets_loss, gramian_loss
from src.losses.orthogonality import orthogonality_loss
from src.utils.unet_inspect import LayerInfo


@dataclass
class LossBreakdown:
    """Individual loss components for logging."""

    total: torch.Tensor
    l_eps: float = 0.0
    l_null: float = 0.0
    l_cond: float = 0.0
    l_kd: float = 0.0
    l_fitnets: float = 0.0
    l_gramian: float = 0.0
    l_orth: float = 0.0


def _get_module(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def compute_composite_loss(
    config: ExperimentConfig,
    noise: torch.Tensor,
    teacher_pred: torch.Tensor,
    student_pred: torch.Tensor,
    student: nn.Module,
    teacher_acts: dict[str, torch.Tensor],
    student_acts: dict[str, torch.Tensor],
    layer_infos: list[LayerInfo],
    skip_infos: list[SkipLayerInfo],
) -> LossBreakdown:
    """Compute full loss based on method selection."""
    device = student_pred.device

    # L_eps: noise prediction MSE (always present)
    l_eps = torch.nn.functional.mse_loss(student_pred, noise)

    l_null = torch.tensor(0.0, device=device)
    l_cond = torch.tensor(0.0, device=device)
    l_kd_val = torch.tensor(0.0, device=device)
    l_fitnets_val = torch.tensor(0.0, device=device)
    l_gramian_val = torch.tensor(0.0, device=device)
    l_orth = torch.tensor(0.0, device=device)

    skip_names = {s.layer_name for s in skip_infos}
    skip_map = {s.layer_name: s for s in skip_infos}

    method = config.method

    # NSA losses (standard_nsa, nsa_diff)
    if method in ("standard_nsa", "nsa_diff"):
        null_count = 0
        cond_count = 0
        for info in layer_infos:
            name = info.name
            if name not in teacher_acts or name not in student_acts:
                continue

            cp_seq = _get_module(student, name)
            if not isinstance(cp_seq, nn.Sequential):
                continue

            w_eff = get_effective_weight(cp_seq)
            t_act = teacher_acts[name]
            s_act = student_acts[name]

            if name in skip_names and method == "nsa_diff":
                # Conditional NSA for skip-receiving layers
                si = skip_map[name]
                l_cond = l_cond + conditional_null_space_loss(
                    w_eff, t_act, s_act, si.decoder_channels,
                )
                cond_count += 1
            else:
                # Standard NSA (for standard_nsa: all layers; for nsa_diff: non-skip)
                l_null = l_null + null_space_loss(w_eff, t_act, s_act)
                null_count += 1

        if null_count > 0:
            l_null = l_null / null_count
        if cond_count > 0:
            l_cond = l_cond / cond_count

    # KD loss
    if method in ("lowrank_kd", "standard_nsa", "nsa_diff", "gramian"):
        l_kd_val = kd_loss(teacher_pred, student_pred)

    # FitNets
    if method == "fitnets":
        l_fitnets_val = fitnets_loss(teacher_acts, student_acts)

    # Gramian
    if method == "gramian":
        l_gramian_val = gramian_loss(teacher_acts, student_acts)

    # Orthogonality
    if method in ("lowrank_kd", "standard_nsa", "nsa_diff"):
        l_orth = orthogonality_loss(student)

    # Aggregate
    total = l_eps
    if method == "lowrank_kd":
        total = total + config.beta * l_kd_val + config.lam * l_orth
    elif method == "standard_nsa":
        total = total + config.alpha * l_null + config.beta * l_kd_val + config.lam * l_orth
    elif method == "nsa_diff":
        total = (
            total
            + config.alpha * l_null
            + config.alpha_s * l_cond
            + config.beta * l_kd_val
            + config.lam * l_orth
        )
    elif method == "fitnets":
        total = total + config.beta * l_fitnets_val
    elif method == "gramian":
        total = total + config.beta * l_gramian_val + config.beta * l_kd_val

    return LossBreakdown(
        total=total,
        l_eps=l_eps.item(),
        l_null=l_null.item(),
        l_cond=l_cond.item(),
        l_kd=l_kd_val.item(),
        l_fitnets=l_fitnets_val.item(),
        l_gramian=l_gramian_val.item(),
        l_orth=l_orth.item(),
    )
