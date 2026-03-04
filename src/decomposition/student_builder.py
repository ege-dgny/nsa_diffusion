"""Build a compressed student model from a teacher UNet via CP decomposition."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.decomposition.cp_decompose import compute_rank, create_cp_sequence
from src.utils.unet_inspect import LayerInfo, discover_compressible_layers


@dataclass(frozen=True)
class SkipLayerInfo:
    """Info about a skip-receiving layer for conditional NSA."""

    layer_name: str        # e.g. "up_blocks.0.resnets.0.conv1"
    decoder_channels: int
    skip_channels: int


def build_student(
    teacher: nn.Module,
    rank_ratio: float = 0.25,
) -> tuple[nn.Module, list[LayerInfo], list[SkipLayerInfo]]:
    """Create student by deep-copying teacher and replacing convs with CP sequences.

    Returns:
        (student, all_layer_infos, skip_layer_infos)
    """
    layers = discover_compressible_layers(teacher)
    student = copy.deepcopy(teacher)

    skip_infos: list[SkipLayerInfo] = []

    for info in layers:
        rank = compute_rank(info.in_channels, info.out_channels, rank_ratio)
        conv = _get_module(student, info.name)

        if not isinstance(conv, nn.Conv2d):
            continue

        cp_seq = create_cp_sequence(conv, rank)
        _set_module(student, info.name, cp_seq)

        if info.is_skip_receiver:
            skip_infos.append(SkipLayerInfo(
                layer_name=info.name,
                decoder_channels=info.decoder_channels,
                skip_channels=info.skip_channels,
            ))

    # Freeze everything that shouldn't train
    # (GroupNorm, timestep embeddings are kept as-is from deepcopy)

    return student, layers, skip_infos


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by dotted path."""
    parts = name.split(".")
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def _set_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    """Set a submodule by dotted path."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement
    else:
        setattr(parent, last, replacement)
