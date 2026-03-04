"""Build a compressed student model from a teacher UNet via CP decomposition."""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from src.decomposition.cp_decompose import compute_rank, cp_decompose_conv
from src.utils.unet_inspect import LayerInfo, discover_compressible_layers


@dataclass(frozen=True)
class SkipLayerInfo:
    """Info about a skip-receiving layer for conditional NSA."""

    layer_name: str        # e.g. "up_blocks.0.resnets.0.conv1"
    decoder_channels: int
    skip_channels: int


def _decompose_one(weight: torch.Tensor, rank: int) -> list[torch.Tensor]:
    """Run CP decomposition for a single layer. Picklable for multiprocessing."""
    _, factors = cp_decompose_conv(weight, rank)
    return [f.cpu() for f in factors]


def _build_cp_sequence(
    c_out: int, c_in: int, kh: int, kw: int,
    rank: int,
    factors: list[torch.Tensor],
    bias: torch.Tensor | None,
) -> nn.Sequential:
    """Assemble a 4-layer CP sequence from pre-computed factors."""
    f_out, f_in, f_h, f_w = factors

    pw_in = nn.Conv2d(c_in, rank, 1, bias=False)
    pw_in.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)

    pad_w = kw // 2
    dw_horiz = nn.Conv2d(rank, rank, (1, kw), padding=(0, pad_w), groups=rank, bias=False)
    dw_horiz.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)

    pad_h = kh // 2
    dw_vert = nn.Conv2d(rank, rank, (kh, 1), padding=(pad_h, 0), groups=rank, bias=False)
    dw_vert.weight.data = f_h.t().unsqueeze(1).unsqueeze(-1)

    pw_out = nn.Conv2d(rank, c_out, 1, bias=bias is not None)
    pw_out.weight.data = f_out.unsqueeze(-1).unsqueeze(-1)
    if bias is not None:
        pw_out.bias.data = bias.clone()

    return nn.Sequential(pw_in, dw_horiz, dw_vert, pw_out)


def build_student(
    teacher: nn.Module,
    rank_ratio: float = 0.25,
    max_workers: int | None = None,
) -> tuple[nn.Module, list[LayerInfo], list[SkipLayerInfo]]:
    """Create student by deep-copying teacher and replacing convs with CP sequences.

    Decompositions run in parallel across CPU cores and are cached to disk.

    Returns:
        (student, all_layer_infos, skip_layer_infos)
    """
    layers = discover_compressible_layers(teacher)
    student = copy.deepcopy(teacher)

    # Collect work items
    work: list[tuple[LayerInfo, int, torch.Tensor]] = []
    for info in layers:
        conv = _get_module(student, info.name)
        if not isinstance(conv, nn.Conv2d):
            continue
        rank = compute_rank(info.in_channels, info.out_channels, rank_ratio)
        work.append((info, rank, conv.weight.data.detach().cpu()))

    # Parallel decomposition
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(work))

    factors_map: dict[str, list[torch.Tensor]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_decompose_one, w, r): info.name
            for info, r, w in work
        }
        with tqdm(total=len(futures), desc="CP decomposing layers", unit="layer", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                name = futures[future]
                factors_map[name] = future.result()
                pbar.update(1)

    # Assemble CP sequences (fast, sequential)
    skip_infos: list[SkipLayerInfo] = []
    for info, rank, _ in work:
        conv = _get_module(student, info.name)
        factors = [f.to(device=conv.weight.device, dtype=conv.weight.dtype) for f in factors_map[info.name]]
        c_out, c_in, kh, kw = conv.weight.shape
        bias = conv.bias.data if conv.bias is not None else None
        cp_seq = _build_cp_sequence(c_out, c_in, kh, kw, rank, factors, bias)
        _set_module(student, info.name, cp_seq)

        if info.is_skip_receiver:
            skip_infos.append(SkipLayerInfo(
                layer_name=info.name,
                decoder_channels=info.decoder_channels,
                skip_channels=info.skip_channels,
            ))

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
