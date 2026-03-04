"""CP decomposition of conv layers and effective weight extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac


tl.set_backend("pytorch")


@torch.no_grad()
def cp_decompose_conv(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Decompose a 4D conv weight via CP (PARAFAC).

    Args:
        weight: (C_out, C_in, kH, kW) tensor
        rank: CP rank

    Returns:
        (weights_vector, [f_out, f_in, f_h, f_w]) — tensorly format
    """
    # SVD init gives better decomposition; falls back to random on MPS/CPU (macOS segfault)
    init = "svd" if torch.cuda.is_available() else "random"
    _, factors = parafac(weight.float().cpu(), rank=rank, init=init, n_iter_max=100)
    # Move factors back to original device/dtype
    factors = [f.to(device=weight.device, dtype=weight.dtype) for f in factors]
    # factors: [f_out (C_out,R), f_in (C_in,R), f_h (kH,R), f_w (kW,R)]
    return None, factors


def create_cp_sequence(
    conv: nn.Conv2d,
    rank: int,
) -> nn.Sequential:
    """Replace a Conv2d with a 4-layer CP sequence.

    Sequence: pw_in (1x1) -> dw_horiz (1xkW) -> dw_vert (kHx1) -> pw_out (1x1)
    Bias from original conv is placed on pw_out.
    """
    c_out, c_in, kh, kw = conv.weight.shape
    has_bias = conv.bias is not None

    _, (f_out, f_in, f_h, f_w) = cp_decompose_conv(conv.weight.data, rank)

    # 1. Pointwise in: C_in -> R
    pw_in = nn.Conv2d(c_in, rank, 1, bias=False)
    pw_in.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)  # (R, C_in, 1, 1)

    # 2. Depthwise horizontal: 1 x kW
    pad_w = kw // 2
    dw_horiz = nn.Conv2d(rank, rank, (1, kw), padding=(0, pad_w), groups=rank, bias=False)
    dw_horiz.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)  # (R, 1, 1, kW)

    # 3. Depthwise vertical: kH x 1
    pad_h = kh // 2
    dw_vert = nn.Conv2d(rank, rank, (kh, 1), padding=(pad_h, 0), groups=rank, bias=False)
    dw_vert.weight.data = f_h.t().unsqueeze(1).unsqueeze(-1)  # (R, 1, kH, 1)

    # 4. Pointwise out: R -> C_out
    pw_out = nn.Conv2d(rank, c_out, 1, bias=has_bias)
    pw_out.weight.data = f_out.unsqueeze(-1).unsqueeze(-1)  # (C_out, R, 1, 1)
    if has_bias:
        pw_out.bias.data = conv.bias.data.clone()

    return nn.Sequential(pw_in, dw_horiz, dw_vert, pw_out)


def get_effective_weight(cp_seq: nn.Sequential) -> torch.Tensor:
    """Extract channel-mixing matrix W_eff = pw_out @ pw_in.

    Returns: (C_out, C_in) matrix.
    """
    pw_in = cp_seq[0]
    pw_out = cp_seq[-1]
    # pw_in.weight: (R, C_in, 1, 1), pw_out.weight: (C_out, R, 1, 1)
    w_in = pw_in.weight.squeeze(-1).squeeze(-1)   # (R, C_in)
    w_out = pw_out.weight.squeeze(-1).squeeze(-1)  # (C_out, R)
    return w_out @ w_in  # (C_out, C_in)


def compute_rank(in_channels: int, out_channels: int, rank_ratio: float) -> int:
    """Compute CP rank from ratio."""
    return max(1, int(rank_ratio * min(in_channels, out_channels)))


def reconstruction_error(original: torch.Tensor, cp_seq: nn.Sequential, x: torch.Tensor) -> float:
    """Compute relative Frobenius norm error of CP reconstruction.

    Uses a random input x to compare outputs.
    """
    with torch.no_grad():
        conv_orig = nn.Conv2d(
            original.shape[1], original.shape[0],
            (original.shape[2], original.shape[3]),
            padding=(original.shape[2] // 2, original.shape[3] // 2),
            bias=False,
        )
        conv_orig.weight.data = original
        y_orig = conv_orig(x)
        y_cp = cp_seq(x)
        err = torch.norm(y_orig - y_cp) / (torch.norm(y_orig) + 1e-8)
    return err.item()
