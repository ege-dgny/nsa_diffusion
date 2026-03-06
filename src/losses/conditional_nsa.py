"""Conditional null-space loss for skip-receiving layers."""

from __future__ import annotations

import torch


def conditional_null_space_loss(
    w_eff: torch.Tensor,
    teacher_act: torch.Tensor,
    student_act: torch.Tensor,
    decoder_channels: int,
) -> torch.Tensor:
    """Conditional NSA loss with stop-gradient decoupling.

    At skip-receiving conv1, input = cat(decoder_features, skip_features).
    We split the error and apply stop-gradient to decouple paths.

    Args:
        w_eff: (C_out, C_in) effective weight, C_in = decoder_ch + skip_ch
        teacher_act: (B, C_in, H, W) detached teacher activation
        student_act: (B, C_in, H, W) student activation
        decoder_channels: number of decoder channels (first part of concat)

    Returns:
        Scalar loss (decoder_term + encoder_term)
    """
    error = teacher_act - student_act  # (B, C_in, H, W)
    b, c_in, h, w = error.shape

    # Split error into decoder and skip components
    e_dec = error[:, :decoder_channels]       # (B, C_dec, H, W)
    delta_skip = error[:, decoder_channels:]  # (B, C_skip, H, W)

    # Decoder term: sg(delta_skip), gradient flows through e_dec
    e_dec_term = torch.cat([e_dec, delta_skip.detach()], dim=1)

    # Encoder term: sg(e_dec), gradient flows through delta_skip
    e_enc_term = torch.cat([e_dec.detach(), delta_skip], dim=1)

    loss_dec = _project_and_norm(w_eff, e_dec_term)
    loss_enc = _project_and_norm(w_eff, e_enc_term)

    return loss_dec + loss_enc


def _project_and_norm(
    w_eff: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Compute ||W_eff @ e||^2 averaged over spatial positions and batch."""
    # Cast to fp32 to prevent AMP fp16 overflow
    w_eff = w_eff.float()
    # Row-normalize W_eff so loss is scale-invariant (SVD init produces large entries)
    w_eff = w_eff / (w_eff.norm(dim=1, keepdim=True) + 1e-8)
    error = error.float()
    b, c_in, h, w = error.shape
    error_flat = error.permute(0, 2, 3, 1).reshape(-1, c_in)  # (N, C_in)
    projected = error_flat @ w_eff.t()  # (N, C_out)
    return (projected ** 2).mean()
