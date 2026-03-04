"""Discover compressible layers and skip-connection info in a UNet2DModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn


@dataclass(frozen=True)
class LayerInfo:
    """Metadata for a compressible conv layer."""

    name: str               # dotted path, e.g. "down_blocks.0.resnets.0.conv1"
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    is_skip_receiver: bool  # True if this conv1 receives concatenated skip
    skip_channels: int      # number of skip channels (0 if not skip receiver)
    decoder_channels: int   # number of decoder channels (0 if not skip receiver)


def discover_compressible_layers(unet: nn.Module) -> list[LayerInfo]:
    """Walk the UNet and return metadata for every compressible conv.

    Compressible = conv1 or conv2 inside ResnetBlock2D (3x3 convs).
    Skip-receivers = conv1 of resnets in up_blocks (they get cat'd skip input).
    """
    layers: list[LayerInfo] = []

    # Collect encoder output channels per block for skip info
    down_channels = _get_down_block_channels(unet)

    # Down blocks
    for bi, block in enumerate(unet.down_blocks):
        for ri, resnet in enumerate(block.resnets):
            _add_resnet_layers(
                layers, resnet, f"down_blocks.{bi}.resnets.{ri}",
                is_skip_receiver=False,
            )

    # Mid block
    if hasattr(unet, "mid_block") and unet.mid_block is not None:
        for ri, resnet in enumerate(unet.mid_block.resnets):
            _add_resnet_layers(
                layers, resnet, f"mid_block.resnets.{ri}",
                is_skip_receiver=False,
            )

    # Up blocks — these receive skip connections
    for bi, block in enumerate(unet.up_blocks):
        for ri, resnet in enumerate(block.resnets):
            _add_resnet_layers(
                layers, resnet, f"up_blocks.{bi}.resnets.{ri}",
                is_skip_receiver=True,
                down_channels=down_channels,
                up_block_idx=bi,
            )

    return layers


def _get_down_block_channels(unet: nn.Module) -> list[int]:
    """Get output channels of each down block (for skip channel computation)."""
    channels = []
    for block in unet.down_blocks:
        last_resnet = block.resnets[-1]
        channels.append(last_resnet.conv2.out_channels)
    return channels


def _add_resnet_layers(
    layers: list[LayerInfo],
    resnet: nn.Module,
    prefix: str,
    is_skip_receiver: bool,
    down_channels: list[int] | None = None,
    up_block_idx: int = 0,
) -> None:
    """Add conv1 and conv2 from a ResnetBlock2D."""
    conv1 = resnet.conv1
    conv2 = resnet.conv2

    skip_ch = 0
    dec_ch = 0

    if is_skip_receiver:
        # conv1 input = decoder_ch + skip_ch (from concatenation)
        total_in = conv1.in_channels
        # The skip channels come from the corresponding down block
        # up_blocks go in reverse order of down_blocks
        if down_channels is not None:
            n_down = len(down_channels)
            # up_block 0 pairs with down_block (n_down - 1), etc.
            down_idx = n_down - 1 - up_block_idx
            skip_ch = down_channels[down_idx]
            dec_ch = total_in - skip_ch
        else:
            dec_ch = total_in
            skip_ch = 0

    layers.append(LayerInfo(
        name=f"{prefix}.conv1",
        in_channels=conv1.in_channels,
        out_channels=conv1.out_channels,
        kernel_size=tuple(conv1.kernel_size),
        is_skip_receiver=is_skip_receiver,
        skip_channels=skip_ch,
        decoder_channels=dec_ch,
    ))
    layers.append(LayerInfo(
        name=f"{prefix}.conv2",
        in_channels=conv2.in_channels,
        out_channels=conv2.out_channels,
        kernel_size=tuple(conv2.kernel_size),
        is_skip_receiver=False,
        skip_channels=0,
        decoder_channels=0,
    ))


def get_skip_layer_names(layers: Sequence[LayerInfo]) -> list[str]:
    """Return names of layers that receive skip connections."""
    return [l.name for l in layers if l.is_skip_receiver]


def count_params(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_layer_summary(layers: Sequence[LayerInfo]) -> None:
    """Print a table of compressible layers."""
    print(f"{'Name':<45} {'In':>5} {'Out':>5} {'Kernel':>7} {'Skip?':>5} {'SkipCh':>6}")
    print("-" * 80)
    for l in layers:
        skip = "YES" if l.is_skip_receiver else ""
        ks = "x".join(str(k) for k in l.kernel_size)
        print(f"{l.name:<45} {l.in_channels:>5} {l.out_channels:>5} {ks:>7} {skip:>5} {l.skip_channels:>6}")
