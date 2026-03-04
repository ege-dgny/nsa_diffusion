"""Generate images from a trained model using DDPM or DDIM scheduler."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    num_samples: int,
    image_size: int = 32,
    num_channels: int = 3,
    scheduler_type: str = "ddpm",
    num_steps: int = 1000,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Generate images using the diffusion model.

    Returns list of (B, C, H, W) tensors, each in [-1, 1].
    """
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if scheduler_type == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_steps)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        scheduler.set_timesteps(num_steps)

    all_samples: list[torch.Tensor] = []
    remaining = num_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        # Start from pure noise
        sample = torch.randn(
            bs, num_channels, image_size, image_size,
            generator=generator,
        ).to(device)

        for t in scheduler.timesteps:
            t_batch = t.expand(bs).to(device)
            pred = model(sample, t_batch).sample
            sample = scheduler.step(pred, t, sample).prev_sample

        all_samples.append(sample.cpu())
        remaining -= bs

    return all_samples


def save_samples(
    samples: list[torch.Tensor],
    output_dir: str | Path,
) -> int:
    """Save generated samples as individual PNG files. Returns count saved."""
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for batch in samples:
        for img in batch:
            # Rescale from [-1,1] to [0,1]
            img = (img + 1.0) / 2.0
            img = img.clamp(0, 1)
            save_image(img, output_dir / f"{idx:06d}.png")
            idx += 1
    return idx
