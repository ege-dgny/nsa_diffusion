"""Generate images from a trained model using DDPM or DDIM scheduler."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm


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
    num_batches = math.ceil(num_samples / batch_size)
    remaining = num_samples

    batch_pbar = tqdm(range(num_batches), desc="Generating", unit="batch", dynamic_ncols=True)
    for _ in batch_pbar:
        bs = min(batch_size, remaining)
        # Start from pure noise
        sample = torch.randn(
            bs, num_channels, image_size, image_size,
            generator=generator,
        ).to(device)

        for t in tqdm(
            scheduler.timesteps,
            desc=f"  Denoising (batch {num_batches - remaining // batch_size + 1}/{num_batches})",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        ):
            t_batch = t.expand(bs).to(device)
            pred = model(sample, t_batch).sample
            sample = scheduler.step(pred, t, sample).prev_sample

        all_samples.append(sample.cpu())
        remaining -= bs
        batch_pbar.set_postfix(samples=f"{num_samples - remaining}/{num_samples}")

    return all_samples


def save_samples(
    samples: list[torch.Tensor],
    output_dir: str | Path,
) -> int:
    """Save generated samples as individual PNG files. Returns count saved."""
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = sum(b.shape[0] for b in samples)
    idx = 0
    with tqdm(total=total, desc="Saving images", unit="img", dynamic_ncols=True) as pbar:
        for batch in samples:
            for img in batch:
                # Rescale from [-1,1] to [0,1]
                img = (img + 1.0) / 2.0
                img = img.clamp(0, 1)
                save_image(img, output_dir / f"{idx:06d}.png")
                idx += 1
                pbar.update(1)
    return idx
