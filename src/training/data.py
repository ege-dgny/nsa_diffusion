"""CIFAR-10 dataloader for diffusion training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def create_cifar10_loader(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> DataLoader:
    """Create CIFAR-10 training dataloader normalized to [-1, 1]."""
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
