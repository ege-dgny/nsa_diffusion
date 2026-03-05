"""CIFAR-10 dataloader for diffusion training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T


def create_cifar10_loader(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create CIFAR-10 training dataloader normalized to [-1, 1].

    Returns (loader, sampler). Sampler is None for single-GPU.
    Caller must call sampler.set_epoch(epoch) each epoch for proper shuffling.
    """
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

    sampler = None
    shuffle = True
    if rank is not None and world_size is not None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader, sampler
