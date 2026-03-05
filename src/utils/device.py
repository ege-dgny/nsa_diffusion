"""Device auto-detection, AMP configuration, and DDP helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def get_device(preference: str = "") -> torch.device:
    """Auto-detect best available device.

    Priority: preference > CUDA > MPS > CPU.
    """
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """Check if device supports automatic mixed precision."""
    return device.type == "cuda"


def get_autocast_ctx(device: torch.device, enabled: bool = True):
    """Return appropriate autocast context manager."""
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    # MPS and CPU: no AMP
    return _nullcontext()


class _nullcontext:
    """Minimal no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def is_ddp() -> bool:
    """Check if running under torchrun / DDP."""
    return "RANK" in os.environ


def setup_ddp() -> tuple[int, int, int]:
    """Initialize DDP process group. Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Only rank 0 should log/save."""
    return rank == 0
