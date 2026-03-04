"""Cache CP decomposition factors to disk keyed by weight content hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


DEFAULT_CACHE_DIR = Path(".cp_cache")


def _weight_hash(weight: torch.Tensor, rank: int) -> str:
    """Deterministic hash of weight tensor content + rank."""
    data = weight.detach().cpu().float().numpy().tobytes()
    h = hashlib.sha256(data)
    h.update(rank.to_bytes(4, "big"))
    return h.hexdigest()[:16]


def load_cached_factors(
    weight: torch.Tensor,
    rank: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[torch.Tensor] | None:
    """Load cached factors if they exist. Returns None on miss."""
    key = _weight_hash(weight, rank)
    path = cache_dir / f"{key}.pt"
    if not path.exists():
        return None
    factors = torch.load(path, map_location="cpu", weights_only=True)
    return [f.to(device=weight.device, dtype=weight.dtype) for f in factors]


def save_factors_to_cache(
    weight: torch.Tensor,
    rank: int,
    factors: list[torch.Tensor],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Save factors to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _weight_hash(weight, rank)
    path = cache_dir / f"{key}.pt"
    torch.save([f.detach().cpu() for f in factors], path)
