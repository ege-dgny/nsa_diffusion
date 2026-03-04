"""Exponential Moving Average for model parameters."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains shadow parameters as EMA of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self._decay = decay
        self._shadow = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow params with current model params."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self._shadow:
                self._shadow[name].mul_(self._decay).add_(
                    p.data, alpha=1.0 - self._decay
                )

    def apply(self, model: nn.Module) -> None:
        """Swap model params with shadow params (for eval)."""
        self._backup.clear()
        for name, p in model.named_parameters():
            if name in self._shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self._shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model params (after eval)."""
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._shadow = dict(state)
