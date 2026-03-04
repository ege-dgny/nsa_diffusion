"""Forward hooks to capture conv layer inputs for NSA loss computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CapturedActivation:
    """Stores captured input to a conv layer."""

    layer_name: str
    input_tensor: torch.Tensor | None = None


class ActivationCaptureManager:
    """Manages forward hooks for teacher and student networks.

    Teacher activations are detached (no grad needed).
    Student activations keep gradients flowing.
    """

    def __init__(self):
        self._teacher_acts: dict[str, torch.Tensor] = {}
        self._student_acts: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def register_hooks(
        self,
        teacher: nn.Module,
        student: nn.Module,
        layer_names: list[str],
    ) -> None:
        """Register forward hooks on specified layers in both models."""
        for name in layer_names:
            t_mod = _get_module(teacher, name)
            s_mod = _get_module(student, name)

            handle_t = t_mod.register_forward_pre_hook(
                _make_capture_hook(self._teacher_acts, name, detach=True)
            )
            handle_s = s_mod.register_forward_pre_hook(
                _make_capture_hook(self._student_acts, name, detach=False)
            )
            self._handles.extend([handle_t, handle_s])

    @property
    def teacher_activations(self) -> dict[str, torch.Tensor]:
        return self._teacher_acts

    @property
    def student_activations(self) -> dict[str, torch.Tensor]:
        return self._student_acts

    def clear(self) -> None:
        """Clear stored activations to avoid memory leak. Call after each step."""
        self._teacher_acts.clear()
        self._student_acts.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.clear()


def _make_capture_hook(
    storage: dict[str, torch.Tensor],
    name: str,
    detach: bool,
):
    """Create a forward pre-hook that captures the input tensor."""

    def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
        inp = args[0] if isinstance(args, tuple) else args
        if detach:
            storage[name] = inp.detach()
        else:
            storage[name] = inp

    return hook


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get submodule by dotted path."""
    parts = name.split(".")
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod
