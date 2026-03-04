"""Thin wandb + console logging wrapper."""

from __future__ import annotations

import sys
from typing import Any


class Logger:
    """Unified logger for wandb and console output."""

    def __init__(self, use_wandb: bool = False, project: str = "", run_name: str = "", config: dict | None = None):
        self._wandb = None
        if use_wandb:
            import wandb
            self._wandb = wandb
            wandb.init(project=project, name=run_name or None, config=config or {})

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb and console."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        parts = [f"step={step}"] if step is not None else []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts), file=sys.stderr, flush=True)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
