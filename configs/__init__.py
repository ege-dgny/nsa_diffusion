"""Experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal


Method = Literal["lowrank_kd", "standard_nsa", "nsa_diff", "fitnets", "gramian"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""

    # Method
    method: Method = "nsa_diff"
    rank_ratio: float = 0.25

    # Training
    num_steps: int = 100_000
    total_samples: int | None = None  # If set, num_steps is ignored; steps = total_samples // batch_size
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1_000
    ema_decay: float = 0.9999

    # Loss weights (from blueprint Section 5.6)
    alpha: float = 0.1       # standard NSA weight
    alpha_s: float = 0.1     # conditional NSA weight
    beta: float = 0.1        # KD / FitNets / Gramian weight
    lam: float = 0.1         # orthogonality weight

    # Model
    teacher_id: str = "google/ddpm-cifar10-32"
    image_size: int = 32

    # Evaluation
    num_fid_samples: int = 50_000
    eval_every: int = 10_000
    save_every: int = 10_000
    sample_steps: int = 1_000

    # Paths
    output_dir: str = "outputs"
    run_name: str = ""

    # Logging
    use_wandb: bool = False
    wandb_project: str = "nsa-diff"

    # Device
    device: str = ""  # auto-detected if empty
    use_amp: bool = True

    # Data
    num_workers: int = 4

    @property
    def effective_num_steps(self) -> int:
        """Number of training steps. If total_samples is set, steps = total_samples // batch_size; else num_steps."""
        if self.total_samples is not None:
            if self.total_samples < self.batch_size:
                raise ValueError(
                    f"total_samples ({self.total_samples}) must be >= batch_size ({self.batch_size})"
                )
            return self.total_samples // self.batch_size
        return self.num_steps

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> ExperimentConfig:
        """Parse CLI arguments into config."""
        parser = argparse.ArgumentParser(description="NSA-Diff Training")
        for name, fld in cls.__dataclass_fields__.items():
            ftype = fld.type
            if name == "total_samples":
                parser.add_argument("--total_samples", type=int, default=None)
                continue
            if ftype == "bool":
                parser.add_argument(
                    f"--{name}",
                    type=_str_to_bool,
                    default=fld.default,
                )
            elif ftype in ("str", "Method"):
                parser.add_argument(f"--{name}", type=str, default=fld.default)
            elif ftype == "float":
                parser.add_argument(f"--{name}", type=float, default=fld.default)
            elif ftype == "int":
                parser.add_argument(f"--{name}", type=int, default=fld.default)
        parsed = parser.parse_args(args)
        return cls(**vars(parsed))


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")
