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
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1_000
    ema_decay: float = 0.9999

    # Loss weights
    alpha: float = 1.0       # standard NSA weight
    alpha_s: float = 1.0     # conditional NSA weight
    beta: float = 1.0        # KD / FitNets / Gramian weight
    lam: float = 0.01        # orthogonality weight

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

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> ExperimentConfig:
        """Parse CLI arguments into config."""
        parser = argparse.ArgumentParser(description="NSA-Diff Training")
        for name, fld in cls.__dataclass_fields__.items():
            ftype = fld.type
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
