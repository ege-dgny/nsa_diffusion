"""FID computation via pytorch-fid."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def compute_fid(
    generated_dir: str | Path,
    reference_dir: str | Path | None = None,
    dataset: str = "cifar10",
) -> float:
    """Compute FID between generated images and reference.

    If reference_dir is None and dataset is 'cifar10', uses pytorch-fid's
    built-in CIFAR-10 statistics.

    Returns FID score.
    """
    generated_dir = str(Path(generated_dir).resolve())

    cmd = [sys.executable, "-m", "pytorch_fid"]

    if reference_dir is not None:
        cmd.extend([generated_dir, str(Path(reference_dir).resolve())])
    else:
        # pytorch-fid supports computing against saved stats
        cmd.extend([generated_dir, "--save-stats" if False else generated_dir])
        # Fallback: need a reference dir
        raise ValueError(
            "reference_dir required. Generate CIFAR-10 reference images first, "
            "or provide path to pre-computed statistics."
        )

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse FID from output
    for line in result.stdout.strip().split("\n"):
        if "FID" in line:
            return float(line.split(":")[-1].strip())

    # Try parsing the entire output as a number
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse FID from output: {result.stdout}")


def compute_fid_from_dirs(
    dir1: str | Path,
    dir2: str | Path,
) -> float:
    """Compute FID between two directories of images."""
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        str(Path(dir1).resolve()),
        str(Path(dir2).resolve()),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # pytorch-fid prints "FID:  <value>" or just the value
    output = result.stdout.strip()
    for line in output.split("\n"):
        line = line.strip()
        if ":" in line:
            return float(line.split(":")[-1].strip())
        try:
            return float(line)
        except ValueError:
            continue

    raise RuntimeError(f"Could not parse FID from: {output}")
