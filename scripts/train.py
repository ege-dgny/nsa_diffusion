#!/usr/bin/env python3
"""CLI entry point for NSA-Diff training."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import ExperimentConfig
from src.training.trainer import Trainer


def main():
    config = ExperimentConfig.from_args()
    print(f"Method: {config.method}")
    print(f"Rank ratio: {config.rank_ratio}")
    print(f"Steps: {config.num_steps}")
    print(f"Batch size: {config.batch_size}")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
