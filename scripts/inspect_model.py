#!/usr/bin/env python3
"""Print teacher/student architecture, param counts, compression ratios."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DDPMPipeline

from src.decomposition.student_builder import build_student
from src.utils.unet_inspect import (
    discover_compressible_layers,
    print_layer_summary,
    count_params,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect DDPM UNet architecture")
    parser.add_argument("--teacher_id", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--rank_ratio", type=float, default=0.25)
    args = parser.parse_args()

    print("Loading teacher model...")
    pipeline = DDPMPipeline.from_pretrained(args.teacher_id)
    teacher = pipeline.unet

    print("\n=== Teacher Architecture ===")
    teacher_params = count_params(teacher)
    print(f"Total params: {teacher_params:,}")

    print("\n=== Compressible Layers ===")
    layers = discover_compressible_layers(teacher)
    print_layer_summary(layers)

    print(f"\n=== Building Student (rank_ratio={args.rank_ratio}) ===")
    student, _, skip_infos = build_student(teacher, args.rank_ratio)
    student_params = count_params(student)
    print(f"Student params: {student_params:,}")
    print(f"Compression: {teacher_params / student_params:.2f}x")
    print(f"Param reduction: {(1 - student_params / teacher_params) * 100:.1f}%")

    print(f"\n=== Skip-Receiving Layers ({len(skip_infos)}) ===")
    for si in skip_infos:
        print(f"  {si.layer_name}: dec_ch={si.decoder_channels}, skip_ch={si.skip_channels}")


if __name__ == "__main__":
    main()
