#!/usr/bin/env python3
"""Evaluate a trained model: generate samples and compute FID."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DDPMPipeline

from configs import ExperimentConfig
from src.decomposition.student_builder import build_student
from src.evaluation.sample import generate_samples, save_samples
from src.evaluation.benchmark import benchmark_model
from src.training.ema import EMA
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate NSA-Diff model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {})
    rank_ratio = ckpt_config.get("rank_ratio", 0.25)
    teacher_id = ckpt_config.get("teacher_id", "google/ddpm-cifar10-32")

    # Rebuild student architecture
    pipeline = DDPMPipeline.from_pretrained(teacher_id)
    teacher = pipeline.unet
    student, _, _ = build_student(teacher, rank_ratio)

    # Load weights (use EMA if available)
    if "ema_state_dict" in ckpt:
        ema = EMA(student)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply(student)
    else:
        student.load_state_dict(ckpt["student_state_dict"])

    student = student.to(device)
    student.eval()

    # Benchmark
    if args.benchmark:
        teacher_params = sum(p.numel() for p in teacher.parameters())
        result = benchmark_model(student, device, teacher_params=teacher_params)
        print(f"Params: {result.total_params:,}")
        print(f"Compression: {result.compression_ratio:.2f}x")
        print(f"Latency: {result.latency_ms:.2f}ms/step")
        if result.peak_memory_mb > 0:
            print(f"Peak memory: {result.peak_memory_mb:.1f}MB")

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples(
        model=student,
        num_samples=args.num_samples,
        scheduler_type=args.scheduler,
        num_steps=args.num_steps,
        device=device,
        batch_size=args.batch_size,
    )

    count = save_samples(samples, args.output_dir)
    print(f"Saved {count} images to {args.output_dir}")
    print("Run pytorch-fid to compute FID against CIFAR-10 reference.")


if __name__ == "__main__":
    main()
