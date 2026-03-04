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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--teacher", action="store_true", help="Evaluate teacher model directly (no checkpoint needed)")
    parser.add_argument("--teacher_id", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if not args.teacher and args.checkpoint is None:
        parser.error("--checkpoint is required unless --teacher is set")

    device = get_device(args.device)
    print(f"Device: {device}")

    if args.teacher:
        # Evaluate teacher directly
        pipeline = DDPMPipeline.from_pretrained(args.teacher_id)
        model = pipeline.unet.to(device)
        model.eval()
        teacher_params = sum(p.numel() for p in model.parameters())
        print(f"Teacher params: {teacher_params:,}")
    else:
        # Load student checkpoint
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        rank_ratio = ckpt_config.get("rank_ratio", 0.25)
        teacher_id = ckpt_config.get("teacher_id", "google/ddpm-cifar10-32")

        pipeline = DDPMPipeline.from_pretrained(teacher_id)
        teacher = pipeline.unet
        student, _, _ = build_student(teacher, rank_ratio)

        if "ema_state_dict" in ckpt:
            ema = EMA(student)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.apply(student)
        else:
            student.load_state_dict(ckpt["student_state_dict"])

        model = student.to(device)
        model.eval()

        if args.benchmark:
            teacher_params = sum(p.numel() for p in teacher.parameters())
            result = benchmark_model(model, device, teacher_params=teacher_params)
            print(f"Params: {result.total_params:,}")
            print(f"Compression: {result.compression_ratio:.2f}x")
            print(f"Latency: {result.latency_ms:.2f}ms/step")
            if result.peak_memory_mb > 0:
                print(f"Peak memory: {result.peak_memory_mb:.1f}MB")

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples(
        model=model,
        num_samples=args.num_samples,
        scheduler_type=args.scheduler,
        num_steps=args.num_steps,
        device=device,
        batch_size=args.batch_size,
    )

    count = save_samples(samples, args.output_dir)
    print(f"Saved {count} images to {args.output_dir}")

    # Auto-compute FID if cifar10 stats exist
    print("Run: python -m pytorch_fid <output_dir> <cifar10_ref_dir> --device cuda")


if __name__ == "__main__":
    main()
