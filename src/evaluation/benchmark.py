"""Latency, memory, and parameter benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results for a model."""

    total_params: int
    trainable_params: int
    latency_ms: float       # per-step inference latency
    peak_memory_mb: float   # peak GPU memory during inference
    compression_ratio: float


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    image_size: int = 32,
    num_channels: int = 3,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_runs: int = 20,
    teacher_params: int | None = None,
) -> BenchmarkResult:
    """Measure latency and memory for a single denoising step."""
    model.eval()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(batch_size, num_channels, image_size, image_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device).long()

    # Warmup
    with torch.no_grad():
        for _ in tqdm(range(num_warmup), desc="Warmup", unit="run", leave=False, dynamic_ncols=True):
            model(x, t)

    # Sync before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    with torch.no_grad():
        start = time.perf_counter()
        for _ in tqdm(range(num_runs), desc="Benchmarking", unit="run", leave=False, dynamic_ncols=True):
            model(x, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / num_runs) * 1000

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    compression = (teacher_params / total_params) if teacher_params else 1.0

    return BenchmarkResult(
        total_params=total_params,
        trainable_params=trainable_params,
        latency_ms=latency_ms,
        peak_memory_mb=peak_memory_mb,
        compression_ratio=compression,
    )
