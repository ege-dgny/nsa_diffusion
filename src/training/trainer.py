"""Main training loop for NSA-Diff."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from tqdm import tqdm

from configs import ExperimentConfig
from src.decomposition.student_builder import SkipLayerInfo, build_student
from src.hooks.activation_capture import ActivationCaptureManager
from src.losses.composite import compute_composite_loss, LossBreakdown
from src.training.data import create_cifar10_loader
from src.training.ema import EMA
from src.utils.device import (
    get_device, get_autocast_ctx, supports_amp,
    is_ddp, setup_ddp, cleanup_ddp, is_main_process,
)
from src.utils.logging_utils import Logger
from src.utils.unet_inspect import LayerInfo, count_params


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup from 0→1 over warmup_steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


class Trainer:
    """Full training loop for student distillation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # DDP setup
        self.ddp = is_ddp()
        if self.ddp:
            self.rank, self.local_rank, self.world_size = setup_ddp()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = get_device(config.device)

        self.main = is_main_process(self.rank)
        self.use_amp = config.use_amp and supports_amp(self.device)

        # Will be set up in train()
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None
        self.student_unwrapped: nn.Module | None = None
        self.layer_infos: list[LayerInfo] = []
        self.skip_infos: list[SkipLayerInfo] = []
        self.hook_mgr: ActivationCaptureManager | None = None

    def _log(self, msg: str) -> None:
        """Print only on rank 0."""
        if self.main:
            print(msg)

    def train(self) -> None:
        """Execute the full training pipeline."""
        config = self.config
        num_steps = config.effective_num_steps

        # Load teacher
        self.teacher = self._load_teacher()

        # Build student (CP decomposition happens on CPU, then move to device)
        self.student, self.layer_infos, self.skip_infos = build_student(
            self.teacher, config.rank_ratio,
        )
        self.student = self.student.to(self.device)
        self.student.train()

        # Keep unwrapped reference for hooks/losses/EMA
        self.student_unwrapped = self.student

        # Wrap in DDP
        if self.ddp:
            self.student = nn.parallel.DistributedDataParallel(
                self.student, device_ids=[self.local_rank],
            )

        teacher_params = count_params(self.teacher)
        student_params = count_params(self.student_unwrapped)
        self._log(f"Teacher params: {teacher_params:,}")
        self._log(f"Student params: {student_params:,}")
        self._log(f"Compression: {teacher_params / student_params:.2f}x")
        self._log(f"Training steps: {num_steps:,} | Warmup: {config.warmup_steps}")
        self._log(f"Loss weights: alpha={config.alpha} alpha_s={config.alpha_s} "
                   f"beta={config.beta} lam={config.lam}")
        if self.ddp:
            self._log(f"DDP: {self.world_size} GPUs, effective batch = {config.batch_size * self.world_size}")

        # Hooks on unwrapped student (DDP adds module. prefix that breaks name lookup)
        self.hook_mgr = ActivationCaptureManager()
        layer_names = [info.name for info in self.layer_infos]
        self.hook_mgr.register_hooks(self.teacher, self.student_unwrapped, layer_names)

        # Noise scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
        )

        # Optimizer (DDP-wrapped .parameters() works fine)
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps,
        )

        # EMA on unwrapped student
        ema = EMA(self.student_unwrapped, decay=config.ema_decay)

        # AMP scaler
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Data (distributed sampler for DDP)
        loader, sampler = create_cifar10_loader(
            config.batch_size, config.num_workers,
            rank=self.rank if self.ddp else None,
            world_size=self.world_size if self.ddp else None,
        )
        data_iter = _infinite_loader(loader, sampler)

        # Logger (rank 0 only)
        logger = None
        if self.main:
            logger = Logger(
                use_wandb=config.use_wandb,
                project=config.wandb_project,
                run_name=config.run_name or config.method,
                config=vars(config) if not isinstance(config, dict) else config,
            )

        # Output dir
        out_dir = Path(config.output_dir) / (config.run_name or config.method)
        if self.main:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Training loop (tqdm only on rank 0)
        step_range = range(1, num_steps + 1)
        if self.main:
            pbar = tqdm(step_range, desc=f"Training [{config.method}]", unit="step", dynamic_ncols=True)
        else:
            pbar = step_range

        for step in pbar:
            images = next(data_iter).to(self.device)

            # Warmup factor: auxiliary losses ramp linearly 0→1
            wf = _warmup_factor(step, config.warmup_steps)

            # Sample noise and timesteps
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=self.device,
            ).long()

            # Add noise
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_pred = self.teacher(noisy_images, timesteps).sample

            # Student forward
            with get_autocast_ctx(self.device, self.use_amp):
                student_pred = self.student(noisy_images, timesteps).sample

                # Pass unwrapped student to loss (accesses raw parameters)
                breakdown = compute_composite_loss(
                    config=config,
                    noise=noise,
                    teacher_pred=teacher_pred,
                    student_pred=student_pred,
                    student=self.student_unwrapped,
                    teacher_acts=self.hook_mgr.teacher_activations,
                    student_acts=self.hook_mgr.student_activations,
                    layer_infos=self.layer_infos,
                    skip_infos=self.skip_infos,
                    warmup_factor=wf,
                )

            # Backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(breakdown.total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.student.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                breakdown.total.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), config.grad_clip_norm)
                optimizer.step()

            lr_scheduler.step()
            ema.update(self.student_unwrapped)
            self.hook_mgr.clear()

            # Progress bar (rank 0)
            if self.main and step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{breakdown.total.item():.4f}",
                    eps=f"{breakdown.l_eps:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    wf=f"{wf:.2f}",
                )

            # Logging (rank 0)
            if self.main and (step % 100 == 0 or step == 1):
                logger.log({
                    "loss/total": breakdown.total.item(),
                    "loss/eps": breakdown.l_eps,
                    "loss/null": breakdown.l_null,
                    "loss/cond": breakdown.l_cond,
                    "loss/kd": breakdown.l_kd,
                    "loss/orth": breakdown.l_orth,
                    "schedule/lr": optimizer.param_groups[0]["lr"],
                    "schedule/warmup": wf,
                }, step=step)

            # Save checkpoint (rank 0)
            if self.main and step % config.save_every == 0:
                self._save_checkpoint(out_dir, step, ema, optimizer, lr_scheduler)

        # Final save
        if self.main:
            self._save_checkpoint(out_dir, num_steps, ema, optimizer, lr_scheduler)
            logger.finish()
        self.hook_mgr.remove_hooks()
        if self.ddp:
            cleanup_ddp()
        self._log(f"Training complete. Checkpoints in {out_dir}")

    def _load_teacher(self) -> nn.Module:
        """Load pretrained teacher UNet."""
        from diffusers import DDPMPipeline

        pipeline = DDPMPipeline.from_pretrained(self.config.teacher_id)
        teacher = pipeline.unet
        teacher = teacher.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _save_checkpoint(
        self,
        out_dir: Path,
        step: int,
        ema: EMA,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        path = out_dir / f"checkpoint_{step}.pt"
        torch.save({
            "step": step,
            "student_state_dict": self.student_unwrapped.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": vars(self.config) if hasattr(self.config, "__dataclass_fields__") else self.config,
        }, path)
        self._log(f"Saved checkpoint: {path}")


def _infinite_loader(loader, sampler=None):
    """Yield images forever from a DataLoader."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch[0]  # images only, discard labels
        epoch += 1
