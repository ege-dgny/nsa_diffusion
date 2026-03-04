"""Main training loop for NSA-Diff."""

from __future__ import annotations

import math
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
from src.utils.device import get_device, get_autocast_ctx, supports_amp
from src.utils.logging_utils import Logger
from src.utils.unet_inspect import LayerInfo, count_params


class Trainer:
    """Full training loop for student distillation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = get_device(config.device)
        self.use_amp = config.use_amp and supports_amp(self.device)

        # Will be set up in train()
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None
        self.layer_infos: list[LayerInfo] = []
        self.skip_infos: list[SkipLayerInfo] = []
        self.hook_mgr: ActivationCaptureManager | None = None

    def train(self) -> None:
        """Execute the full training pipeline."""
        config = self.config

        # Load teacher
        self.teacher = self._load_teacher()

        # Build student
        self.student, self.layer_infos, self.skip_infos = build_student(
            self.teacher, config.rank_ratio,
        )
        self.student = self.student.to(self.device)
        self.student.train()

        teacher_params = count_params(self.teacher)
        student_params = count_params(self.student)
        print(f"Teacher params: {teacher_params:,}")
        print(f"Student params: {student_params:,}")
        print(f"Compression: {teacher_params / student_params:.2f}x")

        # Hooks
        self.hook_mgr = ActivationCaptureManager()
        layer_names = [info.name for info in self.layer_infos]
        self.hook_mgr.register_hooks(self.teacher, self.student, layer_names)

        # Scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
        )

        # Optimizer & LR schedule
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_steps,
        )

        # EMA
        ema = EMA(self.student, decay=config.ema_decay)

        # AMP scaler
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Data
        loader = create_cifar10_loader(config.batch_size, config.num_workers)
        data_iter = _infinite_loader(loader)

        # Logger
        logger = Logger(
            use_wandb=config.use_wandb,
            project=config.wandb_project,
            run_name=config.run_name or config.method,
            config=vars(config) if not isinstance(config, dict) else config,
        )

        # Output dir
        out_dir = Path(config.output_dir) / (config.run_name or config.method)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        pbar = tqdm(
            range(1, config.num_steps + 1),
            desc=f"Training [{config.method}]",
            unit="step",
            dynamic_ncols=True,
        )
        for step in pbar:
            images = next(data_iter).to(self.device)

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

                breakdown = compute_composite_loss(
                    config=config,
                    noise=noise,
                    teacher_pred=teacher_pred,
                    student_pred=student_pred,
                    student=self.student,
                    teacher_acts=self.hook_mgr.teacher_activations,
                    student_acts=self.hook_mgr.student_activations,
                    layer_infos=self.layer_infos,
                    skip_infos=self.skip_infos,
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
            ema.update(self.student)
            self.hook_mgr.clear()

            # Update progress bar postfix
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{breakdown.total.item():.4f}",
                    eps=f"{breakdown.l_eps:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            # Logging
            if step % 100 == 0:
                logger.log({
                    "loss/total": breakdown.total.item(),
                    "loss/eps": breakdown.l_eps,
                    "loss/null": breakdown.l_null,
                    "loss/cond": breakdown.l_cond,
                    "loss/kd": breakdown.l_kd,
                    "loss/orth": breakdown.l_orth,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=step)

            # Save checkpoint
            if step % config.save_every == 0:
                self._save_checkpoint(out_dir, step, ema, optimizer, lr_scheduler)

        # Final save
        self._save_checkpoint(out_dir, config.num_steps, ema, optimizer, lr_scheduler)
        self.hook_mgr.remove_hooks()
        logger.finish()
        print(f"Training complete. Checkpoints in {out_dir}")

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
            "student_state_dict": self.student.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": vars(self.config) if hasattr(self.config, "__dataclass_fields__") else self.config,
        }, path)
        print(f"Saved checkpoint: {path}")


def _infinite_loader(loader):
    """Yield images forever from a DataLoader."""
    while True:
        for batch in loader:
            yield batch[0]  # images only, discard labels
