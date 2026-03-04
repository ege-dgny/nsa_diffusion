"""Smoke test: one training step doesn't crash, loss is finite, grads are nonzero."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and not hasattr(torch.backends, "mps"),
    reason="Requires model download and reasonable compute",
)


class TestTrainingStep:
    @pytest.fixture(scope="class")
    def setup(self):
        from diffusers import DDPMPipeline, DDPMScheduler
        from src.decomposition.student_builder import build_student
        from src.hooks.activation_capture import ActivationCaptureManager
        from src.losses.composite import compute_composite_loss
        from configs import ExperimentConfig

        device = torch.device("cpu")

        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        teacher = pipeline.unet
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        config = ExperimentConfig(
            method="nsa_diff",
            rank_ratio=0.5,
            num_steps=1,
            batch_size=2,
        )

        student, layers, skip_infos = build_student(teacher, config.rank_ratio)
        student.train()

        hook_mgr = ActivationCaptureManager()
        layer_names = [info.name for info in layers]
        hook_mgr.register_hooks(teacher, student, layer_names)

        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        return {
            "teacher": teacher,
            "student": student,
            "layers": layers,
            "skip_infos": skip_infos,
            "hook_mgr": hook_mgr,
            "scheduler": scheduler,
            "config": config,
            "device": device,
        }

    def test_one_step(self, setup):
        from src.losses.composite import compute_composite_loss

        teacher = setup["teacher"]
        student = setup["student"]
        hook_mgr = setup["hook_mgr"]
        scheduler = setup["scheduler"]
        config = setup["config"]

        x = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (2,)).long()
        noisy = scheduler.add_noise(x, noise, t)

        with torch.no_grad():
            teacher_pred = teacher(noisy, t).sample

        student_pred = student(noisy, t).sample

        breakdown = compute_composite_loss(
            config=config,
            noise=noise,
            teacher_pred=teacher_pred,
            student_pred=student_pred,
            student=student,
            teacher_acts=hook_mgr.teacher_activations,
            student_acts=hook_mgr.student_activations,
            layer_infos=setup["layers"],
            skip_infos=setup["skip_infos"],
        )

        assert torch.isfinite(breakdown.total)
        breakdown.total.backward()

        # Check at least some grads are nonzero
        has_nonzero = False
        for p in student.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero

        hook_mgr.clear()

    def test_loss_components_finite(self, setup):
        from src.losses.composite import compute_composite_loss

        teacher = setup["teacher"]
        student = setup["student"]
        hook_mgr = setup["hook_mgr"]
        scheduler = setup["scheduler"]
        config = setup["config"]

        # Reset grads
        for p in student.parameters():
            if p.grad is not None:
                p.grad.zero_()

        x = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (2,)).long()
        noisy = scheduler.add_noise(x, noise, t)

        with torch.no_grad():
            teacher_pred = teacher(noisy, t).sample
        student_pred = student(noisy, t).sample

        breakdown = compute_composite_loss(
            config=config,
            noise=noise,
            teacher_pred=teacher_pred,
            student_pred=student_pred,
            student=student,
            teacher_acts=hook_mgr.teacher_activations,
            student_acts=hook_mgr.student_activations,
            layer_infos=setup["layers"],
            skip_infos=setup["skip_infos"],
        )

        import math
        assert math.isfinite(breakdown.l_eps)
        assert math.isfinite(breakdown.l_null)
        assert math.isfinite(breakdown.l_cond)
        assert math.isfinite(breakdown.l_kd)
        assert math.isfinite(breakdown.l_orth)

        hook_mgr.clear()
