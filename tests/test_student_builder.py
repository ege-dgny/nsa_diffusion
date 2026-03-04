"""Tests for student builder."""

import pytest
import torch

# These tests require diffusers to be installed and teacher model downloaded.
# Mark as slow/integration tests.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and not hasattr(torch.backends, "mps"),
    reason="Requires GPU for reasonable speed",
)


def _get_teacher():
    """Load a small teacher for testing."""
    from diffusers import DDPMPipeline
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    return pipeline.unet


class TestBuildStudent:
    @pytest.fixture(scope="class")
    def teacher_and_student(self):
        from src.decomposition.student_builder import build_student
        teacher = _get_teacher()
        student, layers, skip_infos = build_student(teacher, rank_ratio=0.25)
        return teacher, student, layers, skip_infos

    def test_same_output_shape(self, teacher_and_student):
        teacher, student, _, _ = teacher_and_student
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])

        with torch.no_grad():
            t_out = teacher(x, t).sample
            s_out = student(x, t).sample

        assert t_out.shape == s_out.shape

    def test_param_reduction(self, teacher_and_student):
        teacher, student, _, _ = teacher_and_student
        t_params = sum(p.numel() for p in teacher.parameters())
        s_params = sum(p.numel() for p in student.parameters())
        assert s_params < t_params

    def test_skip_infos_nonempty(self, teacher_and_student):
        _, _, _, skip_infos = teacher_and_student
        assert len(skip_infos) > 0

    def test_skip_channel_split(self, teacher_and_student):
        _, _, _, skip_infos = teacher_and_student
        for si in skip_infos:
            assert si.decoder_channels > 0
            assert si.skip_channels > 0
