"""Tests for activation capture hooks."""

import pytest
import torch
import torch.nn as nn

from src.hooks.activation_capture import ActivationCaptureManager


class _SimpleModel(nn.Module):
    """Minimal model with named conv layers for testing."""

    def __init__(self):
        super().__init__()
        self.block = nn.ModuleDict({
            "conv1": nn.Conv2d(3, 16, 3, padding=1),
            "conv2": nn.Conv2d(16, 16, 3, padding=1),
        })

    def forward(self, x):
        x = self.block["conv1"](x)
        x = self.block["conv2"](x)
        return x


class TestActivationCapture:
    @pytest.fixture
    def models_and_mgr(self):
        teacher = _SimpleModel()
        student = _SimpleModel()
        mgr = ActivationCaptureManager()
        mgr.register_hooks(teacher, student, ["block.conv1", "block.conv2"])
        return teacher, student, mgr

    def test_captures_teacher_activations(self, models_and_mgr):
        teacher, _, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8)
        teacher(x)
        assert "block.conv1" in mgr.teacher_activations
        assert "block.conv2" in mgr.teacher_activations

    def test_captures_student_activations(self, models_and_mgr):
        _, student, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8)
        student(x)
        assert "block.conv1" in mgr.student_activations

    def test_teacher_detached(self, models_and_mgr):
        teacher, _, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8)
        teacher(x)
        for act in mgr.teacher_activations.values():
            assert not act.requires_grad

    def test_student_has_grad(self, models_and_mgr):
        _, student, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        student(x)
        # Student activations should allow gradient flow
        act = mgr.student_activations["block.conv1"]
        assert act.shape[0] == 2

    def test_clear(self, models_and_mgr):
        teacher, student, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8)
        teacher(x)
        student(x)
        mgr.clear()
        assert len(mgr.teacher_activations) == 0
        assert len(mgr.student_activations) == 0

    def test_correct_shapes(self, models_and_mgr):
        teacher, _, mgr = models_and_mgr
        x = torch.randn(2, 3, 8, 8)
        teacher(x)
        # conv1 input should be (2, 3, 8, 8)
        assert mgr.teacher_activations["block.conv1"].shape == (2, 3, 8, 8)
        # conv2 input should be (2, 16, 8, 8)
        assert mgr.teacher_activations["block.conv2"].shape == (2, 16, 8, 8)

    def test_remove_hooks(self, models_and_mgr):
        teacher, _, mgr = models_and_mgr
        mgr.remove_hooks()
        x = torch.randn(2, 3, 8, 8)
        teacher(x)
        # No activations should be captured after removal
        assert len(mgr.teacher_activations) == 0
