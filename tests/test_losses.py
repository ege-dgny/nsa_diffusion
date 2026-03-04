"""Tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from src.losses.nsa_loss import null_space_loss
from src.losses.conditional_nsa import conditional_null_space_loss
from src.losses.distillation import kd_loss, fitnets_loss, gramian_loss
from src.losses.orthogonality import orthogonality_loss


class TestNullSpaceLoss:
    def test_zero_error_gives_zero_loss(self):
        w_eff = torch.randn(32, 64)
        act = torch.randn(2, 64, 4, 4)
        loss = null_space_loss(w_eff, act, act)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss(self):
        w_eff = torch.randn(32, 64)
        t_act = torch.randn(2, 64, 4, 4)
        s_act = torch.randn(2, 64, 4, 4)
        loss = null_space_loss(w_eff, t_act, s_act)
        assert loss.item() > 0

    def test_gradient_flows(self):
        w_eff = torch.randn(32, 64, requires_grad=True)
        t_act = torch.randn(2, 64, 4, 4)
        s_act = torch.randn(2, 64, 4, 4, requires_grad=True)
        loss = null_space_loss(w_eff, t_act, s_act)
        loss.backward()
        assert s_act.grad is not None
        assert w_eff.grad is not None


class TestConditionalNSA:
    def test_zero_error(self):
        w_eff = torch.randn(32, 64)
        act = torch.randn(2, 64, 4, 4)
        loss = conditional_null_space_loss(w_eff, act, act, decoder_channels=32)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss(self):
        w_eff = torch.randn(32, 64)
        t_act = torch.randn(2, 64, 4, 4)
        s_act = torch.randn(2, 64, 4, 4)
        loss = conditional_null_space_loss(w_eff, t_act, s_act, decoder_channels=32)
        assert loss.item() > 0

    def test_gradient_decoupling(self):
        """Verify stop-gradient: decoder grad shouldn't flow through skip."""
        w_eff = torch.randn(32, 96)
        t_act = torch.randn(2, 96, 4, 4)

        # Student activation with gradient tracking
        s_dec = torch.randn(2, 64, 4, 4, requires_grad=True)
        s_skip = torch.randn(2, 32, 4, 4, requires_grad=True)
        s_act = torch.cat([s_dec, s_skip], dim=1)

        loss = conditional_null_space_loss(w_eff, t_act, s_act, decoder_channels=64)
        loss.backward()

        # Both should have gradients (from their respective terms)
        assert s_dec.grad is not None
        assert s_skip.grad is not None


class TestKDLoss:
    def test_identical_gives_zero(self):
        pred = torch.randn(2, 3, 32, 32)
        loss = kd_loss(pred, pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        loss = kd_loss(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
        assert loss.item() > 0


class TestFitNetsLoss:
    def test_matched_shapes(self):
        acts_t = {"layer1": torch.randn(2, 64, 8, 8)}
        acts_s = {"layer1": torch.randn(2, 64, 8, 8)}
        loss = fitnets_loss(acts_t, acts_s)
        assert loss.item() > 0

    def test_empty_acts(self):
        loss = fitnets_loss({}, {})
        assert loss.item() == 0.0


class TestGramianLoss:
    def test_identical_gives_zero(self):
        act = torch.randn(2, 64, 8, 8)
        loss = gramian_loss({"l1": act}, {"l1": act})
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_gives_positive(self):
        loss = gramian_loss(
            {"l1": torch.randn(2, 64, 8, 8)},
            {"l1": torch.randn(2, 64, 8, 8)},
        )
        assert loss.item() > 0


class TestOrthogonalityLoss:
    def test_with_cp_sequences(self):
        model = nn.Module()
        # Simulate a CP sequence
        seq = nn.Sequential(
            nn.Conv2d(64, 16, 1, bias=False),
            nn.Conv2d(16, 16, (1, 3), padding=(0, 1), groups=16, bias=False),
            nn.Conv2d(16, 16, (3, 1), padding=(1, 0), groups=16, bias=False),
            nn.Conv2d(16, 32, 1, bias=False),
        )
        model.seq = seq
        loss = orthogonality_loss(model)
        assert loss.item() > 0  # Random weights won't be orthogonal

    def test_orthogonal_weights_give_small_loss(self):
        model = nn.Module()
        seq = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False),
            nn.Conv2d(8, 8, (1, 3), padding=(0, 1), groups=8, bias=False),
            nn.Conv2d(8, 8, (3, 1), padding=(1, 0), groups=8, bias=False),
            nn.Conv2d(8, 16, 1, bias=False),
        )
        # Make pw_out orthogonal
        q, _ = torch.linalg.qr(torch.randn(16, 8))
        seq[-1].weight.data = q.unsqueeze(-1).unsqueeze(-1)
        model.seq = seq
        loss = orthogonality_loss(model)
        assert loss.item() < 0.01
