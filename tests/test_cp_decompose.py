"""Tests for CP decomposition."""

import pytest
import torch
import torch.nn as nn

from src.decomposition.cp_decompose import (
    create_cp_sequence,
    get_effective_weight,
    compute_rank,
)


class TestComputeRank:
    def test_basic(self):
        assert compute_rank(128, 256, 0.25) == 32

    def test_min_rank_is_one(self):
        assert compute_rank(4, 4, 0.01) == 1

    def test_symmetric(self):
        assert compute_rank(256, 256, 0.5) == 128


class TestCreateCPSequence:
    @pytest.fixture
    def conv3x3(self):
        return nn.Conv2d(64, 128, 3, padding=1, bias=True)

    def test_output_shape(self, conv3x3):
        rank = 16
        cp_seq = create_cp_sequence(conv3x3, rank)
        x = torch.randn(2, 64, 8, 8)
        y = cp_seq(x)
        assert y.shape == (2, 128, 8, 8)

    def test_four_layers(self, conv3x3):
        cp_seq = create_cp_sequence(conv3x3, 16)
        assert len(cp_seq) == 4

    def test_pw_in_shape(self, conv3x3):
        rank = 16
        cp_seq = create_cp_sequence(conv3x3, rank)
        pw_in = cp_seq[0]
        assert pw_in.weight.shape == (rank, 64, 1, 1)

    def test_pw_out_shape(self, conv3x3):
        rank = 16
        cp_seq = create_cp_sequence(conv3x3, rank)
        pw_out = cp_seq[-1]
        assert pw_out.weight.shape == (128, rank, 1, 1)

    def test_bias_preserved(self, conv3x3):
        cp_seq = create_cp_sequence(conv3x3, 16)
        assert cp_seq[-1].bias is not None

    def test_no_bias(self):
        conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        cp_seq = create_cp_sequence(conv, 16)
        assert cp_seq[-1].bias is None

    def test_param_reduction(self, conv3x3):
        rank = 16
        cp_seq = create_cp_sequence(conv3x3, rank)
        orig_params = sum(p.numel() for p in conv3x3.parameters())
        cp_params = sum(p.numel() for p in cp_seq.parameters())
        assert cp_params < orig_params


class TestGetEffectiveWeight:
    def test_shape(self):
        conv = nn.Conv2d(64, 128, 3, padding=1)
        cp_seq = create_cp_sequence(conv, 16)
        w_eff = get_effective_weight(cp_seq)
        assert w_eff.shape == (128, 64)

    def test_requires_grad(self):
        conv = nn.Conv2d(64, 128, 3, padding=1)
        cp_seq = create_cp_sequence(conv, 16)
        w_eff = get_effective_weight(cp_seq)
        # Should propagate gradient through computation
        assert w_eff.requires_grad
