# NSA-Diff

Null-Space Absorbing Compression for Diffusion U-Nets. Compresses DDPM denoisers via CP decomposition + conditional null-space loss at skip connections + knowledge distillation.

**Teacher:** `google/ddpm-cifar10-32` (35.7M params) → **Student:** 10.3M params (3.49x compression at rank=0.25)

## Method

Standard NSA (Ozdemir et al.) drives teacher–student activation mismatch into the null space of low-rank student weights: `||W_eff · e||² → 0`. This fails at U-Net skip connections where the error has two independent sources (decoder + encoder skip).

**Conditional NSA** decouples them via stop-gradient:

```
L_cond = ||W_eff · (e_dec + sg(δ_skip))||² + ||W_eff · (sg(e_dec) + δ_skip)||²
```

Each path independently learns to push its error into the null space.

### Full Loss

```
L = L_ε + α·L_null + α_s·L_cond + β·L_KD + λ·L_orth
```

| Component | Description |
|-----------|-------------|
| L_ε | Noise prediction MSE (diffusion objective) |
| L_null | Standard NSA on non-skip conv layers |
| L_cond | Conditional NSA on skip-receiving conv1 layers |
| L_KD | Output-level distillation (noise pred matching) |
| L_orth | Orthogonality regularization on CP output factors |

### Baselines

| Method | Losses Used |
|--------|-------------|
| `lowrank_kd` | L_ε + β·L_KD + λ·L_orth |
| `standard_nsa` | L_ε + α·L_null (all layers) + β·L_KD + λ·L_orth |
| `nsa_diff` | L_ε + α·L_null + α_s·L_cond + β·L_KD + λ·L_orth |
| `fitnets` | L_ε + β·L_FitNets |
| `gramian` | L_ε + β·L_Gram + β·L_KD |

## Setup

```bash
conda env create -f environment.yml
conda activate nsa_diff
pip install -e .
```

Or manually:

```bash
conda create -n nsa_diff python=3.10 -y
conda activate nsa_diff
pip install torch torchvision diffusers accelerate tensorly pytorch-fid wandb pytest
```

## Usage

### Inspect model architecture

```bash
python scripts/inspect_model.py --rank_ratio 0.25
```

### Train

```bash
# NSA-Diff (proposed method)
python scripts/train.py --method nsa_diff --num_steps 100000 --batch_size 64

# Smoke test on MPS
python scripts/train.py --method nsa_diff --num_steps 100 --batch_size 8

# Train on a fixed number of samples (same data for any batch size)
python scripts/train.py --method nsa_diff --total_samples 6400000 --batch_size 64
python scripts/train.py --method nsa_diff --total_samples 6400000 --batch_size 128  # 50k steps instead of 100k

# All 5 baselines
bash scripts/run_baselines.sh
```

Key arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `--method` | `nsa_diff` | One of: lowrank_kd, standard_nsa, nsa_diff, fitnets, gramian |
| `--rank_ratio` | `0.25` | CP rank / min(C_in, C_out). Lower = more compression |
| `--num_steps` | `100000` | Training iterations (ignored if `--total_samples` is set) |
| `--total_samples` | — | If set, steps = total_samples // batch_size so the same number of samples is seen for any batch size |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--alpha` | `1.0` | NSA loss weight |
| `--alpha_s` | `1.0` | Conditional NSA weight |
| `--beta` | `1.0` | KD/FitNets/Gramian weight |
| `--lam` | `0.01` | Orthogonality weight |
| `--use_wandb` | `false` | Enable W&B logging |
| `--device` | auto | Force device (cuda/mps/cpu) |

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/nsa_diff/checkpoint_100000.pt \
    --num_samples 50000 \
    --scheduler ddpm \
    --benchmark
```

### Tests

```bash
# Unit tests (fast, no model download)
pytest tests/test_cp_decompose.py tests/test_losses.py tests/test_hooks.py -v

# Integration tests (requires model download)
pytest tests/test_student_builder.py tests/test_training_step.py -v

# All tests
pytest -v
```

## Project Structure

```
configs/
  __init__.py          ExperimentConfig dataclass + CLI parser
  defaults.py          Per-method default loss weights
src/
  decomposition/
    cp_decompose.py    CP decomposition → 4-conv sequence (pw_in, dw_h, dw_v, pw_out)
    student_builder.py Deep-copy teacher, replace convs with CP sequences
  losses/
    nsa_loss.py        ||W_eff · e||²  on non-skip layers
    conditional_nsa.py Stop-gradient decoupled loss for skip receivers
    distillation.py    KD (MSE), FitNets (hint MSE), Gramian (F·Fᵀ matching)
    orthogonality.py   ||UᵀU - I||_F²  on CP output factors
    composite.py       Per-method loss aggregator
  hooks/
    activation_capture.py  Forward pre-hooks on conv layers
  training/
    trainer.py         Training loop (noise sampling, forward, loss, backward, EMA)
    ema.py             Exponential moving average
    data.py            CIFAR-10 dataloader ([-1,1], flip augmentation)
  evaluation/
    sample.py          DDPM/DDIM generation
    fid.py             FID via pytorch-fid
    benchmark.py       Latency, memory, param count
  utils/
    device.py          CUDA/MPS/CPU auto-detection, AMP config
    unet_inspect.py    Discover compressible layers + skip connection info
    logging_utils.py   W&B + console logger
scripts/
  train.py             CLI entry point
  evaluate.py          Generate + FID evaluation
  inspect_model.py     Print architecture + compression stats
  run_baselines.sh     Launch all 5 methods
tests/
  test_cp_decompose.py, test_student_builder.py, test_losses.py,
  test_hooks.py, test_training_step.py
```

## Compression Configurations

| Rank Ratio | Student Params | Compression | Target |
|------------|---------------|-------------|--------|
| 0.50 | ~14M | ~2.5x | Negligible FID loss |
| 0.25 | ~10M | ~3.5x | < 2 FID points |
| 0.125 | ~5M | ~7x | Graceful degradation |

## Device Support

- **CUDA (RTX 3090):** Full training with AMP. Recommended for full runs.
- **MPS (Apple Silicon):** Works for dev/smoke tests. AMP auto-disabled.
- **CPU:** Functional but slow. Integration tests run here.
