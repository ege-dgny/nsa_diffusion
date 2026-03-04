# Full Experiment Pipeline

Step-by-step instructions to reproduce all results for the paper.

**Hardware assumed:** RTX 3090 (or better) for training. MPS for local dev/debugging.

---

## 0. Environment Setup

```bash
conda create -n nsa_diff python=3.10 -y
conda activate nsa_diff
pip install torch torchvision diffusers accelerate tensorly pytorch-fid wandb pytest
pip install -e .
```

Verify:
```bash
python scripts/inspect_model.py --rank_ratio 0.25
```

Expected output: Teacher 35.7M params, Student ~10.3M, 3.49x compression, 12 skip layers.

---

## 1. Smoke Test (local, ~5 min on MPS)

```bash
python scripts/train.py \
    --method nsa_diff \
    --num_steps 100 \
    --batch_size 8 \
    --rank_ratio 0.25 \
    --use_wandb false
```

Check: loss decreases, no crashes, checkpoint saved to `outputs/nsa_diff/`.

---

## 2. Main Comparison: 5 Methods x 3 Compression Levels

This produces the **main results table** (Table 1 in the paper).

### 2a. Train all configurations

Each run: ~2-3 hrs on 3090 at 50K steps. 15 runs total → ~30-45 hrs sequential, or parallelize across GPUs.

```bash
# On RTX 3090 — run each block on a separate GPU if available

for RATIO in 0.50 0.25 0.125; do
    for METHOD in lowrank_kd standard_nsa nsa_diff fitnets gramian; do
        python scripts/train.py \
            --method $METHOD \
            --rank_ratio $RATIO \
            --num_steps 50000 \
            --batch_size 64 \
            --lr 1e-4 \
            --alpha 0.1 \
            --alpha_s 0.1 \
            --beta 0.1 \
            --lam 0.1 \
            --ema_decay 0.9999 \
            --use_wandb true \
            --wandb_project nsa-diff \
            --run_name "${METHOD}_r${RATIO}" \
            --output_dir outputs \
            --save_every 10000 \
            --eval_every 10000
    done
done
```

**Output:** Checkpoints at `outputs/{method}_r{ratio}/checkpoint_50000.pt`

### 2b. Generate 50K samples per model

```bash
for RATIO in 0.50 0.25 0.125; do
    for METHOD in lowrank_kd standard_nsa nsa_diff fitnets gramian; do
        python scripts/evaluate.py \
            --checkpoint outputs/${METHOD}_r${RATIO}/checkpoint_50000.pt \
            --num_samples 50000 \
            --batch_size 256 \
            --scheduler ddpm \
            --num_steps 1000 \
            --output_dir eval_samples/${METHOD}_r${RATIO} \
            --benchmark
    done
done
```

Each generation run: ~40 min on 3090 (1000-step DDPM). 15 runs → ~10 hrs. Parallelize across GPUs.

### 2c. Generate teacher samples (baseline FID reference)

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from diffusers import DDPMPipeline
from src.evaluation.sample import generate_samples, save_samples
import torch

pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
teacher = pipe.unet.to('cuda').eval()

samples = generate_samples(teacher, num_samples=50000, device='cuda', batch_size=256)
save_samples(samples, 'eval_samples/teacher')
print('Teacher samples saved.')
"
```

### 2d. Prepare CIFAR-10 reference images

```bash
python -c "
import torchvision, os
from torchvision.utils import save_image

ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
os.makedirs('eval_samples/cifar10_ref', exist_ok=True)
for i, (img, _) in enumerate(ds):
    torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    save_image(torchvision.transforms.functional.to_tensor(img), f'eval_samples/cifar10_ref/{i:06d}.png')
print(f'Saved {len(ds)} reference images.')
"
```

### 2e. Compute FID for every model

```bash
for RATIO in 0.50 0.25 0.125; do
    for METHOD in lowrank_kd standard_nsa nsa_diff fitnets gramian; do
        echo "=== ${METHOD}_r${RATIO} ==="
        python -m pytorch_fid eval_samples/${METHOD}_r${RATIO} eval_samples/cifar10_ref
    done
done

# Teacher FID
echo "=== teacher ==="
python -m pytorch_fid eval_samples/teacher eval_samples/cifar10_ref
```

Record all FID values into Table 1.

---

## 3. Ablation A: Skip Connection Handling (Table 2)

Fix rank_ratio=0.25. Test 5 variants of skip-layer loss.

```bash
# A1: KD only at skips (= lowrank_kd, already trained)
# Already have: outputs/lowrank_kd_r0.25/

# A2: Standard NSA at skips (= standard_nsa, already trained)
# Already have: outputs/standard_nsa_r0.25/

# A3: Conditional NSA, decoder term only
python scripts/train.py \
    --method nsa_diff --rank_ratio 0.25 --num_steps 50000 --batch_size 64 \
    --alpha 0.1 --alpha_s 0.0 --beta 0.1 --lam 0.1 \
    --run_name ablation_A3_dec_only --use_wandb true

# A4: Conditional NSA, encoder term only
# (Requires code change: swap stop-gradient targets in conditional_nsa.py)
# For now, train same as A3 but note the code modification needed.

# A5: Full conditional NSA (= nsa_diff, already trained)
# Already have: outputs/nsa_diff_r0.25/
```

Generate + FID for A3 (and A4 if implemented):
```bash
python scripts/evaluate.py --checkpoint outputs/ablation_A3_dec_only/checkpoint_50000.pt \
    --num_samples 50000 --output_dir eval_samples/ablation_A3 --benchmark
python -m pytorch_fid eval_samples/ablation_A3 eval_samples/cifar10_ref
```

---

## 4. Ablation B: alpha_s Sensitivity (Figure 2)

Fix rank_ratio=0.25. Sweep alpha_s.

```bash
for ALPHA_S in 0.0 0.01 0.05 0.1 0.5 1.0; do
    python scripts/train.py \
        --method nsa_diff --rank_ratio 0.25 --num_steps 50000 --batch_size 64 \
        --alpha 0.1 --alpha_s $ALPHA_S --beta 0.1 --lam 0.1 \
        --run_name "ablation_B_as${ALPHA_S}" --use_wandb true
done
```

Generate + FID for each:
```bash
for ALPHA_S in 0.0 0.01 0.05 0.1 0.5 1.0; do
    python scripts/evaluate.py \
        --checkpoint "outputs/ablation_B_as${ALPHA_S}/checkpoint_50000.pt" \
        --num_samples 50000 \
        --output_dir "eval_samples/ablation_B_as${ALPHA_S}" --benchmark
    python -m pytorch_fid "eval_samples/ablation_B_as${ALPHA_S}" eval_samples/cifar10_ref
done
```

Plot: FID (y-axis) vs alpha_s (x-axis). Expect inverted-U, optimal ~0.05-0.1.

---

## 5. Benchmarks (Table 3)

Run on target hardware (3090 or Jetson Orin).

```bash
# Teacher benchmark
python -c "
import sys; sys.path.insert(0, '.')
import torch
from diffusers import DDPMPipeline
from src.evaluation.benchmark import benchmark_model

pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
teacher = pipe.unet
device = torch.device('cuda')
r = benchmark_model(teacher, device)
print(f'Teacher: {r.total_params:,} params, {r.latency_ms:.2f}ms/step, {r.peak_memory_mb:.1f}MB')
"

# Student benchmarks
for RATIO in 0.50 0.25 0.125; do
    python scripts/evaluate.py \
        --checkpoint "outputs/nsa_diff_r${RATIO}/checkpoint_50000.pt" \
        --num_samples 0 \
        --benchmark
done
```

Record: params, compression ratio, latency (ms/step), peak memory (MB).

---

## 6. Qualitative Samples (Figure 3)

Generate a grid of samples from teacher and each method for visual comparison.

```bash
for METHOD in teacher lowrank_kd standard_nsa nsa_diff fitnets gramian; do
    if [ "$METHOD" = "teacher" ]; then
        # Use teacher samples already generated
        continue
    fi
    python scripts/evaluate.py \
        --checkpoint "outputs/${METHOD}_r0.25/checkpoint_50000.pt" \
        --num_samples 64 \
        --output_dir "figure_samples/${METHOD}" \
        --scheduler ddpm --num_steps 1000
done
```

Then create grid figure from the first 64 images of each method.

---

## 7. Collect Results

### Table 1: Main Results (5 methods x 3 compressions)

| Model | rho | Params | Compression | FID |
|-------|-----|--------|-------------|-----|
| Teacher | - | 35.7M | 1x | ? |
| lowrank_kd | 0.50 | ? | ? | ? |
| standard_nsa | 0.50 | ? | ? | ? |
| nsa_diff | 0.50 | ? | ? | ? |
| fitnets | 0.50 | ? | ? | ? |
| gramian | 0.50 | ? | ? | ? |
| lowrank_kd | 0.25 | ? | ? | ? |
| standard_nsa | 0.25 | ? | ? | ? |
| nsa_diff | 0.25 | ? | ? | ? |
| fitnets | 0.25 | ? | ? | ? |
| gramian | 0.25 | ? | ? | ? |
| lowrank_kd | 0.125 | ? | ? | ? |
| standard_nsa | 0.125 | ? | ? | ? |
| nsa_diff | 0.125 | ? | ? | ? |
| fitnets | 0.125 | ? | ? | ? |
| gramian | 0.125 | ? | ? | ? |

### Table 2: Ablation A — Skip Connection Handling (rho=0.25)

| Variant | Description | FID |
|---------|-------------|-----|
| A1 | KD only at skips | ? |
| A2 | Standard NSA at skips | ? |
| A3 | Conditional (decoder only) | ? |
| A4 | Conditional (encoder only) | ? |
| A5 | Conditional (both) | ? |

### Table 3: Efficiency Benchmarks

| Model | Params | Compression | Latency (ms) | Memory (MB) |
|-------|--------|-------------|-------------|-------------|
| Teacher | 35.7M | 1x | ? | ? |
| NSA-Diff r=0.50 | ? | ~2.5x | ? | ? |
| NSA-Diff r=0.25 | ? | ~3.5x | ? | ? |
| NSA-Diff r=0.125 | ? | ~7x | ? | ? |

### Figure 2: alpha_s sensitivity curve

Plot from ablation B data.

### Figure 3: Qualitative sample grids

Cherry-pick from `figure_samples/` directories.

---

## Estimated Compute Budget

| Task | Time (1x 3090) |
|------|----------------|
| Training: 15 main runs (50K steps each) | ~30-45 hrs |
| Training: 6 alpha_s ablation runs | ~12-18 hrs |
| Training: 2 ablation A extra runs | ~4-6 hrs |
| Generation: 50K samples x ~23 models | ~15 hrs |
| FID computation: ~23 evaluations | ~1 hr |
| **Total** | **~60-85 hrs sequential** |

With 2 GPUs: ~35-45 hrs. With 4 GPUs: ~20-25 hrs.

---

## Troubleshooting

- **OOM on 3090:** reduce `--batch_size` to 32
- **MPS segfaults on CP decomposition:** already handled (uses `init='random'`)
- **NaN loss:** reduce `--lr` to 5e-5, check `--grad_clip_norm 1.0`
- **W&B not logging:** ensure `wandb login` has been run
- **CIFAR-10 download fails:** manually download to `./data/cifar-10-batches-py/`
