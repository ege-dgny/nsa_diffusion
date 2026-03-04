# NSA-Diff: Null-Space Absorbing Compression for Diffusion U-Nets

## Research Blueprint — CVPR 2026 EDGE / ECV Workshop Submission

---

## 1. Motivation and Problem Statement

### 1.1 Context

NSA-Net (Ozdemir et al., APSIPA ASC 2025) demonstrated that projecting teacher–student activation mismatches onto the null space of low-rank student weight matrices yields a powerful regularizer for network compression. The method achieved up to 90% parameter reduction with accuracy gains on CNNs (LeNet-5, VGG-style, AlexNet) across MNIST, CIFAR-10, and CIFAR-100.

However, NSA-Net's mathematical framework assumes that each layer's input passes through a single linear map before a pointwise nonlinearity. This assumption breaks in two important architectural patterns:

- **Residual connections + LayerNorm** (Transformers): empirically confirmed in NSA-ViT experiments, where the null-space loss consistently hurt performance.
- **Skip connections** (U-Nets): the decoder receives inputs from two distinct computational paths (upsampled decoder features and encoder skip features), producing a compound mismatch that cannot be independently absorbed.

Diffusion models based on U-Net architectures (DDPM, Stable Diffusion) are the primary target for efficient on-device generation. Their convolutional backbone is compatible with NSA's CP decomposition, but the skip connections require a mathematical extension.

### 1.2 Research Question

Can we extend the null-space absorption framework to handle multi-path inputs at U-Net skip connections, enabling principled low-rank compression of diffusion model denoisers?

### 1.3 Contribution Summary

1. **Conditional Null-Space Loss**: a reformulation of the NSA loss that decomposes the compound mismatch at skip connections into independently manageable components via stop-gradient decoupling.
2. **Selective NSA**: a strategy that applies null-space absorption only to mathematically compatible submodules (convolutional blocks), with standard distillation elsewhere.
3. **On-device validation**: end-to-end compression of DDPM U-Nets with FID evaluation and latency/memory benchmarks on NVIDIA Jetson Orin.

---

## 2. Mathematical Framework

### 2.1 Review: Standard NSA Loss

For a student layer with low-rank weight $\hat{W}_i = U_i V_i^\top$, $U_i \in \mathbb{R}^{m \times r}$, $V_i \in \mathbb{R}^{n \times r}$, $r < \min(m,n)$, the null-space absorbing loss is:

$$\mathcal{L}_{\text{null}} = \sum_i \left\| \hat{W}_i (h_i - \hat{h}_i) \right\|_2^2$$

where $h_i, \hat{h}_i$ are teacher and student activations at layer $i$. Letting $e_i = h_i - \hat{h}_i$ and decomposing $e_i = e_i^{\parallel} + e_i^{\perp}$ into row-space and null-space components:

$$\mathcal{L}_{\text{null}} = \left\| \hat{W}_i e_i^{\parallel} \right\|_2^2 \implies e_i^{\parallel} \to 0$$

The loss drives the row-space component of the error to zero, allowing arbitrary mismatches in the null space (dimension $n - r$), which by definition do not affect downstream computation through $\hat{W}_i$.

**Key requirement**: the input to the next layer must pass entirely through $\hat{W}_i$ — no bypassing paths.

### 2.2 The Skip Connection Problem

In a U-Net decoder block at resolution level $j$, the input is formed by combining upsampled decoder features with encoder skip features. Two variants exist:

**Concatenation skip** (standard U-Net):

$$h_j = \sigma\left( W_j \begin{bmatrix} h_{j-1} \\ s_i \end{bmatrix} + b_j \right)$$

where $W_j \in \mathbb{R}^{m \times (n_1 + n_2)}$, $h_{j-1} \in \mathbb{R}^{n_1}$ is the decoder path input, and $s_i \in \mathbb{R}^{n_2}$ is the encoder skip.

**Additive skip** (some modern U-Nets):

$$h_j = \sigma\left( W_j (h_{j-1} + s_i) + b_j \right)$$

In both cases, the total teacher–student mismatch at the input to layer $j$ has two independent sources:

$$e_{\text{total}} = \begin{bmatrix} e_{j-1} \\ \delta_i \end{bmatrix} \quad \text{(concat)} \qquad \text{or} \qquad e_{\text{total}} = e_{j-1} + \delta_i \quad \text{(additive)}$$

where $e_{j-1} = h_{j-1}^T - \hat{h}_{j-1}$ is the decoder mismatch and $\delta_i = s_i^T - \hat{s}_i$ is the encoder skip mismatch.

**Why standard NSA fails**: applying $\|\hat{W}_j \, e_{\text{total}}\|^2$ conflates two errors from different computational graphs. The gradient signal from this loss simultaneously tries to modify encoder parameters (through $\delta_i$) and decoder parameters (through $e_{j-1}$) with no mechanism to ensure that each path's error is individually driven to the null space. Worse, a reduction in $\hat{W}_j e_{j-1}$ could be offset by an increase in $\hat{W}_j \delta_i$ — the loss can decrease without either error actually entering the null space.

### 2.3 Conditional Null-Space Loss (Proposed)

We propose decoupling the two error sources via stop-gradient operations and applying the null-space loss to the total mismatch from each path's perspective.

**Definition.** For a decoder layer $j$ receiving a skip connection, the conditional null-space loss is:

$$\mathcal{L}_{\text{cond-null}}^{(j)} = \underbrace{\left\| \hat{W}_j \left( e_{j-1} + \text{sg}(\delta_i) \right) \right\|^2}_{\text{decoder term}} + \underbrace{\left\| \hat{W}_j \left( \text{sg}(e_{j-1}) + \delta_i \right) \right\|^2}_{\text{encoder term}}$$

where $\text{sg}(\cdot)$ denotes stop-gradient.

**Interpretation of the decoder term**: with $\delta_i$ frozen, the loss asks the decoder to choose $e_{j-1}$ such that the total mismatch's row-space component is minimized. The decoder is free to place $e_{j-1}$ anywhere in the coset $\{v : \hat{W}_j v = -\hat{W}_j \,\text{sg}(\delta_i)\}$. Equivalently:

$$e_{j-1} \in -\text{sg}(\delta_i) + \ker(\hat{W}_j)$$

The decoder compensates for the skip error's row-space component while retaining full freedom in the null-space directions.

**Interpretation of the encoder term**: symmetrically, the encoder adjusts its activations to compensate for the decoder error's row-space component.

**Why stop-gradient is essential**: without it, both terms simplify to $\|\hat{W}_j(e_{j-1} + \delta_i)\|^2$ with gradients flowing through both paths simultaneously. The decoupling ensures that each path independently learns to compensate, preventing the degenerate solution where one path's error masks the other's.

**Gradient derivation for the decoder term** (concatenation case, writing $\hat{W}_j = [\hat{W}_j^{(d)} \mid \hat{W}_j^{(s)}]$):

$$\frac{\partial \mathcal{L}_{\text{dec}}}{\partial \hat{h}_{j-1}} = -2 \, (\hat{W}_j^{(d)})^\top \hat{W}_j \begin{bmatrix} e_{j-1} \\ \text{sg}(\delta_i) \end{bmatrix}$$

The gradient pushes the decoder activation to reduce the row-space projection of the total error, treating the skip error as a fixed offset.

**Gradient derivation for the encoder term**:

$$\frac{\partial \mathcal{L}_{\text{enc}}}{\partial \hat{s}_i} = -2 \, (\hat{W}_j^{(s)})^\top \hat{W}_j \begin{bmatrix} \text{sg}(e_{j-1}) \\ \delta_i \end{bmatrix}$$

### 2.4 Relationship to Standard NSA

At non-skip layers, $\delta_i = 0$ and the conditional loss reduces exactly to the standard NSA loss:

$$\mathcal{L}_{\text{cond-null}}^{(j)} = \left\| \hat{W}_j e_{j-1} \right\|^2 + \left\| \hat{W}_j e_{j-1} \right\|^2 = 2 \left\| \hat{W}_j e_{j-1} \right\|^2$$

which is just a scaled version of the original. So the framework is a strict generalization.

### 2.5 Timestep Conditioning

DDPM U-Nets apply timestep embeddings via addition: $h' = h + \text{MLP}(t)$. Since both teacher and student receive the same timestep, the embedding itself doesn't contribute to the mismatch (assuming we share the timestep MLP or distill it separately). The null-space loss is applied after the timestep addition, so the mismatch $e_i$ already incorporates any differences caused by different internal states.

For FiLM-style conditioning ($h' = \gamma(t) \odot h + \beta(t)$), the mismatch becomes:

$$e_i' = \gamma(t) \odot e_i$$

The element-wise scaling rotates the error vector per-timestep. During training, we average the null-space loss over a batch of uniformly sampled timesteps, which naturally integrates over these rotations. No special handling is required.

### 2.6 Handling GroupNorm

DDPM U-Nets use GroupNorm before convolutions within ResNet blocks. GroupNorm is not a linear operation — it normalizes by per-group mean and variance, which depend on the activation magnitudes. However, GroupNorm operates *within* each ResNet block, and the null-space loss is applied at the *output* of each block (after the full ResNet forward pass). Therefore, GroupNorm affects the value of $\hat{h}_i$ and thus $e_i$, but the null-space absorption argument still holds: the loss $\|\hat{W}_j e_i\|^2$ drives $e_i$'s row-space component to zero regardless of how $e_i$ was computed internally.

This is different from the ViT case where LayerNorm sits *between* layers connected by a residual stream, making the null-space of one layer irrelevant to what the next layer sees.

### 2.7 Full Loss Function

The complete fine-tuning loss for diffusion model compression:

$$\mathcal{L} = \mathcal{L}_{\epsilon} + \alpha \, \mathcal{L}_{\text{null}} + \alpha_s \, \mathcal{L}_{\text{cond-null}} + \beta \, \mathcal{L}_{\text{KD}} + \lambda \, \mathcal{L}_{\text{orth}}$$

where:

- $\mathcal{L}_{\epsilon} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$: standard noise-prediction MSE (diffusion training objective)
- $\mathcal{L}_{\text{null}}$: standard NSA loss on non-skip conv layers
- $\mathcal{L}_{\text{cond-null}}$: conditional null-space loss on skip-receiving decoder layers
- $\mathcal{L}_{\text{KD}} = \| \epsilon_{\text{teacher}}(x_t, t) - \epsilon_\theta(x_t, t) \|^2$: output-level distillation (noise prediction matching)
- $\mathcal{L}_{\text{orth}} = \sum_i \| U_i^\top U_i - I \|_F^2$: orthonormality regularization on left factors

Note: for diffusion models, output-level KD naturally uses MSE on predicted noise rather than KL divergence on logits. WD is not applicable since the output is a continuous vector, not a distribution over classes.

---

## 3. Architecture and Decomposition Details

### 3.1 Target Architecture: DDPM U-Net

The DDPM U-Net (Ho et al., 2020) for 32×32 generation (CIFAR-10) has the following structure:

```
Input: x_t ∈ R^{3×32×32}, t ∈ {1,...,T}

Encoder:
  DownBlock1: 2× ResNetBlock(128→128), Downsample → 16×16
  DownBlock2: 2× ResNetBlock(128→256), Downsample → 8×8
  DownBlock3: 2× ResNetBlock(256→256), SelfAttention(256)

MidBlock:
  ResNetBlock(256→256), SelfAttention(256), ResNetBlock(256→256)

Decoder:
  UpBlock3: 2× ResNetBlock(512→256) [cat with DownBlock3 skip], SelfAttention(256)
  UpBlock2: 2× ResNetBlock(512→256) [cat with DownBlock2 skip], Upsample → 16×16
  UpBlock1: 2× ResNetBlock(384→128) [cat with DownBlock1 skip], Upsample → 32×32

Output: Conv(128→3)
```

Total parameters: ~35.7M (varies by exact configuration).

**Skip connections**: DownBlock $k$ sends its output to UpBlock $k$ via channel concatenation. The first conv in each UpBlock ResNetBlock has input channels = decoder_channels + skip_channels.

### 3.2 ResNet Block Internals

Each ResNetBlock contains:

```
input x
  → GroupNorm → SiLU → Conv3×3 (in_ch → out_ch)     [conv1]
  → GroupNorm → SiLU → Dropout → Conv3×3 (out_ch → out_ch)  [conv2]
  → + shortcut(x)                                      [residual]
```

plus a timestep embedding projection (linear → SiLU → linear, added after conv1).

**Where NSA applies**: to `conv1` and `conv2` within each ResNetBlock. These are 3×3 convolutions, compressed via CP decomposition. The shortcut connection within a ResNet block is a 1×1 conv when channel dimensions change — this is also compressed but with standard SVD since it's spatially trivial.

**Where the conditional null-space loss applies**: to `conv1` of the first ResNetBlock in each UpBlock, since this is the layer that directly processes the concatenated [decoder; skip] input.

### 3.3 CP Decomposition for 3×3 Convolutions

A 3×3 conv kernel $\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 3 \times 3}$ is decomposed as:

$$\hat{\mathcal{W}} = \sum_{r=1}^{R} \lambda_r \, a_r^{(1)} \otimes a_r^{(2)} \otimes a_r^{(3)} \otimes a_r^{(4)}$$

where $a_r^{(1)} \in \mathbb{R}^{C_{\text{out}}}$, $a_r^{(2)} \in \mathbb{R}^{C_{\text{in}}}$, $a_r^{(3)} \in \mathbb{R}^3$, $a_r^{(4)} \in \mathbb{R}^3$.

This decomposes the single 4D convolution into a sequence:

1. **Pointwise conv**: $1 \times 1$ with $C_{\text{in}} \to R$ channels (factor $a^{(2)}$)
2. **Depthwise separable horizontal conv**: $1 \times 3$ with $R$ channels (factor $a^{(4)}$)
3. **Depthwise separable vertical conv**: $3 \times 1$ with $R$ channels (factor $a^{(3)}$)
4. **Pointwise conv**: $1 \times 1$ with $R \to C_{\text{out}}$ channels (factor $a^{(1)}$, scaled by $\lambda_r$)

**Parameter count**: original = $C_{\text{out}} \cdot C_{\text{in}} \cdot 9$, decomposed = $R(C_{\text{out}} + C_{\text{in}} + 3 + 3)$.

**Compression ratio**: $\frac{9 \, C_{\text{out}} \, C_{\text{in}}}{R(C_{\text{out}} + C_{\text{in}} + 6)}$.

### 3.4 Null-Space Structure After CP Decomposition

After CP decomposition, the effective linear map for a conv layer (ignoring spatial dimensions for clarity) can be viewed as $\hat{W} = A^{(1)} \text{diag}(\lambda) (A^{(2)})^\top$ where $A^{(1)} \in \mathbb{R}^{C_{\text{out}} \times R}$ and $A^{(2)} \in \mathbb{R}^{C_{\text{in}} \times R}$.

The null-space of this map has dimension $C_{\text{in}} - R$ (in the channel dimension). This is the subspace where activation mismatches can be freely absorbed. For a 256→256 conv compressed to rank $R=64$, the null-space has dimension 192 — substantial freedom for the student.

### 3.5 Concrete Compression Configurations

| Config | Rank Ratio | Approx. Params | Compression | Target |
|--------|-----------|----------------|-------------|--------|
| **Mild** | $R/\min(C_{\text{in}}, C_{\text{out}}) = 0.5$ | ~14M | ~2.5× | Negligible FID loss |
| **Moderate** | $R/\min(C_{\text{in}}, C_{\text{out}}) = 0.25$ | ~7M | ~5× | < 2 FID points |
| **Aggressive** | $R/\min(C_{\text{in}}, C_{\text{out}}) = 0.125$ | ~3.5M | ~10× | Graceful degradation |

Rank selection per layer: set $R_l = \lceil \rho \cdot \min(C_{\text{in}}^{(l)}, C_{\text{out}}^{(l)}) \rceil$ where $\rho$ is the rank ratio. Alternatively, use energy-based selection: choose $R_l$ to retain fraction $\eta$ of the Frobenius norm energy from the singular value spectrum of the unfolded kernel.

### 3.6 Selective NSA Map

| Module | Decomposition | Loss | Rationale |
|--------|--------------|------|-----------|
| ResNetBlock conv1 (non-skip) | CP | Standard NSA | Sequential conv, no multi-path |
| ResNetBlock conv2 | CP | Standard NSA | Sequential conv within block |
| ResNetBlock conv1 (skip-receiving) | CP | **Conditional NSA** | Concatenated decoder + skip input |
| ResNetBlock shortcut (1×1) | SVD | Standard NSA | Channel projection, no spatial dims |
| Self-attention Q, K, V projections | SVD | KD only | Attention is not a sequential linear map in the NSA sense |
| Self-attention output projection | SVD | KD only | Same |
| Timestep embedding MLP | SVD | KD only | Small, shared between teacher/student |
| GroupNorm | Not compressed | — | Affine params are negligible |
| Final output conv | Not compressed | — | Only 3 output channels, negligible |

---

## 4. Implementation

### 4.1 Codebase

Build on top of the `diffusers` library (Hugging Face). The pretrained DDPM-CIFAR10 model is available as `google/ddpm-cifar10-32`.

**Dependencies**: `torch`, `diffusers`, `accelerate`, `tensorly` (for CP decomposition), `pytorch-fid` (for FID computation), `wandb` (logging).

### 4.2 Step 1: Load Pretrained Teacher

```python
from diffusers import DDPMPipeline, DDPMScheduler

pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
teacher_unet = pipeline.unet
teacher_unet.eval()
for p in teacher_unet.parameters():
    p.requires_grad_(False)
```

### 4.3 Step 2: CP Decomposition of Conv Layers

```python
import tensorly as tl
from tensorly.decomposition import parafac

def cp_decompose_conv(conv_layer, rank):
    """Decompose a Conv2d into CP factors."""
    W = conv_layer.weight.data  # (C_out, C_in, k1, k2)
    tl.set_backend('pytorch')

    # CP decomposition
    weights, factors = parafac(W, rank=rank, init='svd', n_iter_max=100)
    # factors: [a^(1) ∈ R^{C_out×R}, a^(2) ∈ R^{C_in×R},
    #           a^(3) ∈ R^{k1×R}, a^(4) ∈ R^{k2×R}]

    return weights, factors

def replace_conv_with_cp(conv, rank):
    """Replace a single Conv2d with a sequence of 4 convolutions."""
    C_out, C_in, k1, k2 = conv.weight.shape
    weights, (f_out, f_in, f_h, f_w) = cp_decompose_conv(conv, rank)

    # Absorb weights into f_out
    f_out = f_out * weights[None, :]

    # 1. Pointwise: C_in → R
    pw1 = nn.Conv2d(C_in, rank, 1, bias=False)
    pw1.weight.data = f_in.t().unsqueeze(-1).unsqueeze(-1)

    # 2. Depthwise horizontal: 1×k2
    dw_h = nn.Conv2d(rank, rank, (1, k2), padding=(0, k2//2),
                     groups=rank, bias=False)
    dw_h.weight.data = f_w.t().unsqueeze(1).unsqueeze(2)

    # 3. Depthwise vertical: k1×1
    dw_v = nn.Conv2d(rank, rank, (k1, 1), padding=(k1//2, 0),
                     groups=rank, bias=False)
    dw_v.weight.data = f_h.t().unsqueeze(1).unsqueeze(-1)

    # 4. Pointwise: R → C_out
    pw2 = nn.Conv2d(rank, C_out, 1, bias=(conv.bias is not None))
    pw2.weight.data = f_out.unsqueeze(-1).unsqueeze(-1)
    if conv.bias is not None:
        pw2.bias.data = conv.bias.data.clone()

    return nn.Sequential(pw1, dw_h, dw_v, pw2)
```

### 4.4 Step 3: Build Student U-Net

```python
def build_student(teacher_unet, rank_ratio=0.25):
    """Create student by replacing conv layers with CP-decomposed versions."""
    student_unet = copy.deepcopy(teacher_unet)

    skip_receiving_layers = []  # track for conditional NSA

    for name, module in student_unet.named_modules():
        if isinstance(module, ResNetBlock):
            C_in, C_out = module.conv1.in_channels, module.conv1.out_channels
            rank = max(1, int(rank_ratio * min(C_in, C_out)))

            # Check if this block receives a skip connection
            is_skip_receiver = is_decoder_first_block(name)

            module.conv1_decomposed = replace_conv_with_cp(module.conv1, rank)
            module.conv2_decomposed = replace_conv_with_cp(module.conv2, rank)

            if is_skip_receiver:
                skip_receiving_layers.append(name)

    return student_unet, skip_receiving_layers
```

### 4.5 Step 4: Hook-Based Activation Capture

```python
class ActivationCapture:
    """Register hooks to capture intermediate activations."""
    def __init__(self, model, layer_names):
        self.activations = {}
        self.hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self.activations[name] = {
                'input': input[0].detach() if isinstance(input, tuple) else input.detach(),
                'output': output.detach() if not isinstance(output, tuple) else output[0].detach()
            }
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
```

### 4.6 Step 5: Loss Computation

```python
def compute_nsa_loss(student_acts, teacher_acts, student_weights, skip_layers):
    """Compute standard + conditional null-space loss."""
    L_null = 0.0
    L_cond = 0.0

    for layer_name in student_acts:
        h_t = teacher_acts[layer_name]['input']    # teacher input to this layer
        h_s = student_acts[layer_name]['input']     # student input to this layer
        W_hat = get_effective_weight(student_weights, layer_name)

        if layer_name in skip_layers:
            # Split into decoder and skip components
            C_dec = h_t.shape[1] // 2   # after concat, first half is decoder
            e_dec = (h_t[:, :C_dec] - h_s[:, :C_dec])
            delta_skip = (h_t[:, C_dec:] - h_s[:, C_dec:])

            # Decoder term: stop-grad on skip error
            e_total_dec = torch.cat([e_dec, delta_skip.detach()], dim=1)
            # Reshape for matrix multiply: flatten spatial dims
            B, C, H, W_sp = e_total_dec.shape
            e_flat = e_total_dec.reshape(B, C, -1)  # (B, C_in, H*W)

            # Apply W_hat to each spatial location
            # W_hat acts on channel dim: (C_out, C_in)
            L_dec = torch.mean(torch.sum(
                (torch.einsum('oi,bix->box', W_hat, e_flat))**2,
                dim=1
            ))

            # Encoder term: stop-grad on decoder error
            e_total_enc = torch.cat([e_dec.detach(), delta_skip], dim=1)
            e_flat_enc = e_total_enc.reshape(B, C, -1)
            L_enc = torch.mean(torch.sum(
                (torch.einsum('oi,bix->box', W_hat, e_flat_enc))**2,
                dim=1
            ))

            L_cond += L_dec + L_enc
        else:
            # Standard NSA
            e = h_t - h_s
            B, C, H, W_sp = e.shape
            e_flat = e.reshape(B, C, -1)
            L_null += torch.mean(torch.sum(
                (torch.einsum('oi,bix->box', W_hat, e_flat))**2,
                dim=1
            ))

    return L_null, L_cond


def compute_orth_loss(student_unet):
    """Orthonormality regularization on left CP factors."""
    L_orth = 0.0
    for name, module in student_unet.named_modules():
        if hasattr(module, 'conv1_decomposed'):
            # The last pointwise conv's weight is U (left factor)
            U = module.conv1_decomposed[-1].weight.squeeze()  # (C_out, R)
            L_orth += torch.norm(U.T @ U - torch.eye(U.shape[1], device=U.device))**2
            U2 = module.conv2_decomposed[-1].weight.squeeze()
            L_orth += torch.norm(U2.T @ U2 - torch.eye(U2.shape[1], device=U2.device))**2
    return L_orth
```

### 4.7 Step 6: Training Loop

```python
def train_step(student, teacher, x_0, noise_scheduler, optimizer,
               teacher_hooks, student_hooks, skip_layers,
               alpha=0.1, alpha_s=0.1, beta=0.1, lam=0.1):
    """Single training step for NSA-Diff distillation."""
    # Sample timestep and noise
    t = torch.randint(0, noise_scheduler.num_train_timesteps,
                      (x_0.shape[0],), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = noise_scheduler.add_noise(x_0, noise, t)

    # Teacher forward (no grad)
    with torch.no_grad():
        eps_teacher = teacher(x_t, t).sample
        teacher_acts = teacher_hooks.activations.copy()

    # Student forward
    eps_student = student(x_t, t).sample
    student_acts = student_hooks.activations.copy()

    # Losses
    L_eps = F.mse_loss(eps_student, noise)                       # noise prediction
    L_kd = F.mse_loss(eps_student, eps_teacher)                  # output distillation
    L_null, L_cond = compute_nsa_loss(
        student_acts, teacher_acts, student, skip_layers
    )
    L_orth = compute_orth_loss(student)

    # Total loss
    loss = L_eps + alpha * L_null + alpha_s * L_cond + beta * L_kd + lam * L_orth

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'total': loss.item(),
        'eps': L_eps.item(),
        'kd': L_kd.item(),
        'null': L_null.item(),
        'cond_null': L_cond.item(),
        'orth': L_orth.item()
    }
```

### 4.8 Step 7: Full Training Script Skeleton

```python
# Config
RANK_RATIO = 0.25        # moderate compression
NUM_STEPS = 50_000
BATCH_SIZE = 64
LR = 1e-4
ALPHA = 0.1              # standard NSA weight
ALPHA_S = 0.1            # conditional NSA weight
BETA = 0.1               # KD weight
LAMBDA = 0.1             # orthonormality weight
LR_STEP = 10_000
LR_GAMMA = 0.7

# Setup
teacher = load_teacher("google/ddpm-cifar10-32")
student, skip_layers = build_student(teacher, RANK_RATIO)
optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_GAMMA)
dataloader = get_cifar10_dataloader(BATCH_SIZE)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Register hooks on target layers
target_layers = get_resnet_block_names(teacher)
teacher_hooks = ActivationCapture(teacher, target_layers)
student_hooks = ActivationCapture(student, target_layers)

# Training
step = 0
while step < NUM_STEPS:
    for batch in dataloader:
        x_0 = batch[0].to(device)
        metrics = train_step(student, teacher, x_0, noise_scheduler,
                            optimizer, teacher_hooks, student_hooks,
                            skip_layers, ALPHA, ALPHA_S, BETA, LAMBDA)
        scheduler.step()
        step += 1
        if step % 1000 == 0:
            log(metrics)
        if step >= NUM_STEPS:
            break

# Save
save_student(student, f"nsa_diff_r{RANK_RATIO}.pt")
```

### 4.9 Extracting the Effective Weight Matrix

For the null-space loss, we need the effective weight matrix $\hat{W}$ from the CP-decomposed sequence. For a decomposed conv with factors $(A^{(1)}, A^{(2)}, A^{(3)}, A^{(4)})$:

```python
def get_effective_weight(decomposed_conv):
    """Reconstruct effective channel-mixing matrix from CP factors.

    For null-space loss, we only need the channel-dimension action.
    The spatial kernels contribute to *how* the input is mixed but
    the null-space of the channel mixing is what matters for absorption.

    Effective channel matrix: W_eff = A^(1) @ diag(lambda) @ A^(2).T
    """
    pw_in = decomposed_conv[0].weight.squeeze()    # (R, C_in)
    pw_out = decomposed_conv[-1].weight.squeeze()   # (C_out, R)
    W_eff = pw_out @ pw_in   # (C_out, C_in)
    return W_eff
```

Note: this is an approximation — the true effective weight includes the spatial kernels, which form a spatially-varying linear map. However, the null-space of the channel mixing matrix is a subspace of the full null-space (any vector in $\ker(A^{(1)} A^{(2)\top})$ is also in the null space of the full convolution). So using $W_{\text{eff}}$ for the loss is conservative — we penalize mismatches that survive the channel mixing, which is the dominant compression bottleneck.

---

## 5. Experimental Setup

### 5.1 Teacher Models

| Teacher | Dataset | Resolution | Params | Pretrained Source |
|---------|---------|------------|--------|-------------------|
| DDPM U-Net | CIFAR-10 | 32×32 | 35.7M | `google/ddpm-cifar10-32` |
| DDPM U-Net | CelebA | 64×64 | 78.5M | `google/ddpm-celebahq-256` (downscale) or train |

Primary experiments on CIFAR-10 (fastest iteration). CelebA-64 as secondary validation if time permits.

### 5.2 Student Configurations

| Student | Rank Ratio $\rho$ | Approx. Params | Compression | Null-Space Dim |
|---------|-------------------|----------------|-------------|----------------|
| NSA-Diff-50 | 0.50 | ~14M | ~2.5× | $0.5 \cdot C_{\text{in}}$ per layer |
| NSA-Diff-25 | 0.25 | ~7M | ~5× | $0.75 \cdot C_{\text{in}}$ per layer |
| NSA-Diff-12 | 0.125 | ~3.5M | ~10× | $0.875 \cdot C_{\text{in}}$ per layer |

### 5.3 Baselines

| Method | Description | What It Tests |
|--------|-------------|---------------|
| **Low-rank + KD** | CP decomposition + output distillation only ($\alpha = \alpha_s = 0$) | Value of null-space loss |
| **Standard NSA (no skip handling)** | NSA loss on all layers including skip-receivers, no stop-gradient decoupling | Value of conditional formulation |
| **NSA-Diff (proposed)** | Full method with conditional null-space loss at skip layers | Full contribution |
| **FitNets** | $\ell_2$ matching of intermediate activations (with regressor for dimension alignment) | Standard intermediate feature matching |
| **Gramian** | Cross-layer Gramian matrix matching + KD | Correlation-based matching |

All baselines use the same CP decomposition initialization and rank. Only the training loss differs.

### 5.4 Evaluation Metrics

**Generation Quality**:
- **FID** (Fréchet Inception Distance): 50K generated samples vs. CIFAR-10 training set statistics. Lower is better. Use `pytorch-fid` with Inception-v3 features.
- **IS** (Inception Score): measures both quality and diversity. Higher is better.
- **CLIP-FID** (optional): FID computed with CLIP features instead of Inception, more robust to small distribution shifts.

**Efficiency**:
- **Parameter count**: total trainable parameters in the student.
- **FLOPs**: per-step denoising FLOPs (forward pass of U-Net at one timestep). Count multiply-accumulate operations.
- **Model size**: disk footprint in MB (fp32 and fp16).

**On-Device (Jetson Orin)**:
- **Per-step latency**: time for one U-Net forward pass, averaged over 100 runs after 10 warmup runs.
- **End-to-end generation time**: full 1000-step DDPM or 20-step DPM-Solver++ pipeline.
- **Peak GPU memory**: maximum allocated GPU memory during inference.
- **Throughput**: images per second at batch size 1 and batch size 4.

### 5.5 Evaluation Protocol

```python
def evaluate_fid(student_unet, noise_scheduler, num_samples=50000,
                 batch_size=256, num_steps=1000):
    """Generate samples and compute FID."""
    student_unet.eval()
    all_samples = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            x = torch.randn(bs, 3, 32, 32, device=device)

            for t in reversed(range(num_steps)):
                t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
                noise_pred = student_unet(x, t_batch).sample
                x = noise_scheduler.step(noise_pred, t, x).prev_sample

            # Clamp and convert
            x = (x.clamp(-1, 1) + 1) / 2 * 255
            all_samples.append(x.cpu().byte())

    # Save and compute FID
    save_images(all_samples, "generated/")
    fid = compute_fid("generated/", "cifar10_train_stats.npz")
    return fid
```

### 5.6 Training Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimizer | AdamW | $\beta_1=0.9$, $\beta_2=0.999$, weight decay $10^{-4}$ |
| Learning rate | $10^{-4}$ | StepLR with step=10K, $\gamma=0.7$ |
| Batch size | 64 | Per GPU; effective 128 with 2× 4090 |
| Training steps | 50,000 | ~6.4 epochs over CIFAR-10 train set |
| $\alpha$ (NSA weight) | 0.1 | From NSA-Net hyperparameter search |
| $\alpha_s$ (conditional NSA) | sweep $\{0.01, 0.05, 0.1, 0.5\}$ | New hyperparameter |
| $\beta$ (KD weight) | 0.1 | |
| $\lambda$ (orthonormality) | 0.1 | |
| EMA decay | 0.9999 | Exponential moving average of student weights |
| Mixed precision | fp16 | Via `torch.cuda.amp` |

### 5.7 Compute Budget

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| CP decomposition of teacher | 1× 4090 | ~5 minutes |
| Distillation (50K steps, one config) | 2× 4090 | ~2-3 hours |
| FID evaluation (50K samples, 1000 steps) | 1× 4090 | ~40 minutes |
| FID evaluation (50K samples, 20-step DPM-Solver++) | 1× 4090 | ~5 minutes |
| Jetson Orin benchmarking (per model) | Jetson Orin | ~30 minutes |
| Full experiment suite (3 compressions × 5 methods + ablations) | 2× 4090 + A5000 | ~36-48 hours |

**Parallelization strategy**: run different compression ratios on different GPUs simultaneously. Use A5000 for FID evaluation while 4090s train next config.

---

## 6. Experiments and Expected Results

### 6.1 Main Results Table

| Model | FID↓ | IS↑ | Params | FLOPs | Compression |
|-------|------|-----|--------|-------|-------------|
| Teacher (DDPM) | ~3.17 | ~9.46 | 35.7M | X | 1× |
| | | | | | |
| Low-rank + KD (ρ=0.25) | ? | ? | ~7M | ? | ~5× |
| Standard NSA (ρ=0.25) | ? | ? | ~7M | ? | ~5× |
| **NSA-Diff (ρ=0.25)** | ? | ? | ~7M | ? | ~5× |
| FitNets (ρ=0.25) | ? | ? | ~7M | ? | ~5× |
| Gramian (ρ=0.25) | ? | ? | ~7M | ? | ~5× |
| | | | | | |
| Low-rank + KD (ρ=0.125) | ? | ? | ~3.5M | ? | ~10× |
| Standard NSA (ρ=0.125) | ? | ? | ~3.5M | ? | ~10× |
| **NSA-Diff (ρ=0.125)** | ? | ? | ~3.5M | ? | ~10× |
| FitNets (ρ=0.125) | ? | ? | ~3.5M | ? | ~10× |

**Expected outcome**: NSA-Diff should outperform Low-rank + KD and FitNets at the same compression ratio, following the pattern from the original NSA-Net paper. The conditional formulation should outperform standard NSA, especially at higher compression ratios where the decoder must compensate for larger skip-connection errors.

### 6.2 On-Device Benchmarks (Jetson Orin)

| Model | Per-Step Latency | 20-Step Total | Peak Memory | Throughput |
|-------|-----------------|---------------|-------------|------------|
| Teacher | ? ms | ? s | ? MB | ? img/s |
| NSA-Diff (ρ=0.50) | ? ms | ? s | ? MB | ? img/s |
| NSA-Diff (ρ=0.25) | ? ms | ? s | ? MB | ? img/s |
| NSA-Diff (ρ=0.12) | ? ms | ? s | ? MB | ? img/s |

**Jetson Orin benchmark script**:

```python
import torch
import time

def benchmark_on_device(model, input_shape=(1, 3, 32, 32), num_runs=100, warmup=10):
    model.eval()
    x = torch.randn(*input_shape, device='cuda')
    t = torch.randint(0, 1000, (input_shape[0],), device='cuda')

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x, t)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x, t)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x, t)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'peak_memory_mb': peak_mem
    }
```

---

## 7. Ablation Studies

### 7.1 Ablation A: Value of Conditional Null-Space Loss at Skip Connections

Fix $\rho=0.25$. Compare:

| Variant | Skip Layer Handling | FID |
|---------|-------------------|-----|
| A1: KD only at skips | No NSA at skip-receiving layers | ? |
| A2: Standard NSA at skips | $\|\hat{W}_j e_{\text{total}}\|^2$, no stop-grad | ? |
| A3: Conditional NSA (decoder only) | $\|\hat{W}_j(e_{j-1} + \text{sg}(\delta_i))\|^2$ only | ? |
| A4: Conditional NSA (encoder only) | $\|\hat{W}_j(\text{sg}(e_{j-1}) + \delta_i)\|^2$ only | ? |
| A5: **Conditional NSA (both)** | Full $\mathcal{L}_{\text{cond-null}}$ | ? |

**Expected**: A5 > A3 ≈ A4 > A2 > A1. The conditional formulation should help most, with standard NSA at skips potentially hurting due to conflated gradients.

### 7.2 Ablation B: $\alpha_s$ Sensitivity

Fix $\rho=0.25$, vary $\alpha_s \in \{0, 0.01, 0.05, 0.1, 0.5, 1.0\}$.

Plot FID vs. $\alpha_s$. Expected: inverted-U shape, with optimal around 0.05–0.1, similar to the $\alpha$ sensitivity in the original paper. Too large $\alpha_s$ forces too much focus on hidden features; too small loses the benefit.

### 7.3 Ablation C: Selective NSA vs. Full NSA

Fix $\rho=0.25$. Compare:

| Variant | What Gets NSA | FID |
|---------|---------------|-----|
| C1: NSA everywhere | All layers including attention | ? |
| C2: NSA on convs only | Conv blocks only, KD on attention | ? |
| C3: **NSA on convs + conditional at skips** | Full selective strategy | ? |

**Expected**: C3 > C2 > C1. Applying NSA to attention should hurt (similar to ViT findings) because attention's softmax breaks linearity.

### 7.4 Ablation D: Null-Space Dimension Analysis

For each compression level, record:
- Average null-space dimension across layers: $\bar{d}_{\text{null}} = \frac{1}{L}\sum_l (C_{\text{in}}^{(l)} - R_l)$
- FID improvement of NSA-Diff over Low-rank + KD baseline

Plot FID improvement vs. $\bar{d}_{\text{null}}$. **Expected**: larger null spaces (more aggressive compression) benefit more from the null-space loss, because there is more freedom for the student to absorb mismatches.

### 7.5 Ablation E: Decomposition Quality

Compare:
- **Random initialization**: student weights are random low-rank matrices (not from teacher)
- **CP from teacher**: standard CP decomposition (proposed default)
- **Energy-adaptive rank**: choose per-layer rank to retain $\eta$ fraction of Frobenius energy

This validates the importance of good initialization, echoing the NSA-Net paper's argument that random init yields full-rank weights (Rank-Nullity theorem).

---

## 8. Additional Analyses

### 8.1 Singular Value Spectrum Visualization

For each compressed layer, plot the singular value spectrum of:
- Teacher weight (full rank)
- Student weight after CP decomposition (before fine-tuning)
- Student weight after fine-tuning with NSA-Diff

**Expected**: the fine-tuned student should have a cleaner spectrum with sharper drop-off at rank $R$, confirming that the orthonormality regularization and null-space loss maintain the low-rank structure during training.

### 8.2 Error Distribution Analysis

For a batch of inputs, compute $e_i = h_i^T - \hat{h}_i$ at each layer and decompose into row-space and null-space components. Plot:
- $\|e_i^{\parallel}\| / \|e_i\|$ (fraction of error in row-space) over training
- $\|e_i^{\perp}\| / \|e_i\|$ (fraction in null-space) over training

**For skip-receiving layers**, additionally decompose by source:
- Row-space component from decoder mismatch
- Row-space component from skip mismatch
- Cross-term

**Expected**: the null-space loss should drive the row-space fraction to near-zero for non-skip layers, and the conditional loss should reduce the row-space fraction at skip layers more effectively than standard NSA.

### 8.3 Generated Sample Comparison

Side-by-side grids (4×4 or 8×8) of generated images from:
- Teacher
- Low-rank + KD (ρ=0.25)
- NSA-Diff (ρ=0.25)
- NSA-Diff (ρ=0.125)

Use the same initial noise for all models to make comparison fair. This provides qualitative evidence beyond FID.

### 8.4 Denoising Trajectory Comparison

For a fixed initial noise, plot the denoising trajectory at timesteps $t \in \{1000, 750, 500, 250, 100, 0\}$ for teacher vs. student. This shows whether the student's denoising process diverges from the teacher's at certain timestep ranges.

---

## 9. Potential Issues and Mitigations

### 9.1 CP Decomposition Instability

CP decomposition via ALS can be numerically unstable for high ranks or poorly conditioned tensors. Mitigations:
- Use `init='svd'` in tensorly for stable initialization
- Cap rank at $0.5 \cdot \min(C_{\text{in}}, C_{\text{out}})$ to avoid degenerate solutions
- Monitor reconstruction error $\|\mathcal{W} - \hat{\mathcal{W}}\|_F / \|\mathcal{W}\|_F$ after decomposition

### 9.2 Effective Weight Approximation

Using only the channel-mixing matrix $W_{\text{eff}} = A^{(1)} A^{(2)\top}$ for the null-space loss ignores the spatial kernels. If results are suboptimal, consider:
- Unfolding the convolution into a matrix (im2col) and computing the full null-space loss on the unfolded form
- This is more expensive but mathematically exact

### 9.3 Memory During Training

With teacher and student both in memory plus activation storage, memory can be tight. Mitigations:
- Teacher in fp16, student in fp32 (or both in fp16 with loss scaling)
- Gradient checkpointing on the student
- Only store activations at NSA-target layers, not all layers

### 9.4 FID Variance

FID can vary by 0.5–1.0 depending on the random seed for sample generation. Mitigations:
- Always use the same set of initial noise vectors across all evaluations
- Report mean ± std over 3 generation runs with different seeds

### 9.5 EMA

Diffusion models typically use an exponential moving average of weights for generation. Apply EMA to the student during training and evaluate the EMA model. Without EMA, FID can be significantly worse.

---

## 10. Timeline (2-Day Sprint)

### Day 1

| Time | Task | Hardware |
|------|------|----------|
| Morning (3h) | Load pretrained DDPM, implement CP decomposition, build student, verify forward pass produces valid noise predictions | 1× 4090 |
| Midday (2h) | Implement activation hooks, standard NSA loss, conditional null-space loss, orthonormality loss, full training loop | — |
| Afternoon (3h) | Launch distillation runs: (1) Low-rank+KD ρ=0.25, (2) NSA ρ=0.25, (3) NSA-Diff ρ=0.25 | 2× 4090 + A5000 |
| Evening | Runs continue overnight. Start FID generation script. | All GPUs |

### Day 2

| Time | Task | Hardware |
|------|------|----------|
| Morning (2h) | Check overnight runs. Launch ρ=0.125 and ρ=0.5 runs. Compute FIDs for completed models. Launch FitNets baseline. | All GPUs |
| Midday (2h) | Skip-connection ablation (A1–A5). α_s sensitivity sweep. | 2× 4090 |
| Afternoon (2h) | Jetson Orin benchmarks: export models, measure latency/memory. Generate sample grids. | Jetson Orin |
| Evening (2h) | Compile results tables. Write up findings. Generate plots. | — |

### Priorities (if time is short)

1. **Must have**: ρ=0.25 comparison of Low-rank+KD vs. NSA vs. NSA-Diff vs. FitNets with FID
2. **Must have**: At least one Jetson Orin latency measurement
3. **Should have**: ρ=0.125 results showing scaling behavior
4. **Should have**: Skip-connection ablation (A1, A2, A5 minimum)
5. **Nice to have**: Sample grids, denoising trajectories, singular value analysis

---

## 11. Paper Outline (4-Page Extended Abstract)

### Title
**NSA-Diff: Null-Space Absorbing Compression for Diffusion U-Nets with Conditional Skip-Connection Handling**

### Abstract (~150 words)
- NSA-Net compresses CNNs by projecting activation mismatches onto null spaces of low-rank weights
- U-Net skip connections produce compound mismatches from multiple paths, breaking the single-path assumption
- We propose conditional null-space loss using stop-gradient decoupling
- Validated on DDPM-CIFAR10 with FID evaluation and Jetson Orin benchmarks
- Achieve X× compression with Y FID points degradation

### Section 1: Introduction (0.5 pages)
- Efficient on-device diffusion generation is critical (cite EDGE/ECV themes)
- NSA-Net works for sequential architectures but not multi-path
- Contribution: conditional null-space loss for skip connections
- Brief mention of on-device results

### Section 2: Method (1.5 pages)
- 2.1: Problem setup (DDPM U-Net structure, skip connections)
- 2.2: Review of NSA loss (Eq. 11-12 from original paper, 2 sentences)
- 2.3: Skip connection problem (why standard NSA fails, 1 paragraph)
- 2.4: Conditional null-space loss (main contribution, full derivation)
- 2.5: Selective NSA strategy (which layers get which loss)
- Figure 1: Architecture diagram showing where each loss is applied

### Section 3: Experiments (1.5 pages)
- 3.1: Setup (teacher, compression configs, baselines, metrics)
- 3.2: Main results table (FID, params, FLOPs)
- 3.3: On-device results table (Jetson Orin latency, memory)
- 3.4: Key ablation (conditional vs. standard NSA at skip layers)
- Figure 2: Generated samples comparison
- Figure 3: FID vs. compression ratio curve

### Section 4: Conclusion (0.5 pages)
- Summary of contribution
- Connection to published NSA-Net (APSIPA 2025) and NSA-ViT findings
- Future work: scaling to Stable Diffusion, DiT architectures

---

## 12. Key References

- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020 (DDPM)
- Ozdemir et al., "Low-Rank Compression of Neural Network Weights by Null-Space Encouragement," APSIPA ASC 2025 (NSA-Net, this group's prior work)
- Romero et al., "FitNets: Hints for Thin Deep Nets," ICLR 2015
- Kim et al., "BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion," ICML 2023
- Lebedev et al., "Speeding-up CNNs using Fine-tuned CP-Decomposition," ICLR 2015
- Hinton et al., "Distilling the Knowledge in a Neural Network," 2015
- Song et al., "Denoising Diffusion Implicit Models," ICLR 2021 (DDIM — for fast sampling during eval)
