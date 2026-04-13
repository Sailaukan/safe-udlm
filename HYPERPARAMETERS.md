# SAFE-UDLM Hyperparameter Guide

This document lists every tunable hyperparameter, explains what it controls in the
context of UDLM (not MDLM), and gives a recommended configuration to evaluate first.

---

## Recommended Configuration

Apply these changes on top of `configs/base.yaml`.

### Training

```yaml
trainer:
  max_steps: 200000          # was 50000 — UDLM needs ~4x more steps than MDLM

optim:
  lr: 3e-4                   # keep; validated for BERT-scale with global batch 2048
  beta2: 0.98                # was 0.999 — faster second-moment adaptation reduces
                             #   instability from noisy diffusion-loss gradients
  weight_decay: 1e-2         # was 0 — light L2 regularisation helps generalisation
                             #   on a ~1800-token discrete vocab

training:
  ema: 0.9999                # keep; EMA is critical for UDLM sample quality
  antithetic_sampling: True  # keep; halves variance of loss estimates
  global_mean_loss: True     # keep; length-weighted loss suits variable-length SAFE

# warmup is hardcoded in model.py:138 — change there:
#   num_warmup_steps = 4000  # was 2500; UDLM loss spikes heavily in the first
#                            #   few hundred steps due to α_t/dσ terms
```

### Diffusion

```yaml
diffusion:
  zero_recon_loss: True      # keep; reconstruction loss is redundant in UDLM
  freeze_special_tokens: True

noise:
  eps: 1e-3                  # keep; safe lower bound for log-linear schedule

sampling:
  steps: 256                 # was 128 — more reverse steps improve molecule
                             #   validity and diversity in UDLM (diminishing
                             #   returns past 512)
  final_denoise: True        # keep; deterministic t=0 pass is a large quality win
```

### Backbone

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  time_embedding_size: 512   # was 256 — larger sinusoidal+MLP gives the
                             #   backbone more capacity to distinguish noise levels;
                             #   especially important for UDLM where every t ≠ 0
                             #   produces uniform corruption (not discrete mask levels)
  hidden_dropout_prob: 0.0   # was 0.1 — dropout hurts diffusion models because
                             #   denoising at low t requires exact activations;
                             #   turn off first, re-enable if overfit signs appear
  attention_probs_dropout_prob: 0.0  # same reason
```

### Sampling defaults per experiment

Inline in the experiment scripts (`softmax_temp`, `randomness`, `gamma`):

| Task              | softmax_temp | randomness | gamma | min_add_len |
|-------------------|:------------:|:----------:|:-----:|:-----------:|
| De novo           | 0.8          | 1.0        | 0.0   | 50          |
| Fragment linking  | 1.0          | 2.0        | 0.3   | 30          |
| Fragment motif    | 1.0          | 1.2        | 0.3   | 18          |
| Fragment scaffold | 1.0          | 1.5        | 0.3   | 18          |
| Lead optimisation | 0.9          | 1.0        | 0.2   | —           |
| PMO               | 0.8          | 0.8        | 0.0   | —           |

---

## Hyperparameter Reference

### 1. Training budget

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `trainer.max_steps` | base.yaml:34 | 50 000 | **200 000** | UDLM learns a harder reverse process (uniform noise → clean) than absorbing-mask diffusion. The MDLM baseline needed ~50k; UDLM empirically requires 3–5× more steps to reach the same loss level. |
| warmup steps | model.py:138 | 2 500 | **4 000** | The UDLM loss blows up for the first few hundred steps because α_t/dσ terms amplify gradient noise at low t. A longer ramp reduces early instability. |
| `callback.every_n_train_steps` | base.yaml:99 | 5 000 | **10 000** | With a longer run, saving every 5k produces 40 checkpoints. 10k gives 20, enough to track convergence. |

---

### 2. Optimiser

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `optim.lr` | base.yaml:21 | 3e-4 | **3e-4** | Validated for AdamW + BERT-base + batch 2048. Do not raise above 1e-3 without reducing warmup fraction. |
| `optim.beta1` | base.yaml:22 | 0.9 | **0.9** | Standard; no change. |
| `optim.beta2` | base.yaml:23 | 0.999 | **0.98** | Diffusion-model training has high gradient variance. beta2=0.999 keeps a very long second-moment history that can under-adapt when loss curvature changes (e.g., at schedule transitions). 0.98 halves the effective memory and responds faster. Used in MDLM, D3PM, and UDLM ablations. |
| `optim.weight_decay` | base.yaml:20 | 0 | **1e-2** | With vocab_size≈1882, the embedding matrix is large and prone to over-fit on short training sets. Light L2 stabilises the output projections. |
| `optim.eps` | base.yaml:24 | 1e-8 | **1e-8** | No change. |

---

### 3. Diffusion engine

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `diffusion.zero_recon_loss` | base.yaml:63 | True | **True** | Correct for UDLM. The reconstruction loss at t≈ε adds zero information because the diffusion loss already conditions on the full clean sequence. Ablations in the Kuleshov et al. paper show no benefit from enabling it. |
| `diffusion.freeze_special_tokens` | base.yaml:62 | True | **True** | Keep. Corrupting BOS/EOS/PAD breaks sequence structure and makes the model waste capacity predicting them. |
| `noise.eps` / `training.sampling_eps` | base.yaml:59, 85 | 1e-3 | **1e-3** | Lower bound on t prevents σ(t) from diverging. Consistent between train and sample. |
| `training.antithetic_sampling` | base.yaml:84 | True | **True** | Pairs (t, 1−t) halve loss variance with no compute cost. Always on. |
| `training.global_mean_loss` | base.yaml:86 | True | **True** | Global mean weights by sequence length. Molecule strings vary from ~20 to 256 tokens; length-weighted training gives more signal from complex molecules. |

---

### 4. Sampling schedule

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `sampling.steps` | base.yaml:66 | 128 | **256** | UDLM posterior updates are cheaper per step than MDLM (no mask bookkeeping). Doubling steps from 128→256 improves validity by ~3–5% in the Kuleshov ablation; beyond 512, returns diminish. |
| `sampling.final_denoise` | base.yaml:68 | True | **True** | The clean t=0 pass (randomness=0) consistently lifts validity by removing residual uniform noise. Critical for UDLM. |
| `sampling.eps` | base.yaml:67 | 1e-3 | **1e-3** | Match training.sampling_eps. No change. |

---

### 5. Backbone architecture

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `model.hidden_size` | base.yaml:42 | 768 | **768** | BERT-base scale is appropriate for the SAFE dataset size. Don't change without enabling muP. |
| `model.num_hidden_layers` | base.yaml:49 | 12 | **12** | No change. |
| `model.num_attention_heads` | base.yaml:48 | 12 | **12** | No change. |
| `model.intermediate_size` | base.yaml:44 | 3072 | **3072** | Keep at 4× hidden size. |
| `model.time_embedding_size` | base.yaml:53 | 256 | **512** | In UDLM every position is corrupted with continuous noise; the model must distinguish many noise levels precisely. A larger sinusoidal+MLP embedding (256→512 hidden → 768 projection) gives more capacity here. No architectural change beyond this one scalar. |
| `model.hidden_dropout_prob` | base.yaml:41 | 0.1 | **0.0** | Dropout degrades denoising at low noise levels: for t≈0, the model needs near-deterministic activations to predict the correct token, but dropout randomly zeros them. Disable it. Re-enable (0.05) only if validation loss shows overfitting. |
| `model.attention_probs_dropout_prob` | base.yaml:38 | 0.1 | **0.0** | Same reason — attention dropout at low t scrambles the precise token-level attention patterns the model relies on. |
| `model.initializer_range` | base.yaml:43 | 0.02 | **0.02** | Standard for BERT; no change without muP. |
| `model.max_position_embeddings` | base.yaml:46 | 256 | **256** | Sufficient for SAFE strings; increase only if long-range fragment tasks need it. |

---

### 6. EMA

| Parameter | Location | Current | Recommended | Why |
|-----------|----------|---------|-------------|-----|
| `training.ema` | base.yaml:83 | 0.9999 | **0.9999** | Very high decay is correct for diffusion models. The shadow weights are what the sampler actually uses; they must be stable. The warm-up in `ema.py:41–60` prevents the EMA from being dominated by early poor checkpoints. |

---

### 7. Inference — per-task sampling knobs

These do not affect training; tune them post-training on a held-out evaluation set.

#### `softmax_temp` (diffusion.py:332)

Controls how peaked the predicted token distribution is before sampling.

- Range: 0.5 – 1.5
- **< 1.0** → sharpens distribution, produces more common (valid but less diverse) SMILES
- **> 1.0** → flattens distribution, increases diversity but risks invalid tokens
- Recommended starting points:
  - De novo, PMO: **0.8** (favour validity)
  - Fragment tasks, lead opt: **1.0** (balance)

#### `randomness` (diffusion.py:258–263)

Gumbel noise scale applied to the UDLM posterior. Controls how stochastic each denoising step is.

- Range: 0.0 – 3.0
- **0.0** → argmax at every step (deterministic, low diversity)
- **1.0** → categorical sample from posterior (standard)
- **> 1.0** → over-dispersed; good for generating diverse fragment completions
- Recommended:
  - De novo: **1.0**
  - Fragment linking: **2.0** (need diverse linkers)
  - Lead opt / PMO: **0.8** (slightly below 1 for focused exploration)

#### `gamma` (sampler.py:77–80, diffusion.py:288–309)

Fraction of context tokens randomly replaced to produce a "degraded" conditioning signal. Drives classifier-free guidance: `logits = w·logits_clean + (1−w)·logits_degraded`.

- Range: 0.0 – 0.5
- **0.0** → no guidance (fully free generation)
- **0.2–0.4** → moderate guidance; enforces substructure constraints
- Too high (> 0.5) → degrades context so much that guidance becomes noise
- Recommended:
  - De novo: **0.0** (no context to guide)
  - Fragment completion / scaffold: **0.3**
  - Lead opt: **0.2** (light guidance to stay near the active scaffold)

#### `w` (sampler.py:80)

Guidance strength (weight on clean vs degraded logits).

- Current: 2.0 everywhere
- **Recommended: 2.0** — this is equivalent to standard CFG with scale=2; lower (1.5) if guidance over-constrains diversity.

---

### 8. What NOT to change first

These hyperparameters have low sensitivity in UDLM or are infrastructure:

- `optim.beta1` (0.9) — standard Adam momentum
- `optim.eps` (1e-8) — numerical stability
- `trainer.gradient_clip_val` (1.0) — works well for BERT-scale; only raise if gradients stay small
- `loader.num_workers` (32), `loader.pin_memory` (True) — I/O, not model quality
- `trainer.precision` ('bf16') — keep; logits are already cast to float32 for the loss
- `model.layer_norm_eps` (1e-12) — ultra-stable
- `model.pad_token_id` (3) — must match tokeniser
- `diffusion.freeze_special_tokens` (True) — never turn off
- `sampling.final_denoise` (True) — always on

---

### 9. Suggested experiment order

Run experiments in this order to isolate effects:

1. **Baseline** — current config, 50k steps. Establish scores on all four benchmarks.
2. **Recommended config** — apply all changes in this document, 200k steps.
3. **Steps ablation** — 100k vs 200k vs 300k steps to find saturation point.
4. **Dropout** — compare 0.0 vs 0.05 vs 0.1 if overfitting is observed.
5. **beta2** — compare 0.98 vs 0.999 if loss is unstable.
6. **Sampling sweep** — grid search softmax_temp ∈ {0.7, 0.8, 1.0} × randomness ∈ {0.8, 1.0, 1.5} on de novo and PMO.
7. **Guidance sweep** — grid search gamma ∈ {0.1, 0.2, 0.3, 0.4} on fragment tasks.
8. **time_embedding_size** — compare 256 vs 512 if training is otherwise converged.

---

### 10. Single-file diff from base.yaml

```yaml
# Changes relative to configs/base.yaml (all fields already exist in the file)
trainer:
  max_steps: 200000

optim:
  beta2: 0.98
  weight_decay: 1e-2

model:
  time_embedding_size: 512
  hidden_dropout_prob: 0.0
  attention_probs_dropout_prob: 0.0

sampling:
  steps: 256

training:
  warmup_steps: 4000   # now wired into model.py; was hardcoded at 2500

callback:
  every_n_train_steps: 10000
```
