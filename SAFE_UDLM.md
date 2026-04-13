# SAFE-UDLM Migration Note

## Goal

This repository uses a local **uniform discrete diffusion language model (UDLM)** engine as the generative core for SAFE-UDLM, so the model can be trained as a SAFE-native UDLM without changing the outer experiment workflow.

The design target was:

- keep the existing `SafeUDLM` and `Sampler` entrypoints,
- keep the existing task scripts and folder structure,
- replace BioNeMo's masked discrete diffusion assumptions with UDLM's uniform-noise forward and reverse process,
- preserve fragment-conditioned and goal-directed workflows that already depend on mask-marked editable spans.

## What Changed

### 1. BioNeMo MDLM was replaced with a local UDLM engine

Files:

- `src/safe_udlm/diffusion.py`
- `src/safe_udlm/utils/utils_moco.py`
- `src/safe_udlm/model.py`

The new engine implements the UDLM mechanics adapted from the `kuleshov-group/discrete-diffusion-guidance` codebase:

- continuous-time log-linear noise,
- uniform token corruption instead of absorbing `[MASK]` corruption,
- posterior-based reverse diffusion,
- explicit diffusion loss for the uniform discrete process,
- optional zero reconstruction loss behavior, which is the default for UDLM.

`SafeUDLM` still exposes `self.mdlm` for compatibility, but that attribute now points to the local UDLM engine instead of BioNeMo's MDLM class.

### 2. The BERT backbone is now time-conditioned

File:

- `src/safe_udlm/backbone.py`

The original masked diffusion approach did not require explicit timestep conditioning. UDLM does. To keep the same BERT-centered model family, the repository now uses a `TimeConditionedBertForMaskedLM` wrapper:

- sinusoidal timestep embedding,
- MLP projection into hidden size,
- additive or scale-shift conditioning on top of BERT embeddings.

This keeps the model structurally similar to the original while making it valid for UDLM training and sampling.

### 3. Sampling now follows UDLM behavior

File:

- `src/safe_udlm/sampler.py`

The old sampler started from `[MASK]` placeholders and progressively locked tokens in place. The new sampler keeps the same user-facing APIs, but internally:

- editable `[MASK]` spans are converted into uniformly sampled real tokens at the start of generation,
- all editable positions remain mutable across the whole reverse chain,
- context tokens remain frozen for conditional generation,
- the final step optionally performs a clean `t -> 0` denoise pass.

This is the main behavioral change that defines SAFE-UDLM rather than SAFE-MDLM.

### 4. Molecular context guidance was made UDLM-compatible

The existing `gamma`/`w` guidance interface was preserved, but poor-context guidance no longer masks context tokens. It now **randomizes** a fraction of frozen context tokens with uniform non-special vocabulary samples, which is consistent with the UDLM latent space.

### 5. Dependency surface was reduced

Files:

- `pyproject.toml`
- `env/requirements.txt`
- `env/requirements.yaml`

`bionemo-moco` is no longer required. The diffusion code now lives inside the repository.

## Important Behavioral Notes

### Checkpoints

Old MDLM checkpoints are **not architecture-compatible** with the SAFE-UDLM code path. SAFE-UDLM should be trained from scratch on the same SAFE dataset.

### Special tokens

`[BOS]`, `[EOS]`, and `[PAD]` are frozen during diffusion. This keeps sequence boundaries stable and preserves the existing downstream task assumptions.

### Conditioning interface

The experiment scripts remain mostly unchanged because editable regions are still marked with `[MASK]` at the API boundary. The difference is that `[MASK]` is now only a user-side placeholder for "editable span"; the actual UDLM latent starts from randomized real tokens.

## Additional Correctness Fixes Included

- `fragment_completion(..., mask_len=...)` now actually respects `mask_len`, which fixes `addmask`.
- `remask` now guards against invalid negative/zero replacement span lengths.
- runtime dependence on `jaxtyping` for time-sampler annotations was removed.

## Why This Is The Right Migration to UDLM

This migration changes the **diffusion engine**, not the surrounding research interface:

- dataset: unchanged,
- SAFE tokenization: unchanged,
- experiment scripts: unchanged at the entrypoint level,
- fragment-constrained generation flow: unchanged at the API level,
- lead optimization and PMO pipelines: unchanged at the script level.

That means the repo preserves the original research interface, but the generative dynamics are now UDLM-shaped.

## Source Basis

The UDLM loss/sampling/noise design in this migration follows the paper and reference implementation from:

- https://github.com/kuleshov-group/discrete-diffusion-guidance

## Recommended Next Step

Retrain SAFE-UDLM from scratch on the original SAFE dataset, then re-run:

- `scripts/exps/denovo/run.py`
- `scripts/exps/frag/run.py`
- `scripts/exps/pmo/run.py`
- `scripts/exps/lead/run.py`

Only after retraining and benchmarking can the new model be validated as your production SAFE-UDLM baseline.
