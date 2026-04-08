# SAFE-UDLM

<p align="center">
  <img width="860" src="assets/picture.png" alt="SAFE-UDLM banner" />
</p>

SAFE-UDLM is a SAFE-native molecule generation project built around a uniform discrete diffusion language model (UDLM). It supports training and sampling on molecular SAFE sequences for de novo generation, fragment-conditioned generation, PMO-style hit generation, and lead optimization.

This repository started from the GenMol codebase and experiment structure. GenMol was used initially, but the current project replaces the original diffusion core with a local UDLM implementation while keeping most of the outer workflow and script entrypoints familiar.

## Overview

SAFE-UDLM keeps the practical structure of the earlier GenMol project, but shifts the generative engine to uniform discrete diffusion over SAFE sequences.

- SAFE-based molecular sequence modeling
- Local UDLM diffusion engine
- BERT-style backbone with timestep conditioning
- Support for de novo, fragment-constrained, PMO, and lead-optimization workflows

Some modules and paths still use the `genmol` name for compatibility, but the active model path in this repository is SAFE-UDLM.

## Repository Layout

- `src/genmol/`: model, diffusion, sampler, and utility code
- `scripts/train.py`: training entrypoint
- `scripts/preprocess_data.py`: convert SMILES datasets into SAFE-ready training data
- `scripts/exps/denovo/`: de novo generation
- `scripts/exps/frag/`: fragment-constrained generation
- `scripts/exps/pmo/`: goal-directed hit generation
- `scripts/exps/lead/`: goal-directed lead optimization
- `configs/base.yaml`: default training configuration
- `SAFE_UDLM.md`: migration and implementation notes
- `MODEL_CARD.md`: model-facing documentation

## Setup

```bash
git clone https://github.com/Sailaukan/safe-udlm.git
cd safe-udlm
bash env/setup.sh
```

The setup script installs the Python dependencies and the local package. If you prefer manual setup, the package metadata lives in `pyproject.toml`.

## Training

Train from scratch with:

```bash
torchrun --nproc_per_node ${NUM_GPU} scripts/train.py hydra.run.dir=${SAVE_DIR} wandb.name=${EXP_NAME}
```

Default training and sampling settings are defined in `configs/base.yaml`. By default, training artifacts are written under `ckpt/`.

To preprocess a custom SMILES dataset:

```bash
python scripts/preprocess_data.py ${INPUT_PATH} ${DATA_PATH}
```

Then point `data` in `configs/base.yaml` to the processed dataset.

## Inference Workflows

Run the main generation tasks with the existing experiment entrypoints:

```bash
python scripts/exps/denovo/run.py
python scripts/exps/frag/run.py
python scripts/exps/pmo/run.py -o ${ORACLE_NAME}
python scripts/exps/lead/run.py -o ${ORACLE_NAME} -i ${START_MOL_IDX} -d ${SIM_THRESHOLD}
```

Evaluation helpers are included alongside the PMO and lead optimization scripts.

## Project Notes

- Historical references to GenMol remain in some filenames, package names, and legacy documentation.
- The current repository direction is SAFE-UDLM, not the original GenMol masked-diffusion setup.
- Older GenMol checkpoints are not necessarily compatible with the current UDLM-based model path.

## License

Code is licensed under Apache 2.0. Additional license details for weights and third-party components are included under `LICENSE/`.
