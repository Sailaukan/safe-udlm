# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
import random
import warnings

import torch
import lightning as L


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _load_seq_len_list():
    """Load precomputed SAFE sequence-length distribution from data/len.pk."""
    path = os.path.join(ROOT_DIR, 'data/len.pk')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _build_masked_inputs(bos_index, eos_index, mask_index, pad_index,
                         num_samples, min_add_len, seq_len_list=None):
    """Return a (num_samples, max_len) tensor of [BOS, <masks>, EOS, <pads>]."""
    x_new = []
    for _ in range(num_samples):
        if seq_len_list is not None:
            add_len = max(random.choice(seq_len_list) - 2, min_add_len)
        else:
            add_len = min_add_len
        row = torch.cat([
            torch.tensor([bos_index]),
            torch.full((add_len,), mask_index),
            torch.tensor([eos_index]),
        ])
        x_new.append(row)

    pad_len = max(len(r) for r in x_new)
    x_new = [
        torch.cat([r, torch.full((pad_len - len(r),), pad_index)])
        for r in x_new
    ]
    return torch.stack(x_new)


@torch.no_grad()
def _generate_denovo(pl_module, num_samples, softmax_temp, randomness, min_add_len, seq_len_list):
    """
    De-novo generation using the live model weights.
    EMA weights must already be swapped in by the caller.
    Mirrors Sampler.de_novo_generation + Sampler.generate without loading from disk.
    """
    from safe_udlm.utils.utils_chem import safe_to_smiles
    from safe_udlm.utils.bracket_safe_converter import bracketsafe2safe

    device = pl_module.device
    diffusion = pl_module.diffusion
    tokenizer = pl_module.tokenizer

    x = _build_masked_inputs(
        bos_index=pl_module.bos_index,
        eos_index=pl_module.eos_index,
        mask_index=pl_module.mask_index,
        pad_index=pl_module.pad_index,
        num_samples=num_samples,
        min_add_len=min_add_len,
        seq_len_list=seq_len_list,
    ).to(device)

    attention_mask = (x != pl_module.pad_index).long()
    editable_mask = (x == pl_module.mask_index) & attention_mask.bool()

    x = diffusion.initialize_sample(x, editable_mask)

    num_steps = max(diffusion.get_num_steps_confidence(x), 2)
    timestep_grid = diffusion.get_sampling_timesteps(device, num_steps=num_steps)

    for i in range(num_steps):
        t = timestep_grid[i].expand(x.shape[0])
        sigma_t = diffusion.time_conditioning(t)
        logits = pl_module(x, attention_mask=attention_mask, timesteps=sigma_t)
        x = diffusion.step_confidence(
            logits, x, i, num_steps,
            softmax_temp=softmax_temp,
            randomness=randomness,
            editable_mask=editable_mask,
            timestep_grid=timestep_grid,
        )

    if diffusion.final_denoise:
        sigma_0 = torch.zeros(x.shape[0], device=device)
        logits = pl_module(x, attention_mask=attention_mask, timesteps=sigma_0)
        x = diffusion.final_denoise_step(
            logits, x, editable_mask,
            softmax_temp=softmax_temp,
            randomness=0.0,
        )

    decoded = tokenizer.batch_decode(x, skip_special_tokens=True)
    use_bracket = pl_module.config.training.get('use_bracket_safe', False)

    smiles_list = []
    for s in decoded:
        try:
            sm = safe_to_smiles(bracketsafe2safe(s) if use_bracket else s, fix=True)
            if sm:
                sm = sorted(sm.split('.'), key=len)[-1]
                smiles_list.append(sm)
        except Exception:
            pass
    return smiles_list


class DeNovoEvalCallback(L.Callback):
    """
    Runs de-novo molecule generation every `eval_every_n_steps` training steps,
    computes molecule quality metrics, and logs them to W&B (or any active logger).

    Metrics logged (all under the `eval/` prefix):
        validity    — fraction of attempted samples that decoded to valid SMILES
        uniqueness  — fraction of valid samples that are structurally unique
        diversity   — average pairwise Tanimoto distance of unique valid samples
        mean_qed    — mean QED score of unique valid samples
        mean_sa     — mean SA score (lower is easier to synthesise)
        quality     — fraction of attempted samples with QED >= qed_threshold
                      AND SA <= sa_threshold
        n_valid     — raw count of valid samples (useful for sanity checking)
    """

    def __init__(
        self,
        eval_every_n_steps: int = 10_000,
        num_samples: int = 256,
        softmax_temp: float = 0.8,
        randomness: float = 1.0,
        min_add_len: int = 50,
        qed_threshold: float = 0.6,
        sa_threshold: float = 4.0,
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.num_samples = num_samples
        self.softmax_temp = softmax_temp
        self.randomness = randomness
        self.min_add_len = min_add_len
        self.qed_threshold = qed_threshold
        self.sa_threshold = sa_threshold
        self._seq_len_list = None
        self._tdc_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self, trainer, pl_module):
        self._seq_len_list = _load_seq_len_list()
        self._lazy_load_tdc()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.eval_every_n_steps != 0:
            return
        # Only evaluate on rank-0 to avoid duplicate work under DDP.
        if not trainer.is_global_zero:
            return

        # Swap in EMA weights so the sampler uses the stabilised model.
        if pl_module.ema is not None:
            pl_module.ema.store(pl_module.backbone.parameters())
            pl_module.ema.copy_to(pl_module.backbone.parameters())

        pl_module.backbone.eval()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                samples = _generate_denovo(
                    pl_module=pl_module,
                    num_samples=self.num_samples,
                    softmax_temp=self.softmax_temp,
                    randomness=self.randomness,
                    min_add_len=self.min_add_len,
                    seq_len_list=self._seq_len_list,
                )
        finally:
            # Always restore online weights, even if generation crashed.
            if pl_module.ema is not None:
                pl_module.ema.restore(pl_module.backbone.parameters())
            pl_module.backbone.train()

        metrics = self._compute_metrics(samples)

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)

        summary = '  '.join(f'{k.split("/")[-1]}={v:.4f}' for k, v in metrics.items())
        print(f'\n[eval step={step}] {summary}\n')

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _lazy_load_tdc(self):
        if self._tdc_loaded:
            return
        try:
            from tdc import Oracle, Evaluator
            self._oracle_qed = Oracle('qed')
            self._oracle_sa = Oracle('sa')
            self._evaluator_diversity = Evaluator('diversity')
            self._tdc_loaded = True
        except ImportError:
            warnings.warn(
                'PyTDC not installed — QED, SA, and diversity metrics will be skipped. '
                'Install with: pip install PyTDC'
            )

    def _compute_metrics(self, samples: list[str]) -> dict[str, float]:
        n_attempted = self.num_samples
        n_valid = len(samples)
        validity = n_valid / n_attempted

        metrics: dict[str, float] = {
            'eval/validity': validity,
            'eval/n_valid': float(n_valid),
        }

        if n_valid == 0:
            metrics['eval/uniqueness'] = 0.0
            metrics['eval/diversity'] = 0.0
            metrics['eval/mean_qed'] = 0.0
            metrics['eval/mean_sa'] = 0.0
            metrics['eval/quality'] = 0.0
            return metrics

        unique_samples = list(set(samples))
        metrics['eval/uniqueness'] = len(unique_samples) / n_valid

        if not self._tdc_loaded:
            return metrics

        try:
            metrics['eval/diversity'] = float(self._evaluator_diversity(unique_samples))
        except Exception as e:
            warnings.warn(f'Diversity computation failed: {e}')

        try:
            qed_scores = self._oracle_qed(unique_samples)
            sa_scores = self._oracle_sa(unique_samples)
            metrics['eval/mean_qed'] = float(sum(qed_scores) / len(qed_scores))
            metrics['eval/mean_sa'] = float(sum(sa_scores) / len(sa_scores))
            n_quality = sum(
                1 for q, s in zip(qed_scores, sa_scores)
                if q >= self.qed_threshold and s <= self.sa_threshold
            )
            metrics['eval/quality'] = n_quality / n_attempted
        except Exception as e:
            warnings.warn(f'QED/SA computation failed: {e}')

        return metrics
