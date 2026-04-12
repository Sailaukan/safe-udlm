# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Maximal Update Parameterization (muP) for SAFE-UDLM.

muP lets you tune hyperparameters (mainly the learning rate) on a small "base"
model once, then train arbitrarily wide models with the same HPs. Without muP,
the optimal LR drifts as width grows and must be re-tuned per model size.

Reference: Yang et al., "Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer", NeurIPS 2022.
    https://arxiv.org/abs/2203.03466

Scope of this implementation
----------------------------
We apply the Adam-case rules from Table 8 of the paper:
  - init std scaling by parameter role,
  - per-parameter-group learning rates,
  - an output-logit multiplier of 1/m on the MLM decoder.
We do NOT replace the 1/sqrt(d_k) attention scaling with 1/d_k, since that
requires subclassing HuggingFace's BertSelfAttention. In practice, init + LR
+ output multiplier captures the bulk of the HP-transfer benefit for models
under ~1B parameters.

Parameter-role classification for this BERT + time-conditioning backbone:

    input-style    (fan_in fixed, fan_out scales)
        bert.embeddings.word_embeddings.weight
        bert.embeddings.position_embeddings.weight
        bert.embeddings.token_type_embeddings.weight
        sigma_map.mlp.0.weight            # frequency_embedding_size -> hidden

    hidden-style   (both dims scale)
        bert.encoder.layer.*.attention.self.{query,key,value}.weight
        bert.encoder.layer.*.attention.output.dense.weight
        bert.encoder.layer.*.intermediate.dense.weight
        bert.encoder.layer.*.output.dense.weight
        cls.predictions.transform.dense.weight
        sigma_map.mlp.2.weight
        time_projection.weight

    output-style   (fan_in scales, fan_out fixed)
        cls.predictions.decoder.weight     # must be untied from embeddings

    unscaled       (no width dependence)
        all *.bias
        all LayerNorm.{weight,bias}
        cls.predictions.bias

Adam rules for width multiplier m = hidden_size / base_hidden_size:

    role          init std                LR              forward multiplier
    ----          --------                --              ------------------
    input         base_std                base_lr         1
    hidden        base_std / sqrt(m)      base_lr / m     1
    output        base_std / sqrt(m)      base_lr / m     1 / m
    unscaled      (leave default)         base_lr         1

Workflow
--------
1. Pick a small base model (e.g. ``base_hidden_size = 128``, and make sure
   ``intermediate_size`` keeps its usual ratio, e.g. 4x -> 512).
2. Enable muP (``mup.enabled = True``) and set ``mup.base_hidden_size`` to
   that base width.
3. Train the base model (``model.hidden_size = 128``, ``intermediate_size = 512``)
   and sweep the learning rate to find the optimum.
4. Scale up by changing only ``model.hidden_size`` / ``model.intermediate_size``
   (keeping their ratio) and leave the learning rate unchanged. muP rescales
   init/LR/output automatically so the optimum transfers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# Exact parameter names (not prefixes) for roles that can't be inferred from
# a simple suffix rule. Anything not matched here falls through to the
# generic classification in `classify_parameter`.
_INPUT_NAMES = frozenset({
    "bert.embeddings.word_embeddings.weight",
    "bert.embeddings.position_embeddings.weight",
    "bert.embeddings.token_type_embeddings.weight",
    "sigma_map.mlp.0.weight",
})

_OUTPUT_NAMES = frozenset({
    "cls.predictions.decoder.weight",
})


@dataclass
class MupConfig:
    """User-facing muP settings.

    Attributes:
        enabled: master switch. When False, muP is a no-op everywhere.
        base_hidden_size: hidden size of the reference "base" model on which
            hyperparameters were tuned. The width multiplier is
            ``model.hidden_size / base_hidden_size``.
        init_std: base initialization standard deviation. Defaults to
            ``BertConfig.initializer_range`` (0.02) when not overridden.
    """
    enabled: bool = False
    base_hidden_size: int = 128
    init_std: float = 0.02


def width_mult(mup: MupConfig, hidden_size: int) -> float:
    """Return m = hidden_size / base_hidden_size. Equals 1 for the base model."""
    return hidden_size / mup.base_hidden_size


def classify_parameter(name: str, param: torch.Tensor) -> str:
    """Return one of ``'input' | 'hidden' | 'output' | 'unscaled'``."""
    if name in _OUTPUT_NAMES:
        return "output"
    if name in _INPUT_NAMES:
        return "input"
    if "LayerNorm" in name:
        return "unscaled"
    if name.endswith(".bias"):
        return "unscaled"
    if param.dim() < 2:
        # Any remaining 1-D tensor (e.g. cls.predictions.bias) is unscaled.
        return "unscaled"
    return "hidden"


@torch.no_grad()
def apply_mup_init(model: nn.Module, mup: MupConfig, hidden_size: int) -> None:
    """Re-initialize ``model`` in place with muP-correct stds.

    Must be called AFTER HuggingFace's ``post_init()`` (which fills all
    weights with the default ``config.initializer_range`` std). This routine
    overwrites input-, hidden-, and output-style weights; unscaled parameters
    (biases, LayerNorm) are left at their defaults.

    No-op when ``mup.enabled`` is False.
    """
    if not mup.enabled:
        return

    m = width_mult(mup, hidden_size)
    base_std = mup.init_std
    hidden_std = base_std / math.sqrt(m)

    for name, param in model.named_parameters():
        role = classify_parameter(name, param)
        if role == "input":
            param.normal_(mean=0.0, std=base_std)
        elif role == "hidden":
            param.normal_(mean=0.0, std=hidden_std)
        elif role == "output":
            # Init std matches hidden; the 1/m output shrinkage is applied as
            # a forward-pass multiplier, not baked into the weight, so that
            # gradients stay in the same regime as hidden layers.
            param.normal_(mean=0.0, std=hidden_std)
        # 'unscaled' -> leave HF defaults (zeros for biases, ones for LN gain)


def mup_param_groups(
    model: nn.Module,
    mup: MupConfig,
    hidden_size: int,
    base_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build AdamW parameter groups with muP-correct per-group learning rates.

    When muP is disabled, returns a single group (equivalent to the standard
    ``optimizer(model.parameters(), lr=base_lr, weight_decay=weight_decay)``).

    When enabled, splits parameters by role and applies:
        input, unscaled:    lr = base_lr
        hidden, output:     lr = base_lr / m

    Weight decay is applied to all weights, and zeroed for biases/LayerNorm
    (the 'unscaled' bucket) — standard practice, unrelated to muP.
    """
    if not mup.enabled:
        return [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
        }]

    m = width_mult(mup, hidden_size)
    buckets: dict[str, list[nn.Parameter]] = {
        "input": [], "hidden": [], "output": [], "unscaled": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        buckets[classify_parameter(name, param)].append(param)

    groups = [
        {"params": buckets["input"],    "lr": base_lr,     "weight_decay": weight_decay},
        {"params": buckets["hidden"],   "lr": base_lr / m, "weight_decay": weight_decay},
        {"params": buckets["output"],   "lr": base_lr / m, "weight_decay": weight_decay},
        {"params": buckets["unscaled"], "lr": base_lr,     "weight_decay": 0.0},
    ]
    # Drop empty groups so the optimizer doesn't complain.
    return [g for g in groups if len(g["params"]) > 0]


def output_multiplier(mup: MupConfig, hidden_size: int) -> float:
    """Factor the backbone applies to decoder logits in the forward pass.

    Equals 1 when muP is disabled or when training the base model (m = 1).
    """
    if not mup.enabled:
        return 1.0
    return 1.0 / width_mult(mup, hidden_size)
