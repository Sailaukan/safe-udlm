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


import math

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=max(half, 1), dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.view(-1)
        return self.mlp(self._timestep_embedding(timesteps, self.frequency_embedding_size))


class TimeConditionedBertForMaskedLM(BertPreTrainedModel):
    """BERT MLM with explicit timestep conditioning for uniform discrete diffusion."""

    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.time_conditioning = getattr(config, "time_conditioning", True)
        # muP output-logit multiplier. Set to 1/m by SafeUDLM when muP is
        # enabled; stays at 1.0 otherwise, making this a no-op cost.
        self.mup_output_multiplier: float = 1.0

        if self.time_conditioning:
            time_embedding_size = getattr(config, "time_embedding_size", 256)
            self.sigma_map = TimestepEmbedder(
                hidden_size=config.hidden_size,
                frequency_embedding_size=time_embedding_size,
            )
            self.time_projection = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.sigma_map = None
            self.time_projection = None

        self.post_init()

    def _add_time_conditioning(self, hidden_states: torch.Tensor, timesteps: torch.Tensor | None) -> torch.Tensor:
        if not self.time_conditioning:
            return hidden_states

        if timesteps is None:
            timesteps = torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep_embedding = self.sigma_map(timesteps).to(hidden_states.dtype)
        modulation = self.time_projection(timestep_embedding)
        return hidden_states + modulation[:, None]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> MaskedLMOutput:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()

        input_shape = input_ids.size()
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        embedding_output = self._add_time_conditioning(embedding_output, timesteps)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
            device=input_ids.device,
        )
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)
        if self.mup_output_multiplier != 1.0:
            prediction_scores = prediction_scores * self.mup_output_multiplier

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
