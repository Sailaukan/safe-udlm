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


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class LossBreakdown:
    loss: torch.Tensor
    token_loss: torch.Tensor
    diffusion_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    loss_mask: torch.Tensor


class LogLinearNoiseSchedule(nn.Module):
    """Log-linear schedule used by continuous-time UDLM."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.total_noise(t), self.rate_noise(t)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)


class UniformDiscreteDiffusion(nn.Module):
    """Local UDLM engine with the same outer surface the project previously used for MDLM."""

    def __init__(
        self,
        vocab_size: int,
        mask_index: int,
        pad_index: int,
        bos_index: int,
        eos_index: int,
        time_distribution,
        noise_schedule: LogLinearNoiseSchedule,
        zero_recon_loss: bool = True,
        freeze_special_tokens: bool = True,
        sampling_steps: int = 128,
        sampling_eps: float = 1e-3,
        final_denoise: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.time_distribution = time_distribution
        self.noise_schedule = noise_schedule
        self.zero_recon_loss = zero_recon_loss
        self.freeze_special_tokens = freeze_special_tokens
        self.sampling_steps = sampling_steps
        self.sampling_eps = sampling_eps
        self.final_denoise = final_denoise

    def sample_time(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return self.time_distribution.sample(batch_size, device=device)

    def time_conditioning(self, t: torch.Tensor) -> torch.Tensor:
        sigma, _ = self.noise_schedule(t)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        return sigma

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.time_conditioning(t)
        return torch.exp(-sigma)

    def get_frozen_token_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.freeze_special_tokens:
            return torch.zeros_like(x, dtype=torch.bool)
        frozen_mask = x.eq(self.pad_index)
        frozen_mask |= x.eq(self.bos_index)
        frozen_mask |= x.eq(self.eos_index)
        return frozen_mask

    def _q_xt(self, x: torch.Tensor, move_chance: torch.Tensor) -> torch.Tensor:
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
        return torch.where(move_indices, uniform_tensor, x)

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        frozen_mask: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sigma is None:
            sigma = self.time_conditioning(t)
        move_chance = (1 - torch.exp(-sigma))[:, None]
        xt = self._q_xt(x0, move_chance)
        if frozen_mask is not None:
            xt = torch.where(frozen_mask, x0, xt)
        return xt

    def _compute_posterior(
        self,
        x: torch.Tensor,
        xt: torch.Tensor,
        alpha_s: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(x.dtype)

        numerator = (
            alpha_t * self.vocab_size * x * xt_one_hot
            + (alpha_ts - alpha_t) * xt_one_hot
            + d_alpha * x
            + (1 - alpha_ts) * (1 - alpha_s) / self.vocab_size
        )
        denominator = alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None]) + (1 - alpha_t)
        posterior = numerator / denominator.clamp_min(1e-12)
        return posterior.clamp_min(1e-12)

    def _token_losses(
        self,
        log_probs: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma, dsigma = self.noise_schedule(t)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if dsigma.ndim > 1:
            dsigma = dsigma.squeeze(-1)

        alpha_t = torch.exp(-sigma)[:, None, None]
        alpha_t_prime = (-dsigma * torch.exp(-sigma))[:, None, None]
        x0_one_hot = F.one_hot(x0, self.vocab_size).to(log_probs.dtype)
        x_theta = log_probs.exp()

        x_bar = self.vocab_size * alpha_t * x0_one_hot + (1 - alpha_t)
        x_bar_theta = self.vocab_size * alpha_t * x_theta + (1 - alpha_t)

        x_bar_zt = torch.gather(x_bar, -1, xt[..., None]).clamp_min(1e-12)
        x_bar_theta_zt = torch.gather(x_bar_theta, -1, xt[..., None]).clamp_min(1e-12)

        term1 = (self.vocab_size / x_bar_zt) - (self.vocab_size / x_bar_theta_zt)
        term2 = (
            (x_bar / x_bar_zt)
            * (
                x_bar_theta_zt.log()
                - x_bar_theta.clamp_min(1e-12).log()
                + x_bar.clamp_min(1e-12).log()
                - x_bar_zt.log()
            )
        ).sum(dim=-1, keepdim=True)

        diffusion_loss = (alpha_t_prime / alpha_t.clamp_min(1e-12) / self.vocab_size * (term1 - term2)).squeeze(-1)
        reconstruction_loss = -torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
        token_loss = diffusion_loss if self.zero_recon_loss else diffusion_loss + reconstruction_loss
        return token_loss, diffusion_loss, reconstruction_loss

    def loss_terms(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        frozen_mask: torch.Tensor | None = None,
        global_mean: bool = False,
    ) -> LossBreakdown:
        log_probs = logits.log_softmax(dim=-1)
        token_loss, diffusion_loss, reconstruction_loss = self._token_losses(log_probs, x0, xt, t)

        if mask is None:
            loss_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            loss_mask = mask.to(torch.bool)
        if frozen_mask is not None:
            loss_mask &= ~frozen_mask

        masked_token_loss = token_loss * loss_mask
        if global_mean:
            loss = masked_token_loss.sum() / loss_mask.sum().clamp_min(1)
        else:
            loss = masked_token_loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)

        return LossBreakdown(
            loss=loss,
            token_loss=token_loss,
            diffusion_loss=diffusion_loss,
            reconstruction_loss=reconstruction_loss,
            loss_mask=loss_mask,
        )

    def loss(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        global_mean: bool = False,
        frozen_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_terms(
            logits=logits,
            x0=x0,
            xt=xt,
            t=t,
            mask=mask,
            frozen_mask=frozen_mask,
            global_mean=global_mean,
        ).loss

    def initialize_sample(self, template: torch.Tensor, editable_mask: torch.Tensor) -> torch.Tensor:
        xt = template.clone()
        prior = torch.randint(0, self.vocab_size, template.shape, device=template.device)
        xt[editable_mask] = prior[editable_mask]
        return xt

    def get_num_steps_confidence(self, _: torch.Tensor | None = None) -> int:
        return max(int(self.sampling_steps), 2)

    def get_sampling_timesteps(self, device: torch.device | str, num_steps: int | None = None) -> torch.Tensor:
        num_steps = self.get_num_steps_confidence() if num_steps is None else max(int(num_steps), 2)
        return torch.linspace(1.0, self.sampling_eps, num_steps + 1, device=device)

    def _sample_with_randomness(self, probs: torch.Tensor, randomness: float) -> torch.Tensor:
        if randomness <= 0:
            return probs.argmax(dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        gumbel = -torch.log(-torch.log(torch.rand_like(log_probs).clamp_min(1e-12)))
        return (log_probs + randomness * gumbel).argmax(dim=-1)

    def _sample_non_special_tokens(self, count: int, device: torch.device) -> torch.Tensor:
        samples = torch.randint(0, self.vocab_size, (count,), device=device)
        special_mask = (
            samples.eq(self.pad_index)
            | samples.eq(self.bos_index)
            | samples.eq(self.eos_index)
            | samples.eq(self.mask_index)
        )
        while special_mask.any():
            samples[special_mask] = torch.randint(
                0,
                self.vocab_size,
                (int(special_mask.sum().item()),),
                device=device,
            )
            special_mask = (
                samples.eq(self.pad_index)
                | samples.eq(self.bos_index)
                | samples.eq(self.eos_index)
                | samples.eq(self.mask_index)
            )
        return samples

    def degrade_context(self, x: torch.Tensor, frozen_mask: torch.Tensor, gamma: float) -> torch.Tensor:
        if gamma <= 0:
            return x

        editable_context = frozen_mask.clone()
        editable_context &= ~x.eq(self.pad_index)
        editable_context &= ~x.eq(self.bos_index)
        editable_context &= ~x.eq(self.eos_index)

        degraded = x.clone()
        for batch_idx in range(x.shape[0]):
            candidate_ids = editable_context[batch_idx].nonzero(as_tuple=True)[0]
            if candidate_ids.numel() == 0:
                continue
            num_replace = min(candidate_ids.numel(), int(candidate_ids.numel() * gamma))
            if num_replace == 0 and gamma > 0:
                num_replace = 1
            if num_replace == 0:
                continue
            selected = candidate_ids[torch.randperm(candidate_ids.numel(), device=x.device)[:num_replace]]
            degraded[batch_idx, selected] = self._sample_non_special_tokens(num_replace, x.device)
        return degraded

    def step_confidence(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        step_idx: int,
        num_steps: int,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        editable_mask: torch.Tensor | None = None,
        timestep_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timestep_grid is None:
            timestep_grid = self.get_sampling_timesteps(xt.device, num_steps=num_steps)

        t = timestep_grid[step_idx].expand(xt.shape[0])
        s = timestep_grid[step_idx + 1].expand(xt.shape[0])
        sigma_t = self.time_conditioning(t)
        sigma_s = self.time_conditioning(s)
        alpha_t = torch.exp(-sigma_t)[:, None, None]
        alpha_s = torch.exp(-sigma_s)[:, None, None]

        log_probs = (logits / max(softmax_temp, 1e-6)).log_softmax(dim=-1)
        posterior = self._compute_posterior(log_probs.exp(), xt, alpha_s=alpha_s, alpha_t=alpha_t)
        xs = self._sample_with_randomness(posterior, randomness=randomness)

        if editable_mask is None:
            return xs
        return torch.where(editable_mask, xs, xt)

    def final_denoise_step(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        editable_mask: torch.Tensor,
        softmax_temp: float = 1.0,
        randomness: float = 0.0,
    ) -> torch.Tensor:
        probs = (logits / max(softmax_temp, 1e-6)).softmax(dim=-1)
        xs = self._sample_with_randomness(probs, randomness=randomness)
        return torch.where(editable_mask, xs, xt)
