# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exponential moving average of model weights (config key ema_decay)."""

import torch


class ModelEMA:
    """fp32 EMA of the TRAINABLE parameters of `model` (the unwrapped
    module, not the DDP wrapper).

    Frozen parameters and buffers are not tracked: they are taken from the
    live model when the EMA state dict is built, so a frozen-frontend
    stage-1 only stores the head. Under DDP all ranks hold identical
    parameters after each step, so keeping the EMA on rank 0 only is exact.

    ema = decay * ema + (1 - decay) * param after every optimizer step;
    with warmup, decay_t = min(decay, (1 + t) / (10 + t)).
    """

    def __init__(self, model, decay=0.9999, warmup=False):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.names = [n for n, p in model.named_parameters()
                      if p.requires_grad]
        params = dict(model.named_parameters())
        self.shadow = [params[n].detach().clone().float() for n in self.names]

    def current_decay(self):
        if self.warmup:
            t = self.num_updates
            return min(self.decay, (1.0 + t) / (10.0 + t))
        return self.decay

    @torch.no_grad()
    def update(self, model):
        decay = self.current_decay()
        params = dict(model.named_parameters())
        current = [params[n].detach().float() for n in self.names]
        torch._foreach_lerp_(self.shadow, current, 1.0 - decay)
        self.num_updates += 1

    @torch.no_grad()
    def model_state_dict(self, model):
        """Full state dict of `model` with the EMA weights substituted; same
        format as model_N.pt, so extract.py can load it directly."""
        state = model.state_dict()
        for n, s in zip(self.names, self.shadow):
            state[n] = s.to(dtype=state[n].dtype).clone()
        return state

    @torch.no_grad()
    def load_model_state_dict(self, state, num_updates=0):
        """Restore the shadow from a saved ema_model_N.pt state dict."""
        missing = [n for n in self.names if n not in state]
        assert not missing, 'EMA state lacks {}'.format(missing[:5])
        for n, s in zip(self.names, self.shadow):
            s.copy_(state[n].to(device=s.device, dtype=s.dtype))
        self.num_updates = int(num_updates)
