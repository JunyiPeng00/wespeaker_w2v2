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
"""omniASR wav2vec 2.0 SSL encoder frontend for speaker verification.

Meta's Omnilingual ASR W2V encoders (omniASR-W2V-300M/1B/3B/7B) are plain
pre-LN ("stable layer norm") wav2vec 2.0 Transformers. They are loaded with
`transformers.Wav2Vec2Model` from a HF conversion of the official fairseq2
checkpoint (aadel4/omniASR-W2V-1B: all 805 encoder tensors are bit-identical
to facebook/omniASR-W2V-1B, checked 2026-10-18).

Input: 16 kHz waveform, per-utterance layer norm (zero mean, unit variance,
no affine), as in the omniASR pipeline (`normalize_audio=True`).
Output: the input of layer 1 (CNN features + projection + conv positions)
and every layer output, stacked as (B, D, T, L) for the MHFA backends,
T = floor((N - 400) / 320) + 1 frames at 20 ms.

Pre-LN residual streams are not normalised and their scale grows with depth,
whereas W2V-BERT conformer blocks end in a LayerNorm. MHFA mixes layers with a
softmax before any normalisation, so by default every state gets a
parameter-free LayerNorm over D (`layer_norm_states`).
"""

import contextlib
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class OmniASRSVFrontend(nn.Module):
    """Wav2Vec2 (omniASR W2V) encoder wrapper exposing all hidden states.

    Args:
        upstream_args: {'path_or_url': <dir with config.json +
            model.safetensors>}
        frozen: freeze the encoder (no gradient, forward under no_grad, and
            the encoder stays in eval mode even when the model is in train
            mode).
        frame_shift / frame_length: 20 ms (used by the dataset pipeline to
            turn num_frms into samples).
        gradient_checkpointing: activation checkpointing of the transformer
            layers (full fine-tuning).
        disable_dropout: set every encoder dropout and LayerDrop to 0 (the
            pretraining config has 0.1).
        layer_norm_states: parameter-free LayerNorm on every stacked state.
        normalize_waveform: per-utterance waveform layer norm.
        max_input_seconds: inputs are truncated to this length.
        attn_implementation: 'sdpa' (default) or 'eager'.
    """

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
        gradient_checkpointing: bool = False,
        disable_dropout: bool = True,
        layer_norm_states: bool = True,
        normalize_waveform: bool = True,
        max_input_seconds: float = 10.0,
        attn_implementation: str = 'sdpa',
        download_dir: Optional[str] = None,  # unused, config compatibility
    ):
        super().__init__()
        path = upstream_args['path_or_url']
        self.frozen = frozen
        self.max_input_samples = int(max_input_seconds * sample_rate)
        self.layer_norm_states = layer_norm_states
        self.attn_implementation = attn_implementation
        self.normalize_waveform = normalize_waveform

        kwargs = dict(apply_spec_augment=False, mask_time_prob=0.0,
                      mask_feature_prob=0.0)
        if disable_dropout:
            kwargs.update(layerdrop=0.0, hidden_dropout=0.0,
                          attention_dropout=0.0, activation_dropout=0.0,
                          feat_proj_dropout=0.0)
        self.upstream = Wav2Vec2Model.from_pretrained(
            path, local_files_only=True,
            attn_implementation=attn_implementation, **kwargs)
        cfg = self.upstream.config
        assert cfg.do_stable_layer_norm, 'expects a pre-LN wav2vec 2.0'
        # Not used by forward(); removing it keeps DDP from complaining about
        # a parameter without gradient.
        if hasattr(self.upstream, 'masked_spec_embed'):
            delattr(self.upstream, 'masked_spec_embed')
        # The final encoder LayerNorm is not used either: the last stacked
        # state is the raw output of the last layer (normalised here by
        # layer_norm_states when enabled).
        if hasattr(self.upstream.encoder, 'layer_norm'):
            self.upstream.encoder.layer_norm = nn.Identity()
        self.d_model = cfg.hidden_size
        self.n_hidden_states = cfg.num_hidden_layers + 1

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()
        else:
            for param in self.upstream.parameters():
                param.requires_grad_(True)
            self.gradient_checkpointing = gradient_checkpointing
            if gradient_checkpointing:
                self.upstream.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.upstream.eval()
        return self

    def output_size(self) -> int:
        return self.d_model

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        if not self.layer_norm_states:
            return h
        return F.layer_norm(h.float(), h.shape[-1:]).to(h.dtype)

    def forward(
        self, input_wav: torch.Tensor,
        input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """input_wav (B, N) -> ((B, D, T, L), None)."""
        x = input_wav[:, :self.max_input_samples]
        if input_lengths is not None:
            lens = input_lengths.clamp(max=x.shape[1])
            if bool((lens < x.shape[1]).any()):
                return self._forward_ragged(x, lens)
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            if self.normalize_waveform:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    x = F.layer_norm(x.float(), x.shape[-1:])
            feats = self.upstream.feature_extractor(x).transpose(1, 2)
            h = self.upstream.feature_projection(feats)[0]
            h = h + self.upstream.encoder.pos_conv_embed(h)
            hidden_states = [self._norm(h)]
            for layer in self.upstream.encoder.layers:
                out = layer(h, attention_mask=None)
                # transformers <5 returns a tuple (hidden_states, attn)
                h = out[0] if isinstance(out, (tuple, list)) else out
                hidden_states.append(self._norm(h))
        return torch.stack(hidden_states, dim=0).permute(1, 3, 2, 0), None

    def _forward_ragged(self, x: torch.Tensor, lens: torch.Tensor):
        """Zero-padded batch: per-utterance waveform norm over the valid
        samples; CNN frames whose receptive field lies inside the valid
        samples are exact; padded frames are zeroed before the conv position
        encoder and masked in self-attention. Returns frame lengths."""
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            valid_s = torch.arange(x.shape[1], device=x.device)[None, :] \
                < lens[:, None]
            if self.normalize_waveform:
                lens_l = lens.tolist()  # one host sync
                with torch.autocast(device_type=x.device.type, enabled=False):
                    x = torch.stack([torch.nn.functional.pad(
                        F.layer_norm(x[i, :n].float(), (n,)),
                        (0, x.shape[1] - n))
                        for i, n in enumerate(lens_l)])
            x = x.masked_fill(~valid_s, 0.0)
            feats = self.upstream.feature_extractor(x).transpose(1, 2)
            flens = self.upstream._get_feat_extract_output_lengths(
                lens).to(torch.long)
            mask = torch.arange(feats.shape[1], device=x.device)[None, :] \
                < flens[:, None]
            h = self.upstream.feature_projection(feats)[0]
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
            h = h + self.upstream.encoder.pos_conv_embed(h)
            if self.attn_implementation == 'sdpa':
                # boolean keep-mask: SDPA needs a float mask in the query
                # dtype, which differs under autocast
                ext = mask[:, None, None, :].expand(
                    -1, 1, mask.shape[1], mask.shape[1])
            else:
                ext = (1.0 - mask[:, None, None, :].to(h.dtype)) \
                    * torch.finfo(h.dtype).min
                ext = ext.expand(-1, 1, mask.shape[1], mask.shape[1])
            hidden_states = [self._norm(h)]
            for layer in self.upstream.encoder.layers:
                out = layer(h, attention_mask=ext)
                h = out[0] if isinstance(out, (tuple, list)) else out
                hidden_states.append(self._norm(h))
        return torch.stack(hidden_states, dim=0).permute(1, 3, 2, 0), flens
