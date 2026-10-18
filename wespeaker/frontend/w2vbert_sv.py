# Copyright (c) 2024-2026 Junyi Peng (pengjy@fit.vut.cz)
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
"""W2V-BERT 2.0 frontend for speaker verification (`w2vbert_sv`).

Uses the official `transformers.Wav2Vec2BertModel` (the former in-tree copy
with HardConcrete pruning hooks was structurally identical for `prune: false`
and has been removed) and a torch re-implementation of the SeamlessM4T
feature extractor so that fbank computation runs on the GPU instead of in
numpy on the CPU inside every training step.

Outputs the stack of all hidden states (input to layer 1 + 24 layer outputs)
as (B, D, T, L) for the MHFA backends.
"""

import contextlib
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel
from transformers.audio_utils import mel_filter_bank, window_function


class SeamlessFbank(nn.Module):
    """Torch port of `transformers.SeamlessM4TFeatureExtractor`.

    80-dim Kaldi-style log-mel fbank (povey window, 25 ms / 10 ms, pre-emphasis
    0.97, DC removal, 512-point FFT, mel floor 1.19e-7), per-utterance
    per-bin mean/variance normalization (unbiased variance, eps 1e-7), then
    stride-2 frame stacking to 160-dim / 20 ms frames. Device-agnostic;
    the numpy original computes the FFT in float64, this runs in float32.
    """

    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80,
                 stride: int = 2):
        super().__init__()
        self.frame_length = 400
        self.hop_length = 160
        self.fft_length = 512
        self.stride = stride
        self.preemphasis = 0.97
        self.mel_floor = 1.192092955078125e-07
        mel = mel_filter_bank(num_frequency_bins=self.fft_length // 2 + 1,
                              num_mel_filters=num_mel_bins,
                              min_frequency=20,
                              max_frequency=sample_rate // 2,
                              sampling_rate=sample_rate,
                              norm=None,
                              mel_scale="kaldi",
                              triangularize_in_mel_space=True)
        self.register_buffer('mel_filters',
                             torch.tensor(mel, dtype=torch.float32),
                             persistent=False)  # (257, num_mel_bins)
        self.register_buffer(
            'window',
            torch.tensor(window_function(self.frame_length, "povey",
                                         periodic=False),
                         dtype=torch.float32),
            persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) float waveform in [-1, 1] -> (B, T', 2 * num_mel_bins)."""
        with torch.autocast(device_type=wav.device.type, enabled=False):
            min_len = self.frame_length + self.hop_length * (2 * self.stride - 1)
            if wav.shape[-1] < min_len:  # < 2 stacked frames: pad with zeros
                wav = torch.nn.functional.pad(wav, (0, min_len - wav.shape[-1]))
            wav = wav.float() * 32768.0  # Kaldi compliance: int16 scale
            frames = wav.unfold(-1, self.frame_length, self.hop_length)
            frames = frames - frames.mean(-1, keepdim=True)  # remove DC
            frames = torch.cat(
                [frames[..., :1] * (1.0 - self.preemphasis),
                 frames[..., 1:] - self.preemphasis * frames[..., :-1]],
                dim=-1)
            frames = frames * self.window
            power = torch.fft.rfft(frames, n=self.fft_length).abs().pow(2)
            mel = torch.clamp(power @ self.mel_filters, min=self.mel_floor)
            mel = mel.log()
            mean = mel.mean(dim=1, keepdim=True)
            var = mel.var(dim=1, keepdim=True, unbiased=True)
            mel = (mel - mean) / torch.sqrt(var + 1e-7)
            b, t, c = mel.shape
            t2 = t - t % self.stride
            return mel[:, :t2].reshape(b, t2 // self.stride, c * self.stride)


class W2VBertSVFrontend(nn.Module):
    """W2V-BERT 2.0 encoder wrapper exposing all hidden states.

    Args:
        upstream_args: {'path_or_url': <dir with config.json +
            model.safetensors>, 'name': <ignored, kept for config
            compatibility>}
        frozen: freeze the encoder (parameters get no gradient, forward runs
            under no_grad).
        frame_shift / frame_length: 20 ms (informational, used by the dataset
            pipeline to turn num_frms into samples).
        gradient_checkpointing: activation checkpointing of the conformer
            layers (needed for full fine-tuning at bs 48 x 4 s on 40 GB).
        feature_extractor: 'torch' (GPU fbank, default) or 'hf' (the original
            numpy extractor on the CPU, for equivalence checks only).
        max_input_seconds: inputs are truncated to this length (the original
            recipe used 10 s).
    """

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
        gradient_checkpointing: bool = False,
        feature_extractor: str = 'torch',
        max_input_seconds: float = 10.0,
        download_dir: Optional[str] = None,  # unused, config compatibility
    ):
        super().__init__()
        path = upstream_args['path_or_url']
        self.frozen = frozen
        self.max_input_samples = int(max_input_seconds * sample_rate)
        self.feature_extractor_type = feature_extractor

        if feature_extractor == 'torch':
            self.fbank = SeamlessFbank(sample_rate=sample_rate)
        elif feature_extractor == 'hf':
            from transformers import SeamlessM4TFeatureExtractor
            self.fbank = None
            self.hf_feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                path, local_files_only=True)
        else:
            raise ValueError(f'unknown feature_extractor {feature_extractor}')

        self.upstream = Wav2Vec2BertModel.from_pretrained(
            path, local_files_only=True)
        # Not used by the layer loop below; removing it keeps DDP from
        # complaining about a parameter without gradient.
        if hasattr(self.upstream, 'masked_spec_embed'):
            delattr(self.upstream, 'masked_spec_embed')
        self.d_model = self.upstream.config.hidden_size
        self.n_hidden_states = self.upstream.config.num_hidden_layers + 1

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()
        else:
            for param in self.upstream.parameters():
                param.requires_grad_(True)
            # Wav2Vec2BertEncoderLayer subclasses GradientCheckpointingLayer,
            # so the per-layer __call__ in forward() honours the flag.
            self.gradient_checkpointing = gradient_checkpointing
            if gradient_checkpointing:
                self.upstream.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})

    def output_size(self) -> int:
        return self.d_model

    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor_type == 'torch':
            return self.fbank(wav)
        feats = self.hf_feature_extractor(
            wav.detach().cpu().numpy(), sampling_rate=16000,
            return_tensors="pt", padding=False, truncation=False,
            return_attention_mask=False)
        return feats.input_features.to(wav.device)

    def forward(
        self, input_wav: torch.Tensor,
        input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """input_wav (B, T) -> ((B, D, T', L), None | frame lengths).

        If input_lengths marks a ragged (zero-padded) batch, every utterance
        gets its own fbank + normalisation, padded frames are masked in the
        attention and conv modules (as transformers' Wav2Vec2BertEncoder
        does) and the valid frame counts are returned for the backend.
        Equal lengths (training) take the unmasked path unchanged.
        """
        x = input_wav[:, :self.max_input_samples]
        if input_lengths is not None:
            lens = input_lengths.clamp(max=x.shape[1])
            if bool((lens < x.shape[1]).any()):
                return self._forward_ragged(x, lens)
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            feats = self.extract_features(x)
            feats = feats.to(next(self.upstream.parameters()).dtype)
            h = self.upstream.feature_projection(feats)[0]
            hidden_states = [h]
            for layer in self.upstream.encoder.layers:
                out = layer(h)
                # transformers <5 returns a tuple (hidden_states, attn);
                # >=5 returns the tensor itself (indexing [0] would then
                # silently select batch element 0)
                h = out[0] if isinstance(out, (tuple, list)) else out
                hidden_states.append(h)
        return torch.stack(hidden_states, dim=0).permute(1, 3, 2, 0), None

    def _forward_ragged(self, x: torch.Tensor, lens: torch.Tensor):
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            lens_l = lens.tolist()  # one host sync instead of one per utterance
            feats = [self.extract_features(x[i:i + 1, :lens_l[i]])[0]
                     for i in range(x.shape[0])]
            flens = torch.tensor([f.shape[0] for f in feats], device=x.device)
            feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
            mask = torch.arange(feats.shape[1], device=x.device)[None, :] \
                < flens[:, None]
            feats = feats.to(next(self.upstream.parameters()).dtype)
            h = self.upstream.feature_projection(feats)[0]
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
            ext = (1.0 - mask[:, None, None, :].to(h.dtype)) \
                * torch.finfo(h.dtype).min
            ext = ext.expand(-1, 1, mask.shape[1], mask.shape[1])
            hidden_states = [h]
            for layer in self.upstream.encoder.layers:
                out = layer(h, attention_mask=ext, conv_attention_mask=mask)
                h = out[0] if isinstance(out, (tuple, list)) else out
                hidden_states.append(h)
        return torch.stack(hidden_states, dim=0).permute(1, 3, 2, 0), flens
