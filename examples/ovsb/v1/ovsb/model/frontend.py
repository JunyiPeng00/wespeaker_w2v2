"""W2V-BERT 2.0 frontend with low-ratio time masking of the input frames.

In training, random spans of the projected input frames (the encoder input,
after the feature projection) are set to zero, in the style of wav2vec 2.0
fine-tuning. For an utterance with ``L`` valid frames, mask ratio ``p`` and
span length ``s``:

    spans   n = floor(L * p / s + u),  u ~ U[0, 1)   (stochastic rounding)
    starts  floor(v * max(1, L - s + 1)),  v ~ U[0, 1), one per span

and every frame covered by a span and inside the valid length is masked.
The expected masked fraction is about ``p`` for ``L >= s`` (overlapping
spans only lower it); an utterance shorter than ``s`` gets one truncated
span with probability ``p * L / s``. Padded frames are never masked.
"""
from typing import Any, List, Mapping, Optional

import torch

from wespeaker.frontend.w2vbert_sv import W2VBertSVFrontend


def draw_time_mask(n_frames: int, valid: torch.Tensor, prob: float,
                   span: int) -> torch.Tensor:
    """Draw the boolean frame mask (B, n_frames) described above.

    Args:
        n_frames: padded number of frames ``T`` of the batch.
        valid: (B,) fp32 valid frames per row, on the target device.
        prob: expected fraction ``p`` of masked valid frames.
        span: frames ``s`` per masked span.

    Two draws from the global generator of ``valid.device``, in this order:
    ``rand(B)`` for the span counts, then ``rand(B, K)`` for the starts with
    ``K = floor(T * p / s) + 1``, the largest possible count. No host sync.
    """
    dev = valid.device
    batch = valid.shape[0]
    n = torch.floor(valid * prob / span
                    + torch.rand(batch, device=dev)).long()
    max_n = int(n_frames * prob / span) + 1
    starts = (torch.rand(batch, max_n, device=dev)
              * (valid[:, None] - span + 1).clamp(min=1.0)).long()
    active = torch.arange(max_n, device=dev)[None, :] < n[:, None]
    idx = starts[:, :, None] + torch.arange(span, device=dev)[None, None]
    hit = torch.zeros(batch, n_frames + span, dtype=torch.int32, device=dev)
    hit.scatter_add_(1, idx.reshape(batch, -1),
                     active[:, :, None].expand(-1, -1, span)
                     .reshape(batch, -1).to(torch.int32))
    return (hit[:, :n_frames] > 0) & (
        torch.arange(n_frames, device=dev)[None, :] < valid[:, None])


class TimeMaskedW2VBertSV(W2VBertSVFrontend):
    """``W2VBertSVFrontend`` that zeroes random frame spans in training.

    Args:
        time_mask: ``{'prob': p, 'span': s}``; ``p`` in [0, 0.5] is the
            expected masked fraction of the valid frames (default 0, no
            masking), ``s >= 1`` the span length in frames (default 10).
        *args, **kwargs: passed to ``W2VBertSVFrontend`` unchanged.

    Masking needs a trainable encoder and is skipped in eval mode.
    """

    def __init__(self, *args: Any,
                 time_mask: Optional[Mapping[str, Any]] = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        conf = dict(time_mask or {})
        self.time_mask_prob = float(conf.pop('prob', 0.0))
        self.time_mask_span = int(conf.pop('span', 10))
        if conf:
            raise ValueError(f'unknown time_mask keys {sorted(conf)}')
        if not 0.0 <= self.time_mask_prob <= 0.5:
            raise ValueError('time_mask.prob must be in [0, 0.5], got '
                             f'{self.time_mask_prob}')
        if self.time_mask_span < 1:
            raise ValueError('time_mask.span must be >= 1, got '
                             f'{self.time_mask_span}')
        if self.frozen and self.time_mask_prob > 0:
            raise ValueError('time_mask needs a trainable encoder')
        # device tensor [masked frames, valid frames], summed over batches
        self._masked_valid: Optional[torch.Tensor] = None

    def _mask_frames(self, h: torch.Tensor,
                     flens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Zero random spans of the valid frames of ``h`` (B, T, D).

        ``flens`` (B,) are the valid frames per row of a padded batch; None
        means every row has ``T`` valid frames. Identity in eval mode and
        with ``prob`` 0 (no random draw then).
        """
        if not self.training or self.time_mask_prob <= 0:
            return h
        batch, n_frames, _ = h.shape
        valid = (flens.to(h.device).float() if flens is not None
                 else torch.full((batch,), float(n_frames), device=h.device))
        mask = draw_time_mask(n_frames, valid, self.time_mask_prob,
                              self.time_mask_span)
        seen = torch.stack([mask.sum(), valid.sum().long()])
        self._masked_valid = seen if self._masked_valid is None \
            else self._masked_valid + seen
        return h.masked_fill(mask.unsqueeze(-1), 0.0)

    @property
    def time_mask_stats(self) -> List[int]:
        """[masked, valid] frames seen by this process since construction.

        Cumulative over epochs and local to this process (one DDP rank);
        reading it costs one host sync.
        """
        if self._masked_valid is None:
            return [0, 0]
        return self._masked_valid.tolist()
