"""Sub-centre ArcFace head with a per-row, duration-scaled target margin.

The core head applies one margin ``m`` to every row. Here row ``i`` gets
``m_i = m * g_i`` with ``g_i = min(1, len_i / (ref_sec * sample_rate))``, so
a crop shorter than ``ref_sec`` is asked for a proportionally smaller
angular margin. With ``c`` the largest sub-centre cosine per class and
``y_i`` the label of row ``i``:

    target logit:      phi(c, m_i) = cos(theta + m_i)        if c > cos(pi - m_i)
                                     c - (1 + cos(pi - m_i))  otherwise
    hard negatives:    cos(theta - mp)   for the k_top largest c_j, j != y_i
    all other logits:  c

all multiplied by ``scale`` (``easy_margin`` uses ``phi = c`` for ``c <= 0``
instead). The cosines are clamped to ``[-1 + 1e-7, 1 - 1e-7]`` before any
``sqrt(1 - c^2)``, whose derivative is infinite at ``|c| = 1``.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F

from wespeaker.models.projections import ArcMarginProduct_intertopk_subcenter

COS_EPS = 1e-7


def margin_scale_from_len(wav_len: torch.Tensor, ref_sec: float,
                          sample_rate: int = 16000) -> torch.Tensor:
    """``min(1, wav_len / (ref_sec * sample_rate))`` per row, fp32.

    Args:
        wav_len: (B,) valid samples per row.
        ref_sec: duration from which a row gets the full margin.
        sample_rate: samples per second of ``wav_len``.
    """
    ref = float(ref_sec) * sample_rate
    return (wav_len.float() / ref).clamp(max=1.0)


class DurationMarginSubcenterHead(ArcMarginProduct_intertopk_subcenter):
    """Core sub-centre head whose target margin can be scaled per row.

    Construction, the margin schedule (``update``) and the ``weight``
    parameter are those of the core class, so checkpoints are interchangeable
    with it. ``forward`` without ``margin_scale`` is the core forward.
    """

    def forward(self, input: torch.Tensor, label: torch.Tensor,
                margin_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Margin logits (B, out_features).

        Args:
            input: (B, in_features) fp32 embeddings.
            label: (B,) class labels.
            margin_scale: (B,) factors ``g_i`` on the target margin, or None
                for the core forward (one margin for all rows).
        """
        if margin_scale is None:
            return super().forward(input, label)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
        cosine, _ = torch.max(cosine, 2)  # (B, out_features)
        cosine = cosine.clamp(-1.0 + COS_EPS, 1.0 - COS_EPS)

        target = label.view(-1, 1).long()
        m = cosine.new_full((cosine.shape[0], 1), self.m) \
            * margin_scale.view(-1, 1).to(cosine.dtype)
        c = cosine.gather(1, target)
        s = torch.sqrt(1.0 - c * c)
        phi = c * torch.cos(m) - s * torch.sin(m)
        if self.easy_margin:
            phi = torch.where(c > 0, phi, c)
        else:
            edge = torch.cos(math.pi - m)
            phi = torch.where(c > edge, phi, c - (1.0 + edge))

        one_hot = input.new_zeros(cosine.size()).scatter_(1, target, 1)
        output = cosine * (1.0 - one_hot) + one_hot * phi
        if self.k_top > 0:
            sine = torch.sqrt(1.0 - cosine * cosine)
            phi_mp = cosine * self.cos_mp + sine * self.sin_mp
            _, top_k_index = torch.topk(cosine - 2 * one_hot, self.k_top)
            top_k = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)
            output = output + top_k * (phi_mp - cosine)
        return output * self.scale
