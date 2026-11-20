"""The training objective: margin cross-entropy plus the two pair terms.

    total = CE(logits, labels)
            + w_cons * L_cons    (only when the batch has anchor rows)
            + w_pauc * L_pauc    (only when the batch has both pair kinds)

* ``CE`` is the mean cross-entropy of the margin logits over every row of
  the batch: plain rows, anchors and views alike.
* ``L_cons`` (``consistency.pair_consistency_loss``) is computed on the
  L2-normalised fp32 embeddings; it exists on pair batches only.
* ``L_pauc`` (``pair_auc.pair_auc_loss``) is computed on every batch, on the
  raw embeddings minus the running centre once the centre is warmed up. The
  centre is updated with the raw embeddings of the current batch first.

All three terms are evaluated in fp32 with autocast disabled.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ovsb.losses.consistency import pair_consistency_loss
from ovsb.losses.pair_auc import RunningCenter, pair_auc_loss

Components = Dict[str, Optional[torch.Tensor]]
Stats = Dict[str, float]


class LossBundle:
    """Combine the cross-entropy, consistency and pair-AUC terms.

    Args:
        consistency_weight: ``w_cons``; 0 leaves the term out.
        pair_auc_weight: ``w_pauc``; 0 leaves the term (and the centre) out.
        pair_auc_scale: the factor on the cosine differences in ``L_pauc``.
        center: running centre subtracted from the embeddings in ``L_pauc``,
            or None to score the uncentred cosine.
    """

    def __init__(self, consistency_weight: float = 0.0,
                 pair_auc_weight: float = 0.0, pair_auc_scale: float = 16.0,
                 center: Optional[RunningCenter] = None) -> None:
        self.consistency_weight = float(consistency_weight)
        self.pair_auc_weight = float(pair_auc_weight)
        self.pair_auc_scale = float(pair_auc_scale)
        self.center = center if self.pair_auc_weight > 0 else None

    def __call__(
            self,
            logits: torch.Tensor,
            emb: torch.Tensor,
            labels: torch.Tensor,
            pair_role: torch.Tensor,
            arm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Components, Stats]:
        """Evaluate the objective on one batch.

        Args:
            logits: (B, C) margin logits of the classifier head.
            emb: (B, D) raw (unnormalised) embeddings the logits came from.
            labels: (B,) class labels on the device of ``logits``.
            pair_role: (B,) roles, 0 plain row, 1 anchor, 2 view of the
                anchor on the row before; all zeros on a plain batch.
            arm: (B,) sampler arm index of each row; needed with a centre.

        Returns:
            (total, components, stats). ``components`` maps 'ce', 'cons',
            'pauc' to the detached unweighted terms; a term that did not
            enter ``total`` on this batch is None. ``stats`` holds the pair
            counts and mean cosines of this batch.
        """
        with torch.autocast(emb.device.type, enabled=False):
            emb = emb.float()
            total = F.cross_entropy(logits.float(), labels)
            parts: Components = {'ce': total.detach(), 'cons': None,
                                 'pauc': None}
            stats: Stats = {}
            if self.consistency_weight > 0:
                cons, n_views = pair_consistency_loss(
                    F.normalize(emb, dim=1), pair_role)
                stats['cons_views'] = n_views
                if cons is not None:
                    total = total + self.consistency_weight * cons
                    parts['cons'] = cons.detach()
            if self.pair_auc_weight > 0:
                scored = emb
                if self.center is not None:
                    if arm is None:
                        raise ValueError(
                            'the pair-AUC centre needs the arm of every row')
                    scored = self.center(emb, arm)
                pauc, pair_stats = pair_auc_loss(
                    scored, labels, self.pair_auc_scale, pair_role)
                stats.update(pauc_pos=pair_stats.n_pos,
                             pauc_neg=pair_stats.n_neg,
                             pauc_cos_pos=pair_stats.pos_mean,
                             pauc_cos_neg=pair_stats.neg_mean)
                if pauc is not None:
                    total = total + self.pair_auc_weight * pauc
                    parts['pauc'] = pauc.detach()
        return total, parts, stats
