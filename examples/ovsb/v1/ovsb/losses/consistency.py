"""Long-to-short consistency loss on pair batches.

A pair batch is a sequence of (anchor, view) couples on adjacent rows:
``pair_role == 1`` marks a long clean anchor and the row right after it
(``pair_role == 2``) is a short view cut from the same utterance. With
L2-normalised embeddings ``e`` the loss is

    L = mean over anchors a of  1 - <e[a + 1], stopgrad(e[a])>

so the short view is pulled towards the embedding of its own anchor and the
anchor receives no gradient from this term.
"""
from typing import Optional, Tuple

import torch


def pair_consistency_loss(
        emb: torch.Tensor,
        pair_role: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
    """Mean ``1 - cos(view, stopgrad(anchor))`` over the couples of a batch.

    Args:
        emb: (B, D) L2-normalised fp32 embeddings.
        pair_role: (B,) integer roles, 0 plain row, 1 anchor, 2 view. The
            view of the anchor on row ``i`` is row ``i + 1``.

    Returns:
        (loss, number of views). The loss is ``None`` when the batch has no
        anchor (a plain batch), so that the caller adds nothing.
    """
    anchors = torch.nonzero(pair_role.to(emb.device) == 1).squeeze(1)
    n_views = int(anchors.numel())
    if n_views == 0:
        return None, 0
    views = emb[anchors + 1]
    targets = emb[anchors].detach()
    return (1.0 - (views * targets).sum(1)).mean(), n_views
