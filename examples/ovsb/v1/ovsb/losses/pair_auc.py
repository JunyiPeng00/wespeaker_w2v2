"""Pooled pairwise verification loss and the running centre it scores on.

With ``s_ij = cos(x_i, x_j)`` over the rows of one batch, target pairs
``P`` (same label, ``i < j``) and nontarget pairs ``N`` (different labels,
``i < j``), the loss is

    L = mean over p in P of  log(1 + sum over n in N of exp(a (s_n - s_p)))

i.e. ``softplus(logsumexp(a s_N) - a s_p)`` averaged over the target pairs.
The pooled log-sum-exp acts as one soft threshold shared by every target
pair of the batch, whatever the length or condition of its two rows. A
consistency anchor and its own view (``pair_role`` 1 followed by 2 on the
next row) are the same utterance and are not a target pair.

Test-time scoring is the cosine after subtracting a corpus mean embedding.
``RunningCenter`` supplies the training-time counterpart: a detached,
bias-corrected exponential moving average of the mean raw embedding of the
batch rows of one sampler arm, subtracted from every row before the pair
scores are computed.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F


class PairAucStats(NamedTuple):
    """Pair counts and mean cosines of one batch (means are 0.0 if unset)."""
    n_pos: int
    n_neg: int
    pos_mean: float
    neg_mean: float


def pair_auc_loss(
        emb: torch.Tensor,
        labels: torch.Tensor,
        scale: float,
        pair_role: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], PairAucStats]:
    """``mean_p softplus(logsumexp_n(scale * s_n) - scale * s_p)``.

    Args:
        emb: (B, D) embeddings, any float dtype; cast to fp32 and
            L2-normalised here, so ``s`` is the cosine matrix.
        labels: (B,) class labels on the device of ``emb``.
        scale: the factor ``a`` on the cosine differences.
        pair_role: optional (B,) roles (0 plain, 1 anchor, 2 view); each
            (anchor ``i``, view ``i + 1``) couple is removed from the pairs.

    Returns:
        (loss, stats). The loss is ``None`` when the batch has no target
        pair or no nontarget pair. Gradient flows to both pair kinds.
    """
    b = emb.shape[0]
    e = F.normalize(emb.float(), dim=1)
    s = e @ e.t()
    valid = torch.triu(
        torch.ones(b, b, dtype=torch.bool, device=emb.device), 1)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    if pair_role is not None:
        pr = pair_role.to(emb.device)
        idx = torch.nonzero(pr[:-1].eq(1) & pr[1:].eq(2)).squeeze(1)
        if idx.numel() > 0:  # anchor i and its view i + 1: same utterance
            valid[idx, idx + 1] = False
    pos = s[valid & same]
    neg = s[valid & ~same]
    n_pos, n_neg = int(pos.numel()), int(neg.numel())
    if n_pos == 0 or n_neg == 0:
        return None, PairAucStats(n_pos, n_neg, 0.0, 0.0)
    loss = F.softplus(torch.logsumexp(scale * neg, 0) - scale * pos).mean()
    return loss, PairAucStats(n_pos, n_neg, float(pos.mean().item()),
                              float(neg.mean().item()))


class RunningCenter:
    """Bias-corrected EMA of the mean raw embedding of one arm's rows.

    For the ``n``-th batch that holds at least one row of the arm, with
    ``m_n`` the mean of those rows (fp32, detached) and momentum ``mu``:

        S_n = mu * S_{n-1} + (1 - mu) * m_n,   S_0 = 0
        mean_n = S_n / (1 - mu ** n)

    so the first batches form a plain running average instead of being
    dominated by the zero start. ``center`` subtracts ``mean`` once ``n``
    has reached ``warmup``; before that it returns its input unchanged.
    The mean carries no gradient. A batch whose arm-row mean is not finite
    leaves the state untouched and is counted in ``n_skipped``.

    One object belongs to one process (one DDP rank) and is meant to live
    for the whole run; ``state_dict`` / ``load_state_dict`` carry it across
    a restart.
    """

    def __init__(self, arm_index: int, momentum: float = 0.99,
                 warmup: int = 20) -> None:
        self.arm = int(arm_index)
        self.momentum = float(momentum)
        self.warmup = int(warmup)
        if not 0.0 < self.momentum < 1.0:
            raise ValueError(f'momentum must be in (0, 1), got {momentum}')
        if self.warmup < 1:
            raise ValueError(f'warmup must be >= 1, got {warmup}')
        self.acc: Optional[torch.Tensor] = None   # S_n
        self.mean: Optional[torch.Tensor] = None  # S_n / (1 - mu ** n)
        self.n_upd = 0       # n: batches that updated the state
        self.n_rows = 0      # arm rows seen in those batches
        self.n_skipped = 0   # batches skipped for a non-finite arm mean
        self.row_norm = 0.0  # mean L2 norm of the arm rows, last update

    @torch.no_grad()
    def update(self, emb: torch.Tensor, arm: torch.Tensor) -> bool:
        """Fold the arm rows of ``emb`` (B, D) into the running mean.

        Returns True when the state changed: False for a batch without a
        row of the arm and for a non-finite arm-row mean (also counted).
        """
        rows = emb.detach().float()[arm.to(emb.device) == self.arm]
        if rows.shape[0] == 0:
            return False
        m = rows.mean(0)
        if not bool(torch.isfinite(m).all()):
            self.n_skipped += 1
            return False
        if self.acc is None:
            self.acc = torch.zeros_like(m)
        elif self.acc.device != m.device:  # restored state: follow the rows
            self.acc = self.acc.to(m.device)
        self.acc.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)
        self.n_upd += 1
        # a new tensor each time, so graphs built on the old mean are intact
        self.mean = self.acc / (1.0 - self.momentum ** self.n_upd)
        self.n_rows += int(rows.shape[0])
        self.row_norm = float(rows.norm(dim=1).mean())
        return True

    @property
    def ready(self) -> bool:
        """True once ``warmup`` batches have updated the mean."""
        return self.mean is not None and self.n_upd >= self.warmup

    def center(self, emb: torch.Tensor) -> torch.Tensor:
        """``emb.float() - mean`` when ready, else ``emb`` itself."""
        if not self.ready:
            return emb
        return emb.float() - self.mean.to(emb.device)

    def __call__(self, emb: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
        """Update with this batch, then centre it (update comes first)."""
        self.update(emb, arm)
        return self.center(emb)

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to continue the average after a restart."""
        return {
            'arm': self.arm,
            'momentum': self.momentum,
            'warmup': self.warmup,
            'acc': None if self.acc is None else self.acc.detach().cpu().clone(),
            'n_upd': self.n_upd,
            'n_rows': self.n_rows,
            'n_skipped': self.n_skipped,
            'row_norm': self.row_norm,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a ``state_dict``; the settings must match this object."""
        for key in ('arm', 'momentum', 'warmup'):
            if state[key] != getattr(self, key):
                raise ValueError(
                    'centre state was saved with %s=%r, this object has %r'
                    % (key, state[key], getattr(self, key)))
        self.n_upd = int(state['n_upd'])
        self.n_rows = int(state['n_rows'])
        self.n_skipped = int(state['n_skipped'])
        self.row_norm = float(state['row_norm'])
        if state['acc'] is None:
            self.acc, self.mean = None, None
        else:
            self.acc = state['acc'].detach().float().clone()
            self.mean = self.acc / (1.0 - self.momentum ** self.n_upd)

    def describe(self) -> str:
        """One log line with the state of the average."""
        if self.mean is None:
            return ('pair-auc centre: no row of the arm yet (%d skipped)'
                    % self.n_skipped)
        return ('pair-auc centre: %s, %d updates / %d rows, %d skipped, '
                '|mean| %.4f, mean row norm of the arm (last update) %.4f'
                % ('on' if self.ready else 'warm-up', self.n_upd,
                   self.n_rows, self.n_skipped, float(self.mean.norm()),
                   self.row_norm))
