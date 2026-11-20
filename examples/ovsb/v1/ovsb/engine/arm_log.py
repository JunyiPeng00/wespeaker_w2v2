"""Per-arm training log.

At every log interval one ``ARMLOG`` line per sampler arm, accumulated over
the interval and summed over all processes:

    ARMLOG epoch <e> batch <b> arm <name> {json}

with, per arm: ``rows`` and ``share`` of all rows; ``ce`` (mean
cross-entropy of the margin logits) and ``acc`` (argmax == label); ``view``
(share of consistency views); zero-run shares of the crop before
augmentation (``zr_lsb50`` rows with more than half of the samples at the
LSB level, ``zr_run50ms`` rows with a long run of exact zeros, ``zr_dith``
rows dithered, ``zr_bypass`` rows that skipped the codec chain);
``chain`` (share of rows through the codec chain) and ``codec`` shares;
energy gate (``eg_gated`` share of gated rows, ``eg_redraw`` redraws and
``eg_fail`` failures per gated row); ``aug`` (realised augmentation
types); ``ptype`` (positive-pair types of the arm's speaker groups, taken
from the first row of each group that is not a consistency view).

Costs one (B x C) cross-entropy on detached logits and a few bincounts per
step.
"""
import json
import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from ovsb.config.schema import AUG_TYPES, CODECS, Config
from ovsb.engine.logging import all_reduce_sum

_BASE = ('rows', 'ce_sum', 'correct', 'views', 'zr_lsb50', 'zr_run',
         'zr_dith', 'zr_byp', 'chain', 'eg_rows', 'eg_redraw', 'eg_fail')


class ArmLogger:
    """Accumulates per-arm sums of the batches and prints them.

    Args:
        arm_names: sampler arms in index order (``batch['arm']`` indexes
            this list).
        ptypes: names of the positive-pair types in code order
            (``batch['ptype']`` indexes this list).
        device: training device of this process, used for the all-reduce.
    """

    def __init__(self, arm_names: Sequence[str], ptypes: Sequence[str],
                 device: torch.device) -> None:
        self.names = list(arm_names)
        self.ptypes = list(ptypes)
        self.device = device
        self.na = len(self.names)
        self.cols = list(_BASE) + ['aug_' + t for t in AUG_TYPES] + \
            ['codec_' + c for c in CODECS] + ['pt_' + p for p in self.ptypes]
        self.ix = {c: i for i, c in enumerate(self.cols)}
        self.acc = torch.zeros(self.na, len(self.cols), dtype=torch.float64)

    def _add_counts(self, col0: int, arm: torch.Tensor, code: torch.Tensor,
                    n_codes: int, mask: Optional[torch.Tensor] = None) -> None:
        ok = code >= 0  # negative code: not applicable to the row
        if mask is not None:
            ok &= mask
        if not bool(ok.any()):
            return
        if int(code[ok].max()) >= n_codes:
            raise ValueError('code %d outside the table of %d names' %
                             (int(code[ok].max()), n_codes))
        flat = arm[ok] * n_codes + code[ok]
        cnt = torch.bincount(flat, minlength=self.na * n_codes).view(
            self.na, n_codes).to(torch.float64)
        self.acc[:, col0:col0 + n_codes] += cnt

    @torch.no_grad()
    def add(self, batch: Mapping[str, Any], logits: torch.Tensor,
            labels: torch.Tensor) -> None:
        """Add one batch.

        Args:
            batch: collated batch with the log fields (``arm``,
                ``pair_role``, ``kpos``, ``ptype``, ``aug_code``,
                ``chain_code``, ``zr_*``, ``eg_*``).
            logits: (B, C) margin logits of the step.
            labels: (B,) class labels on the device of ``logits``.
        """
        arm = batch['arm'].long().cpu()
        out = logits.detach().float()
        ce = F.cross_entropy(out, labels, reduction='none').cpu().double()
        cor = (out.argmax(1) == labels).cpu().double()
        role = batch['pair_role'].long().cpu()
        a = self.acc

        def add(col: str, values: torch.Tensor) -> None:
            a[:, self.ix[col]].index_add_(0, arm, values.cpu().double())

        add('rows', torch.ones_like(ce))
        add('ce_sum', ce)
        add('correct', cor)
        add('views', role == 2)
        add('zr_lsb50', batch['zr_lsb'].double() > 0.5)
        add('zr_run', batch['zr_run'])
        add('zr_dith', batch['zr_dith'])
        add('zr_byp', batch['zr_bypass'])
        gate = batch['eg_redraw'].long().cpu()  # -1: row not gated
        add('eg_rows', gate >= 0)
        add('eg_redraw', gate.clamp(min=0))
        add('eg_fail', batch['eg_fail'])
        self._add_counts(self.ix['aug_' + AUG_TYPES[0]], arm,
                         batch['aug_code'].long().cpu(), len(AUG_TYPES))
        chain = batch['chain_code'].long().cpu()
        add('chain', chain >= 0)
        self._add_counts(self.ix['codec_' + CODECS[0]], arm, chain,
                         len(CODECS))
        group = (batch['kpos'].long().cpu() == 0) & (role != 2)
        self._add_counts(self.ix['pt_' + self.ptypes[0]], arm,
                         batch['ptype'].long().cpu(), len(self.ptypes), group)

    def flush(self, logger: Optional[logging.Logger] = None, epoch: int = 0,
              batch: int = 0) -> Optional[Dict[str, Dict[str, Any]]]:
        """Sum over all processes, log and reset.

        Every process must call it at the same step; pass ``logger`` on the
        process that prints (the others return None).

        Returns:
            arm name -> the logged values, for the arms that had rows.
        """
        acc = all_reduce_sum(self.acc, self.device)
        self.acc = torch.zeros_like(self.acc)
        if logger is None:
            return None
        total = float(acc[:, self.ix['rows']].sum())
        res = {}
        for k, name in enumerate(self.names):
            r = {c: float(acc[k, i]) for c, i in self.ix.items()}
            rows = r['rows']
            if rows == 0:
                continue
            d = dict(rows=int(rows), share=round(rows / max(total, 1.0), 4),
                     ce=round(r['ce_sum'] / rows, 4),
                     acc=round(r['correct'] / rows, 4),
                     view=round(r['views'] / rows, 4),
                     zr_lsb50=round(r['zr_lsb50'] / rows, 4),
                     zr_run50ms=round(r['zr_run'] / rows, 4),
                     zr_dith=round(r['zr_dith'] / rows, 4),
                     zr_bypass=round(r['zr_byp'] / rows, 4),
                     chain=round(r['chain'] / rows, 4))
            gated = r['eg_rows']
            if gated > 0:
                d['eg_gated'] = round(gated / rows, 4)
                d['eg_redraw'] = round(r['eg_redraw'] / gated, 4)
                d['eg_fail'] = round(r['eg_fail'] / gated, 4)
            for key, prefix, table in (('aug', 'aug_', AUG_TYPES),
                                       ('codec', 'codec_', CODECS),
                                       ('ptype', 'pt_', self.ptypes)):
                counts = {t: r[prefix + t] for t in table}
                n = sum(counts.values())
                if n > 0:
                    d[key] = {t: round(v / n, 4)
                              for t, v in counts.items() if v}
            res[name] = d
        for name, d in res.items():
            logger.info('ARMLOG epoch %d batch %d arm %s %s' % (
                epoch, batch, name, json.dumps(d, sort_keys=True)))
        return res


def make_arm_logger(cfg: Config, ptypes: Sequence[str],
                    device: torch.device) -> ArmLogger:
    """``ArmLogger`` for the sampler arms of ``cfg``, in arm-code order."""
    return ArmLogger(cfg.data.sampler.arm_order, ptypes, device)
