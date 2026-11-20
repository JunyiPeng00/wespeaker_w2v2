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
"""Speaker sampling weights of the episodic sampler.

Inside an arm a speaker of weight w is drawn w times as often as a speaker
of weight 1. Arms without a weight rule draw their speakers uniformly and
are not "weighted". The rules of an arm's policy are applied in this order:

1. ``w_bridge``: speakers with two or more genres get this weight.
2. ``dialect_mass``: the 3D-Speaker dialect speakers share ``boost`` x
   (their number) of mass equally per dialect, then every weight of the arm
   is capped at ``cap_x_median`` x the arm's median weight.
3. ``spk_weight_file``: the weight of every listed speaker is multiplied by
   its entry. With ``spk_weight_share`` the sampler additionally rescales
   the listed speakers inside every sampler cell (see
   :func:`share_weights`) so that they hold exactly that share of the
   cell's mass.

Everything here is a deterministic function of a
:class:`~ovsb.data.speakers.SpeakerTable`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ovsb.data.speakers import DialectMass, SpeakerTable


@dataclass(frozen=True, eq=False)
class SpeakerWeights:
    """Speaker weights of every arm.

    Attributes:
        spk_w: (S,) float64 weight per speaker (1.0 unless a rule applies),
            or None when no arm is weighted.
        arm_weighted: (A,) bool; the sampler draws the speakers of a
            weighted arm from weight-proportional pools.
        spk_share: (A,) float64 per-cell share of the listed speakers of the
            arm (nan = no share rule), or None when no arm has one.
        spk_listed: (S,) bool speaker is in its arm's weight file, or None
            when no arm has a share rule.
        info: counts for the log.
    """
    spk_w: Optional[np.ndarray]
    arm_weighted: np.ndarray
    spk_share: Optional[np.ndarray]
    spk_listed: Optional[np.ndarray]
    info: Dict[str, Any] = field(default_factory=dict)


def bridge_weights(w: np.ndarray, spk: np.ndarray, spk_ngenre: np.ndarray,
                   w_bridge: float) -> int:
    """Set the weight of the multi-genre speakers among ``spk`` (in place).

    Returns the number of such speakers.
    """
    bridge = spk[spk_ngenre[spk] >= 2]
    w[bridge] = float(w_bridge)
    return len(bridge)


def dialect_mass_weights(w: np.ndarray, spk: np.ndarray, spk_dial: np.ndarray,
                         rule: DialectMass) -> Optional[Tuple[int, int, int]]:
    """Apply the dialect-mass rule to the arm speakers ``spk`` (in place).

    Returns ``(dialect speakers, dialects, capped speakers)``, or None when
    the arm has no dialect speaker (nothing is changed, not even the cap).
    """
    dsp = spk[spk_dial[spk] > 0]
    dials, cnt = np.unique(spk_dial[dsp], return_counts=True)
    if not len(dials):
        return None
    per_dial = float(rule.boost) * len(dsp) / len(dials)
    for d, n in zip(dials, cnt):
        w[dsp[spk_dial[dsp] == d]] = per_dial / n
    cap = float(rule.cap_x_median) * float(np.median(w[spk]))
    n_cap = int((w[spk] > cap).sum())
    w[spk] = np.minimum(w[spk], cap)
    return len(dsp), len(dials), n_cap


def read_weight_file(path: str) -> Dict[str, float]:
    """Read a ``<speaker> <multiplier>`` file."""
    mult = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                mult[p[0]] = float(p[1])
    return mult


def file_weights(w: np.ndarray, spk: np.ndarray, spk_names: np.ndarray,
                 mult: Dict[str, float]) -> np.ndarray:
    """Multiply the weights of the listed speakers among ``spk`` (in place).

    Returns the bool mask over ``spk`` of the speakers found in ``mult``.
    """
    listed = np.zeros(len(spk), bool)
    for j, s in enumerate(spk):
        m = mult.get(str(spk_names[s]))
        if m is not None:
            w[s] *= m
            listed[j] = True
    return listed


def share_weights(w: np.ndarray, listed: np.ndarray,
                  share: float) -> np.ndarray:
    """Rescale weights so the listed speakers hold ``share`` of the mass.

    Args:
        w: weights of the speakers of one arm in one sampler cell.
        listed: bool mask over ``w`` of the listed speakers.
        share: target share of the total mass, in (0, 1).

    Returns ``w`` itself when either group has no mass, else a copy in which
    the listed weights are scaled by one factor (relative weights inside
    each group are kept).
    """
    ml, mr = float(w[listed].sum()), float(w[~listed].sum())
    if ml <= 0 or mr <= 0:
        return w
    w = w.copy()
    w[listed] *= share / (1.0 - share) * mr / ml
    return w


def build_weights(table: SpeakerTable) -> SpeakerWeights:
    """Speaker weights of every arm from the policies of ``table``."""
    n_arms = len(table.arm_names)
    w = np.ones(table.n_spk, np.float64)
    arm_weighted = np.zeros(n_arms, bool)
    spk_share = None
    spk_listed = None
    info = {}
    for a, pol in enumerate(table.policy):
        if pol is None:
            continue
        name = table.arm_names[a]
        spk = table.arm_speakers(a)
        if pol.w_bridge and float(pol.w_bridge) != 1.0:
            info['%s_bridge_speakers' % name] = bridge_weights(
                w, spk, table.spk_ngenre, pol.w_bridge)
            arm_weighted[a] = True
        if pol.dialect_mass:
            res = dialect_mass_weights(w, spk, table.spk_dial,
                                       pol.dialect_mass)
            if res is not None:
                arm_weighted[a] = True
                info['%s_dialect_speakers' % name] = res[0]
                info['%s_dialects' % name] = res[1]
                info['%s_weight_capped' % name] = res[2]
        if pol.spk_weight_file:
            listed = file_weights(w, spk, table.spk_names,
                                  read_weight_file(pol.spk_weight_file))
            arm_weighted[a] = True
            info['%s_spk_weight_file_hits' % name] = int(listed.sum())
            if pol.spk_weight_share is not None:
                if spk_share is None:
                    spk_share = np.full(n_arms, np.nan)
                    spk_listed = np.zeros(table.n_spk, bool)
                spk_share[a] = float(pol.spk_weight_share)
                spk_listed[spk[listed]] = True
                info['%s_spk_weight_share' % name] = spk_share[a].item()
    return SpeakerWeights(spk_w=w if arm_weighted.any() else None,
                          arm_weighted=arm_weighted, spk_share=spk_share,
                          spk_listed=spk_listed, info=info)
