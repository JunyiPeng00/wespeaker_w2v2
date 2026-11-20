"""Loss bundle: composition of the three terms and source-trainer parity."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from ovsb.losses.bundle import LossBundle  # noqa: E402
from ovsb.losses.consistency import pair_consistency_loss  # noqa: E402
from ovsb.losses.pair_auc import RunningCenter, pair_auc_loss  # noqa: E402
from ovsb.model.head import (DurationMarginSubcenterHead,  # noqa: E402
                             margin_scale_from_len)

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')

ARMS = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
CENTER_ARM, MOMENTUM, WARMUP = 1, 0.99, 6
FEAT, DIM, CLASSES, K = 40, 32, 60, 3
CONS_W, PAUC_W, PAUC_SCALE, REF_SEC, SR, MARGIN = 0.5, 0.2, 16.0, 3.0, 16000, 0.2

# Source side: the source trainer's own epoch loop (run_epoch) on a list of
# batches. The model is a stand-in (pass-through frontend, one linear layer
# as the embedding network, the source head), the optimizer records the
# gradients instead of stepping and the total loss is read where the loop
# checks it for finiteness. Nothing of the loss path is re-implemented here.
SOURCE_DRIVER = r'''
import logging
import sys
import types
import numpy as np
import torch
try:
    import torchnet  # noqa: F401
except ImportError:  # only its two meters are used, for logging
    class _Meter:
        def __init__(self, *a, **k):
            self.n, self.total = 0, 0.0

        def add(self, value, *a):
            self.n += 1
            self.total += float(value) if np.ndim(value) == 0 else 0.0

        def value(self):
            return self.total / max(self.n, 1), 0.0
    tnt = types.ModuleType('torchnet')
    tnt.meter = types.ModuleType('torchnet.meter')
    tnt.meter.AverageValueMeter = tnt.meter.ClassErrorMeter = _Meter
    sys.modules['torchnet'], sys.modules['torchnet.meter'] = tnt, tnt.meter
import wespeaker
from wespeaker.models.projections import get_projection
from wespeaker.utils import executor

torch.set_num_threads(4)
data, out = np.load(sys.argv[1]), {'tree': np.array(wespeaker.__file__)}
steps = int(data['steps'])
configs = {
    'num_epochs': 1, 'log_batch_interval': 5, 'enable_amp': False,
    'amp_dtype': 'bfloat16', 'grad_clip': None,
    'consistency_weight': float(data['cons_w']),
    'pair_auc': {'weight': float(data['pauc_w']),
                 'scale': float(data['pauc_scale']),
                 'center': {'arm': str(data['center_arm']),
                            'momentum': float(data['momentum']),
                            'warmup': int(data['warmup'])}},
    'dataset_args': {'frontend': 'w2vbert_sv', 'cmvn': False,
                     'resample_rate': int(data['sr']),
                     'episodic_sampler': {'arms': {
                         str(a): {} for a in data['arms']}}},
}


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.from_numpy(data['embed_w'])
        self.embed = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.projection = get_projection({
            'project_type': 'arc_margin_intertopk_subcenter',
            'embed_dim': w.shape[0], 'num_class': int(data['classes']),
            'scale': 32.0, 'easy_margin': False, 'K': int(data['k']),
            'mp': 0.06, 'k_top': 5, 'do_lm': False,
            'duration_margin': {'ref_sec': float(data['ref_sec'])}})
        with torch.no_grad():
            self.embed.weight.copy_(w)
            self.projection.weight.copy_(torch.from_numpy(data['head_w']))
        self.projection.update(float(data['margin']))
        self.embeds = None

    def frontend(self, wavs, wavs_len):
        return wavs, None

    def forward(self, features):
        self.embeds = self.embed(features)
        self.embeds.retain_grad()
        return self.embeds


class Model:  # what the loop sees as the DDP-wrapped model
    def __init__(self, net):
        self.module = net

    def train(self):
        self.module.train()

    def parameters(self):
        return self.module.parameters()

    def __call__(self, *args):
        return self.module(*args)


class Recorder:  # optimizer stand-in: keeps the gradients of each step
    def __init__(self, net):
        self.net, self.step_no = net, 0

    def zero_grad(self):
        self.net.zero_grad()

    def step(self):
        t = self.step_no
        out['g_emb%d' % t] = self.net.embeds.grad.numpy().copy()
        out['g_embed_w%d' % t] = self.net.embed.weight.grad.numpy().copy()
        out['g_head_w%d' % t] = self.net.projection.weight.grad.numpy().copy()
        self.step_no += 1


class Schedule:
    def step(self, cur_iter):
        pass

    def get_lr(self):
        return 0.0

    def get_margin(self):
        return float(data['margin'])


losses, finite_check = [], executor.all_ranks_finite


def record_loss(value):
    losses.append(float(value.detach()))
    out['loss%d' % (len(losses) - 1)] = value.detach().numpy().copy()
    return finite_check(value)


executor.all_ranks_finite = record_loss
batches = []
for t in range(steps):
    n = data['label%d' % t].shape[0]
    batches.append({
        'key': ['utt%d' % r for r in range(n)],
        'label': torch.from_numpy(data['label%d' % t]),
        'wav': torch.from_numpy(data['feat%d' % t]).unsqueeze(1),
        'wav_len': torch.from_numpy(data['len%d' % t]),
        'label_b': torch.full((n,), -1, dtype=torch.long),
        'lam': torch.ones(n, dtype=torch.float64),
        'pair_role': torch.from_numpy(data['role%d' % t]),
        'arm': torch.from_numpy(data['arm%d' % t]),
    })
net = Net()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
executor.run_epoch(
    batches, steps, Model(net), torch.nn.CrossEntropyLoss(), Recorder(net),
    Schedule(), Schedule(), 1, logging.getLogger('source'),
    torch.amp.GradScaler('cpu', enabled=False), torch.device('cpu'), configs)
assert len(losses) == steps and ('g_emb%d' % (steps - 1)) in out
np.savez(sys.argv[2], **out)
'''


def run_source(driver, tmp_path, inputs):
    """Run `driver` under the source tree; returns its saved arrays."""
    script, inp, out = (tmp_path / n for n in ('drv.py', 'in.npz', 'out.npz'))
    script.write_text(driver)
    np.savez(inp, **inputs)
    env = dict(os.environ, PYTHONPATH=SOURCE_TREE, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(script), str(inp), str(out)],
                   check=True, env=env, cwd=str(tmp_path))
    res = dict(np.load(out))
    assert os.path.abspath(str(res.pop('tree'))).startswith(
        os.path.abspath(SOURCE_TREE))
    return res


def make_batch(gen, kind, offset, n=48):
    """One batch (feat, label, wav_len, pair_role, arm).

    kind 'plain': two rows per speaker, random lengths, all roles 0.
    kind 'pair': (6 s anchor, short view) couples, two couples per speaker.
    """
    feat = torch.randn(n, FEAT, generator=gen) + offset
    arm = torch.randint(0, len(ARMS), (n // 2,), generator=gen) \
        .repeat_interleave(2)
    if kind == 'plain':
        label = torch.randperm(CLASSES, generator=gen)[:n // 2] \
            .repeat_interleave(2)
        role = torch.zeros(n, dtype=torch.long)
        wav_len = torch.randint(SR // 2, 8 * SR, (n,), generator=gen)
    else:
        label = torch.randperm(CLASSES, generator=gen)[:n // 4] \
            .repeat_interleave(4)
        role = torch.tensor([1, 2] * (n // 2))
        wav_len = torch.randint(SR, 3 * SR, (n,), generator=gen)
        wav_len[0::2] = 6 * SR
        feat[1::2] = feat[0::2] + 0.5 * torch.randn(n // 2, FEAT,
                                                    generator=gen)
    return feat, label, wav_len, role, arm


def make_head(head_w):
    head = DurationMarginSubcenterHead(DIM, CLASSES, scale=32.0, margin=0.0,
                                       K=K, mp=0.06, k_top=5)
    with torch.no_grad():
        head.weight.copy_(head_w)
    head.update(MARGIN)
    return head


def test_total_is_the_weighted_sum_of_the_terms():
    gen = torch.Generator().manual_seed(0)
    head = make_head(torch.randn(CLASSES * K, DIM, generator=gen))
    bundle = LossBundle(CONS_W, PAUC_W, PAUC_SCALE)
    for kind in ('plain', 'pair'):
        feat, label, wav_len, role, _ = make_batch(gen, kind, 0.0)
        emb = feat[:, :DIM].clone().requires_grad_(True)
        logits = head(emb, label, margin_scale_from_len(wav_len, REF_SEC, SR))
        total, parts, stats = bundle(logits, emb, label, role)
        ce = F.cross_entropy(logits, label)      # every row, mean
        pauc = pair_auc_loss(emb, label, PAUC_SCALE, role)[0]
        ref = ce + PAUC_W * pauc
        assert torch.equal(parts['ce'], ce) and torch.equal(parts['pauc'], pauc)
        if kind == 'pair':
            cons = pair_consistency_loss(F.normalize(emb, dim=1), role)[0]
            ref = ce + CONS_W * cons + PAUC_W * pauc
            assert torch.equal(parts['cons'], cons)
            assert stats['cons_views'] == 24
            # 12 speakers x (2 anchors, 2 views): 6 same-label pairs each,
            # minus the 2 same-utterance couples
            assert stats['pauc_pos'] == 12 * 4
        else:
            assert parts['cons'] is None and stats['cons_views'] == 0
            assert stats['pauc_pos'] == 24
        assert torch.equal(total, ref)
        assert not any(v.requires_grad for v in parts.values()
                       if v is not None)
        total.backward()
        assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0


def test_zero_weights_leave_cross_entropy_only():
    gen = torch.Generator().manual_seed(1)
    head = make_head(torch.randn(CLASSES * K, DIM, generator=gen))
    feat, label, wav_len, role, arm = make_batch(gen, 'pair', 0.0)
    emb = feat[:, :DIM]
    logits = head(emb, label, margin_scale_from_len(wav_len, REF_SEC, SR))
    center = RunningCenter(CENTER_ARM)
    total, parts, stats = LossBundle(center=center)(
        logits, emb, label, role, arm)
    assert torch.equal(total, F.cross_entropy(logits, label))
    assert parts['cons'] is None and parts['pauc'] is None and not stats
    assert center.n_upd == 0     # no pair-AUC term: the centre is not fed


def test_center_is_updated_before_it_is_used():
    gen = torch.Generator().manual_seed(2)
    head = make_head(torch.randn(CLASSES * K, DIM, generator=gen))
    center = RunningCenter(CENTER_ARM, momentum=0.9, warmup=2)
    bundle = LossBundle(0.0, PAUC_W, PAUC_SCALE, center)
    offset = 3.0 * torch.randn(FEAT, generator=gen)
    for step in range(4):
        feat, label, wav_len, role, arm = make_batch(gen, 'plain', offset)
        arm[:4] = CENTER_ARM
        emb = feat[:, :DIM]
        logits = head(emb, label, margin_scale_from_len(wav_len, REF_SEC, SR))
        total, parts, _ = bundle(logits, emb, label, role, arm)
        assert center.n_upd == step + 1
        scored = emb if step == 0 else emb - center.mean  # this batch's mean
        ref = pair_auc_loss(scored, label, PAUC_SCALE, role)[0]
        assert torch.equal(parts['pauc'], ref)
    with pytest.raises(ValueError):
        bundle(logits, emb, label, role)   # a centre without the arms


def test_half_precision_inputs_are_scored_in_fp32():
    gen = torch.Generator().manual_seed(3)
    head = make_head(torch.randn(CLASSES * K, DIM, generator=gen))
    feat, label, wav_len, role, arm = make_batch(gen, 'pair', 1.0)
    emb16 = feat[:, :DIM].to(torch.bfloat16)
    logits = head(emb16.float(), label,
                  margin_scale_from_len(wav_len, REF_SEC, SR))
    make = lambda: LossBundle(  # noqa: E731
        CONS_W, PAUC_W, PAUC_SCALE, RunningCenter(CENTER_ARM, warmup=1))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        total, parts, _ = make()(logits, emb16, label, role, arm)
    ref, ref_parts, _ = make()(logits, emb16.float(), label, role, arm)
    assert total.dtype == torch.float32 and torch.equal(total, ref)
    assert all(torch.equal(parts[k], ref_parts[k]) for k in parts)


@needs_source
def test_parity_with_source_trainer(tmp_path):
    gen = torch.Generator().manual_seed(2026)
    embed_w = torch.randn(DIM, FEAT, generator=gen) / FEAT ** 0.5
    head_w = torch.randn(CLASSES * K, DIM, generator=gen)
    offset = 1.5 * torch.randn(FEAT, generator=gen)
    kinds = ['plain', 'pair', 'plain', 'plain', 'pair', 'pair'] * 3
    batches = [make_batch(gen, kind, offset) for kind in kinds]
    batches[3][4][batches[3][4] == CENTER_ARM] = 0   # no centre-arm row
    inputs = {
        'steps': np.array(len(batches)), 'embed_w': embed_w.numpy(),
        'head_w': head_w.numpy(), 'classes': np.array(CLASSES),
        'k': np.array(K), 'ref_sec': np.array(REF_SEC), 'sr': np.array(SR),
        'margin': np.array(MARGIN), 'cons_w': np.array(CONS_W),
        'pauc_w': np.array(PAUC_W), 'pauc_scale': np.array(PAUC_SCALE),
        'arms': np.array(ARMS), 'center_arm': np.array(ARMS[CENTER_ARM]),
        'momentum': np.array(MOMENTUM), 'warmup': np.array(WARMUP)}
    for t, (feat, label, wav_len, role, arm) in enumerate(batches):
        inputs.update({'feat%d' % t: feat.numpy(), 'label%d' % t:
                       label.numpy(), 'len%d' % t: wav_len.numpy(),
                       'role%d' % t: role.numpy(), 'arm%d' % t: arm.numpy()})
    src = run_source(SOURCE_DRIVER, tmp_path, inputs)

    embed = torch.nn.Linear(FEAT, DIM, bias=False)
    with torch.no_grad():
        embed.weight.copy_(embed_w)
    head = make_head(head_w)
    center = RunningCenter(CENTER_ARM, MOMENTUM, WARMUP)
    bundle = LossBundle(CONS_W, PAUC_W, PAUC_SCALE, center)
    worst = {'loss': 0.0, 'g_emb': 0.0, 'g_embed_w': 0.0, 'g_head_w': 0.0}
    rel = dict(worst)
    centred = 0
    for t, (feat, label, wav_len, role, arm) in enumerate(batches):
        embed.zero_grad()
        head.zero_grad()
        emb = embed(feat)
        emb.retain_grad()
        logits = head(emb.float(), label,
                      margin_scale_from_len(wav_len, REF_SEC, SR))
        centred += int(center.n_upd + 1 >= WARMUP)
        total, parts, _ = bundle(logits, emb, label, role, arm)
        total.backward()
        assert (parts['cons'] is not None) == (kinds[t] == 'pair')
        assert parts['pauc'] is not None
        new = {'loss': total.detach(), 'g_emb': emb.grad,
               'g_embed_w': embed.weight.grad, 'g_head_w': head.weight.grad}
        for name, value in new.items():
            ref = src['%s%d' % (name, t)]
            diff = float(np.abs(value.numpy() - ref).max())
            worst[name] = max(worst[name], diff)
            rel[name] = max(rel[name], diff / float(np.abs(ref).max()))
    assert center.ready and centred >= 10 and center.n_upd == len(batches) - 1
    print('bundle vs source run_epoch: %d steps (%d pair batches, %d with '
          'the centre on); max abs diff %s; max rel diff %s'
          % (len(batches), kinds.count('pair'), centred,
             {k: '%.3g' % v for k, v in worst.items()},
             {k: '%.3g' % v for k, v in rel.items()}))
    assert max(rel.values()) <= 1e-6
    assert max(worst.values()) == 0.0   # in fact bit-identical
