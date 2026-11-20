"""Model builder: keys, parameter order, initialisation, source parity.

Unit tests build a tiny random Wav2Vec2-BERT in a temporary directory.
Parity tests run the original trainer's construction in a subprocess with
``PYTHONPATH=$OVSB_SOURCE_TREE`` (the two ``wespeaker`` packages never share
a process) and need

  OVSB_SOURCE_TREE     root of the source repository
  OVSB_SOURCE_RECIPE   its recipe directory (holds conf/, exp/, data/, assets/)

The real-model check also needs ``OVSB_PARITY_REF`` (a reference dump of
the source tree made by its parity harness: state-dict hashes and the
embeddings of a fixed synthetic batch) and ``OVSB_REAL_MODEL=1``; it builds
the full model on the CPU (about 5 GB of memory).

This module also holds the helpers the other engine tests share.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import transformers
import yaml

from conftest import RECIPE_DIR

from ovsb.config import load_legacy
from ovsb.config.load import apply_overrides, read_tree
from ovsb.config.schema import build_config
from ovsb.model.build import build_model
from ovsb.model.frontend import TimeMaskedW2VBertSV
from ovsb.model.head import DurationMarginSubcenterHead
from ovsb.model.head_record import (HeadMismatchError, check_run_head,
                                    head_sidecar_path, verify_head)

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
SOURCE_RECIPE = os.environ.get('OVSB_SOURCE_RECIPE')
PARITY_REF = os.environ.get('OVSB_PARITY_REF')
RUN_DIR = os.path.join('exp', 's3', 'base_v9c')
AUTHORED = os.path.join('conf', 's3_base_v9c.yaml')
needs_source = pytest.mark.skipif(
    not (SOURCE_TREE and SOURCE_RECIPE),
    reason='set OVSB_SOURCE_TREE and OVSB_SOURCE_RECIPE for parity tests')
needs_real_model = pytest.mark.skipif(
    not (SOURCE_RECIPE and PARITY_REF
         and os.environ.get('OVSB_REAL_MODEL') == '1'),
    reason='set OVSB_SOURCE_RECIPE, OVSB_PARITY_REF and OVSB_REAL_MODEL=1 '
           'for the real-model checks')

TINY_CLASSES = 40
TINY_MODEL = {'feat_dim': 32, 'nb_layer': 3, 'head_nb': 4, 'embed_dim': 16,
              'compression_dim': 8}
TINY_LAYERS = 2


# ------------------------------------------------------------ shared helpers
def make_tiny_assets(path):
    """Save a 3-layer random Wav2Vec2-BERT (width 32) to ``path``."""
    torch.manual_seed(0)
    config = transformers.Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
        intermediate_size=64, feature_projection_input_dim=160,
        conv_depthwise_kernel_size=5, left_max_position_embeddings=8,
        right_max_position_embeddings=4, add_adapter=False, vocab_size=8)
    transformers.Wav2Vec2BertModel(config).save_pretrained(str(path))
    return str(path)


def tiny_config(assets, *overrides):
    """The reference recipe config shrunk to the tiny encoder."""
    tree = read_tree(os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml'))
    sets = ['model.frontend.assets=%s' % assets,
            'model.frontend.num_layers=%d' % TINY_LAYERS]
    sets += ['model.backend.%s=%d' % kv for kv in TINY_MODEL.items()]
    return build_config(apply_overrides(tree, sets + list(overrides)))


def write_tiny_legacy(tmp_dir, assets, center_warmup=None):
    """The reference run's saved config shrunk to the tiny encoder.

    Returns (path of the saved-config YAML, recipe ``Config``); the source
    drivers read the YAML, the recipe side uses the ``Config``.
    """
    with open(os.path.join(SOURCE_RECIPE, RUN_DIR, 'config.yaml')) as fin:
        tree = yaml.safe_load(fin)
    data = tree['dataset_args']
    data['frontend_num_layers'] = TINY_LAYERS
    data['w2vbert_sv_args']['upstream_args']['path_or_url'] = assets
    tree['model_args'].update(TINY_MODEL)
    tree['projection_args'].update(embed_dim=TINY_MODEL['embed_dim'],
                                   num_class=TINY_CLASSES)
    if center_warmup is not None:
        tree['pair_auc']['center']['warmup'] = center_warmup
    # a saved config has its keys sorted; the source code takes the arm
    # codes from the order of the mapping, so restore the run's order
    authored = os.path.join(SOURCE_RECIPE, AUTHORED)
    with open(authored) as fin:
        order = list(yaml.safe_load(fin)['dataset_args']['episodic_sampler']
                     ['arms'])
    sampler = data['episodic_sampler']
    sampler['arms'] = {name: sampler['arms'][name] for name in order}
    path = os.path.join(str(tmp_dir), 'tiny_legacy.yaml')
    with open(path, 'w') as fout:
        yaml.safe_dump(tree, fout, sort_keys=False)
    cfg = load_legacy(path, authored)
    assert cfg.data.sampler.arm_order == order
    return path, cfg


# Source side: the model, optimiser and schedulers as the original
# wespeaker/bin/train.py builds them for this config (same statements, same
# order), without DDP and CUDA.
SOURCE_PRELUDE = r'''
import sys
import numpy as np
import torch
import yaml
import wespeaker
import wespeaker.utils.schedulers as schedulers
from wespeaker.frontend import frontend_class_dict
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils import layer_weights as LW
from wespeaker.utils import phase1 as P1
from wespeaker.utils.frontend_utils import maybe_truncate_frontend

torch.set_num_threads(4)
with open(sys.argv[1]) as fin:
    configs = yaml.safe_load(fin)


class Wrapper(torch.nn.Module):
    """Stands in for DistributedDataParallel on one process."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def build_model():
    data = configs['dataset_args']
    frontend = frontend_class_dict[data['frontend']](
        **data[data['frontend'] + '_args'],
        sample_rate=data['resample_rate'])
    configs['model_args']['feat_dim'] = frontend.output_size()
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    model.add_module('frontend', frontend)
    maybe_truncate_frontend(model, data, configs['model_args'].get('nb_layer'))
    P1.attach_ptp(model, configs)
    configs['projection_args']['embed_dim'] = P1.projection_embed_dim(configs)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    model.add_module('projection', get_projection(configs['projection_args']))
    return model


def build_optim(model):
    world = len(configs['gpus'])
    batch_size = configs['dataloader_args']['batch_size']
    epoch_iter = configs['dataset_args']['sample_num_per_epoch'] \
        // world // batch_size
    ddp = Wrapper(model)
    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = LW.build_optimizer(ddp, model, configs)
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    configs['scheduler_args']['scale_ratio'] = 1.0 * world * batch_size / 64
    scheduler = getattr(schedulers, configs['scheduler'])(
        optimizer, model=model, **configs['scheduler_args'])
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    return ddp, optimizer, scheduler, margin, epoch_iter
'''


def run_source(driver, tmp_path, config_path, *args):
    """Run PRELUDE + ``driver`` under the source tree.

    The driver gets ``config_path`` as argv[1], an output path as argv[2]
    and ``args`` after it, and must ``torch.save`` a dict with the key
    'tree' (``wespeaker.__file__``) to the output path.
    """
    script, out = tmp_path / 'source_driver.py', tmp_path / 'source_out.pt'
    script.write_text(SOURCE_PRELUDE + driver)
    env = dict(os.environ, PYTHONPATH=SOURCE_TREE, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(script), config_path, str(out)]
                   + [str(a) for a in args], check=True, env=env,
                   cwd=str(tmp_path))
    res = torch.load(str(out), weights_only=False)
    assert os.path.abspath(res.pop('tree')).startswith(
        os.path.abspath(SOURCE_TREE))
    return res


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    return make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))


# ------------------------------------------------------------------- unit
def test_keys_shapes_and_parameter_order(tiny_assets):
    model = build_model(tiny_config(tiny_assets), TINY_CLASSES)
    names = [n for n, _ in model.named_parameters()]
    tops = [n.split('.')[0] for n in names]
    first_fe, last_fe = tops.index('frontend'), \
        len(tops) - 1 - tops[::-1].index('frontend')
    # backend parameters, then the encoder, then the head
    assert 'frontend' not in tops[:first_fe] and first_fe == 14
    assert set(tops[first_fe:last_fe + 1]) == {'frontend'}
    assert names[last_fe + 1:] == ['projection.weight']
    assert names[:2] == ['weights_k', 'weights_v']
    state = model.state_dict()
    assert state['projection.weight'].shape == (3 * TINY_CLASSES, 16)
    assert state['weights_k'].shape == (TINY_LAYERS + 1,)
    assert state['pooling_fc.weight'].shape == (16, 4 * 8)
    layers = {int(k.split('.')[4]) for k in state
              if k.startswith('frontend.upstream.encoder.layers.')}
    assert layers == set(range(TINY_LAYERS))
    assert not any('masked_spec_embed' in k for k in state)


def test_modules_and_head_margin(tiny_assets):
    cfg = tiny_config(tiny_assets, 'model.head.margin=0.3',
                      'model.frontend.time_mask={prob: 0.1, span: 4}')
    model = build_model(cfg, TINY_CLASSES)
    assert isinstance(model.frontend, TimeMaskedW2VBertSV)
    assert (model.frontend.time_mask_prob, model.frontend.time_mask_span) \
        == (0.1, 4)
    assert model.frontend.gradient_checkpointing is True
    assert model.frontend.n_hidden_states == TINY_LAYERS + 1
    head = model.projection
    assert isinstance(head, DurationMarginSubcenterHead)
    assert (head.K, head.k_top, head.scale, head.easy_margin) == \
        (3, 5, 32.0, False)
    assert head.m == head.margin == 0.3
    # the hard-negative penalty follows the margin: mp * margin / 0.2
    assert head.cos_mp == pytest.approx(np.cos(0.06 * 0.3 / 0.2))
    assert all(p.requires_grad for p in model.parameters())


def test_forward_shapes(tiny_assets):
    model = build_model(tiny_config(tiny_assets), TINY_CLASSES).eval()
    wav = 0.1 * torch.randn(3, 12000, generator=torch.Generator().manual_seed(1))
    lens = torch.tensor([12000, 9000, 6400])
    with torch.no_grad():
        feats, flens = model.frontend(wav, lens)
        emb = model(feats, flens)
        logits = model.projection(emb, torch.tensor([0, 1, 2]),
                                  torch.ones(3))
    assert feats.shape[:2] == (3, 32) and feats.shape[3] == TINY_LAYERS + 1
    assert emb.shape == (3, 16) and logits.shape == (3, TINY_CLASSES)


def test_same_seed_same_model(tiny_assets):
    cfg = tiny_config(tiny_assets)
    states = []
    for _ in range(2):
        torch.manual_seed(5)
        states.append(build_model(cfg, TINY_CLASSES).state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_wrong_feat_dim_is_refused(tiny_assets):
    cfg = tiny_config(tiny_assets, 'model.backend.feat_dim=64')
    with pytest.raises(ValueError, match='model.backend.feat_dim'):
        build_model(cfg, TINY_CLASSES)


# ----------------------------------------------------------- source parity
BUILD_DRIVER = r'''
torch.manual_seed(int(sys.argv[3]))
model = build_model()
head = model.projection
torch.save({
    'tree': wespeaker.__file__,
    'state': model.state_dict(),
    'params': [n for n, _ in model.named_parameters()],
    'next_draw': int(torch.empty((), dtype=torch.int64).random_()),
    'head': [head.m, head.cos_mp, head.sin_mp, head.scale, head.K,
             head.k_top, head.easy_margin],
    'repr_children': [n for n, _ in model.named_children()],
}, sys.argv[2])
'''


@needs_source
def test_tiny_model_equals_source_build(tiny_assets, tmp_path):
    """Same seed -> same keys (in order), same initial values and the same
    state of the global generator afterwards (the DataLoader draws its
    worker seed from it right after the model is built)."""
    legacy, cfg = write_tiny_legacy(tmp_path, tiny_assets)
    src = run_source(BUILD_DRIVER, tmp_path, legacy, 11)
    torch.manual_seed(11)
    model = build_model(cfg, TINY_CLASSES)
    next_draw = int(torch.empty((), dtype=torch.int64).random_())
    state = model.state_dict()
    assert list(state) == list(src['state'])
    assert [n for n, _ in model.named_parameters()] == src['params']
    assert [n for n, _ in model.named_children()] == src['repr_children']
    for key, ref in src['state'].items():
        assert state[key].dtype == ref.dtype and torch.equal(state[key], ref), key
    assert next_draw == src['next_draw']
    head = model.projection
    # the source head gets its margin from the margin scheduler; compare
    # the fields that the constructor fixes
    assert [head.scale, head.K, head.k_top, head.easy_margin] == src['head'][3:]


# ------------------------------------------------- real model vs reference
SYNTH_SEED = 20261120


def synth_wave(rng, n, sr=16000):
    """The reference harness's speech-like test signal (peak 0.1)."""
    t = np.arange(n) / sr
    f0 = rng.uniform(90, 220) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(
        1, 3) * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(n)
    for h in range(1, 25):
        x += rng.uniform(0.2, 1.0) / h * np.sin(h * ph + rng.uniform(0, 6.28))
    x *= 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
    x += 0.02 * rng.standard_normal(n)
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


def synth_batches(arm_names, sr=16000):
    """The reference harness's two fixed batches (plain, pair).

    plain: 4 speakers x 2 rows of 1.5-3 s padded to 3 s; pair: 2 speakers
    x 2 utterances x (3 s anchor, 1.5-2 s view).
    """
    rng = np.random.default_rng(SYNTH_SEED)
    arm = {n: i for i, n in enumerate(arm_names)}
    pad_to = 3 * sr

    def sample(wav, label, arm_name, pair_role, kpos):
        w = torch.from_numpy(wav).unsqueeze(0)
        n = w.shape[1]
        return dict(label=label, wav=torch.nn.functional.pad(w, (0, pad_to - n)),
                    wav_len=n, pair_role=pair_role, arm=arm[arm_name],
                    kpos=kpos, ptype=0, aug_code=0, chain_code=-1, zr_lsb=0.0,
                    zr_run=0, zr_dith=0, zr_bypass=0, eg_redraw=-1, eg_fail=0)

    labels = [100, 50000, 10100, 120000]
    arms = ['vox2', 'vox2', 'cn1', 'mandarin_t1']
    secs = [3.0, 1.5, 2.25, 3.0, 1.8, 2.6, 2.0, 2.9]
    plain = [sample(synth_wave(rng, int(s * sr)), labels[i // 2],
                    arms[i // 2], 0, i % 2) for i, s in enumerate(secs)]
    pair = []
    for u, view in enumerate([1.5, 2.0, 1.6, 1.9]):
        long = synth_wave(rng, 5 * sr)
        n = int(view * sr)
        off = int(rng.integers(0, 5 * sr - n))
        lab, name = (labels[0], 'vox2') if u < 2 else (labels[2], 'cn1')
        pair.append(sample(long[:pad_to], lab, name, 1, u % 2))
        pair.append(sample(long[off:off + n], lab, name, 2, u % 2))
    collate = torch.utils.data.default_collate
    return collate(plain), collate(pair)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(
        np.ascontiguousarray(tensor.numpy()).tobytes()).hexdigest()


def load_reference():
    """The 'model' and 'schedule' checks of the reference dump."""
    with open(PARITY_REF) as fin:
        ref = json.load(fin)
    assert ref['model']['device'] == 'cpu' and not ref['model']['amp']
    return ref


def real_config():
    """Config of the reference run; call with the source recipe as cwd."""
    return load_legacy(os.path.join(RUN_DIR, 'config.yaml'), AUTHORED)


def build_real_model(cfg, epoch=14):
    """The reference run's model with the weights of ``model_<epoch>.pt``."""
    model = build_model(cfg, cfg.derived.num_class)
    path = os.path.join(RUN_DIR, 'models', 'model_%d.pt' % epoch)
    ckpt = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    built = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert built == {k: tuple(v.shape) for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model


def read_speakers(path):
    with open(path) as fin:
        return {line.split()[1] for line in fin if line.strip()}


@needs_real_model
def test_real_model_keys_load_and_embeddings(monkeypatch, tmp_path):
    monkeypatch.chdir(SOURCE_RECIPE)
    ref = load_reference()['model']
    cfg = real_config()
    model = build_model(cfg, cfg.derived.num_class)
    state = model.state_dict()
    shapes = {k: list(v.shape) for k, v in state.items()}
    want = ref['state_dict']
    assert len(shapes) == want['n_keys'] == 723
    assert sha_text('\n'.join(sorted(shapes))) == want['keys_sha256']
    assert sha_text(json.dumps([[k, shapes[k]] for k in sorted(shapes)])) \
        == want['shapes_sha256']
    assert sha_text(json.dumps([[k, str(v.dtype)] for k, v in
                                sorted(state.items())])) == want['dtypes_sha256']
    assert [n for n, _ in model.named_children()] == want['top_level_modules']
    assert sum(p.numel() for p in model.parameters()) == want['n_params_total']
    del model, state

    model = build_real_model(cfg).eval()
    plain, _ = synth_batches(cfg.data.sampler.arm_order)
    assert tensor_sha(plain['wav']) == ref['synthetic']['plain']['wav']['sha256']
    wav, lens = plain['wav'].squeeze(1), plain['wav_len'].long()
    with torch.no_grad():
        feats, flens = model.frontend(wav, lens)
        emb = model(feats, flens).double()
    want = torch.tensor(ref['eval']['emb_ragged_batch'], dtype=torch.float64)
    norm = torch.nn.functional.normalize
    diff_raw = (emb - want).abs().max().item()
    diff_unit = (norm(emb, dim=1) - norm(want, dim=1)).abs().max().item()
    print('\nreal-model embeddings vs reference: max abs diff %.3g (raw, row '
          'norms %.0f-%.0f), %.3g (L2-normalised)' % (
              diff_raw, want.norm(dim=1).min(), want.norm(dim=1).max(),
              diff_unit))
    assert diff_unit <= 1e-5
    assert torch.allclose(emb, want, rtol=2e-6, atol=1e-5)


@needs_real_model
def test_real_head_record(monkeypatch, tmp_path):
    monkeypatch.chdir(SOURCE_RECIPE)
    cfg = real_config()
    rows = cfg.derived.num_class * cfg.model.head.K
    spks = read_speakers(cfg.data.lists.train_label)
    init = os.path.join(RUN_DIR, 'models', 'model_0.pt')
    with open(head_sidecar_path(init)) as fin:
        meta = json.load(fin)
    verify_head(meta, spks, rows, 'sidecar')
    weight = torch.load(init, map_location='cpu', mmap=True,
                        weights_only=True)['projection.weight']
    assert weight.shape[0] == rows
    assert check_run_head(init, str(tmp_path), spks, rows) == 'sidecar'
    with open(tmp_path / 'head_spk.json') as fin:
        run = json.load(fin)
    with open(os.path.join(RUN_DIR, 'models', 'head_spk.json')) as fin:
        old_run = json.load(fin)
    assert {k: run[k] for k in ('spk_sha256', 'num_classes', 'rows',
                                'verified_from')} == \
        {k: old_run[k] for k in ('spk_sha256', 'num_classes', 'rows',
                                 'verified_from')}
    # a later checkpoint of the run has no sidecar: the run record is used
    assert check_run_head(os.path.join(str(tmp_path), 'model_3.pt'),
                          str(tmp_path), spks, rows) == 'run'
    for key, value in (('spk_sha256', '0' * 64), ('rows', rows - 3),
                       ('num_classes', cfg.derived.num_class - 1)):
        fake = str(tmp_path / ('tampered_%s.pt' % key))
        with open(head_sidecar_path(fake), 'w') as fout:
            json.dump(dict(meta, **{key: value}), fout)
        with pytest.raises(HeadMismatchError):
            check_run_head(fake, str(tmp_path / 'other'), spks, rows)
    with pytest.raises(HeadMismatchError):
        verify_head(meta, spks - {next(iter(spks))}, rows, 'one speaker less')
