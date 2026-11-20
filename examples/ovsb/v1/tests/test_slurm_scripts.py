"""Shell scripts of the recipe: syntax, forbidden options, the launcher.

``local/slurm/train_segment.sbatch`` is run as a plain bash script with a
stand-in ``python`` first on PATH. The stand-in answers the three things
the launcher asks of python: the launch plan (``ovsb.bin.train --plan``),
the trainer (``torch.distributed.run``: a process that starts one "training
process" as its child, in a new session as torchrun does) and the drift
gate. The dry-run test uses the real interpreter and the real config.
"""
import glob
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import time

import pytest

from conftest import RECIPE_DIR

LAUNCHER = os.path.join('local', 'slurm', 'train_segment.sbatch')
SCRIPTS = sorted(
    glob.glob(os.path.join(RECIPE_DIR, '*.sh'))
    + glob.glob(os.path.join(RECIPE_DIR, 'local', '**', '*.sh'),
                recursive=True)
    + glob.glob(os.path.join(RECIPE_DIR, 'local', '**', '*.sbatch'),
                recursive=True))
NEW_TEXT_FILES = [
    'run.sh', LAUNCHER, 'local/slurm/site.env.example',
    'local/slurm/README.md', 'README.md', 'ovsb/bin/train.py',
    'ovsb/engine/loop.py', 'ovsb/engine/checkpoint.py',
    'ovsb/engine/signals.py', 'ovsb/engine/seeding.py']


def read(path):
    with open(os.path.join(RECIPE_DIR, path)) as fin:
        return fin.read()


# ----------------------------------------------------------------- static
def test_scripts_are_found():
    names = [os.path.relpath(p, RECIPE_DIR) for p in SCRIPTS]
    assert 'run.sh' in names and LAUNCHER in names
    assert os.path.join('local', 'slurm', 'eval.sbatch') in names


@pytest.mark.parametrize('script', SCRIPTS,
                         ids=lambda p: os.path.relpath(p, RECIPE_DIR))
def test_bash_syntax(script):
    subprocess.run(['bash', '-n', script], check=True)


@pytest.mark.skipif(shutil.which('shellcheck') is None,
                    reason='shellcheck is not installed')
@pytest.mark.parametrize('script', ['run.sh', LAUNCHER])
def test_shellcheck(script):
    subprocess.run(['shellcheck', '-s', 'bash', '-x', script], check=True,
                   cwd=RECIPE_DIR)


@pytest.mark.parametrize('script', SCRIPTS,
                         ids=lambda p: os.path.relpath(p, RECIPE_DIR))
def test_no_forbidden_options(script):
    with open(script) as fin:
        lines = fin.read().splitlines()
    for line in lines:
        if line.startswith('#SBATCH'):
            # resources and site names belong on the command line
            assert not re.search(
                r'--cpus-per-task|--mem\b|--mem=|--mem-per|\s-c\s|'
                r'--account|\s-A\s|--partition|\s-p\s|--time|--output|'
                r'--gpus|--nodes', line), line
        code = line.split('#', 1)[0]
        assert not re.search(r'\bset\s+-[a-z]*u', code), line
        assert not re.search(r'set\s+-o\s+nounset', code), line


def test_launcher_shell_options():
    text = read(LAUNCHER)
    assert re.search(r'^set -o pipefail$', text, re.M)
    assert 'cd "${SLURM_SUBMIT_DIR:-.}"' in text
    sbatch = [x for x in text.splitlines() if x.startswith('#SBATCH')]
    assert sbatch == ['#SBATCH --job-name ovsb_train']


@pytest.mark.parametrize('path', NEW_TEXT_FILES)
def test_no_site_specifics(path):
    text = '\n'.join(line for line in read(path).splitlines()
                     if 'Copyright (c)' not in line)
    for pattern in (r'/scratch/', r'/mnt/', r'/home/', r'OPEN-\d', r'qgpu',
                    r'karolina', r'it4i', r'acn\d', os.environ.get('USER',
                                                                   'nobody')):
        assert not re.search(pattern, text, re.I), (path, pattern)


def test_readme_documents_the_chain():
    text = read('local/slurm/README.md')
    assert '--dependency=afterany:' in text and '--signal B:TERM@300' in text
    assert 'OVSB_DRY_RUN=1' in text
    example = read('local/slurm/site.env.example')
    for name in ('OVSB_CONDA_ENV', 'OVSB_MODULES', 'EXP', 'CONFIG', 'INIT',
                 'EPOCHS_PER_SEG', 'DRIFT_REF', 'DRIFT_GATE',
                 'EXPECT_GLOBAL_BATCH', 'AUG_TO_SHM', 'TMPDIR'):
        assert re.search(r'^%s=' % name, example, re.M), name


# ------------------------------------------------- the launcher, stand-ins
SHIM = '''\
#!/bin/bash
# stand-in for "python" (tests/test_slurm_scripts.py)
echo "python $*" >> "{fake}/calls"
case "$*" in
  *"--plan"*) cat "{fake}/plan"; exit "$(cat "{fake}/plan_rc")" ;;
  *"torch.distributed.run"*) exec "{python}" "{fake}/torchrun.py" ;;
  *"drift_gate.py"*) exit "$(cat "{fake}/gate_rc")" ;;
esac
echo "unexpected python call: $*" >&2
exit 99
'''

# torchrun: starts the training process in a new session and waits for it.
# It has no handler for SIGUSR1 (the real one has none either): if the
# launcher signalled it, it would die and leave the mark below.
TORCHRUN = '''\
import os, signal, subprocess, sys
fake = os.path.dirname(os.path.abspath(__file__))


def marked(signum, frame):
    open(os.path.join(fake, 'torchrun_got_%d' % signum), 'w').close()


for signum in (signal.SIGTERM, signal.SIGUSR1):
    signal.signal(signum, marked)
proc = subprocess.Popen([sys.executable, os.path.join(fake, 'rank.py')],
                        start_new_session=True)
sys.exit(proc.wait())
'''

# the training process: "trains" epochs (a marker file each), stops at
# SIGUSR1 with a partial state, like ovsb.bin.train
RANK = '''\
import json, os, signal, sys, time
fake = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(fake, 'rank.json')) as fin:
    conf = json.load(fin)
models = os.path.join(conf['exp'], 'models')
stop = []
signal.signal(signal.SIGUSR1, lambda s, f: stop.append(s))
open(os.path.join(fake, 'rank_ready'), 'w').close()
for epoch in range(conf['first'], conf['first'] + conf['epochs']):
    end = time.time() + conf['epoch_sec']
    while time.time() < end and not stop:
        time.sleep(0.05)
    if stop:
        open(os.path.join(models, 'partial_state.pt'), 'w').close()
        sys.exit(0)
    open(os.path.join(models, 'model_%d.pt' % epoch), 'w').close()
    open(os.path.join(models, 'train_state_%d.pt' % epoch), 'w').close()
if conf['final']:
    open(os.path.join(models, 'final_model.pt'), 'w').close()
sys.exit(conf['rc'])
'''


class Site:
    """A temporary run with the stand-in python and a site file."""

    def __init__(self, tmp_path, gate_rc=0, epoch_done=0, epochs=1,
                 epoch_sec=0.3, final=False, rc=0, num_epochs=3, batch=48):
        self.tmp = str(tmp_path)
        self.fake = os.path.join(self.tmp, 'fake')
        self.exp = os.path.join(self.tmp, 'exp')
        os.makedirs(os.path.join(self.fake, 'bin'))
        os.makedirs(os.path.join(self.exp, 'models'))
        shim = os.path.join(self.fake, 'bin', 'python')
        with open(shim, 'w') as fout:
            fout.write(SHIM.format(fake=self.fake, python=sys.executable))
        os.chmod(shim, os.stat(shim).st_mode | stat.S_IXUSR)
        for name, text in (('torchrun.py', TORCHRUN), ('rank.py', RANK)):
            with open(os.path.join(self.fake, name), 'w') as fout:
                fout.write(text)
        self.write('gate_rc', str(gate_rc))
        self.write('plan_rc', '0')
        self.write('calls', '')
        with open(os.path.join(self.fake, 'rank.json'), 'w') as fout:
            json.dump(dict(exp=self.exp, first=epoch_done + 1, epochs=epochs,
                           epoch_sec=epoch_sec, final=final, rc=rc), fout)
        self.write('plan', textwrap.dedent('''\
            PLAN_EXP_DIR={exp}
            PLAN_NUM_EPOCHS={num_epochs}
            PLAN_NUM_AVG=1
            PLAN_WORLD_SIZE=1
            PLAN_BATCH_SIZE={batch}
            PLAN_GLOBAL_BATCH={batch}
            PLAN_SAVE_OPTIMIZER_STATE=1
            PLAN_INIT_CHECKPOINT={tmp}/init.pt
            PLAN_NOISE_LMDB=data/musan/lmdb
            PLAN_REVERB_LMDB=data/rirs/lmdb
            PLAN_RESUME_KIND={kind}
            PLAN_EPOCH_DONE={done}
            PLAN_NEXT_EPOCH={next}
            PLAN_NEXT_STEP=0
            ''').format(exp=self.exp, tmp=self.tmp, num_epochs=num_epochs,
                        batch=batch, done=epoch_done, next=epoch_done + 1,
                        kind='epoch' if epoch_done else 'fresh'))
        for name in ('init.pt', 'ref.pt'):
            open(os.path.join(self.tmp, name), 'w').close()
        self.settings = dict(
            EXP=self.exp, CONFIG='conf/stage2_v9c.yaml',
            INIT=os.path.join(self.tmp, 'init.pt'), EPOCHS_PER_SEG='1',
            DRIFT_REF=os.path.join(self.tmp, 'ref.pt'), DRIFT_GATE='0.08',
            EXPECT_GLOBAL_BATCH='48', AUG_TO_SHM='0', TMPDIR=self.tmp)

    def write(self, name, text):
        with open(os.path.join(self.fake, name), 'w') as fout:
            fout.write(text)

    def calls(self):
        with open(os.path.join(self.fake, 'calls')) as fin:
            return fin.read().splitlines()

    def trained(self):
        return any('torch.distributed.run' in c for c in self.calls())

    def gate_calls(self):
        return sum('drift_gate.py' in c for c in self.calls())

    def models(self):
        return sorted(os.listdir(os.path.join(self.exp, 'models')))

    def env(self, **extra):
        site_env = os.path.join(self.tmp, 'site.env')
        with open(site_env, 'w') as fout:
            for name, value in self.settings.items():
                if value is not None:
                    fout.write('%s=%s\n' % (name, value))
        env = {k: v for k, v in os.environ.items()
               if k not in self.settings and not k.startswith(('SLURM_',
                                                               'OVSB_'))}
        env.update(PATH=os.path.join(self.fake, 'bin') + os.pathsep
                   + env['PATH'], OVSB_SITE_ENV=site_env, OVSB_POLL_SEC='0.2')
        env.pop('TMPDIR', None)
        env.update(extra)
        return env

    def start(self, **extra):
        return subprocess.Popen(['bash', LAUNCHER], cwd=RECIPE_DIR,
                                env=self.env(**extra), text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

    def run(self, **extra):
        proc = self.start(**extra)
        out, _ = proc.communicate(timeout=120)
        return proc.returncode, out


def test_segment_runs_and_gates_every_epoch(tmp_path):
    site = Site(tmp_path, epochs=2)
    rc, out = site.run()
    assert rc == 0, out
    assert site.models() == ['model_1.pt', 'model_2.pt', 'train_state_1.pt',
                             'train_state_2.pt']
    assert site.gate_calls() >= 2  # after epoch 1 and at the end
    assert 'segment done; submit the same command again' in out
    train = [c for c in site.calls() if 'torch.distributed.run' in c]
    assert len(train) == 1
    for text in ('--standalone --nnodes=1 --nproc_per_node=1 -m '
                 'ovsb.bin.train --config conf/stage2_v9c.yaml',
                 '--set run.exp_dir=%s' % site.exp, '--set run.gpus=[0]',
                 '--set model.init.checkpoint=%s/init.pt' % site.tmp,
                 '--resume --epochs_this_run 1'):
        assert text in train[0], text
    gate = [c for c in site.calls() if 'drift_gate.py' in c][0]
    assert gate == 'python local/drift_gate.py --exp %s --ref %s/ref.pt ' \
        '--gate 0.08' % (site.exp, site.tmp)
    with open(os.path.join(site.exp, 'segments.tsv')) as fin:
        assert fin.read().split()[1:] == ['none', '1', '0']
    assert not glob.glob(os.path.join(site.fake, 'torchrun_got_*'))


def test_finished_run_reports_the_final_model(tmp_path):
    site = Site(tmp_path, epoch_done=2, final=True)
    rc, out = site.run()
    assert rc == 0 and 'all 3 epochs are done' in out, out
    assert site.gate_calls() >= 2  # before training too: epochs exist


def test_missing_drift_reference_stops_before_training(tmp_path):
    site = Site(tmp_path)
    os.remove(os.path.join(site.tmp, 'ref.pt'))
    rc, out = site.run()
    assert rc == 1 and 'DRIFT_REF' in out and 'does not exist' in out
    assert not site.trained() and site.gate_calls() == 0
    assert site.models() == []
    site.settings['DRIFT_REF'] = None  # not set at all
    rc, out = site.run()
    assert rc == 1 and 'not set: DRIFT_REF' in out
    assert not site.trained()


@pytest.mark.parametrize('name', ['EXP', 'CONFIG', 'INIT', 'EPOCHS_PER_SEG',
                                  'DRIFT_GATE', 'EXPECT_GLOBAL_BATCH',
                                  'AUG_TO_SHM', 'TMPDIR'])
def test_every_setting_is_required(tmp_path, name):
    site = Site(tmp_path)
    site.settings[name] = None
    rc, out = site.run()
    assert rc == 1 and 'not set: %s' % name in out, out
    assert site.calls() == []


@pytest.mark.parametrize('gate_rc', [3, 2])
def test_gate_failure_after_an_epoch_stops_the_run(tmp_path, gate_rc):
    """Exceeded (3) and error (2): the trainer is asked to stop through
    SIGUSR1 to the training process, the job ends with the gate's status."""
    site = Site(tmp_path, gate_rc=gate_rc, epochs=50, epoch_sec=0.4)
    rc, out = site.run()
    assert rc == gate_rc, out
    assert 'after epoch 1' in out
    assert ('DRIFT GATE EXCEEDED' in out) == (gate_rc == 3)
    assert ('DRIFT GATE ERROR' in out) == (gate_rc == 2)
    assert 'partial_state.pt' in site.models()  # it saved, then stopped
    assert 'model_5.pt' not in site.models()
    assert site.gate_calls() == 1
    assert not glob.glob(os.path.join(site.fake, 'torchrun_got_*'))


@pytest.mark.parametrize('gate_rc', [3, 2, 1])
def test_gate_failure_before_training_trains_nothing(tmp_path, gate_rc):
    site = Site(tmp_path, gate_rc=gate_rc, epoch_done=1)
    rc, out = site.run()
    assert rc == gate_rc and 'before training' in out, out
    assert not site.trained()


def test_gate_failure_at_the_end_is_the_exit_status(tmp_path):
    """The trainer ends between two polls: the last gate still decides."""
    site = Site(tmp_path, epochs=1, epoch_sec=0.0)
    site.write('gate_rc', '3')
    rc, out = site.run(OVSB_POLL_SEC='1')
    assert rc == 3 and 'DRIFT GATE EXCEEDED' in out, out


def test_trainer_failure_is_the_exit_status(tmp_path):
    site = Site(tmp_path, rc=7)
    rc, out = site.run()
    assert rc == 7 and 'the trainer failed' in out, out


def test_wrong_global_batch_and_refused_config(tmp_path):
    site = Site(tmp_path, batch=32)
    rc, out = site.run()
    assert rc == 1 and 'EXPECT_GLOBAL_BATCH is 48' in out, out
    assert not site.trained()
    site = Site(tmp_path / 'b')
    site.write('plan_rc', '2')
    rc, out = site.run()
    assert rc == 1 and 'the trainer refused' in out
    assert not site.trained()
    site = Site(tmp_path / 'c')
    os.remove(os.path.join(site.tmp, 'init.pt'))
    rc, out = site.run()
    assert rc == 1 and 'INIT' in out and not site.trained()


def test_sigterm_reaches_the_training_process_as_sigusr1(tmp_path):
    """The time-limit signal: SIGTERM to the batch shell only."""
    site = Site(tmp_path, epochs=50, epoch_sec=30)
    proc = site.start()
    ready = os.path.join(site.fake, 'rank_ready')
    deadline = time.time() + 60
    while not os.path.exists(ready) and time.time() < deadline:
        time.sleep(0.05)
    assert os.path.exists(ready)
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert 'SIGTERM: asking the training processes to save and stop' in out
    assert 'stop request: term' in out
    assert site.models() == ['partial_state.pt']
    # torchrun saw neither signal; only its child was signalled
    assert not glob.glob(os.path.join(site.fake, 'torchrun_got_*'))
    assert site.gate_calls() == 0  # the next job gates before it trains


def test_dry_run_prints_the_commands(tmp_path):
    """Real interpreter, real config: checks run, nothing is started."""
    site = Site(tmp_path)
    site.settings.update(EXPECT_GLOBAL_BATCH='384', AUG_TO_SHM='1')
    env = site.env(OVSB_DRY_RUN='1', SLURM_GPUS_ON_NODE='8',
                   PATH=os.path.dirname(sys.executable) + os.pathsep
                   + os.environ['PATH'])
    proc = subprocess.run(['bash', LAUNCHER], cwd=RECIPE_DIR, env=env,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=300)
    out = proc.stdout
    assert proc.returncode == 0, out
    lines = [x[2:] for x in out.splitlines() if x.startswith('+ ')]
    assert len(lines) == 5, out
    assert lines[0].startswith('mkdir -p /dev/shm/ovsb_aug_')
    assert re.fullmatch(r'cp data/musan/lmdb/data.mdb /dev/shm/ovsb_aug_\d+'
                        r'/noise/', lines[1])
    assert re.fullmatch(r'cp data/rirs/lmdb/data.mdb /dev/shm/ovsb_aug_\d+'
                        r'/reverb/', lines[2])
    assert lines[3].startswith(
        'python -m torch.distributed.run --standalone --nnodes=1 '
        '--nproc_per_node=8 -m ovsb.bin.train --config conf/stage2_v9c.yaml '
        '--set run.exp_dir=%s --set run.gpus=[0,1,2,3,4,5,6,7] ' % site.exp)
    assert re.search(r'--set data\.augment\.noise_lmdb=/dev/shm/ovsb_aug_\d+'
                     r'/noise --set data\.augment\.reverb_lmdb=/dev/shm/'
                     r'ovsb_aug_\d+/reverb --resume --epochs_this_run 1$',
                     lines[3])
    assert lines[4] == 'python local/drift_gate.py --exp %s --ref %s/ref.pt ' \
        '--gate 0.08' % (site.exp, site.tmp)
    assert 'fresh: next epoch 1 step 0 of 14 epochs' in out
    assert site.models() == [] and site.calls() == []
    assert not glob.glob('/dev/shm/ovsb_aug_*/noise')
    # the same dry run without the reference: stops in the checks
    os.remove(os.path.join(site.tmp, 'ref.pt'))
    proc = subprocess.run(['bash', LAUNCHER], cwd=RECIPE_DIR, env=env,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=300)
    assert proc.returncode == 1 and 'DRIFT_REF' in proc.stdout
    assert '+ ' not in proc.stdout


def test_run_sh_prints_the_cluster_stages(tmp_path):
    """GPU stages print a command and submit nothing."""
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep
               + os.environ['PATH'])
    proc = subprocess.run(
        ['bash', 'run.sh', '--stage', '2', '--stop_stage', '2'],
        cwd=RECIPE_DIR, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode == 0, proc.stdout
    assert 'wespeaker/bin/train.py --config conf/stage1_frozen.yaml' in \
        proc.stdout
    for stage, text in ((4, 'bash local/init/build_init.sh --stage 3 '
                         '--stop_stage 3'),
                        (6, 'local/slurm/train_segment.sbatch'),
                        (8, 'local/slurm/eval.sbatch')):
        proc = subprocess.run(
            ['bash', 'run.sh', '--stage', str(stage), '--stop_stage',
             str(stage)], cwd=RECIPE_DIR, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        assert proc.returncode == 0, proc.stdout
        assert text in proc.stdout, proc.stdout
    assert 'sbatch --account <account> --partition <partition>' in proc.stdout
