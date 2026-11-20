"""Run provenance written once at the start of a training process.

``<exp_dir>/run_meta_<SLURM_JOB_ID or pid>.json`` holds the git commit and
dirty flag of the recipe directory, the sha256 of the config file, of the
training lists and of the initial checkpoint's head record, the job id,
seed, number of processes, host and time. A missing file or a missing git
is recorded as null: provenance never stops a run.
"""
import hashlib
import json
import os
import socket
import subprocess
import time
from typing import Any, Dict, Optional

from ovsb.config.schema import Config
from ovsb.model.head_record import head_sidecar_path


def sha256_file(path: Optional[str], bufsize: int = 1 << 24) -> Optional[str]:
    """sha256 of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as fin:
            for block in iter(lambda: fin.read(bufsize), b''):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, TypeError):
        return None


def git_info(cwd: str) -> Dict[str, Any]:
    """{'commit', 'dirty', 'branch'} of the git tree at ``cwd``, or nulls."""
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(['git', '-C', cwd] + list(args),
                                  capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = run('status', '--porcelain', '--untracked-files=no')
    return dict(commit=run('rev-parse', 'HEAD') or None,
                dirty=None if status is None else bool(status),
                branch=run('rev-parse', '--abbrev-ref', 'HEAD') or None)


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, 'r', encoding='utf8') as fin:
            return json.load(fin)
    except (OSError, ValueError):
        return None


def collect_run_meta(cfg: Config, config_path: Optional[str] = None,
                     checkpoint: Optional[str] = None,
                     world_size: Optional[int] = None,
                     cwd: Optional[str] = None) -> Dict[str, Any]:
    """The provenance record (see the module docstring).

    Args:
        cfg: the recipe config.
        config_path: the config file the run was started with.
        checkpoint: the checkpoint the run loads.
        world_size: number of training processes.
        cwd: recipe directory (default: the working directory).
    """
    cwd = cwd or os.getcwd()
    lists = cfg.data.lists
    side = head_sidecar_path(checkpoint) if checkpoint else None
    return dict(
        time=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        host=socket.gethostname(),
        recipe_dir=os.path.abspath(cwd),
        git=git_info(cwd),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_job_name=os.environ.get('SLURM_JOB_NAME'),
        world_size=world_size,
        seed=cfg.run.seed,
        exp_dir=cfg.run.exp_dir,
        config=dict(path=os.path.abspath(config_path) if config_path else None,
                    sha256=sha256_file(config_path)),
        checkpoint=dict(path=os.path.abspath(checkpoint) if checkpoint else None,
                        sha256_head_json=sha256_file(side),
                        head_json=_read_json(side) if side else None),
        data={name: dict(path=getattr(lists, name),
                         sha256=sha256_file(getattr(lists, name)))
              for name in ('train_label', 'spk2corpus', 'spk2lang')},
    )


def write_run_meta(cfg: Config, config_path: Optional[str] = None,
                   checkpoint: Optional[str] = None,
                   world_size: Optional[int] = None,
                   cwd: Optional[str] = None) -> Optional[str]:
    """Write the record into ``cfg.run.exp_dir``; returns its path.

    Returns None when nothing could be written.
    """
    try:
        meta = collect_run_meta(cfg, config_path, checkpoint, world_size, cwd)
        tag = os.environ.get('SLURM_JOB_ID') or 'pid%d' % os.getpid()
        os.makedirs(cfg.run.exp_dir, exist_ok=True)
        path = os.path.join(cfg.run.exp_dir, 'run_meta_%s.json' % tag)
        with open(path + '.tmp', 'w', encoding='utf8') as fout:
            json.dump(meta, fout, indent=1, sort_keys=True, default=str)
        os.replace(path + '.tmp', path)
        return path
    except Exception:  # noqa: B902  provenance must never stop a run
        return None
