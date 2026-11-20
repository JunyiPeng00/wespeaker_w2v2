"""Process isolation for the torchaudio / FFmpeg calls of the channel chain.

:class:`IsolatedEffector` is a drop-in for
``torchaudio.io.AudioEffector(...).apply()``: the very same torchaudio /
libav code runs in a persistent helper process (one per DataLoader worker,
started lazily on the first call), fed over a pipe. A native crash (SIGSEGV,
SIGABRT, ...) or a hang of the helper then kills the helper only: the caller
restarts it and retries the same call once with the same parameters; if the
retry fails too, ``apply()`` raises :class:`HelperFailure`, which
``ovsb.data.channel.apply_channel`` handles like any other effector failure
(the codec is skipped and the failure is counted). A Python exception
raised by ``AudioEffector`` inside the helper is re-raised as
:class:`EffectorError` without a retry (deterministic, as in-process).

The helper is a fresh interpreter (fork + exec of ``sys.executable``), so it
inherits no threads, locks or CUDA state of the DataLoader worker; it
imports only numpy / torch / torchaudio, and the handshake checks that
torchaudio and the libav* versions are the ones of the calling process.

Random parameters are drawn by the caller before the call (nothing in here
touches ``random``), so the caller's RNG consumption does not depend on
whether the helper crashed.

Any other failure of the transport or of the helper start (an unexpected
exception anywhere between spawning the helper and decoding its reply)
kills and reaps the helper, is counted (``transport_errors`` /
``start_failures``) and takes the same retry-once-then-HelperFailure path:
no exception leaves a half-initialised helper in place. Pipe deadlines use
``poll()``, which works for any descriptor number (``select()`` fails above
FD_SETSIZE = 1024, and a DataLoader worker's tar reader may hold thousands
of shard descriptors).

Owner death: the helper arms ``prctl(PR_SET_PDEATHSIG, SIGKILL)`` first
thing at start-up, then checks ``getppid()`` against the owner pid it was
given (closes the race with an owner that died before the signal was
armed); without prctl a watchdog thread polls ``getppid()``. So the helper
dies with its DataLoader worker even in the middle of a hung request. (The
death signal follows the owner THREAD that spawned the helper; a helper
started from a short-lived thread dies with that thread and is restarted at
the next call, counted as a crash.)

Counters are per process (= per DataLoader worker): :func:`get_stats`;
``calls == ok + child_errors + fallbacks``. :func:`stats_line` formats them
for a log line.

This file is also the helper's entry point; it is run as a plain script, so
it must not import the ``ovsb`` package at module level::

    python -P <this file> --ffx-helper --owner <pid>

(stdin: requests, stdout: replies, stderr: shared with the worker log).
"""

import logging
import math
import os
import pickle
import select
import signal
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT = 60.0  # s per call; a 6 s crop takes < 0.1 s
START_TIMEOUT = 600.0  # s for the helper's interpreter start + imports
_LEN = struct.Struct('<Q')
_PIPE_SZ = 1 << 20  # F_SETPIPE_SZ target (a 6 s crop is 384 KB)
_F_SETPIPE_SZ = 1031
_WATCHDOG_S = 1.0  # getppid() poll period of the owner watchdog (no prctl)
_PR_SET_PDEATHSIG = 1
_MAX_EVENT_LINES = 20  # first N failure events per process logged in full
_EVENT_LOG_EVERY = 300.0  # s, then at most one line per interval


class EffectorError(RuntimeError):
    """AudioEffector raised a Python exception inside the helper."""


class HelperFailure(RuntimeError):
    """The helper died / timed out on the call and on its single retry."""


class IsolationConfigError(BaseException):
    """The helper does not run the caller's torchaudio / libav (a setup
    error, raised past `except Exception` so that it is never silent)."""


_STAT_KEYS = ('calls', 'ok', 'child_errors', 'crashes', 'timeouts',
              'transport_errors', 'start_failures', 'retries', 'retry_ok',
              'fallbacks', 'starts')
_STATS = {'pid': None}
_HELPER = None
_INJECT = []  # test hook: per-attempt actions, see set_inject()
_EVENTS = {'n': 0, 'last': 0.0, 'suppressed': 0}


def _stats():
    if _STATS['pid'] != os.getpid():  # new (forked) process: own counters
        _STATS.clear()
        _STATS.update({k: 0 for k in _STAT_KEYS})
        _STATS['pid'] = os.getpid()
        _EVENTS.update(n=0, last=0.0, suppressed=0)
    return _STATS


def get_stats() -> Dict[str, int]:
    """Counters of this process (DataLoader worker).

    calls = ok + child_errors + fallbacks; crashes / timeouts /
    transport_errors / start_failures count failed attempts, retries /
    retry_ok the second attempts, starts the helper processes started.
    """
    s = dict(_stats())
    s.pop('pid', None)
    return s


def stats_line() -> str:
    """The counters of :func:`get_stats` as ``key=value`` words."""
    return ' '.join('%s=%d' % (k, v) for k, v in get_stats().items())


def set_inject(actions: List[Optional[str]]) -> None:
    """Test hook: the next attempts send these actions to the helper, one per
    attempt ('segv' = the helper SIGSEGVs itself after reading the request,
    'hang' = sleeps forever, 'sleep:<s>' = sleeps <s> s first, None =
    normal)."""
    _INJECT[:] = list(actions)


def _worker_id():
    try:
        import torch.utils.data as tud
        wi = tud.get_worker_info()
        return None if wi is None else wi.id
    except Exception:  # noqa
        return None


def _event(level, msg):
    """Rate-limited failure log (first _MAX_EVENT_LINES, then one line per
    _EVENT_LOG_EVERY s carrying the number of suppressed events)."""
    now = time.time()
    _EVENTS['n'] += 1
    if _EVENTS['n'] <= _MAX_EVENT_LINES or \
            now - _EVENTS['last'] >= _EVENT_LOG_EVERY:
        sup = _EVENTS['suppressed']
        _EVENTS['suppressed'] = 0
        _EVENTS['last'] = now
        logging.log(level, 'ffmpeg_isolation[pid %d worker %s]: %s%s | %s',
                    os.getpid(), _worker_id(), msg,
                    ' (+%d events not logged)' % sup if sup else '',
                    stats_line())
    else:
        _EVENTS['suppressed'] += 1


# ---------------------------------------------------------------------------
# pipe I/O with a deadline


class _Dead(Exception):

    def __init__(self, kind, detail=''):
        super().__init__(kind, detail)
        self.kind, self.detail = kind, detail


def _left(deadline, what):
    """seconds left (None = no deadline); raises _Dead on expiry"""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise _Dead('timeout', what)
    return left


def _classify(kind, rc):
    """'eof' with a signal death -> 'crash'"""
    if kind.endswith('eof') and rc is not None and rc < 0:
        try:
            sig = signal.Signals(-rc).name
        except ValueError:
            sig = 'signal %d' % -rc
        return kind[:-3] + 'crash', sig
    return kind, 'rc %s' % rc


def _wait(fd, events, deadline, what):
    """poll() one descriptor until it is ready (any fd number, unlike
    select()); returns the revents; raises _Dead('timeout') at the deadline
    (None = wait forever). EINTR is retried by Python (PEP 475)."""
    p = select.poll()
    p.register(fd, events)
    while True:
        left = _left(deadline, what)
        ms = -1 if left is None else max(1, int(math.ceil(left * 1000.0)))
        ready = p.poll(ms)
        if ready:
            return ready[0][1]


def _write_all(fd, data, deadline):
    view = memoryview(data)
    while view:
        # POLLERR / POLLHUP: the write below reports the broken pipe
        _wait(fd, select.POLLOUT, deadline, 'write')
        try:
            k = os.write(fd, view)
        except BlockingIOError:
            continue
        except (BrokenPipeError, ConnectionResetError) as e:
            raise _Dead('eof', 'write: %s' % e)
        view = view[k:]


def _read_exact(fd, n, deadline):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        # POLLHUP without data: the read below returns 0 (EOF)
        _wait(fd, select.POLLIN, deadline, 'read')
        try:
            k = os.readv(fd, [view[got:]])
        except BlockingIOError:
            continue
        except ConnectionResetError as e:
            raise _Dead('eof', 'read: %s' % e)
        if k == 0:
            raise _Dead('eof', 'read')
        got += k
    return buf


def _send_msg(fd, obj, payload, deadline):
    head = pickle.dumps(obj, protocol=4)
    _write_all(fd, _LEN.pack(len(head)) + head + _LEN.pack(len(payload)),
               deadline)
    if payload:
        _write_all(fd, payload, deadline)


def _recv_msg(fd, deadline):
    (n, ) = _LEN.unpack(_read_exact(fd, _LEN.size, deadline))
    obj = pickle.loads(_read_exact(fd, n, deadline))
    (m, ) = _LEN.unpack(_read_exact(fd, _LEN.size, deadline))
    payload = _read_exact(fd, m, deadline) if m else b''
    return obj, payload


# ---------------------------------------------------------------------------
# parent side


_IDENTITY = {}


def _local_identity():
    if _IDENTITY:
        return dict(_IDENTITY)
    import torchaudio
    import torchaudio.utils.ffmpeg_utils as fu
    _IDENTITY.update(torchaudio=os.path.realpath(torchaudio.__file__),
                     libav=fu.get_versions())
    return dict(_IDENTITY)


class _Helper:
    """One helper process, owned by the process that started it."""

    def __init__(self):
        self.owner = os.getpid()
        self.proc = None
        self.wfd = self.rfd = None
        self.guard = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        st = _stats()
        st['starts'] += 1
        cmd = [sys.executable, '-P', os.path.abspath(__file__),
               '--ffx-helper', '--owner', str(os.getpid())]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None,
                                     close_fds=True, bufsize=0)
        self.wfd = self.proc.stdin.fileno()
        self.rfd = self.proc.stdout.fileno()
        os.set_blocking(self.wfd, False)
        try:
            import fcntl
            fcntl.fcntl(self.wfd, _F_SETPIPE_SZ, _PIPE_SZ)
            fcntl.fcntl(self.rfd, _F_SETPIPE_SZ, _PIPE_SZ)
        except Exception:  # noqa (optional: bigger pipe = fewer wake-ups)
            pass
        hello, _ = _recv_msg(self.rfd, time.monotonic() + START_TIMEOUT)
        mine = _local_identity()
        if hello.get('torchaudio') != mine['torchaudio'] or \
                hello.get('libav') != mine['libav']:
            self.kill()
            raise IsolationConfigError(
                'ffmpeg helper runs other code than its caller: helper %s, '
                'caller %s' % (hello, mine))
        self.guard = hello.get('owner_guard')
        if _worker_id() in (None, 0) and st['starts'] == 1:
            logging.info('ffmpeg_isolation: helper pid %d started for pid %d '
                         '(worker %s; owner guard %s; torchaudio %s, '
                         'libavcodec %s)', self.proc.pid, os.getpid(),
                         _worker_id(), self.guard, mine['torchaudio'],
                         mine['libav'].get('libavcodec'))

    def kill(self):
        """SIGKILL (if alive) + reap; returns the returncode."""
        if self.proc is None:
            return None
        rc = self.proc.poll()
        if rc is None:
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                rc = self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = None
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:  # noqa
                pass
        self.proc = None
        self.wfd = self.rfd = None
        return rc

    def forget(self):
        """Inherited from the parent by fork: close our copies of the pipe
        ends, never signal the parent's helper."""
        if self.proc is not None:
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:  # noqa
                    pass
            self.proc = None
            self.wfd = self.rfd = None

    def call(self, req, payload, timeout):
        """-> ('ok', reply, output tensor) | ('err', reply, None); raises
        _Dead on a dead / hung helper or on ANY other failure of the start,
        the transport or the reply decoding (the helper is then killed and
        reaped, never left half-initialised). Non-Exception BaseExceptions
        (KeyboardInterrupt, IsolationConfigError) also kill the helper and
        propagate."""
        if self.proc is not None and self.proc.poll() is not None:
            # died while idle (between two calls)
            kind, why = _classify('eof', self.kill())
            raise _Dead(kind, '%s (helper died between calls)' % why)
        try:
            if self.proc is None:
                try:
                    self.start()
                except _Dead as d:
                    raise _Dead('start_' + d.kind, 'start: ' + d.detail)
                except Exception as e:  # noqa
                    raise _Dead('start_transport_error', 'start: %s: %s' % (
                        type(e).__name__, e))
            deadline = time.monotonic() + timeout
            _send_msg(self.wfd, req, payload, deadline)
            rep, data = _recv_msg(self.rfd, deadline)
            if rep['status'] == 'err':
                return 'err', rep, None
            if rep['status'] != 'ok':
                raise ValueError('bad reply status %r' % (rep['status'], ))
            import numpy as np
            import torch
            y = np.frombuffer(bytes(data), dtype=np.dtype(rep['dtype']))
            return 'ok', rep, torch.from_numpy(y.reshape(rep['shape']).copy())
        except _Dead as d:
            kind, why = _classify(d.kind, self.kill())
            raise _Dead(kind, '%s (%s)' % (why, d.detail))
        except BaseException as e:
            rc = self.kill()
            if not isinstance(e, Exception):
                raise
            raise _Dead('transport_error', '%s: %s (helper killed, rc %s)' % (
                type(e).__name__, e, rc))


def _helper():
    global _HELPER
    if _HELPER is None or _HELPER.owner != os.getpid():
        if _HELPER is not None:
            _HELPER.forget()
        _HELPER = _Helper()
    return _HELPER


def shutdown() -> None:
    """Stop this process's helper (it also exits on its own when the pipe
    closes, i.e. when the DataLoader worker exits)."""
    global _HELPER
    if _HELPER is not None and _HELPER.owner == os.getpid():
        _HELPER.kill()
    _HELPER = None


def helper_pid() -> Optional[int]:
    """Pid of this process's live helper, else None."""
    h = _HELPER
    if h is None or h.owner != os.getpid() or not h.alive():
        return None
    return h.proc.pid


def _codec_fields(codec_config):
    if codec_config is None:
        return None
    import dataclasses
    return dataclasses.asdict(codec_config)


class IsolatedEffector:
    """Same constructor / apply() as torchaudio.io.AudioEffector (the
    arguments apply_channel uses); the effector itself is built and applied
    inside the helper process. timeout: seconds per call; dump_dir: where
    inputs that killed the helper are saved (npz)."""

    def __init__(self, effect: Optional[str] = None,
                 format: Optional[str] = None, *,
                 encoder: Optional[str] = None, codec_config: Any = None,
                 pad_end: bool = True, timeout: Optional[float] = None,
                 dump_dir: Optional[str] = None) -> None:
        self.spec = dict(effect=effect, format=format, encoder=encoder,
                         codec_config=_codec_fields(codec_config),
                         pad_end=pad_end)
        self.timeout = float(timeout or DEFAULT_TIMEOUT)
        self.dump_dir = dump_dir

    def apply(self, waveform: Any, sample_rate: int,
              output_sample_rate: Optional[int] = None) -> Any:
        """Every call ends in exactly one of ok / child_errors /
        fallbacks: an unexpected Exception anywhere in _apply (request
        preparation, logging, dump) that did not already count a terminal
        outcome is counted as transport_errors + fallbacks and re-raised as
        HelperFailure, i.e. the no-effector path of apply_channel.
        BaseExceptions (KeyboardInterrupt, IsolationConfigError) abort the
        process and propagate uncounted."""
        st = _stats()
        st['calls'] += 1
        done0 = st['ok'] + st['child_errors'] + st['fallbacks']
        try:
            return self._apply(st, waveform, sample_rate, output_sample_rate)
        except Exception as e:
            if st['ok'] + st['child_errors'] + st['fallbacks'] != done0:
                raise
            st['transport_errors'] += 1
            st['fallbacks'] += 1
            _event(logging.WARNING, 'unexpected %r outside the helper '
                   'transport on %s; sample falls back to the no-effector '
                   'path' % (e, self.spec))
            raise HelperFailure('unexpected: %r' % (e,)) from e

    def _apply(self, st, waveform, sample_rate, output_sample_rate):
        import numpy as np
        arr = np.ascontiguousarray(waveform.detach().cpu().numpy())
        req = dict(op='apply', spec=self.spec, sample_rate=int(sample_rate),
                   output_sample_rate=output_sample_rate,
                   shape=arr.shape, dtype=arr.dtype.str, inject=None)
        payload = arr.tobytes()
        for attempt in (0, 1):
            req['inject'] = _INJECT.pop(0) if _INJECT else None
            try:
                status, rep, y = _helper().call(req, payload, self.timeout)
            except _Dead as d:
                if d.kind.startswith('start_'):
                    st['start_failures'] += 1
                if 'timeout' in d.kind:
                    st['timeouts'] += 1
                elif 'transport_error' in d.kind:
                    st['transport_errors'] += 1
                else:
                    st['crashes'] += 1
                self._dump(arr, d, attempt)
                if attempt == 0:
                    st['retries'] += 1
                    _event(logging.WARNING, 'helper %s on %s (%d samples @ '
                           '%d Hz, peak %.3g, finite %s): %s; restarting, '
                           'retrying once' % (
                               d.kind, self.spec, arr.shape[0], sample_rate,
                               float(np.max(np.abs(arr))) if arr.size else 0,
                               bool(np.isfinite(arr).all()), d.detail))
                    continue
                st['fallbacks'] += 1
                _event(logging.WARNING, 'retry failed too (%s: %s) on %s; '
                       'sample falls back to the no-effector path' % (
                           d.kind, d.detail, self.spec))
                raise HelperFailure('%s: %s' % (d.kind, d.detail))
            if status == 'err':
                st['child_errors'] += 1
                raise EffectorError(rep.get('error', '?'))
            st['ok'] += 1
            if attempt == 1:
                st['retry_ok'] += 1
            return y

    def _dump(self, arr, dead, attempt):
        if not self.dump_dir:
            return
        try:
            import numpy as np
            os.makedirs(self.dump_dir, exist_ok=True)
            st = _stats()
            path = os.path.join(self.dump_dir, 'ffx_%d_%d_%d.npz' % (
                os.getpid(), st['crashes'] + st['timeouts'] +
                st['transport_errors'], attempt))
            np.savez(path, x=arr, spec=repr(self.spec), kind=dead.kind,
                     detail=dead.detail)
        except Exception as e:  # noqa (never fatal)
            logging.debug('ffmpeg_isolation: dump failed: %s', e)


# ---------------------------------------------------------------------------
# helper side


def _arm_owner_guard(owner):
    """Die with the owner, even mid-request: PR_SET_PDEATHSIG = SIGKILL
    (kernel-delivered, works while a native call hangs), then the race
    check (owner already gone = reparented); without prctl, or with
    OVSB_FFX_NO_PDEATHSIG=1 (test hook), a daemon thread polls
    getppid() every _WATCHDOG_S s (needs the GIL, i.e. a native hang that
    holds the GIL is covered by PDEATHSIG only). Returns the mechanism."""
    armed = False
    if os.environ.get('OVSB_FFX_NO_PDEATHSIG') != '1':
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            armed = libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL),
                               0, 0, 0) == 0
        except Exception:  # noqa
            armed = False
    if owner and os.getppid() != owner:
        os._exit(0)  # the owner died before the death signal was armed
    if not owner:
        return 'none'
    if armed:
        return 'pdeathsig'
    import threading

    def watch():
        while True:
            time.sleep(_WATCHDOG_S)
            if os.getppid() != owner:
                os._exit(0)
    threading.Thread(target=watch, name='ffx-owner-watchdog',
                     daemon=True).start()
    return 'watchdog'


def _helper_main(owner=None):
    guard = _arm_owner_guard(owner)
    # protocol on a private copy of the stdout pipe; anything a library
    # prints to stdout goes to stderr instead of corrupting the protocol
    wfd = os.dup(1)
    os.dup2(2, 1)
    rfd = os.dup(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    # no core dump: a crash here is handled by the caller (restart +
    # retry), and writing a core of this process (~0.3 GB, systemd-coredump)
    # would hold the crashed helper alive past the caller's timeout
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from torchaudio.io import AudioEffector, CodecConfig
    forever = None
    try:
        _send_msg(wfd, dict(status='hello', pid=os.getpid(),
                            owner_guard=guard, **_local_identity()), b'',
                  forever)
        while True:
            try:
                req, payload = _recv_msg(rfd, forever)
            except _Dead:
                return  # caller gone (EOF)
            inject = req.get('inject')
            if inject == 'segv':
                os.kill(os.getpid(), signal.SIGSEGV)
            elif inject == 'hang':
                while True:
                    time.sleep(3600)
            elif inject and inject.startswith('sleep:'):
                time.sleep(float(inject[6:]))
            try:
                x = np.frombuffer(payload, dtype=np.dtype(
                    req['dtype'])).reshape(req['shape']).copy()
                spec = req['spec']
                cfg = spec['codec_config']
                ae = AudioEffector(
                    effect=spec['effect'], format=spec['format'],
                    encoder=spec['encoder'],
                    codec_config=CodecConfig(**cfg) if cfg is not None
                    else None, pad_end=spec['pad_end'])
                y = ae.apply(torch.from_numpy(x), req['sample_rate'],
                             req['output_sample_rate'])
                y = np.ascontiguousarray(y.numpy())
                rep, out = dict(status='ok', shape=y.shape,
                                dtype=y.dtype.str), y.tobytes()
            except Exception as e:  # noqa
                rep, out = dict(status='err',
                                error='%s: %s' % (type(e).__name__, e)), b''
            _send_msg(wfd, rep, out, forever)
    except _Dead:
        return


if __name__ == '__main__' and '--ffx-helper' in sys.argv:
    _owner = None
    if '--owner' in sys.argv:
        _owner = int(sys.argv[sys.argv.index('--owner') + 1])
    _helper_main(_owner)
