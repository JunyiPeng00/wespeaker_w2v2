"""Stop flag: SIGUSR1 / SIGTERM, all-reduce, forked children."""
import os
import signal
import time

import pytest
import torch
import torch.distributed as dist

from ovsb.engine.signals import STOP_SIGNALS, StopFlag

CPU = torch.device('cpu')


@pytest.fixture()
def flag():
    stop = StopFlag().install()
    yield stop
    stop.restore()


def test_stop_signals():
    assert STOP_SIGNALS == (signal.SIGUSR1, signal.SIGTERM)


@pytest.mark.parametrize('signum', STOP_SIGNALS)
def test_signal_sets_the_flag_and_nothing_else(flag, signum):
    assert flag.received is None and not flag.poll(CPU)
    os.kill(os.getpid(), signum)
    time.sleep(0.01)  # the handler runs between two bytecodes
    assert flag.received == signal.Signals(signum).name
    assert flag.poll(CPU) and flag.agreed
    os.kill(os.getpid(), signal.SIGUSR1)  # a second signal changes nothing
    assert flag.received == signal.Signals(signum).name and flag.poll(CPU)


def test_restore_puts_the_old_handlers_back():
    before = [signal.getsignal(s) for s in STOP_SIGNALS]
    stop = StopFlag().install()
    assert all(signal.getsignal(s) != b for s, b in zip(STOP_SIGNALS, before))
    stop.restore()
    assert [signal.getsignal(s) for s in STOP_SIGNALS] == before


def test_request_from_code(flag):
    flag.request('segment')
    assert flag.received == 'segment' and flag.poll(CPU)


def test_poll_is_sticky_and_stops_reducing(flag, monkeypatch):
    flag.request()
    assert flag.poll(CPU)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', lambda *a, **k: 1 / 0)
    assert flag.poll(CPU)  # agreed: no further all-reduce


def test_poll_takes_the_maximum_over_processes(flag, monkeypatch):
    """A request on another process stops this one at the same poll."""
    seen = []

    def all_reduce(tensor, op=None):
        seen.append((int(tensor.item()), op))
        tensor.fill_(max(int(tensor.item()), others.pop(0)))

    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', all_reduce)
    others = [0, 1]
    assert not flag.poll(CPU)       # nobody asked
    assert flag.received is None
    assert flag.poll(CPU)           # the other process did
    assert seen == [(0, dist.ReduceOp.MAX), (0, dist.ReduceOp.MAX)]
    assert flag.received is None and flag.agreed


def test_forked_child_keeps_the_default_action(flag):
    """A DataLoader worker inherits the handler; a signal must still
    terminate it (the DataLoader stops stuck workers with SIGTERM)."""
    for signum in STOP_SIGNALS:
        pid = os.fork()
        if pid == 0:  # child: wait to be signalled
            try:
                while True:
                    time.sleep(0.05)
            finally:
                os._exit(7)
        time.sleep(0.2)
        os.kill(pid, signum)
        _, status = os.waitpid(pid, 0)
        assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signum
    assert flag.received is None  # the parent was not signalled
