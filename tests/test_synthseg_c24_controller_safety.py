"""Local resource-control software contracts; no Torch, MRI, compiler or GPU."""
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


def safety_module():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "validation/smri_cpu/seg_columns_c24_integration_prepare_20261006/controller_safety.py"
    spec = importlib.util.spec_from_file_location("C24_controller_safety_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unreaped_child_is_explicit_failure_and_receives_lock(monkeypatch, tmp_path):
    safety = safety_module()
    captures = []
    process = SimpleNamespace(pid=123456789, wait=lambda timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("mock", timeout)))
    def popen(command, **kwargs):
        captures.append(kwargs)
        return process
    monkeypatch.setattr(safety.subprocess, "Popen", popen)
    monkeypatch.setattr(safety.os, "killpg", lambda pid, sig: None)
    with (tmp_path / "lock").open("a+b") as lock, (tmp_path / "log").open("wb") as log:
        row = safety.locked_child(["never-executed"], {}, log, lock.fileno(), 1)
        assert captures[0]["pass_fds"] == (lock.fileno(),)
        assert captures[0]["env"]["FNIT_VALIDATION_LOCK_FD"] == str(lock.fileno())
    assert row["returncode"] == 124 and row["child_reaped"] is False
    assert row["cleanup_trigger"] == "TimeoutExpired" and row["cleanup_errors"]


def test_controller_interrupt_cleans_child_and_carries_original_receipt(monkeypatch, tmp_path):
    safety = safety_module()
    waits = []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise KeyboardInterrupt("declared controller interrupt")
        return -9
    monkeypatch.setattr(safety.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123456789, wait=wait))
    killed = []
    monkeypatch.setattr(safety.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with (tmp_path / "lock").open("a+b") as lock, (tmp_path / "log").open("wb") as log:
        with pytest.raises(KeyboardInterrupt) as caught:
            safety.locked_child(["never-executed"], {}, log, lock.fileno(), 1)
    assert killed == [(123456789, signal.SIGKILL)] and waits == [1, 5]
    assert caught.value.child_receipt["child_reaped"] is True
    assert caught.value.child_receipt["cleanup_trigger"] == "KeyboardInterrupt"


def test_inherited_same_lock_survives_parent_descriptor_close(tmp_path):
    safety = safety_module()
    path = tmp_path / "lock"
    parent = path.open("a+b")
    fcntl.flock(parent, fcntl.LOCK_EX)
    descriptor = parent.fileno()
    launcher = safety.child_subprocess_lock_wrapper(subprocess.Popen, descriptor)
    child = launcher([sys.executable, "-c", "import os,sys,time; os.fstat(int(sys.argv[1])); print('ready',flush=True); time.sleep(3)", str(descriptor)],
                     stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        parent.close()
        with path.open("a+b") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        with path.open("a+b") as competitor:
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        parent.close()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def test_deadline_raises_without_waiting_for_next_arm():
    safety = safety_module()
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="outer deadline"):
        with safety.controller_deadline(.01):
            signal.pause()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
