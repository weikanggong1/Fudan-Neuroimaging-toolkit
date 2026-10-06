"""Validation-only deadlines and inherited resource locks; no scientific imports."""
from contextlib import contextmanager
import os
import signal
import subprocess
import time


@contextmanager
def controller_deadline(seconds):
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise TimeoutError("declared controller outer deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def locked_child(command, environment, log, lock_fd, wait_seconds, sample=None):
    environment = {**environment, "FNIT_VALIDATION_LOCK_FD": str(lock_fd)}
    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                               pass_fds=(lock_fd,))
    row = {"leader_PID": process.pid, "child_reaped": False, "returncode": None,
           "lock_FD_inherited": True, "cleanup_trigger": None, "cleanup_errors": []}
    started = time.monotonic()
    try:
        if sample is None:
            code = process.wait(timeout=wait_seconds)
        else:
            while process.poll() is None:
                if time.monotonic() - started >= wait_seconds:
                    raise subprocess.TimeoutExpired(command, wait_seconds)
                sample(process.pid)
                time.sleep(.5)
            code = process.wait(timeout=5)
        row.update(returncode=code, child_reaped=True)
    except BaseException as error:
        row["cleanup_trigger"] = type(error).__name__
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as cleanup_error:
            row["cleanup_errors"].append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
        try:
            process.wait(timeout=5)
            row["child_reaped"] = True
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            row["cleanup_errors"].append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
        row["returncode"] = 124
        # An unreaped child owns the same inherited descriptor and keeps the
        # shared lock after the controller closes its own descriptor.
        if not isinstance(error, subprocess.TimeoutExpired):
            error.child_receipt = row
            raise
    return row


def child_subprocess_lock_wrapper(original, lock_fd):
    """Pass the inherited lock to compiler/version subprocesses in this worker."""
    os.fstat(lock_fd)
    def inherited(*args, **kwargs):
        descriptors = set(kwargs.pop("pass_fds", ()))
        descriptors.add(lock_fd)
        kwargs["pass_fds"] = tuple(sorted(descriptors))
        return original(*args, **kwargs)
    return inherited
