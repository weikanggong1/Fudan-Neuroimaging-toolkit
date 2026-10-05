"""Own CPU adapter dependency, cache and bounded build controls."""
import errno
import os
import signal
import subprocess

import pytest

from fnit.synthmorph import _cpu_eigen as eigen


@pytest.mark.parametrize('minor,valid', [(0, True), (90, False)])
def test_eigen_version_pin_and_precision_flags(tmp_path, monkeypatch, minor, valid):
    include = tmp_path / 'include'
    (include / 'Eigen/src/Core/util').mkdir(parents=True)
    (include / 'Eigen/Core').write_text('// test header')
    (include / 'Eigen/src/Core/util/Macros.h').write_text(
        '#define EIGEN_WORLD_VERSION 3\n#define EIGEN_MAJOR_VERSION 4\n#define EIGEN_MINOR_VERSION ' + str(minor) + '\n')
    monkeypatch.setenv('FNIT_EIGEN_INCLUDE', str(include))
    monkeypatch.setenv('CXX', '/bin/true -ffast-math')
    if valid:
        _, compiler, _, flags, _, _ = eigen._build_inputs()
        assert compiler[-1] == '-ffast-math'
        assert flags[-2:] == ['-fno-fast-math', '-ffp-contract=off']
    else:
        with pytest.raises(RuntimeError, match='requires Eigen 3.4.0'):
            eigen._build_inputs()


def test_cache_directory_and_files_are_private(tmp_path):
    cache = tmp_path / 'cache'
    eigen._private_directory(cache)
    assert cache.stat().st_mode & 0o777 == 0o700
    file = cache / 'build.json'
    with eigen._cache_file(file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 'w') as stream:
        stream.write('{}')
    assert file.stat().st_mode & 0o777 == 0o600
    assert eigen._cache_digest(file)


def test_private_directory_does_not_require_nofollow_chmod(tmp_path, monkeypatch):
    def unsupported(*arguments, **keywords):
        raise NotImplementedError('chmod: follow_symlinks unavailable on this platform')
    monkeypatch.setattr(eigen.os, 'chmod', unsupported)
    cache = tmp_path / 'private-cache'
    eigen._private_directory(cache)
    assert cache.stat().st_mode & 0o777 == 0o700


def test_private_directory_rejects_unenforced_permissions(tmp_path, monkeypatch):
    cache = tmp_path / 'public-cache'
    cache.mkdir(mode=0o755)
    monkeypatch.setattr(eigen.os, 'fchmod', lambda descriptor, mode: None)
    with pytest.raises(RuntimeError, match='must enforce directory mode 0700'):
        eigen._private_directory(cache)


def test_cache_rejects_symlink_directory_file_and_hardlink(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    directory = tmp_path / 'cache'
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match='not a symlink'):
        eigen._private_directory(directory)
    file = target / 'payload'
    file.write_bytes(b'unchanged')
    symlink = tmp_path / 'link'
    symlink.symlink_to(file)
    with pytest.raises(OSError):
        eigen._cache_digest(symlink)
    hardlink = tmp_path / 'hardlink'
    os.link(file, hardlink)
    with pytest.raises(RuntimeError, match='private, owned and regular'):
        eigen._cache_digest(hardlink)
    assert file.read_bytes() == b'unchanged'


def test_lock_contention_is_nonblocking_and_bounded(tmp_path, monkeypatch):
    import fcntl
    times = iter((0., 21.))
    monkeypatch.setattr(eigen.time, 'monotonic', lambda: next(times))
    def busy(descriptor, options):
        assert options & fcntl.LOCK_NB
        raise OSError(errno.EAGAIN, 'already locked')
    monkeypatch.setattr(fcntl, 'flock', busy)
    with (tmp_path / 'lock').open('w') as lock:
        with pytest.raises(TimeoutError, match='lock timed out'):
            eigen._cache_lock(lock)


def test_compile_timeout_kills_only_its_new_process_group(monkeypatch):
    class Process:
        pid = 123456789
        returncode = 0
        calls = 0
        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                assert timeout == 60
                raise subprocess.TimeoutExpired(['compiler'], timeout)
            assert timeout == 2
            return '', ''
    child = Process()
    def create(command, **options):
        assert options['start_new_session'] is True
        return child
    killed = []
    monkeypatch.setattr(eigen.subprocess, 'Popen', create)
    monkeypatch.setattr(eigen.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(TimeoutError, match='compilation exceeded 60'):
        eigen._compile(['compiler'])
    assert killed == [(child.pid, signal.SIGTERM)]


def test_compile_timeout_remains_bounded_after_kill(monkeypatch):
    class Pipe:
        closed = False
        def close(self):
            self.closed = True
    class Process:
        pid = 123456789
        stdout, stderr = Pipe(), Pipe()
        timeouts = []
        def communicate(self, timeout=None):
            self.timeouts.append(timeout)
            raise subprocess.TimeoutExpired(['compiler'], timeout)
    child = Process()
    monkeypatch.setattr(eigen.subprocess, 'Popen', lambda *args, **kwargs: child)
    killed = []
    monkeypatch.setattr(eigen.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(TimeoutError, match='compilation exceeded 60'):
        eigen._compile(['compiler'])
    assert child.timeouts == [60, 2, 2]
    assert child.stdout.closed and child.stderr.closed
    assert killed == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
