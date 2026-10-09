"""Build and verify FNIT's pinned, source-built MRtrix tracking executable.

This module never searches for an installed MRtrix executable.  Official source
archives and licences stay in the external cache, outside the FNIT source tree.
The existing Conda C/C++ compiler and make are the only build tools required.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen

from fnit._release_assets import release_url_for


SOURCE_COMMIT = "026e850d171ec2a12f09865d31b8332d23d7ecf6"
SCHEMA_VERSION = 1
CONFIG_ISOLATION_ENV = "FNIT_MRTRIX_ISOLATED_CONFIG"
CONFIG_PATCH_SHA256 = "d6e755afa3c1edd67b5b16699e696ab9da3e517e523917d3d7b9dacc12379960"
ARCHIVES = {
    "mrtrix": {
        "filename": f"mrtrix3-{SOURCE_COMMIT}.tar.gz",
        "folder": f"mrtrix3-{SOURCE_COMMIT}",
        "url": f"https://codeload.github.com/MRtrix3/mrtrix3/tar.gz/{SOURCE_COMMIT}",
        "size": 2550496,
        "sha256": "5ee1dc1c8a2e302e5321c77bbb26f6ec8c0891d63877f83e30ff7cc099701990",
    },
    "eigen": {
        "filename": "eigen-3.4.0.tar.gz", "folder": "eigen-3.4.0",
        "url": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
        "size": 2705005,
        "sha256": "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    },
    "zlib": {
        "filename": "zlib-1.3.1.tar.gz", "folder": "zlib-1.3.1",
        "url": "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
        "size": 1512791,
        "sha256": "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    },
}
SOURCE_FILES = {
    "configure": "96d5a14424afd3ca5259663301c921fe31a6b779118479e219134102126b14ee",
    "build": "4e59144b62af4cddd2f837e4b7d9141496a605687f91a617532fc21fdb4e04dc",
    "LICENCE.txt": "fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85",
    "core/file/config.cpp": "06e787a036f13892224b1d4368bd655f6639d776e73504feacfc3dcf8feb83fa",
}


def _configuration_patch_metadata() -> dict:
    return {"file": "core/file/config.cpp",
            "original_sha256": SOURCE_FILES["core/file/config.cpp"],
            "patched_sha256": CONFIG_PATCH_SHA256,
            "environment_variable": CONFIG_ISOLATION_ENV, "enabled_value": "1",
            "scope": "skip automatic system/user config reads only; defaults and CLI -config unchanged"}


def _patch_isolated_configuration(source: Path) -> dict:
    """Wrap only automatic configuration reads; leave all numerical code intact."""
    path = source / "core/file/config.cpp"
    if _sha256(path) != SOURCE_FILES["core/file/config.cpp"]:
        raise ValueError("Pinned configuration source differs before isolation patch")
    text = path.read_text()
    begin = '      const char* sysconf_location = getenv ("MRTRIX_CONFIGFILE");\n'
    end = '      auto opt = App::get_options ("config");\n'
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Pinned configuration isolation patch anchors differ")
    start, stop = text.index(begin), text.index(end)
    if start >= stop:
        raise ValueError("Pinned configuration isolation patch region differs")
    guard = ('      // FNIT: isolate automatic config reads; retain upstream defaults and CLI.\n'
             f'      const char* fnit_isolated = getenv ("{CONFIG_ISOLATION_ENV}");\n'
             '      if (!fnit_isolated || std::string(fnit_isolated) != "1") {\n')
    text = text[:start] + guard + text[start:stop] + '      }\n\n' + text[stop:]
    if hashlib.sha256(text.encode()).hexdigest() != CONFIG_PATCH_SHA256:
        raise ValueError("Pinned configuration isolation patch result differs")
    path.write_text(text)
    return _configuration_patch_metadata()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_signature() -> str:
    """Prevent reuse of an ARCH=native build on a different CPU ISA."""
    fields = [platform.system(), platform.machine()]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith(("model name", "flags", "Features", "CPU architecture")):
                fields.append(line)
            if not line.strip() and len(fields) > 2:
                break
    return hashlib.sha256("\n".join(fields).encode()).hexdigest()


def _builder_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        cache_dir = os.environ.get("FNIT_NATIVE_CACHE")
    if cache_dir is None:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base / "fnit" / "connectome-native"
    return Path(cache_dir).expanduser().resolve()


def _runtime_name() -> str:
    content = f"{SOURCE_COMMIT}:{_builder_sha256()}:{_cpu_signature()}"
    return "mrtrix-tckgen-" + hashlib.sha256(content.encode()).hexdigest()[:20]


@contextmanager
def _build_lock(root: Path, name: str, timeout: float = 3600):
    import fcntl

    lock = root / (name + ".lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock, flags, 0o600)
    started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Timed out waiting for FNIT native tracking build lock")
                time.sleep(0.1)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _verify_asset(path: Path, asset: dict) -> None:
    if (not path.is_file() or path.is_symlink() or
            path.stat().st_size != asset["size"] or _sha256(path) != asset["sha256"]):
        raise ValueError(f"Pinned source archive failed size/SHA-256 verification: {path.name}")


def _download_archive(root: Path, asset: dict, allow_download: bool) -> Path:
    downloads = root / "downloads"
    downloads.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = downloads / asset["filename"]
    if target.exists():
        _verify_asset(target, asset)
        return target
    if not allow_download:
        raise FileNotFoundError(f"Verified source archive missing from native cache: {target.name}")
    # Consult the packaged fixed Release catalog before using upstream.  At
    # introduction none of these three source archives is in assets-v1.
    mirror = release_url_for(asset["sha256"], size=asset["size"])
    urls = ([mirror] if mirror else []) + [asset["url"]]
    last_error = None
    for url in dict.fromkeys(urls):
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "FNIT-native-tracking"})
            with tempfile.NamedTemporaryFile(dir=downloads, delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(request, timeout=120) as response:
                    shutil.copyfileobj(response, stream)
            _verify_asset(temporary, asset)
            os.replace(temporary, target)
            return target
        except (OSError, ValueError) as error:
            last_error = error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError("Could not download verified native source archive: " + target.name) from last_error


def _extract_archive(archive: Path, asset: dict, destination: Path) -> Path:
    _verify_asset(archive, asset)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        if asset is ARCHIVES["mrtrix"] and stream.pax_headers.get("comment") != SOURCE_COMMIT:
            raise ValueError("MRtrix GitHub archive PAX commit differs from the pinned commit")
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if (not parts or parts[0] != asset["folder"] or ".." in parts or
                    member.name.startswith("/") or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe or unexpected member in pinned source archive")
        stream.extractall(destination)
    return destination / asset["folder"]


def _compiler(variable: str, names: tuple[str, ...]) -> list[str]:
    supplied = os.environ.get(variable)
    if supplied:
        command = shlex.split(supplied)
        if not command:
            raise ValueError(f"Empty {variable} compiler command")
        executable = shutil.which(command[0])
    else:
        command = []
        prefix = str(Path(sys.prefix) / "bin")
        executable = next((value for name in names
                           if (value := shutil.which(name, path=prefix))), None)
        if executable is None:
            executable = next((value for name in names
                               if (value := shutil.which(name))), None)
    if executable is None:
        raise RuntimeError(f"Missing {variable} compiler; install the FNIT Conda environment")
    return [str(Path(executable).resolve()), *command[1:]]


def _build_environment(cxx: list[str], cc: list[str], jobs: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("CXX", "CC", "LINK", "CXX_ARGS", "LINK_ARGS", "CFLAGS", "CPPFLAGS",
                "CXXFLAGS", "LDFLAGS", "LINKFLAGS", "LINKLIB_FLAGS", "EIGEN_CFLAGS",
                "ZLIB_CFLAGS", "ZLIB_LINKFLAGS", "LIBRARY_PATH", "LD_LIBRARY_PATH",
                "CPATH", "CPLUS_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH",
                "TIFF_CFLAGS", "TIFF_LINKFLAGS", "PNG_CFLAGS", "PNG_LINKFLAGS",
                "PYTHONPATH", "MRTRIX_CONFIGFILE", "MRTRIX_RNG_SEED"):
        environment.pop(key, None)
    environment.update({
        "CXX": shlex.join(cxx), "CC": shlex.join(cc), "ARCH": "native",
        "NUMBER_OF_PROCESSORS": str(jobs), "CUDA_VISIBLE_DEVICES": "",
        "PATH": os.pathsep.join(dict.fromkeys((str(Path(sys.prefix) / "bin"),
                    str(Path(cxx[0]).parent), str(Path(cc[0]).parent), "/usr/bin", "/bin"))),
    })
    return environment


def _run(command: list[str], cwd: Path, environment: dict[str, str], log: Path) -> dict:
    started = time.perf_counter()
    with log.open("wb") as stream:
        process = subprocess.run(command, cwd=cwd, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"FNIT native tracking build failed; inspect {log}")
    return {"argv": command, "seconds": time.perf_counter() - started,
            "returncode": process.returncode, "log": log.name}


def _build_runtime(staging: Path, archives: dict[str, Path], jobs: int) -> None:
    cxx = _compiler("CXX", ("x86_64-conda-linux-gnu-c++", "g++", "clang++", "c++"))
    cc = _compiler("CC", ("x86_64-conda-linux-gnu-cc", "gcc", "clang", "cc"))
    environment = _build_environment(cxx, cc, jobs)
    make = shutil.which("make", path=environment["PATH"])
    if make is None:
        raise RuntimeError("Missing make; install the FNIT Conda environment")
    source = {name: _extract_archive(archives[name], asset, staging / "source")
              for name, asset in ARCHIVES.items()}
    mrtrix = source["mrtrix"]
    for name, expected in SOURCE_FILES.items():
        if _sha256(mrtrix / name) != expected:
            raise ValueError("Pinned MRtrix source file differs: " + name)
    configuration_patch = _patch_isolated_configuration(mrtrix)
    logs = staging / "logs"
    logs.mkdir()
    zlib_prefix = staging / "dependencies" / "zlib-1.3.1"
    zlib_env = environment.copy()
    zlib_env["CFLAGS"] = "-O3 -fPIC"
    records = [
        _run(["/bin/sh", "configure", "--static", "--prefix=" + str(zlib_prefix)],
             source["zlib"], zlib_env, logs / "zlib-configure.log"),
        _run([make, "-j", str(jobs)], source["zlib"], zlib_env, logs / "zlib-build.log"),
        _run([make, "install"], source["zlib"], zlib_env, logs / "zlib-install.log"),
    ]
    environment.update({
        "EIGEN_CFLAGS": "-isystem " + shlex.quote(str(source["eigen"])),
        "ZLIB_CFLAGS": "-isystem " + shlex.quote(str(zlib_prefix / "include")),
        "ZLIB_LINKFLAGS": shlex.quote(str(zlib_prefix / "lib" / "libz.a")),
    })
    records += [
        _run([sys.executable, "configure", "-nogui", "-conda"], mrtrix,
             environment, logs / "configure.log"),
        _run([sys.executable, "build", "bin/tckgen"], mrtrix,
             environment, logs / "build.log"),
    ]
    (staging / "bin").mkdir()
    (staging / "lib").mkdir()
    shutil.copy2(mrtrix / "bin" / "tckgen", staging / "bin" / "tckgen")
    for library in (mrtrix / "lib").glob("libmrtrix*"):
        if library.is_file():
            shutil.copy2(library, staging / "lib" / library.name)
    shutil.copy2(mrtrix / "LICENCE.txt", staging / "MRtrix-LICENCE.txt")
    binary = staging / "bin" / "tckgen"
    runtime_environment = environment.copy()
    runtime_environment[CONFIG_ISOLATION_ENV] = "1"
    runtime_environment["LD_LIBRARY_PATH"] = os.pathsep.join((
        str(staging / "lib"), str(Path(sys.prefix) / "lib")))
    version = subprocess.check_output([str(binary), "-version"], env=runtime_environment,
                                      text=True, stderr=subprocess.STDOUT).strip()
    tracked = [binary, staging / "MRtrix-LICENCE.txt", *(staging / "lib").glob("libmrtrix*")]
    manifest = {
        "schema_version": SCHEMA_VERSION, "purpose": "FNIT source-built MRtrix tckgen",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": ARCHIVES["mrtrix"]["sha256"],
        "source_archives": ARCHIVES, "source_file_sha256": SOURCE_FILES,
        "configuration_isolation": configuration_patch,
        "builder_sha256": _builder_sha256(), "cpu_signature": _cpu_signature(),
        "binary_sha256": _sha256(binary), "binary_size": binary.stat().st_size,
        "runtime_files": {str(path.relative_to(staging)): {"sha256": _sha256(path),
                          "size": path.stat().st_size} for path in tracked},
        "version": version, "license": "MPL-2.0", "arch": "native",
        "configure_args": ["-nogui", "-conda"], "build_target": "bin/tckgen",
        "compiler": {"cxx_argv": cxx, "cc_argv": cc,
            "version": subprocess.check_output([*cxx, "--version"], env=environment, text=True)},
        "jobs": jobs, "build_records": records,
        "dependency_linking": "pinned zlib static; MRtrix library in FNIT cache lib directory",
        "installed_mrtrix_used": False,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def native_runtime_manifest(binary: str | Path) -> dict:
    """Verify the managed executable, companion libraries and source identity.

    Returns JSON-compatible provenance.  A corrupted cache is an error; no
    system executable or automatic replacement is substituted.
    """
    binary = Path(binary).expanduser().absolute()
    if binary.is_symlink() or binary.name != "tckgen" or binary.parent.name != "bin":
        raise ValueError("Expected a regular FNIT-managed bin/tckgen executable")
    root = binary.parent.parent
    if root.is_symlink() or binary.parent.is_symlink() or (root / "lib").is_symlink():
        raise ValueError("FNIT native runtime directories must not be symlinks")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("FNIT native runtime manifest missing")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION, "purpose": "FNIT source-built MRtrix tckgen",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": ARCHIVES["mrtrix"]["sha256"],
        "builder_sha256": _builder_sha256(), "cpu_signature": _cpu_signature(),
        "installed_mrtrix_used": False, "source_file_sha256": SOURCE_FILES,
        "configuration_isolation": _configuration_patch_metadata(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("FNIT native runtime manifest source/builder/CPU identity differs")
    files = manifest.get("runtime_files", {})
    libraries = {"lib/" + path.name for path in (root / "lib").glob("libmrtrix*")}
    expected_files = {"bin/tckgen", "MRtrix-LICENCE.txt", *libraries}
    if not libraries or set(files) != expected_files:
        raise ValueError("FNIT native runtime manifest incomplete")
    for name, asset in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid FNIT native runtime file path")
        path = root / relative
        _verify_asset(path, asset)
    if (not os.access(binary, os.X_OK) or _sha256(binary) != manifest.get("binary_sha256") or
            binary.stat().st_size != manifest.get("binary_size")):
        raise ValueError("FNIT native tracking executable checksum or permission differs")
    return manifest


def ensure_tckgen(cache_dir: str | Path | None = None, *, jobs: int | None = None,
                  allow_download: bool = True) -> Path:
    """Return a verified FNIT-owned executable, building it once if necessary.

    cache_dir: external build/cache root; defaults to FNIT_NATIVE_CACHE, then
    XDG_CACHE_HOME/fnit/connectome-native. jobs: positive build concurrency
    (default min(8, CPU count)); this does not set tracking thread count.
    allow_download=False requires the three pinned archives in cache/downloads
    on first build; an already verified build works completely offline.
    """
    if platform.system() != "Linux":
        raise RuntimeError("FNIT native tracking currently supports Linux/WSL and the Conda environment")
    jobs = min(8, os.cpu_count() or 1) if jobs is None else jobs
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    root = _cache_root(cache_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = _runtime_name()
    runtime = root / name
    binary = runtime / "bin" / "tckgen"
    with _build_lock(root, name):
        if runtime.exists():
            native_runtime_manifest(binary)
            return binary
        archives = {key: _download_archive(root, asset, allow_download)
                    for key, asset in ARCHIVES.items()}
        staging = Path(tempfile.mkdtemp(prefix="." + name + ".build-", dir=root))
        try:
            _build_runtime(staging, archives, jobs)
            native_runtime_manifest(staging / "bin" / "tckgen")
            os.replace(staging, runtime)
            native_runtime_manifest(binary)
        except BaseException:
            # Keep private compiler diagnostics without publishing a valid runtime.
            failed = root / (name + ".failed-" + str(time.time_ns()))
            if staging.exists():
                os.replace(staging, failed)
            raise
    return binary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="FNIT external native cache directory")
    parser.add_argument("--jobs", type=int, default=None, help="Build jobs, default at most 8")
    parser.add_argument("--offline", action="store_true", help="Never download missing source archives")
    args = parser.parse_args(argv)
    binary = ensure_tckgen(args.cache_dir, jobs=args.jobs, allow_download=not args.offline)
    print(json.dumps({"binary": str(binary), "manifest": native_runtime_manifest(binary)}, indent=2))


if __name__ == "__main__":
    main()
