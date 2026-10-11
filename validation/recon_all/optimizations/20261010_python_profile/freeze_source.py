"""冻结独立 Python 整例验证的源码，不打包原版程序、权重、影像或许可证。

输入为本地仓库根目录、尚不存在的 tar 路径和验证标签；输出 tar、同名
manifest.json 与 sha256.json。记录实际 Git HEAD、未提交文件及每个冻结
文件的 SHA-256；这些 SHA 描述真正执行的工作树，不把 HEAD 当作全部代码。
只包含 src 下 Python/声明的文本资产和本次验证工具/测试；无坐标或影像输入。
路径不存在、输出已存在或冻结期间 HEAD 改变时抛异常；没有对应官方 CLI。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def freeze_source(*, repository: Path, archive: Path, label: str) -> dict:
    """返回冻结清单及 archive_sha256；各文件字节捕获后直接进入 tar。"""
    repository = repository.resolve()
    if not (repository / "src/fnit").is_dir():
        raise FileNotFoundError(repository / "src/fnit")
    if archive.exists():
        raise FileExistsError(archive)
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all").splitlines()
    paths = {p for p in (repository / "src").rglob("*")
             if p.is_file() and p.suffix in {".py", ".json", ".txt", ".mat"}
             and "__pycache__" not in p.parts}
    for relative in ("tests/recon_all", "tools/recon_python_execution_audit"):
        paths.update((repository / relative).rglob("*.py"))
    paths.update(repository / relative for relative in (
        "tools/benchmark_recon_torch_end_to_end.py",
        "tools/evaluate_recon_volume_prefix.py",
        "validation/recon_all/optimizations/20261010_python_profile/freeze_source.py"))
    captured = {str(p.relative_to(repository)): p.read_bytes() for p in sorted(paths)}
    manifest = {"label": label, "git_commit": commit,
                "git_branch": git("branch", "--show-current"),
                "working_tree_status": dirty,
                "scope": "Python profile source freeze; upstream stage programs and clinical/licensed assets excluded",
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in captured.items()}}
    if git("rev-parse", "HEAD") != commit:
        raise RuntimeError("Git HEAD changed during source capture")
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as stream:
        for name, data in {**captured, "SOURCE_MANIFEST.json": manifest_bytes}.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            stream.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = archive.with_suffix(".manifest.json")
    manifest_path.write_bytes(manifest_bytes)
    receipt = {"label": label, "git_commit": commit, "archive_bytes": archive.stat().st_size,
               "archive_sha256": digest, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "files": len(captured)}
    archive.with_suffix(".sha256.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--label", required=True)
    options = parser.parse_args()
    print(json.dumps(freeze_source(repository=options.repository, archive=options.archive,
                                   label=options.label), indent=2))
