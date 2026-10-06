"""Download only selected, licensed connectome atlas templates."""

from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from fnit._release_assets import release_url_for


def _required(atlas: str) -> tuple[str, ...]:
    if atlas in ("fs-aparc", "fs-aparc-a2009s"):
        return ()
    scale = 4 if atlas.endswith("tian-s4") else 1
    if atlas in ("aparc+tian-s1", "aparc.a2009s+tian-s1"):
        parcels = None
    elif atlas.startswith("schaefer") and "+tian-" in atlas:
        parcels = int(atlas.split("+", 1)[0].removeprefix("schaefer"))
        if (parcels, scale) not in ((200, 1), (500, 4), (1000, 4)):
            raise ValueError(f"unsupported atlas: {atlas}")
    else:
        raise ValueError(f"no redistributable FNIT template for {atlas}")
    names = (f"Tian_Subcortex_S{scale}_3T.nii.gz",
             f"Tian_Subcortex_S{scale}_3T_label.txt")
    if parcels is not None:
        names += tuple(f"{hemi}.Schaefer2018_{parcels}Parcels_7Networks_order.annot"
                       for hemi in ("lh", "rh"))
    return names


def _valid(path: Path, *, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def install_connectome_atlases(atlases: tuple[str, ...] | list[str],
                               output_dir: str | Path) -> Path:
    """Prefer verified FNIT Release Tian/Schaefer files, with source fallback."""
    manifest = json.loads(files("fnit.connectome").joinpath(
        "atlas_manifest.json").read_text())
    root = Path(output_dir).expanduser().resolve()
    required = dict.fromkeys(name for atlas in atlases for name in _required(atlas))
    root.mkdir(parents=True, exist_ok=True)
    for name in required:
        entry = manifest["files"][name]
        target = root / name
        if _valid(target, **{key: entry[key] for key in ("size", "sha256")}):
            continue
        if target.exists():
            raise ValueError(f"atlas file has unexpected size or SHA-256: {target}")
        release_url = release_url_for(entry["sha256"], size=entry["size"])
        urls = ([release_url] if release_url else []) + [entry["url"]]
        last_error = None
        for url in dict.fromkeys(urls):
            temporary = None
            try:
                with urlopen(url, timeout=120) as source:
                    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                        temporary = Path(stream.name)
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            stream.write(chunk)
                if not _valid(temporary, **{key: entry[key] for key in ("size", "sha256")}):
                    raise ValueError(f"downloaded atlas failed verification: {name}")
                os.replace(temporary, target)
                break
            except (OSError, ValueError) as error:
                last_error = error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            raise ValueError(f"Could not download verified atlas: {name}") from last_error
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(install_connectome_atlases(args.atlas, args.output_dir))
