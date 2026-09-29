"""Resolve local checkpoints and explicitly download verified official files."""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


# Keep each upstream filename, URL, size, and SHA-256 together.
WEIGHT_FILES = {
    "synthstrip.1.pt": (
        "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
        30851709, "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"),
    "synthstrip.nocsf.1.pt": (
        "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt",
        30851709, "62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28"),
    "synthmorph.affine.2.h5": (
        "https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.2.h5",
        51455312, "1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6"),
    "synthmorph.deform.3.h5": (
        "https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.deform.3.h5",
        3508630424, "95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536"),
    "synthmorph.rigid.1.h5": (
        "https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.rigid.1.h5",
        51656152, "284c145fce47e98ecf3fdeda2163f646ac3ebb0240e87dd50d71d879f4d5b3af"),
    "WMH-SynthSeg_v10_231110.pth": (
        "https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth",
        790531383, "0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988"),
    "synthsr_v20_230130.h5": (
        "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/f08/bc9/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5",
        106163752, "a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b"),
    "synthsr_lowfield_v20_230130.h5": (
        "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/de0/799/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5",
        106163752, "a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84"),
    "synthsr_v10_210712.h5": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthsr/synthsr_v10_210712.h5",
        53075984, "2fd59e96196388360eba95254fb6dfc9eb9eb8638018b590575e47e0a387f255"),
    "synthseg_2.0.h5": (
        "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/bee/241/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5",
        53079152, "f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e"),
    "synthseg_parc_2.0.h5": (
        "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/p0/0f/SHA256E-s53090840--83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684.0.h5/SHA256E-s53090840--83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684.0.h5",
        53090840, "83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684"),
    "synthseg_segmentation_labels_2.0.npy": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_labels_2.0.npy",
        348, "5ef25ec33fe917ac99f30b8f2185b2d77121136ee411b9c4970c0b59be615ed8"),
    "synthseg_segmentation_names_2.0.npy": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_names_2.0.npy",
        7168, "234eb6d514e10d6ebd748a8b30a1d12d9426fd874c607e37852406fae8f290fc"),
    "synthseg_topological_classes_2.0.npy": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_topological_classes_2.0.npy",
        348, "650b4b96834485c1e6d7421de4af74da80d861e6b2a39ef1164389bde3a5e14a"),
    "entowm.fsm31.t1.nstd00-30.nstd21-108.h5": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/entowm.fsm31.t1.nstd00-30.nstd21-108.h5",
        3296904, "9be55798498331f655acd75d4f0cd5036463e0f497bbb239be0167d6a9129a07"),
    "entowm.ctab": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/entowm.ctab",
        318, "fa46a74e7c5385b6e474553acbb34f536dac640c52586ec4193a0ea9739948f1"),
    "mca-dura.both-lh.nstd21.fhs.h5": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/mca-dura.both-lh.nstd21.fhs.h5",
        3294856, "da6a7b994e3e804cc3dc0e98e965c28a802ddcd38fd9b5c680d75cef285657b0"),
    "mca-dura.ctab": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/mca-dura.ctab",
        116, "77faedc95badab7b01ab8ef71889724c5eda33a0862fefa845b5ba33b8bc3e13"),
    "vsinus.no-sp.m.all.nstd10-070.h5": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/vsinus.no-sp.m.all.nstd10-070.h5",
        3296904, "3d78948741306a31337468c86be55821913edb73855116fcb063b61135b90f12"),
    "sclimbic.volstats.csv": (
        "https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/sclimbic.volstats.csv",
        500, "691b8e1a1d74668b65a0571e2854a4a83c484438107693d71eef5a081d17380b"),
}

MODEL_FILES = {
    "synthstrip": ("synthstrip.1.pt",),
    "fast-vbm": ("synthstrip.1.pt", "synthmorph.deform.3.h5"),
    "fmri": ("synthstrip.1.pt", "synthmorph.deform.3.h5"),
    "synthstrip-nocsf": ("synthstrip.nocsf.1.pt",),
    "synthmorph-rigid": ("synthmorph.rigid.1.h5",),
    "synthmorph-affine": ("synthmorph.affine.2.h5",),
    "synthmorph-deform": ("synthmorph.deform.3.h5",),
    "synthmorph-joint": ("synthmorph.affine.2.h5", "synthmorph.deform.3.h5"),
    "wmh-synthseg": ("WMH-SynthSeg_v10_231110.pth",),
    "synthseg": (
        "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
        "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy"),
    "synthseg-plus": (
        "synthseg_2.0.h5", "synthseg_parc_2.0.h5",
        "synthseg_segmentation_labels_2.0.npy",
        "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy"),
    "recon-all": (
        "synthstrip.1.pt", "synthmorph.affine.2.h5", "synthmorph.deform.3.h5",
        "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
        "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy",
        "entowm.fsm31.t1.nstd00-30.nstd21-108.h5", "entowm.ctab",
        "mca-dura.both-lh.nstd21.fhs.h5", "vsinus.no-sp.m.all.nstd10-070.h5"),
    "synthsr": ("synthsr_v20_230130.h5",),
    "synthsr-lowfield": ("synthsr_lowfield_v20_230130.h5",),
    "synthsr-v1": ("synthsr_v10_210712.h5",),
}

# This release contains byte-identical upstream files. Keep the upstream URLs
# above as the fallback and the record of each original source.
RELEASE_BASE = ("https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/"
                "releases/download/assets-v1")
RELEASE_CHECKSUMS = {name: entry[2] for name, entry in WEIGHT_FILES.items()}
DEFORM_PARTS = (
    ("synthmorph.deform.3.h5.part00", 1900000000,
     "1960594f0929801321de64eaa9b2f116a7d4205cd74c778e55225101e6f1d290"),
    ("synthmorph.deform.3.h5.part01", 1608630424,
     "dc4c53b5014ad60c5be82d3d8d275d46f2f1d32dd0ab01bcfdec20b65c9f1b48"),
)


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "fnit"


def configured_dir():
    config = cache_dir() / "weights.json"
    if not config.is_file():
        return None
    value = json.loads(config.read_text(encoding="utf-8"))["directory"]
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Invalid weights directory in {config}")
    return Path(value)


def save_config(directory):
    config = cache_dir() / "weights.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    part = config.with_name(config.name + ".tmp")
    part.write_text(json.dumps({"directory": str(directory)}, indent=2) + "\n", encoding="utf-8")
    part.replace(config)


def resolve_weights(filename, explicit=None):
    """Find weights without network access; explicit path and env take priority."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        path = path / filename if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    environment = os.environ.get("FNIT_WEIGHTS")
    if environment and (Path(environment) / filename).is_file():
        return Path(environment) / filename
    roots = [configured_dir(), cache_dir()]
    for root in roots:
        if root and (Path(root) / filename).is_file():
            return Path(root) / filename
    raise FileNotFoundError(
        f"{filename}: run tools/setup_weights.py, provide weights=, or set "
        "FNIT_WEIGHTS to the weights directory")


def verify_file(path, size, sha256):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == sha256


def _download_verified(url, size, sha256, final):
    """Download one URL with HTTP Range and publish only a verified file."""
    if verify_file(final, size, sha256):
        return final
    part = final.with_name(final.name + ".part")
    if part.exists() and part.stat().st_size > size:
        part.unlink()

    for attempt in range(2):
        for transient in range(3):
            offset = part.stat().st_size if part.exists() else 0
            try:
                if offset < size:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with urlopen(Request(url, headers=headers), timeout=60) as response:
                        if offset and response.status == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                                 response.headers.get("Content-Range", ""))
                            if not match or int(match[1]) != offset or int(match[3]) != size:
                                raise ValueError(f"Unexpected HTTP Content-Range for {final.name}")
                            mode = "ab"
                        elif response.status == 200:
                            mode = "wb"  # A server that ignores Range sends the whole file.
                        else:
                            raise ValueError(f"Unexpected HTTP status {response.status} for {final.name}")
                        with part.open(mode) as stream:
                            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                                stream.write(block)
                break
            except (TimeoutError, ConnectionError):
                if transient == 2:
                    raise
        if verify_file(part, size, sha256):
            part.replace(final)
            return final
        part.unlink(missing_ok=True)
        if attempt:
            raise ValueError(f"Downloaded checkpoint failed size or SHA-256 verification: {final.name}")
    raise AssertionError("unreachable")


def download_file(filename, directory, verify_only=False):
    """Prefer the pinned release; fall back to the official file and verify SHA-256."""
    upstream, size, sha256 = WEIGHT_FILES[filename]
    final = directory / filename
    if verify_file(final, size, sha256):
        return final
    if verify_only:
        raise ValueError(f"Missing or invalid checkpoint: {final}")
    directory.mkdir(parents=True, exist_ok=True)
    if RELEASE_CHECKSUMS.get(filename) == sha256:
        try:
            if filename == "synthmorph.deform.3.h5":
                chunks = [_download_verified(f"{RELEASE_BASE}/{name}", part_size,
                                             part_hash, directory / name)
                          for name, part_size, part_hash in DEFORM_PARTS]
                temporary = final.with_name(final.name + ".part")
                with temporary.open("wb") as output:
                    for chunk in chunks:
                        with chunk.open("rb") as source:
                            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if not verify_file(temporary, size, sha256):
                    raise ValueError(f"Release parts failed SHA-256 verification: {filename}")
                temporary.replace(final)
                return final
            return _download_verified(f"{RELEASE_BASE}/{filename}", size, sha256, final)
        except (OSError, ValueError):
            pass
    return _download_verified(upstream, size, sha256, final)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download verified FreeSurfer weights and configure their location")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=MODEL_FILES, action="append", metavar="MODEL",
                       help="Model to set up; repeat to select multiple (default: all official weights)")
    group.add_argument("--all", action="store_true", help="Set up all official weights (default)")
    parser.add_argument("--dest", type=Path, help="Weight directory; saved for future API/CLI calls")
    parser.add_argument("--verify-only", action="store_true", help="Check files without downloading or changing config")
    args = parser.parse_args(argv)
    destination = (args.dest or os.environ.get("FNIT_WEIGHTS")
                   or configured_dir() or cache_dir())
    destination = Path(destination).expanduser().resolve()
    models = args.model or MODEL_FILES.keys()
    names = dict.fromkeys(name for model in models for name in MODEL_FILES[model])
    for name in names:
        path = download_file(name, destination, verify_only=args.verify_only)
        print(f"Verified {path}")
    if not args.verify_only:
        save_config(destination)
        print(f"Configured weight directory: {destination}")


if __name__ == "__main__":
    main()
