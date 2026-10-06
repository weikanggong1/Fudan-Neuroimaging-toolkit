"""优先从已核验 FNIT Release 部署投影资源，保留固定作者来源回退。"""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from fnit._release_assets import release_url_for

COMMIT = "b69b822a15e2a94f1e439606552fc44b6858cf3c"
BASE = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/{COMMIT}/"
# Only assets in FNIT's license-reviewed publication catalogue use its Release.
FILES = {
    "left_mni.surf.gii": (
        "data/templates/surface/fs_LR_32k/fsaverage.L.midthickness_mni.32k_fs_LR.surf.gii",
        723249, "ac51edc0f61ee988c6d941e073ae3275ef5da509233df9309bdf586b3b31838a"),
    "right_mni.surf.gii": (
        "data/templates/surface/fs_LR_32k/fsaverage.R.midthickness_mni.32k_fs_LR.surf.gii",
        710702, "6e1c9842efb303945abe0cd780422a9812d1eab59c08d3fa3c72276ec31e6a25"),
    "cortex_estimate.nii.gz": (
        "stable_projects/registration/Wu2017_RegistrationFusion/bin/"
        "liberal_cortex_masks_FS5.3/FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz",
        207362, "e4d788be332be76d7429855aba8f20c02693625f400905573e9063b4001f0e2b"),
}


def _install_file(output: Path, name: str, relative: str, size: int, digest: str) -> Path:
    destination = output / name
    if destination.exists():
        data = destination.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"resource size/SHA-256 mismatch: {name}")
        return destination
    release_url = release_url_for(digest, size=size)
    urls = ([release_url] if release_url else []) + [BASE + relative]
    last_error = None
    for url in dict.fromkeys(urls):
        temporary = None
        try:
            hasher = hashlib.sha256()
            with urlopen(url, timeout=120) as source:
                with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
                    temporary = Path(stream.name)
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        stream.write(chunk)
                        hasher.update(chunk)
            if temporary.stat().st_size != size or hasher.hexdigest() != digest:
                raise ValueError(f"resource size/SHA-256 mismatch: {name}")
            os.replace(temporary, destination)
            return destination
        except (OSError, ValueError) as error:
            last_error = error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError(f"Could not download verified resource: {name}") from last_error


def prepare_projection_assets(output_dir, reference):
    """下载两个 fsLR32k MNI 中层表面，并将皮层掩膜放到 reference 网格。

    reference 必须是 FSL MNI152 / MNI152NLin6Asym 的 3D 或 4D NIfTI。
    返回三个路径键 left_surface、right_surface、cortical_mask。安装阶段
    的 mask 重采样使用 CPU 最近邻；推断时不联网。
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, (relative, size, digest) in FILES.items():
        _install_file(output, name, relative, size, digest)
    original = nib.load(str(output / "cortex_estimate.nii.gz"))
    cortex = nib.Nifti1Image(np.asarray(original.dataobj)[..., 0], original.affine)
    target = nib.load(str(reference))
    remapped = resample_from_to(cortex, (target.shape[:3], target.affine), order=0)
    mask = nib.Nifti1Image((np.asarray(remapped.dataobj) > 0).astype(np.uint8), target.affine)
    mask.header.set_xyzt_units(xyz=target.header.get_xyzt_units()[0])
    mask_path = output / "cortical_mask.nii.gz"
    nib.save(mask, str(mask_path))
    return {"left_surface": output / "left_mni.surf.gii",
            "right_surface": output / "right_mni.surf.gii", "cortical_mask": mask_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deploy pinned CBIG MNI projection assets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", required=True, help="FSL MNI152 3D/4D target grid")
    args = parser.parse_args(argv)
    for key, path in prepare_projection_assets(args.output_dir, args.reference).items():
        print(f"{key}: {path.resolve()}")


if __name__ == "__main__":
    main()
