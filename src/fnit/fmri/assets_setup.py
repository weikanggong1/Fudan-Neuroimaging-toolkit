"""Download the public HCP fsLR templates used by the FNIT surface workflow.

Usage: python -m fnit.fmri.assets_setup --output-dir /absolute/path/hcp_templates

The output keeps HCPpipelines paths under ``global/templates/`` and ``MSMConfig/``.
Subject-specific surfaces and registrations are not included.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


HCP_COMMIT = "f8cac6892f88bdf889d644711ff038198eb81533"  # v4.7.0
BASE_URL = f"https://raw.githubusercontent.com/Washington-University/HCPpipelines/{HCP_COMMIT}/"
FALLBACK_URL = f"https://cdn.jsdelivr.net/gh/Washington-University/HCPpipelines@{HCP_COMMIT}/"
MESH = "global/templates/standard_mesh_atlases/"

# (HCPpipelines-relative path, SHA-256 of the released file)
ASSETS = (
    ("LICENSE.md", "2c686e38603c7e1525e2a98a6de9d15ae6766e4ff850a2e759fbb2de7c6c6445"),
    ("global/config/FreeSurferAllLut.txt", "74b5615738f1790c274ea51069090f7596035b06a68f7c6ecbd657cc69089ab6"),
    ("global/config/FreeSurferSubcorticalLabelTableLut.txt", "9a3032695f9257246048353a202eb7b0c7f60c04972e2d5e5b42a0247cce757e"),
    (MESH + "L.sphere.32k_fs_LR.surf.gii", "1846b053f870405466776d004d714cc1da0cec7361761c65a864782dd09f30a8"),
    (MESH + "R.sphere.32k_fs_LR.surf.gii", "1a898433a9f1070e4e0435d4db966776ecc3291bfcd444efc7910aeffb91561c"),
    (MESH + "L.atlasroi.32k_fs_LR.shape.gii", "4ac9199dab151ccdc2a35bdddb5bac4f4907da7dfebd90807c09b82e2eb9d512"),
    (MESH + "R.atlasroi.32k_fs_LR.shape.gii", "698f46b399f5a89829f83cc697e32dc8841031fbf31ffef75aaaa0c4a16c3015"),
    (MESH + "L.atlasroi.164k_fs_LR.shape.gii", "5c8e5346626624ed2233850e15f7562ff15b9e110aa845a7096fb1ae1c0202c4"),
    (MESH + "R.atlasroi.164k_fs_LR.shape.gii", "1b48f6f83a5e49ad624cda15fe808e6770585ff9d762c1478a517020784eec9e"),
    (MESH + "L.refsulc.164k_fs_LR.shape.gii", "10f5e3fd775bb2f77278b0b69126f6f67abefb3c76de6f1233128d9914ddcb5c"),
    (MESH + "R.refsulc.164k_fs_LR.shape.gii", "4eefc7807cdd9076cfc9ffc933da3d0c100a4770d5359c3e5f4fb37d71097520"),
    (MESH + "fs_L/fsaverage.L.sphere.164k_fs_L.surf.gii", "9bf5b673cc1c4738b1137ac53ff724ff0ef45e12f5550448efa2da50bb521c32"),
    (MESH + "fs_R/fsaverage.R.sphere.164k_fs_R.surf.gii", "a2fada23394286212f38cd44303bce0b3252759e09fe693f92d463286ca28b45"),
    (MESH + "fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii", "a92b7c63eff6d04d8ce74b38e9f0060390b88d24b7c109bf13b194f0e8a54221"),
    (MESH + "fs_R/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii", "d4b5808a75df1d9499f9f8339931fd385766ba625953823d662cb11f4b5e68cf"),
    (MESH + "fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii", "6968ef7d9638de54ade78cdbb7f0c6f25025c60477f72456ab2fb77c53a85cfa"),
    (MESH + "fsaverage.R_LR.spherical_std.164k_fs_LR.surf.gii", "85e718e424a1d87521fb0f3942538db16d8c745fe5b9ee20405baac2aff1dbb1"),
    (MESH + "Avgwmparc.nii.gz", "c8d80a4a0327daf3855168ce36ee3a5f4b9c660fd38d4ebf9865d2b915887a1b"),
    ("global/templates/91282_Greyordinates/Atlas_ROIs.2.nii.gz", "764c5c0139c37f4e0ec288525e8a83f0d5d6821bc82fefcc979c1ac0c35b1cd4"),
)

MSMALL_ASSETS = (
    ("MSMConfig/MSMSulcStrainFinalconf", "46b250404cb2570b4f645d8e53c30fabde799663d61761d61cf54ff110318203"),
    ("MSMConfig/MSMAllStrainFinalconf1to1_1to3_1", "ec9348a2bd2aea5ce2bf1a7387997dabb3443784c15a017faa4871e4ea38cc2e"),
    ("MSMConfig/MSMAllStrainFinalconf1to1_1to3_2", "646f100e7826d0c285f379f2a9121f0c0c8cd61a3d2d200cc241c416617b8555"),
    (MESH + "Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii", "66f5726a4d4e02189e2d28643cf4017f1bcd4b7c0ef07a6282e05180b73de070"),
    ("global/templates/MSMAll/DeDriftingGroup.L.sphere.DeDriftMSMAll.164k_fs_LR.surf.gii", "e07fe3f9d7e508f60a85a0230c1680610f23521c89d204f79000093abd51149e"),
    ("global/templates/MSMAll/DeDriftingGroup.R.sphere.DeDriftMSMAll.164k_fs_LR.surf.gii", "27c4ab1d7773fbaf90e917f6ae992fc37c6400f81603b239f8faa357ae276fb3"),
    ("global/templates/MSMAll/Q1-Q6_RelatedParcellation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii", "7dda1cf9ca08a098fc9e7e14918be65936d10fc95dbd0e3f0ee77e3100aa899b"),
    ("global/templates/MSMAll/Q1-Q6_RelatedParcellation210.atlas_Topographic_ROIs.32k_fs_LR.dscalar.nii", "5b7dc5bbb339aa97a998eb638d915968eddb42179aad2273e7a121c0753ad14f"),
    ("global/templates/MSMAll/Q1-Q6_RelatedParcellation210.atlas_Topography.32k_fs_LR.dscalar.nii", "35f86f3da38e085e4ec52e158b244ebdb00896d88820273b0eb971fdb0211300"),
    ("global/templates/MSMAll/rfMRI_REST_Atlas_MSMAll_2_d41_WRN_DeDrift_hp2000_clean_PCA.ica_d40_ROW_vn/Weights.txt", "3b01bec7f345d61ab62798cfba2630dfb07363379260f61f8b5b27bf5cb69ce5"),
    ("global/templates/MSMAll/rfMRI_REST_Atlas_MSMAll_2_d41_WRN_DeDrift_hp2000_clean_PCA.ica_d40_ROW_vn/melodic_oIC.dscalar.nii", "399f299bdde45720a37e650bf1306a771ffe179a8cdb914fdd297d28a16e9805"),
)

FMRIPREP_ASSETS = (
    ("fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz",
     "9c25e63edec37b3876756b749a3f0127511c6b63bf2855060a44007bb479b987"),
)
FMRIPREP_BASE = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_one(output_dir: Path, relative_path: str, expected_sha256: str,
                 opener=urlopen, base_urls=(BASE_URL, FALLBACK_URL)) -> Path:
    destination = output_dir / relative_path
    if destination.exists():
        if _sha256(destination) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch in existing file: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for base_url in base_urls:
        temporary = None
        try:
            remote_name = relative_path.split("/")[-1] if base_urls == (FMRIPREP_BASE,) else relative_path
            with opener(base_url + remote_name, timeout=60) as source:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
                    temporary = Path(target.name)
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
            if _sha256(temporary) != expected_sha256:
                raise ValueError(f"SHA-256 mismatch in download: {relative_path}")
            os.replace(temporary, destination)
            return destination
        except (OSError, ValueError) as error:
            last_error = error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError(f"Could not download verified asset: {relative_path}") from last_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Absolute destination directory")
    parser.add_argument("--msmall", action="store_true", help="Install public MSMAll d40 templates and MSM configuration")
    parser.add_argument("--fmriprep", action="store_true", help="Install the TemplateFlow HCP dseg for 91k CIFTI")
    args = parser.parse_args(argv)
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be an absolute path")
    for relative_path, digest in ASSETS + (MSMALL_ASSETS if args.msmall else ()):
        path = _install_one(args.output_dir, relative_path, digest)
        print(path)
    if args.fmriprep:
        for relative_path, digest in FMRIPREP_ASSETS:
            print(_install_one(args.output_dir, relative_path, digest,
                               base_urls=(FMRIPREP_BASE,)))


if __name__ == "__main__":
    main()
