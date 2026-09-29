"""Install the pinned HCP 2017 and CBIG RF-ANTs conversion assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


HCP_BASE = ("https://raw.githubusercontent.com/Washington-University/HCPpipelines/"
            "f8cac6892f88bdf889d644711ff038198eb81533/"
            "global/templates/standard_mesh_atlases/")
CBIG_BASE = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
             "v0.18.1-Update_stable_project_unit_test/"
             "stable_projects/registration/Wu2017_RegistrationFusion/bin/")

# SHA-256 values of the published HCP 2017 atlas files and CBIG FS5.3 RF-ANTs maps.
HCP_FILES = dict(line.split() for line in """
resample_fsaverage/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii 5c3281b8d0afef386ee3e79aa51fa075d345986a0846846a11a4f3ded8d434d9
resample_fsaverage/fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii 7c8f29373e1ad8a30f4159b743cf724cc01e4e601b7af0dfd7ff22bb560a23be
resample_fsaverage/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii 38b13855694098fde897ef3984f0420bd6bce4eac1562ce0c285bcaf565d56b8
resample_fsaverage/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii d3c03ce6efe36e0bc85d85a6f24bdfcc51da7427b19e28b79783ae8062e88e02
resample_fsaverage/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii 7fd8185a51bdae23f565b07d28150ce0100aec60c6e72fc70e740462633a3f2a
resample_fsaverage/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii 0b757ddcb61fa9570220a655977a3c3eaab53eee323331c832bec979ad35add4
resample_fsaverage/fsaverage_std_sphere.L.164k_fsavg_L.surf.gii 9bf5b673cc1c4738b1137ac53ff724ff0ef45e12f5550448efa2da50bb521c32
resample_fsaverage/fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii fbbccfe4bbcec0619fa6549cfe3f3460c761b0becc7097092379d98a684a52e0
resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii 31987d139b6cdf1040188d1795bf23c213c57f8e84fb45f2c04823d996b0defc
resample_fsaverage/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii e06c0f3e6069f9244e4dd74a4265323be3324a96c0f2bf9edd5ad9734d170f6d
L.sphere.32k_fs_LR.surf.gii 1846b053f870405466776d004d714cc1da0cec7361761c65a864782dd09f30a8
resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.59k_fs_LR.surf.gii 6d25494ddb49de4b782fa3fb91b05075fd64325cb329020fc166adc330c23122
resample_fsaverage/fs_LR.L.midthickness_va_avg.59k_fs_LR.shape.gii 72f2c95a137d57010da03414d70208b327ac8a573a315d6ceab38ce2e950312e
L.sphere.59k_fs_LR.surf.gii 65d38a5f938c2667f94f8978af4750ed66f76ce9060884bafccc3e9cdc52d178
resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.164k_fs_LR.surf.gii 1f3b09b813fe32c934a565264dccc357629834259fa34f39de5c8c00dc0fec92
resample_fsaverage/fs_LR.L.midthickness_va_avg.164k_fs_LR.shape.gii 5334b611efd10601ccaa5a3469610723387f0caa9cc040f9ec4928870d2968f0
fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii 6968ef7d9638de54ade78cdbb7f0c6f25025c60477f72456ab2fb77c53a85cfa
resample_fsaverage/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii b12139d0bb2e606596ab4ed27b52e8cc3e090fb61577915bccd5b430bbb8eb8d
resample_fsaverage/fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii fe6da9656d50de88b3177db6864b9b88478859471c4966fa7a58ae45b1ab9e52
resample_fsaverage/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii a2b9f152da32bf387e33e58f0a6a8931ebca69be66e198178298d66d68d98808
resample_fsaverage/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii c039f3035e0730feb49774f8956c80ecd85288ad0a9fbc2706c538dc73ddde8a
resample_fsaverage/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii f0a093489fc2248905148c904b883302c3b806105845328fd49b96fe4e7ce658
resample_fsaverage/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii 56495d5114d9b31c0a58bddfd5a961ed51c974c96ca745afeb119406aca18384
resample_fsaverage/fsaverage_std_sphere.R.164k_fsavg_R.surf.gii a2fada23394286212f38cd44303bce0b3252759e09fe693f92d463286ca28b45
resample_fsaverage/fsaverage.R.midthickness_va_avg.164k_fsavg_R.shape.gii 2cda969b6295e96f510aad9d1df329afc36859aba7ee95564b1217b43f30ee03
resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii 8c1f67fa8fb3024ad3b292be995d5c9bae07dc4dbdd53fbac48e511e1366553d
resample_fsaverage/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii cb4b7680b8637c81398a99d4a034ee1680510ad6528d6948b4aa9c67b3e81867
R.sphere.32k_fs_LR.surf.gii 1a898433a9f1070e4e0435d4db966776ecc3291bfcd444efc7910aeffb91561c
resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.59k_fs_LR.surf.gii 41a471a8fc5bff1330289646c61d95b7fe3e6c57297092e6af513c8c822ad6b6
resample_fsaverage/fs_LR.R.midthickness_va_avg.59k_fs_LR.shape.gii 1b7395b8fd83f5b6e14f4209ed1aaf72a7351b360845ac73cee4e31a0dfd3ebc
R.sphere.59k_fs_LR.surf.gii 6df50dfa10d872f0aca5b4e9c96b6dbf0546131fec469f918445433cab805d60
resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.164k_fs_LR.surf.gii f024337147073263dc11e56603afcb9dae2c48cf8869809388d5985eb95a585a
resample_fsaverage/fs_LR.R.midthickness_va_avg.164k_fs_LR.shape.gii 97006956e0a29913c3507fc89bb4cfcb4558b8f94be8c8f54b889d103290024b
fsaverage.R_LR.spherical_std.164k_fs_LR.surf.gii 85e718e424a1d87521fb0f3942538db16d8c745fe5b9ee20405baac2aff1dbb1
""".strip().splitlines())

CBIG_FILES = dict(line.split() for line in """
final_warps_FS5.3/lh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat 3961b1e1f04621f8c1961ac8e4e5385813e47579214e0ccd5e62d685265205fd
final_warps_FS5.3/rh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat c44a8a824ade7c7c2203f2cd91cc6dced7c064d4371a45270cc50bb02d182319
final_warps_FS5.3/allSub_fsaverage_to_FSL_MNI152_FS4.5.0_RF_ANTs_avgMapping.vertex.mat 91da461449b2d85e82b8e7f945bdb3d57121a48a2e3a8610e4c742e3bb26a89e
liberal_cortex_masks_FS5.3/FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz e4d788be332be76d7429855aba8f20c02693625f400905573e9063b4001f0e2b
""".strip().splitlines())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install(base: str, remote: str, destination: Path, digest: str) -> Path:
    if destination.exists():
        if _sha256(destination) != digest:
            raise ValueError(f"Existing asset has a different SHA-256: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(base + remote, timeout=120) as source:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(chunk)
        if _sha256(temporary) != digest:
            raise ValueError(f"Downloaded asset has a different SHA-256: {remote}")
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_space_assets(output_dir: str | Path) -> Path:
    """Download and verify the official RF-ANTs and HCP conversion files."""
    root = Path(output_dir).expanduser().resolve()
    for relative, digest in HCP_FILES.items():
        print(_install(HCP_BASE, relative, root / "hcp_2017" / relative, digest))
    for relative, digest in CBIG_FILES.items():
        print(_install(CBIG_BASE, relative, root / "rf_ants" / Path(relative).name, digest))
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    install_space_assets(args.output_dir)


if __name__ == "__main__":
    main()
