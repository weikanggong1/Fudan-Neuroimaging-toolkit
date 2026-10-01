"""Public feature imports resolve to their implementation objects."""
import importlib
import inspect
import os
import re
from fnmatch import fnmatchcase
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,names", [
    ("synthstrip", ("SynthStrip", "StripResult")),
    ("synthmorph", ("SynthMorph", "RegistrationResult", "apply_transform",
                    "convert_warp_to_fsl")),
    ("wmh_synthseg", ("WMHSynthSeg", "WMHResult")),
    ("synthseg_parc", ("SynthSeg", "SynthSegResult")),
    ("synthsr", ("SynthSR", "SynthSRResult", "SynthSRImage")),
    ("fast", ("TorchFAST", "FASTResult", "FASTConfig", "FASTTensorResult", "segment_t1")),
    ("flirt", ("FLIRTResult", "TorchFLIRT",
               "flirt_to_world_affine", "flirt_to_world_pull",
               "voxel_to_fsl_scaled_mm", "world_to_flirt_affine")),
    ("mcflirt", ("TorchMCFLIRT", "MCFLIRTResult")),
    ("fnirt", ("TorchFNIRT", "TorchFNIRTResult", "FNIRTConfig",
               "GMFNIRTConfig", "T1FNIRTConfig", "TBSSFNIRTConfig",
               "resolve_fnirt_config")),
    ("applywarp", ("TorchApplyWarp", "ApplyWarpResult")),
    ("topup", ("TorchTOPUP", "TOPUPResult", "TOPUPConfig",
               "run_ukb_topup")),
    ("eddy", ("TorchEDDY", "EDDYResult", "EDDYConfig",
              "run_ukb_eddy")),
    ("dtifit", ("TorchDTIFIT", "DTIFITResult", "select_shell")),
    ("amico_noddi", ("TorchAMICONODDI", "AMICONODDIResult",
                     "AMICONODDIConfig")),
    ("mmorf", ("TorchMMORF", "MMORFResult", "MMORFConfig",
               "apply_mmorf_warp", "run_mmorf")),
    ("dmri_pipeline", ("DMRIPipeline", "DMRIPipelineResult",
                       "STANDARD_MAP_NAMES")),
    ("bedpostx", ("TorchBEDPOSTX", "BedpostXResult")),
    ("probtrackx", ("TorchProbtrackX", "ProbTrackXResult")),
    ("fast_vbm", ("FastVBM", "FastVBMResult", "VBMRegistrationResult")),
    ("connectome", ("UKBConnectome_pipeline", "UKBConnectome", "ConnectomeResult")),
])
def test_top_level_exports_are_feature_objects(module, names):
    package = importlib.import_module("fnit")
    feature = importlib.import_module(f"fnit.{module}")
    for name in names:
        assert getattr(package, name) is getattr(feature, name)


def test_mshbm_module_level_api_is_explicit():
    package = importlib.import_module("fnit")
    module = importlib.import_module("fnit.mshbm")
    for name in ("load_assets", "profiles_from_timeseries", "parcellate"):
        assert callable(getattr(module, name))
        assert not hasattr(package, name)


@pytest.mark.parametrize(
    ("documentation_slug", "source_slug"),
    (
        ("synthstrip", "synthstrip"),
        ("synthmorph", "synthmorph"),
        ("wmh_synthseg", "wmh_synthseg"),
        ("synthseg", "synthseg_parc"),
        ("synthsr", "synthsr"),
        ("fast", "fast"),
        ("flirt", "flirt"),
        ("mcflirt", "mcflirt"),
        ("fnirt", "fnirt"),
        ("applywarp", "applywarp"),
        ("fast_vbm", "fast_vbm"),
        ("topup", "topup"),
        ("eddy", "eddy"),
        ("dtifit", "dtifit"),
        ("amico_noddi", "amico_noddi"),
        ("mmorf", "mmorf"),
        ("dmri_pipeline", "dmri_pipeline"),
        ("bedpostx", "bedpostx"),
        ("probtrackx", "probtrackx"),
        ("mshbm", "mshbm"),
    ),
)
def test_in_scope_feature_documentation_exists(documentation_slug, source_slug):
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / documentation_slug / "README.md").is_file()
    assert (root / "src" / "fnit" / source_slug / "README.md").is_file()


def test_in_scope_console_script_targets_resolve():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text()
    block = project.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    excluded = {
        "fnit-setup-recon-all-assets",
        "fnit-recon-all",
        "fnit-normalize",
        "fnit-normalize-aseg",
        "fnit-fmri",
        "fnit-setup-fmri-surface-assets",
    }
    scripts = re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', block, re.MULTILINE)
    assert scripts
    for script_name, target in scripts:
        if script_name in excluded:
            continue
        module_name, attribute = target.split(":", 1)
        entry = getattr(importlib.import_module(module_name), attribute)
        assert callable(entry), f"{script_name}: {target} is not callable"


def test_removed_batch_api_is_absent():
    package = importlib.import_module("fnit")
    for name in (
        "BatchRunner", "BatchResult", "run_batch",
        "prepare_ukb_topup", "prepare_ukb_eddy",
        "TorchTBSS", "TBSSResult", "TBSSConfig",
    ):
        assert not hasattr(package, name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fnit.batch")
    source = Path(__file__).resolve().parents[1] / "src"
    removed = subprocess.run(
        [sys.executable, "-S", "-c", "import freesurfer_torch"],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert removed.returncode != 0

    apply_transform = package.apply_transform
    assert "device" not in inspect.signature(apply_transform).parameters


def test_top_level_import_stays_lightweight():
    code = "import sys; import fnit; assert not {'torch', 'surfa', 'h5py', 'nibabel', 'tensorflow'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_features_and_root_help_do_not_require_banned_packages():
    code = """
import importlib.abc
import sys

class BlockBannedPackages(importlib.abc.MetaPathFinder):
    roots = {
        "surfa", "dipy", "trx", "nipype", "fmriprep", "smriprep",
        "qsiprep", "qsirecon", "mriqc", "cpac",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0].lower() in self.roots:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockBannedPackages())
import fnit.synthstrip
import fnit.synthmorph
import fnit.wmh_synthseg
import fnit.synthseg_parc
import fnit.synthsr
import fnit.fast
import fnit.fast_vbm
import fnit.flirt
import fnit.fnirt
import fnit.applywarp
import fnit.topup
import fnit.eddy
import fnit.dtifit
import fnit.amico_noddi
import fnit.mmorf
import fnit.dmri_pipeline
import fnit.bedpostx
import fnit.probtrackx
import fnit.mshbm
from fnit import cli
try:
    cli.main(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("fnit --help did not exit")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_feature_sources_do_not_import_banned_packages():
    from setuptools.config.pyprojecttoml import read_configuration

    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "src" / "fnit"
    distribution_excludes = read_configuration(
        project_root / "pyproject.toml", expand=False
    )["tool"]["setuptools"]["packages"]["find"].get("exclude", [])
    assert not (root / "gems" / "native_samseg").exists()
    excluded = {"recon_all", "connectome", "fmri", "_vendor_fsl"}
    banned = {
        "surfa", "dipy", "trx", "nipype", "fmriprep", "smriprep",
        "qsiprep", "qsirecon", "mriqc", "cpac",
    }
    violations = []
    scanned = []
    for path in root.rglob("*.py"):
        package = ".".join(("fnit", *path.relative_to(root).parent.parts))
        if any(fnmatchcase(package, pattern) for pattern in distribution_excludes):
            continue
        if excluded.intersection(path.relative_to(root).parts):
            continue
        scanned.append(path)
        source = path.read_text()
        for line_number, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip().lower()
            for package in banned:
                if (stripped.startswith(f"import {package}")
                        or stripped.startswith(f"from {package}")):
                    violations.append(
                        f"{path.relative_to(root)}:{line_number}: {line.strip()}"
                    )
    assert root / "gems" / "pipeline.py" in scanned
    assert root / "gems" / "output.py" in scanned
    assert not violations, "\n".join(violations)


def test_surfa_is_confined_to_excluded_feature_extras():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text()
    default_dependencies = project.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "surfa" not in default_dependencies
    assert 'recon-all-python-stages = [' in project
    assert '"surfa==0.6.3"]' in project
    assert 'connectome-legacy = ["surfa==0.6.3"]' in project

    default_environment = (root / "environment.yml").read_text()
    recon_environment = (root / "environment-recon-all-cpp.yml").read_text()
    assert "surfa==" not in default_environment
    assert "- -e .\n" in default_environment
    assert "surfa==" not in recon_environment
    assert "-e .[recon-all-python-stages]" in recon_environment
