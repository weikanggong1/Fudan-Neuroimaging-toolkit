"""Verify real fresh-cache installer downloads using anonymous FNIT Release GETs.

Only the two pending-license Caret meshes are pre-seeded from previously
verified official downloads. No MRI pipeline, model inference or GPU is run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import urlopen as original_urlopen

import nibabel as nib
import numpy as np

import fnit.connectome.assets as connectome
import fnit.fmri.assets_setup as fmri
import fnit.mshbm.assets_setup as mshbm
import fnit.space_assets as space


RELEASE_BASE = (
    "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/"
    "releases/download/assets-v1/"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AuditedResponse:
    def __init__(self, response, record):
        self.response, self.record = response, record
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        chunk = self.response.read(size)
        self.size += len(chunk)
        self.digest.update(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.response.close()
        self.record.update(bytes_received=self.size, sha256=self.digest.hexdigest(),
                           status="complete" if kind is None else "read_failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="other_v1")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    stage = args.stage.resolve()
    destination = stage / "fresh-install" / args.attempt
    if destination.exists():
        raise FileExistsError("The attempt directory already exists; use a fresh attempt name")
    destination.mkdir(parents=True)
    report = {
        "schema": "fnit.assets_release.other_install.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "scope": "Real fresh-cache installers; no MRI pipeline, inference or GPU",
        "authentication": "Anonymous urllib GET; no credential headers",
        "release_policy": "Only fixed FNIT Release request URLs accepted; fallback is prohibited",
        "source_sha256": {}, "downloads": [], "modules": {},
        "pending_license_preseeded": [], "passed": False,
    }
    for relative in ("src/fnit/fmri/assets_setup.py", "src/fnit/space_assets.py",
                     "src/fnit/connectome/assets.py", "src/fnit/mshbm/assets_setup.py",
                     "src/fnit/_release_assets.py", "src/fnit/_release_asset_catalog.json"):
        report["source_sha256"][relative] = sha256(repo / relative)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    module_name = None

    def strict_opener(url, timeout):
        if not isinstance(url, str) or not url.startswith(RELEASE_BASE):
            raise RuntimeError("An installer attempted a non-FNIT Release download")
        record = {"module": module_name, "request_url": url, "status": "requested"}
        report["downloads"].append(record)
        try:
            response = original_urlopen(url, timeout=timeout)
            record.update(http_status=response.status,
                          final_host=urlsplit(response.geturl()).hostname)
            if response.status != 200:
                response.close()
                raise RuntimeError("Release response was not HTTP 200")
            return AuditedResponse(response, record)
        except Exception as error:
            record.update(status="request_failed", error_type=type(error).__name__)
            save()
            # RuntimeError deliberately prevents installers from trying upstream.
            raise RuntimeError("Anonymous Release GET failed") from error

    def record_files(root, expected):
        result = []
        for relative, (size, digest) in expected.items():
            path = root / relative
            actual_size, actual_digest = path.stat().st_size, sha256(path)
            if actual_digest != digest or (size is not None and actual_size != size):
                raise ValueError("Installed resource differs from its pinned content")
            result.append({"path": relative, "size": actual_size,
                           "sha256": actual_digest, "verified": True})
        return result

    catalog = json.loads((repo / "src/fnit/_release_asset_catalog.json").read_text())
    by_digest = {entry["sha256"]: entry for entry in catalog["assets"]
                 if entry.get("status") == "published"}
    fmri_install = fmri._install_one

    def traced_fmri_install(*parameters, **options):
        return fmri_install(*parameters, opener=strict_opener, **options)

    originals = (fmri._install_one, space.urlopen, connectome.urlopen, mshbm.urlopen)
    fmri._install_one = traced_fmri_install
    space.urlopen = connectome.urlopen = mshbm.urlopen = strict_opener
    started = time.perf_counter()
    save()
    try:
        module_name = "fmri"
        module_root = destination / module_name
        module_start = time.perf_counter()
        fmri.main(["--output-dir", str(module_root), "--msmall", "--fmriprep"])
        expected = {relative: (by_digest[digest]["size"], digest)
                    for relative, digest in fmri.ASSETS + fmri.MSMALL_ASSETS
                    + fmri.MSMALL_LOW_DIM_ASSETS + fmri.FMRIPREP_ASSETS}
        report["modules"][module_name] = {
            "initial_cache_files": 0, "source_paths": len(expected),
            "files": record_files(module_root, expected),
            "elapsed_seconds": time.perf_counter() - module_start,
        }
        save()

        module_name = "space"
        module_root = destination / module_name
        module_start = time.perf_counter()
        space.install_space_assets(module_root)
        expected = {"hcp_2017/" + relative: (by_digest[digest]["size"], digest)
                    for relative, digest in space.HCP_FILES.items()}
        expected.update({"rf_ants/" + Path(relative).name:
                         (by_digest[digest]["size"], digest)
                         for relative, digest in space.CBIG_FILES.items()})
        report["modules"][module_name] = {
            "initial_cache_files": 0, "source_paths": len(expected),
            "files": record_files(module_root, expected),
            "elapsed_seconds": time.perf_counter() - module_start,
        }
        save()

        module_name = "connectome"
        module_root = destination / module_name
        module_start = time.perf_counter()
        connectome.install_connectome_atlases(
            ["schaefer200+tian-s1", "schaefer500+tian-s4", "schaefer1000+tian-s4"], module_root)
        manifest = json.loads(connectome.files("fnit.connectome").joinpath(
            "atlas_manifest.json").read_text())
        expected = {name: (entry["size"], entry["sha256"])
                    for name, entry in manifest["files"].items()}
        report["modules"][module_name] = {
            "initial_cache_files": 0, "source_paths": len(expected),
            "files": record_files(module_root, expected),
            "elapsed_seconds": time.perf_counter() - module_start,
        }
        save()

        module_name = "mshbm"
        module_root = destination / module_name
        module_root.mkdir()
        for name in ("left_mni.surf.gii", "right_mni.surf.gii"):
            source = stage / "files" / ("mshbm--" + name)
            _, size, digest = mshbm.FILES[name]
            if source.stat().st_size != size or sha256(source) != digest:
                raise ValueError("Pending-license official mesh did not match its content guard")
            if digest in by_digest:
                raise ValueError("Pending-license mesh unexpectedly appears in FNIT published catalogue")
            shutil.copyfile(source, module_root / name)
            report["pending_license_preseeded"].append({
                "module": module_name, "path": name, "size": size, "sha256": digest,
                "retrieval": "Previously verified official CBIG staging download, preseeded",
                "mirrored_to_FNIT": False, "fresh_cache_item": False,
            })
        reference = destination / "fmri/fmriprep/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"
        module_start = time.perf_counter()
        output = mshbm.prepare_projection_assets(module_root, reference)
        expected = {name: (size, digest) for name, (_, size, digest) in mshbm.FILES.items()}
        reference_image, mask_image = nib.load(reference), nib.load(output["cortical_mask"])
        mask_values = np.asarray(mask_image.dataobj)
        grid_matches = (mask_image.shape == reference_image.shape[:3]
                        and np.array_equal(mask_image.affine, reference_image.affine))
        if (not grid_matches or mask_image.get_data_dtype() != np.dtype(np.uint8)
                or not np.isin(mask_values, (0, 1)).all() or not mask_values.any()):
            raise ValueError("Projection mask did not preserve the real reference grid or mask format")
        report["modules"][module_name] = {
            "initial_cache_files": 2, "source_paths": 3,
            "fresh_release_source_paths": 1, "preseeded_pending_source_paths": 2,
            "files": record_files(module_root, expected),
            "reference": {"resource": reference.name, "sha256": sha256(reference),
                          "shape": list(reference_image.shape),
                          "affine": reference_image.affine.tolist(),
                          "dtype": str(reference_image.get_data_dtype()),
                          "units": list(reference_image.header.get_xyzt_units())},
            "generated_mask": {"path": "cortical_mask.nii.gz",
                               "size": output["cortical_mask"].stat().st_size,
                               "sha256": sha256(output["cortical_mask"]),
                               "shape": list(mask_image.shape),
                               "affine": mask_image.affine.tolist(),
                               "dtype": str(mask_image.get_data_dtype()),
                               "units": list(mask_image.header.get_xyzt_units()),
                               "nonzero_voxels": int(np.count_nonzero(mask_values)),
                               "grid_matches_reference": grid_matches},
            "elapsed_seconds": time.perf_counter() - module_start,
        }
        expected_counts = {"fmri": 48, "space": 38, "connectome": 10, "mshbm": 1}
        for name, count in expected_counts.items():
            records = [item for item in report["downloads"] if item["module"] == name]
            if len(records) != count or any(item["status"] != "complete" for item in records):
                raise ValueError("Unexpected module download count or incomplete response")
            for item in records:
                filename = item["request_url"].removeprefix(RELEASE_BASE)
                matching = [entry for entry in catalog["assets"] if entry["name"] == filename]
                if len(matching) != 1 or (item["bytes_received"], item["sha256"]) != (
                        matching[0]["size"], matching[0]["sha256"]):
                    raise ValueError("Release response differed from the published catalogue")
        report["fresh_release_GET_count"] = len(report["downloads"])
        report["fresh_release_bytes"] = sum(item["bytes_received"] for item in report["downloads"])
        report["passed"] = True
    except Exception as error:
        report["failure"] = {"module": module_name, "error_type": type(error).__name__,
                             "message": "Installer verification failed; failure preserved without private paths"}
        raise
    finally:
        fmri._install_one, space.urlopen, connectome.urlopen, mshbm.urlopen = originals
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({"passed": report["passed"],
                      "fresh_release_GET_count": report["fresh_release_GET_count"],
                      "preseeded_pending_meshes": len(report["pending_license_preseeded"])}))


if __name__ == "__main__":
    main()
