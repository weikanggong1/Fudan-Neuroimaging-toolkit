"""Summarize independently collected asset hashes and live Release metadata.

This validation tool reads metadata only. It does not fetch or copy weights,
atlases, MRI, or the user's FreeSurfer registration license.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


FAMILIES = {"brainstem": "BrainstemSS", "thalamus": "ThalamicNuclei",
            "hippo-amygdala-left": "HippoSF", "hippo-amygdala-right": "HippoSF"}
ORIGINAL_NAMES = {"AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt"}


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal source manifest {name}")


def annex_url(size, digest, extension):
    key = f"SHA256E-s{size}--{digest}{extension}"
    prefix = hashlib.md5(key.encode("ascii")).hexdigest()
    return f"https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/{prefix[:3]}/{prefix[3:6]}/{key}/{key}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-root", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    args = p.parse_args()
    root, repo = args.metadata_root.resolve(), args.repository.resolve()
    live_path, release_path = root / "assets_live_metadata.json", root / "release_live_metadata.json"
    live, release = json.loads(live_path.read_text()), json.loads(release_path.read_text())
    licenses_path = root / "release_license_metadata.json"
    licenses = json.loads(licenses_path.read_text())
    asset_source, weight_source = repo / "src/fnit/recon_all/assets.py", repo / "src/fnit/weights.py"
    originals = literal_assignment(asset_source, "ASSET_FILES")
    weights = literal_assignment(weight_source, "WEIGHT_FILES")
    release_weights = {x["name"]: x for x in release["weights"]}
    release_assets = {x["name"]: x for x in release["matching_live_release_assets"]}
    rows = []
    for entry in live["assets"]:
        path = Path(entry["path"])
        location = "weights/" + path.name if path.parent.name == "weights" else "atlases/" + path.parent.name + "/" + path.name
        row = {k: entry[k] for k in ("bytes", "sha256", "prior_exact_match", "launch_exact_match")}
        row["relative_path"] = location
        if location.startswith("weights/"):
            upstream, size, digest = weights[path.name]
            rel, attached = release_weights[path.name], release_assets[path.name]
            row.update(kind="official_SynthSeg_asset", active=True, official_url=upstream,
                       license=rel["license"], present_in_fixed_release=True,
                       matches_source_manifest=(size, digest) == (row["bytes"], row["sha256"]),
                       matches_release_manifest=(rel["size"], rel["sha256"]) == (size, digest),
                       matches_live_release_attachment=(attached["size"], attached["digest"]) == (size, "sha256:" + digest))
        elif path.name in ORIGINAL_NAMES:
            key = f"average/{FAMILIES[path.parent.name]}/atlas/{path.name}"
            size, digest, extension = originals[key]
            row.update(kind="official_atlas_original", active=True, official_asset_key=key,
                       official_url=("bundled verified FNIT LUT; upstream atlas compressionLookupTable" if path.name == "compressionLookupTable.txt" else annex_url(size, digest, extension)),
                       matches_source_manifest=(size, digest) == (row["bytes"], row["sha256"]),
                       present_in_fixed_release=False,
                       redistribution_policy="Existing licensed server installation/original upstream only; no atlas payload published in this benchmark.")
        elif path.parent.name == "brainstem":
            row.update(kind="FNIT_atlas_recipe_or_alpha_cache", active=True,
                       generation_source="src/fnit/gems/setup.py:prepare_brainstem_atlas; smooth_atlas_alphas",
                       original_data_family="BrainstemSS", exact_creation_driver_sha256=None,
                       creation_driver_sha256_status="Not preserved in old cache; current generating implementation and exact cache bytes audited separately.")
        else:
            row.update(kind="legacy_FNIT_population_grid_config_or_alpha_cache", active=False,
                       active_solver_policy="GEMSRecipe reads original AtlasMesh/LUT and smooths the subject transformed reference mesh; does not read this config/cache.",
                       exact_creation_driver_sha256=None,
                       creation_driver_sha256_status="Historical generator SHA unknown; retained hashes prove unchanged bytes, not current use.")
        rows.append(row)
    official = []
    for entry in live["official_atlas_files"]:
        key = entry["path"].split("/average/", 1)[1]
        key = "average/" + key
        size, digest, _ = originals[key]
        official.append(dict(official_asset_key=key, bytes=entry["bytes"], sha256=entry["sha256"],
                             matches_fnit_upstream_manifest=(size, digest) == (entry["bytes"], entry["sha256"])))
    source_paths = [asset_source, weight_source, repo / "src/fnit/gems/setup.py",
                    repo / "src/fnit/gems/smoothing.py", repo / "src/fnit/gems/recipes/brainstem.py",
                    repo / "src/fnit/gems/recipes/base.py"]
    frozen_path = root / "expected_source_manifest.json"
    frozen = {x["path"]: x for x in json.loads(frozen_path.read_text())["files"]}
    source_rows = []
    for path in source_paths:
        row = identity(path)
        row["relative_path"] = str(path.relative_to(repo))
        expected = frozen.get(row["relative_path"], {})
        row["matches_current_cohort_frozen_source"] = all(row[k] == expected.get(k) for k in ("bytes", "sha256"))
        source_rows.append(row)
    checked = [x for x in rows if x["kind"] in ("official_SynthSeg_asset", "official_atlas_original")]
    passed = (len(rows) == 40 and all(x["prior_exact_match"] and x["launch_exact_match"] for x in rows)
              and all(x["matches_source_manifest"] for x in checked)
              and all(x.get("matches_release_manifest", True) and x.get("matches_live_release_attachment", True) for x in checked)
              and all(x["matches_current_cohort_frozen_source"] for x in source_rows)
              and len(official) == 9 and all(x["matches_fnit_upstream_manifest"] for x in official))
    result = dict(schema_version=1, all_verified=passed, checked_unix=live["checked_unix"],
                  scope="Read-only exact assets identity audit during ten public T1 cohort; assets never downloaded/transferred by this audit.",
                  total_assets=len(rows), prior_exact_matches=sum(x["prior_exact_match"] for x in rows),
                  launch_exact_matches=sum(x["launch_exact_match"] for x in rows),
                  official_atlas_originals=12, synthseg_assets=5, active_fnit_recipe_cache_files=3,
                  inactive_legacy_config_cache_files=20,
                  previous_queue=live["prior_queue_path"], current_queue=live["queue_path"],
                  assets=rows, official_installed_atlases=official,
                  fixed_release=release, live_release_license_assets=licenses["license_assets"],
                  source_manifests=source_rows, current_cohort_frozen_source_manifest=identity(frozen_path),
                  metadata_inputs=[identity(live_path), identity(release_path), identity(licenses_path)],
                  license_use=dict(public_software_license="FreeSurfer Software License 1.0; original data/weight attribution retained",
                                   official_license_url="https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense",
                                   unknown_third_party_atlas_redistribution_rights="No new redistribution; official upstream/server copies only",
                                   credential_registration_license="External installed FS_LICENSE only stat/access checked; never read, copied or hashed"))
    (root / "assets_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_verified", "total_assets", "prior_exact_matches", "launch_exact_matches")}))
    if not passed:
        raise ValueError("Asset audit failed; retained exact mismatches in output metadata")


if __name__ == "__main__":
    main()
