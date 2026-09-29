"""Run the complete ABIDE I+II 2 mm gray-matter BWAS benchmark privately."""

import csv
import hashlib
import json
from pathlib import Path
from time import monotonic, sleep

import nibabel as nib
import numpy as np

from fnit.bwas import run_bwas


def main():
    # These private paths are supplied by the server-side launch script.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--min-valid-voxels", type=int, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--prepared-cache-dir", type=Path)
    parser.add_argument("--fwhm", type=float)
    parser.add_argument("--validate-direct-ols", action="store_true")
    args = parser.parse_args()
    with (args.bids_root / "participants.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1778:
        raise ValueError(f"expected all 1778 matched ABIDE I+II subjects, found {len(rows)}")
    deadline = monotonic()+12*3600
    while True:
        ready = sum((args.bids_root / "private_qc" / f"{row['participant_id']}.npy").is_file()
                    and any((args.bids_root / row["participant_id"]).glob(
                        "**/*desc-clean_bold.nii.gz")) for row in rows)
        if ready == len(rows):
            break
        if monotonic() > deadline:
            raise TimeoutError(f"only {ready}/{len(rows)} BIDS inputs completed")
        print(f"Waiting for 2 mm BIDS preparation: {ready}/{len(rows)}", flush=True)
        sleep(60)
    source_mask = args.bids_root / "group_space-MNI152NLin6Asym_res-2_desc-graymatter_mask.nii.gz"
    image = nib.load(str(source_mask))
    mask = np.asarray(image.dataobj) != 0
    if not 0 < args.min_valid_voxels <= int(mask.sum()):
        raise ValueError("min-valid-voxels must lie within the gray-matter mask size")
    eligible = []
    common = np.ones(int(mask.sum()), dtype=bool)
    for row in rows:
        qc = np.load(args.bids_root / "private_qc" / f"{row['participant_id']}.npy")
        valid = np.unpackbits(qc)[:len(common)].astype(bool)
        if int(valid.sum()) >= args.min_valid_voxels:
            eligible.append(row)
            common &= valid
    if not eligible:
        raise ValueError("no subjects passed the prespecified coverage threshold")
    mask[mask] = common
    if not mask.any():
        raise ValueError("no common gray-matter voxels across all subjects")
    final_mask = args.bids_root / (
        f"group_space-MNI152NLin6Asym_res-2_desc-qc{args.min_valid_voxels}GrayMatter_mask.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine), str(final_mask))
    print(f"Eligible subjects: {len(eligible)}/{len(rows)}; common gray-matter voxels: "
          f"{int(mask.sum())}", flush=True)
    site_columns = tuple(name for name in rows[0] if name.startswith("site_"))
    selected_tsv = (args.bids_root / "private_qc" /
                    f"participants_minvalid{args.min_valid_voxels}.tsv")
    with selected_tsv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(eligible)
    result = run_bwas(
        args.bids_root, selected_tsv, final_mask,
        args.output_root, phenotype="case", covariates=("age", "sex", *site_columns),
        cdt=5.0, block_size=2048, subject_block_size=16,
        num_workers=8, device=args.device, fwhm=args.fwhm,
        cache_root=args.cache_root, _prepared_cache_dir=args.prepared_cache_dir,
        validate_direct_ols=args.validate_direct_ols,
    )
    metadata = json.loads(result.metadata.read_text())
    public = {
        "dataset": "ABIDE I+II real preprocessed BOLD, resampled to 2 mm MNI152 gray matter",
        "matched_subjects": len(rows), "subjects": len(eligible),
        "excluded_for_coverage": len(rows)-len(eligible),
        "minimum_valid_gray_voxels_per_subject": args.min_valid_voxels,
        "cases": sum(int(row["case"]) for row in eligible),
        "controls": sum(int(row["case"]) == 0 for row in eligible),
        "sites": len(site_columns)+1,
        "gray_prior_voxels": int(np.asarray(image.dataobj).astype(bool).sum()),
        "common_gray_voxels": result.voxels,
        "all_unordered_voxel_pairs": result.voxels*(result.voxels-1)//2,
        "suprathreshold_edges": result.suprathreshold_edges,
        "cdt": 5.0,
        "fwhm_voxels": metadata["FWHMInVoxels"],
        "fnit_elapsed_seconds": metadata["ElapsedSeconds"],
        "peak_cuda_allocated_bytes": metadata["PeakCUDAAllocatedBytes"],
        "direct_ols_validation": metadata.get("DirectOLSValidation"),
        "upstream_source_sha256": hashlib.sha256(args.upstream_source.read_bytes()).hexdigest(),
    }
    report = args.output_root / "validation_summary.public.json"
    report.write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public, indent=2), flush=True)


if __name__ == "__main__":
    main()
