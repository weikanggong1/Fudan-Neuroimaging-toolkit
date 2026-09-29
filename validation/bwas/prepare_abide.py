"""Private ABIDE I+II benchmark adapter: existing 3 mm BOLD to BIDS 2 mm GM BOLD.

No participant-level files produced here belong in the public FNIT repository.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def records(root, phenotype_i, phenotype_ii):
    first = {int(row["SUB_ID"]): row for row in
             csv.DictReader(phenotype_i.open(encoding="latin1")) if row["SUB_ID"]}
    second = {int(row["SUB_ID"]): row for row in
              csv.DictReader(phenotype_ii.open(encoding="latin1")) if row["SUB_ID"]}
    found = []
    patterns = [
        (root / "abide1/dparsf/filt_noglobal/func_preproc", "*_func_preproc.nii.gz",
         first, "AGE_AT_SCAN", "ABIDEI"),
        (root / "abide2", "*.nii.gz", second, "AGE_AT_SCAN ", "ABIDEII"),
    ]
    for directory, pattern, lookup, age_column, cohort in patterns:
        for file in directory.glob(pattern):
            match = re.search(r"(\d{5,7})(?:_func_preproc)?\.nii\.gz$", file.name)
            if match is None or int(match.group(1)) not in lookup:
                raise ValueError(f"missing phenotype match for {file}")
            row = lookup[int(match.group(1))]
            age, sex, diagnosis = float(row[age_column]), int(row["SEX"]), row["DX_GROUP"]
            if not np.isfinite(age) or sex not in (1, 2) or diagnosis not in ("1", "2"):
                raise ValueError(f"incomplete phenotype for {file}")
            found.append({"participant_id": "sub-" + match.group(1),
                          "source": str(file), "case": int(diagnosis == "1"),
                          "age": age, "sex": sex,
                          "site": cohort + "-" + row["SITE_ID"]})
    found.sort(key=lambda row: row["participant_id"])
    if len({row["participant_id"] for row in found}) != len(found):
        raise ValueError("duplicate participant IDs across ABIDE I and II")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abide-root", type=Path, required=True)
    parser.add_argument("--phenotype-i", type=Path, required=True)
    parser.add_argument("--phenotype-ii", type=Path, required=True)
    parser.add_argument("--gray-prior", type=Path, required=True)
    parser.add_argument("--mni-reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    rows = records(args.abide_root, args.phenotype_i, args.phenotype_ii)
    reference = nib.load(str(args.mni_reference))
    prior = nib.load(str(args.gray_prior))
    gray = np.squeeze(np.asanyarray(prior.dataobj))
    if gray.shape != reference.shape or not np.allclose(prior.affine, reference.affine):
        raise ValueError("2 mm gray prior and MNI reference grids differ")
    mask = gray > 0.5
    if not np.allclose(reference.header.get_zooms()[:3], 2):
        raise ValueError("reference must have 2 mm voxels")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    mask_file = output / "group_space-MNI152NLin6Asym_res-2_desc-graymatter_mask.nii.gz"
    if not mask_file.exists():
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), str(mask_file))
    sites = sorted({row["site"] for row in rows})
    columns = [f"site_{number:02d}" for number in range(1, len(sites))]
    participants = output / "participants.tsv"
    if not participants.exists():
        with participants.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["participant_id", "case", "age", "sex", *columns])
            for row in rows:
                writer.writerow([row["participant_id"], row["case"], row["age"],
                                 row["sex"], *(int(row["site"] == site) for site in sites[1:])])
        (output / "dataset_description.json").write_text(json.dumps({
            "Name": "Private ABIDE I+II 2 mm gray-matter BWAS benchmark",
            "BIDSVersion": "1.11.1", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "FNIT BWAS benchmark adapter"}],
        }, indent=2) + "\n")
        (output / "private_manifest.json").write_text(json.dumps({
            "records": rows, "site_reference": sites[0],
            "site_columns": dict(zip(columns, sites[1:])),
            "gray_prior_sha256": hashlib.sha256(args.gray_prior.read_bytes()).hexdigest(),
        }, indent=2) + "\n")
    coordinates = np.column_stack(np.where(mask))
    world = reference.affine @ np.column_stack((coordinates, np.ones(len(coordinates)))).T
    for row in rows[args.start:args.stop]:
        subject = row["participant_id"]
        dest = (output / subject / "func" /
                f"{subject}_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz")
        qc = output / "private_qc" / f"{subject}.npy"
        if dest.exists() and qc.exists():
            continue
        start = perf_counter()
        image = nib.load(row["source"])
        points = (np.linalg.inv(image.affine) @ world)[:3]
        data = np.asarray(image.dataobj, dtype=np.float32)
        volume = np.zeros((*mask.shape, data.shape[3]), dtype=np.float32)
        extracted = np.empty((len(coordinates), data.shape[3]), dtype=np.float32)
        for frame in range(data.shape[3]):
            extracted[:, frame] = map_coordinates(data[..., frame], points,
                                                   order=1, mode="constant", cval=0)
        volume[mask] = extracted
        dest.parent.mkdir(parents=True, exist_ok=True)
        image_2mm = nib.Nifti1Image(volume, reference.affine)
        image_2mm.set_qform(reference.affine, code=4)
        image_2mm.set_sform(reference.affine, code=4)
        nib.save(image_2mm, str(dest))
        qc.parent.mkdir(parents=True, exist_ok=True)
        np.save(qc, np.packbits(np.isfinite(extracted).all(axis=1) &
                               (extracted.std(axis=1) > 0)))
        print(subject, image.shape[3], dest.stat().st_size,
              round(perf_counter()-start, 2), flush=True)


if __name__ == "__main__":
    main()
