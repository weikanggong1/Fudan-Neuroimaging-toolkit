"""Read-only ten-case BIDS/staged/oracle PE, readout, frame, mask and gradient audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--routes",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads(args.manifest.read_text());routes=json.loads(args.routes.read_text())
    report={"manifest_sha256":sha(args.manifest),"routes_sha256":sha(args.routes),
            "dataset":manifest["dataset"],"license":manifest["license"],
            "scope":"read-only raw hashes and actual staged/oracle input contract",
            "software":{"FSL_version":Path("/public/software/apps/FSL/6.0.7.4/etc/fslversion").read_text().strip()},
            "cases":{}}
    for case in manifest["cases"]:
        case_id=case["case_id"];route=routes["cases"][case_id]
        official=Path(route["official_case_directory"])
        actual=Path(route["actual_FNIT_case_directory"])/"connectome/preproc"
        frozen_files=case["input_files"]
        hashes=[{"path":f["path"],"matches_frozen_SHA256":sha(f["path"])==f["sha256"]}
                for f in frozen_files]
        metadata={}
        for f in frozen_files:
            if f["kind"] in ("dwi_json","reverse_dwi_json") or "_dwi.json" in f["path"]:
                d=json.loads(Path(f["path"]).read_text())
                metadata[Path(f["path"]).name]={k:d.get(k) for k in
                     ("PhaseEncodingDirection","TotalReadoutTime","EffectiveEchoSpacing")}
        contract=json.loads(Path(route["verified_contract_expected_path"]).read_text())
        bvals=np.loadtxt(official/"raw/AP.bval").reshape(-1)
        bvecs=np.loadtxt(official/"raw/AP.bvec")
        ap=nib.load(str(official/"raw/AP.nii.gz"));pa=nib.load(str(official/"raw/PA.nii.gz"))
        mask=nib.load(str(official/"mask/nodif_brain_mask.nii.gz"))
        packed_official=nib.load(str(official/"topup/B0_AP_PA.nii.gz"))
        packed_actual=nib.load(str(actual/"topup/B0_AP_PA.nii.gz"))
        packed_equal=np.array_equal(np.asarray(packed_official.dataobj),np.asarray(packed_actual.dataobj))
        acq=np.loadtxt(official/"topup/acqparams.txt")
        acq_actual=np.loadtxt(actual/"topup/acqparams.txt")
        official_index=np.loadtxt(official/"eddy/eddy_index.txt").reshape(-1)
        actual_index=np.loadtxt(actual/"eddy/eddy_index.txt").reshape(-1)
        raw_matches_staged=[]
        for f in frozen_files:
            if "acq-AP_dwi" in f["path"] and f["path"].endswith(("nii.gz","bvec","bval")):
                suffix=Path(f["path"]).name.split("dwi.")[-1]
                staged=official/"raw"/f"AP.{suffix}"
                raw_matches_staged.append({"kind":f["kind"],"same_raw_bytes":sha(staged)==f["sha256"]})
        report["cases"][case_id]={"raw_hashes":hashes,"raw_metadata":metadata,
            "raw_matches_official_staging":raw_matches_staged,
            "AP_shape":list(ap.shape),"PA_shape":list(pa.shape),
            "AP_voxel_sizes":list(map(float,ap.header.get_zooms()[:3])),
            "AP_affine_determinant":float(np.linalg.det(ap.affine[:3,:3])),
            "mask_same_AP_grid":mask.shape==ap.shape[:3] and np.allclose(mask.affine,ap.affine,atol=1e-5,rtol=0),
            "TOPUP_same_packed_voxels":packed_equal,
            "TOPUP_same_packed_header_affine":np.array_equal(packed_actual.affine,packed_official.affine),
            "acqparams":acq.tolist(),"same_actual_official_acqparams":np.array_equal(acq,acq_actual),
            "same_actual_official_index":np.array_equal(official_index,actual_index),
            "index_values":np.unique(official_index).tolist(),
            "bvec_shape":list(bvecs.shape),"bval_count":int(bvals.size),
            "b0_indices":np.flatnonzero(bvals<100).tolist(),
            "ref_scan_no":contract["ref_scan_no"],
            "ref_is_b0":bool(bvals[contract["ref_scan_no"]]<100),
            "BIDS_readout_rounding":"original .0266003 was staged as legacy .0266 on both sides; untouched this round"}
        print(case_id,packed_equal,np.array_equal(acq,acq_actual),all(x["matches_frozen_SHA256"] for x in hashes),flush=True)
    report["all_raw_hashes_match"]=all(x["matches_frozen_SHA256"] for row in report["cases"].values() for x in row["raw_hashes"])
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
