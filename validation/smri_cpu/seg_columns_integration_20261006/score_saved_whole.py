"""Saved CC0 labels/CSV/header score and 2D figure only; no inference/native/GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time


def identity(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):digest.update(block)
    return {"bytes":Path(path).stat().st_size,"sha256":digest.hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","workspace","run","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    os.umask(0o077)
    if args.output.exists():raise RuntimeError("new posthoc report required")
    resource.setrlimit(resource.RLIMIT_AS,(8_000_000_000,8_000_000_000))
    plan=json.loads((args.workspace/"PLAN.json").read_text())
    queue=json.loads((args.run/"whole_cpu/QUEUE.json").read_text())
    if queue["status"]!="complete" or len(queue["jobs"])!=4:raise RuntimeError("CPU four-arm queue complete required before reading reference")
    for row in queue["jobs"]:
        assert row["returncode"]==0 and row["immediate_full_file_SHA_gate_passed"]
    native=plan["existing_official_reference"]
    comparator=args.root/"workspaces/smri_cpu_20261004/t2_seg/compare_outputs.py"
    if identity(comparator)["sha256"]!="61eaa50d008e1958f48c5f718d6ad6805fccf7142a91de83c266a91733b7bbf8":raise RuntimeError("existing comparator bytes changed")
    paths={"baseline":args.run/"whole_cpu/A1_baseline/segmentation.nii.gz",
           "candidate":args.run/"whole_cpu/B2_warm/segmentation.nii.gz",
           "official":args.root/native["seg33_map_fnit_relative"]}
    csvs={"baseline":args.run/"whole_cpu/A1_baseline/volumes.csv",
          "candidate":args.run/"whole_cpu/B2_warm/volumes.csv",
          "official":args.root/native["seg33_csv_fnit_relative"]}
    before={"map/"+name:identity(path) for name,path in paths.items()}
    before.update({"csv/"+name:identity(path) for name,path in csvs.items()})
    assert before["map/official"]["sha256"]==native["map_sha256"] and before["csv/official"]["sha256"]==native["csv_sha256"]
    started=time.monotonic()
    import nibabel as nib
    import numpy as np
    spec=importlib.util.spec_from_file_location("saved_fnit_compare",comparator)
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    images={name:nib.load(path) for name,path in paths.items()}
    def header_compare(first,second):
        a,b=images[first],images[second]
        fields={name:a.header[name].tobytes()==b.header[name].tobytes() for name in a.header.keys()}
        controls={"shape":a.shape==b.shape,"affine":np.array_equal(a.affine,b.affine),
          "dtype":a.get_data_dtype()==b.get_data_dtype(),"zooms":a.header.get_zooms()==b.header.get_zooms(),
          "pixdim":np.array_equal(a.header["pixdim"],b.header["pixdim"]),
          "qform_code":a.header["qform_code"]==b.header["qform_code"],
          "sform_code":a.header["sform_code"]==b.header["sform_code"],
          "qform":np.array_equal(a.header.get_qform(),b.header.get_qform()),
          "sform":np.array_equal(a.header.get_sform(),b.header.get_sform()),
          "xyzt_units":np.array_equal(a.header["xyzt_units"],b.header["xyzt_units"]),
          "intent":a.header.get_intent()==b.header.get_intent(),
          "binaryblock":a.header.binaryblock==b.header.binaryblock,
          "extensions":[(e.get_code(),e._raw) for e in a.header.extensions]==[(e.get_code(),e._raw) for e in b.header.extensions]}
        return {"complete_NIfTI_struct_fields_equal":fields,"ordinary_13_controls":{k:bool(v) for k,v in controls.items()},
           "all_13_exact":all(controls.values()),"all_struct_fields_exact":all(fields.values()),
           "gzip_file_SHA_equal":before["map/"+first]["sha256"]==before["map/"+second]["sha256"],
           "affine_max_abs_mm":float(np.abs(a.affine-b.affine).max())}
    pairs={}
    for first,second in (("baseline","candidate"),("official","baseline"),("official","candidate")):
        row={"map":compare.compare_segmentation(paths[first],paths[second]),
             "soft_CSV":compare.compare_csv(csvs[first],csvs[second]),"header":header_compare(first,second)}
        if first=="baseline":
            assert row["map"]["different_voxels"]==0 and row["soft_CSV"]["max_absolute_difference_mm3"]==0
            assert row["header"]["all_13_exact"] and row["header"]["all_struct_fields_exact"]
            assert row["header"]["gzip_file_SHA_equal"] and before["csv/baseline"]==before["csv/candidate"]
        pairs[first+"_vs_"+second]=row
    official_old=pairs["official_vs_baseline"]["map"];official_new=pairs["official_vs_candidate"]["map"]
    assert official_old["comparison_status"]==official_new["comparison_status"]=="compared"
    assert official_old["different_voxels"]==official_new["different_voxels"]
    data={name:np.asanyarray(image.dataobj) for name,image in images.items()}
    old_error=data["official"]!=data["baseline"];new_error=data["official"]!=data["candidate"]
    changed={"introduced":int(np.count_nonzero(~old_error&new_error)),"removed":int(np.count_nonzero(old_error&~new_error)),
             "retained":int(np.count_nonzero(old_error&new_error)),
             "retained_with_changed_label":int(np.count_nonzero(old_error&new_error&(data["baseline"]!=data["candidate"])))}
    assert changed["introduced"]==changed["removed"]==changed["retained_with_changed_label"]==0
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    canonical={name:nib.as_closest_canonical(image) for name,image in images.items()}
    arrays={name:np.asanyarray(image.dataobj) for name,image in canonical.items()}
    assert all(arrays[name].shape==arrays["official"].shape for name in arrays)
    foreground=np.where(arrays["official"]!=0)
    index=[int((v.min()+v.max())//2) for v in foreground]
    labels=sorted(set(int(v) for value in arrays.values() for v in np.unique(value)))
    colors=plt.get_cmap("hsv",len(labels)+1)(np.arange(len(labels)))[:,:3]
    if 0 in labels:colors[labels.index(0)]=0
    figure,axes=plt.subplots(3,3,figsize=(9,8),layout="constrained")
    for row,name in enumerate(("baseline","candidate","official")):
        for axis in range(3):
            plane=np.take(arrays[name],index[axis],axis=axis).T
            axes[row,axis].imshow(colors[np.searchsorted(labels,plane)],origin="lower",interpolation="nearest")
            axes[row,axis].set_title(name+" / "+("sagittal","coronal","axial")[axis]+" @ "+str(index[axis]),fontsize=10)
            axes[row,axis].axis("off")
    figure.suptitle("CC0 case02 / new complete CPU output; RAS display without interpolation",fontsize=12)
    png=args.output.parent/"case02_current_cpu_labels.png"
    if png.exists():raise RuntimeError("new figure path required")
    figure.savefig(png,dpi=140);plt.close(figure)
    after={"map/"+name:identity(path) for name,path in paths.items()}
    after.update({"csv/"+name:identity(path) for name,path in csvs.items()})
    assert after==before
    report={"schema":"fnit_columns_saved_whole_posthoc/v1","status":"saved_output_score_passed_no_inference",
       "collector":identity(__file__),"PLAN":identity(args.workspace/"PLAN.json"),"queue":identity(args.run/"whole_cpu/QUEUE.json"),
       "comparator":identity(comparator),"dataset":"CC0 OpenNeuro ds003138 v1.0.1 case02",
       "reference_read_only_after_complete_inference":True,"new_inference_native_or_GPU_calls":0,
       "saved_files_before":before,"saved_files_after":after,"saved_files_unchanged":True,
       "pairs":pairs,"official_error_changes":changed,
       "figure":{"file":png.name,**identity(png),"source_map_SHA256":{k:before["map/"+k]["sha256"] for k in paths},
         "canonical_display_only_no_resampling":True,"slice_indices":index,"new_current_candidate_output":True},
       "RSS_maximum_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       "posthoc_observation_seconds":time.monotonic()-started,"clock_is_not_model_benchmark":True}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"old_new_voxels":pairs["baseline_vs_candidate"]["map"]["different_voxels"],
       "official_candidate_voxels":official_new["different_voxels"],"official_CSV_max_abs_mm3":pairs["official_vs_candidate"]["soft_CSV"]["max_absolute_difference_mm3"]}))


if __name__=="__main__":main()
