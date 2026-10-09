"""用nibabel只读比较已保存的前段体积和主要表面，分开压缩字节和解码几何。

输入control/candidate已完成run根、重复--case和新--output。体积保持conform
网格/affine/dtype；表面比较surface RAS毫米有序坐标、面与空间头。
没有重新计算重建阶段，不读取官方参考，也不设置新的验收阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-root", "candidate-root", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--case",action="append",required=True)
    args=parser.parse_args()
    if args.output.exists() or any(Path(case).name != case for case in args.case):
        raise ValueError("new output and basename cases required")
    report={"scope":"read-only decoded saved-input/geometry checks; not new producer execution", "script_sha256":sha(Path(__file__)),"cases":{}}
    for case in args.case:
        left,right=(root/(case+"-candidate")/"subject"for root in (args.control_root,args.candidate_root))
        volumes={}
        for name in ("orig","nu","filled"):
            a,b=(nib.load(str(root/"mri"/(name+".mgz")))for root in (left,right))
            x,y=np.asarray(a.dataobj),np.asarray(b.dataobj)
            row={"control_raw_sha256":sha(left/"mri"/(name+".mgz")),"candidate_raw_sha256":sha(right/"mri"/(name+".mgz")),
                 "shape_equal":x.shape==y.shape,"dtype_equal":x.dtype==y.dtype,"affine_equal":bool(np.array_equal(a.affine,b.affine)),
                 "voxel_differences":int(np.count_nonzero(x!=y))if x.shape==y.shape else None,
                 "maximum_absolute_voxel_error":float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))if x.shape==y.shape else None}
            volumes[name]=row
        surfaces={}
        for hemisphere in ("lh","rh"):
            for name in ("orig.premesh","orig","smoothwm","inflated","sphere","sphere.reg","white","pial"):
                a,b=(root/"surf"/(hemisphere+"."+name)for root in (left,right))
                av,af,ah=nib.freesurfer.io.read_geometry(str(a),read_metadata=True)
                bv,bf,bh=nib.freesurfer.io.read_geometry(str(b),read_metadata=True)
                coordinate_equal=av.shape==bv.shape and np.array_equal(av,bv)
                header_keys=set(ah)|set(bh)
                header_equal={key:key in ah and key in bh and bool(np.array_equal(ah[key],bh[key]))for key in header_keys if key!="filename"}
                surfaces[hemisphere+"."+name]={"control_raw_sha256":sha(a),"candidate_raw_sha256":sha(b),
                    "vertices_control":len(av),"vertices_candidate":len(bv),"faces_control":len(af),"faces_candidate":len(bf),
                    "ordered_coordinates_equal":bool(coordinate_equal),"ordered_faces_equal":bool(np.array_equal(af,bf)),
                    "geometry_header_fields_equal":header_equal,"metadata_filename_equal":ah.get("filename")==bh.get("filename"),
                    "geometry_header_scope":"all parsed geometry header fields; source filename reported separately",
                    "maximum_coordinate_error_mm":float(np.max(np.abs(av-bv)))if av.shape==bv.shape else None}
        report["cases"][case]={"volumes":volumes,"surfaces":surfaces,
            "all_checked_volume_voxels_dtype_shape_affine_equal":all(row["voxel_differences"]==0 and row["dtype_equal"] and row["affine_equal"]for row in volumes.values()),
            "all_checked_surface_geometry_equal":all(row["ordered_coordinates_equal"] and row["ordered_faces_equal"] and all(row["geometry_header_fields_equal"].values())for row in surfaces.values())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({case:{key:row[key]for key in ("all_checked_volume_voxels_dtype_shape_affine_equal","all_checked_surface_geometry_equal")}for case,row in report["cases"].items()}),flush=True)


if __name__=="__main__":
    main()
