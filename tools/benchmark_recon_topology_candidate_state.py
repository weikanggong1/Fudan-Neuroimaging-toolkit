"""冻结同输入 DVS 保存/恢复；参考只用于恢复之后的比较，不进入生产。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from fnit.recon_all.topology_candidate_state import record_candidate_state, restore_candidate_state


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(offsets, values, count, name):
    if (offsets.dtype.kind not in "iu" or offsets.shape != (count + 1,)
            or offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
        raise ValueError(name + " storage offsets inconsistent")
    return [values[int(offsets[i]):int(offsets[i+1])].copy() for i in range(count)]


def load_state_capture(path):
    """FNITDVS1原真实调用phase→同网格字段/原选择；仅参考解析器。"""
    raw=Path(path).read_bytes()
    if raw[:8]!=b"FNITDVS1":raise ValueError("complete FNITDVS1 state capture required")
    offset=8
    def read(dtype,count,shape=None):
        nonlocal offset
        dt=np.dtype(dtype)
        if count<0 or offset+count*dt.itemsize>len(raw):raise ValueError("truncated DVS capture")
        a=np.frombuffer(raw,dt,count=count,offset=offset).copy();offset+=count*dt.itemsize
        return a.reshape(shape) if shape else a
    header=read('<i4',9);n,nf,en,ei,k,h,b,nmap,phase=map(int,header)
    if min(n,nf,en,ei,k,h,b,nmap,phase)<0 or max(n,nf,en,ei,k,h,b,nmap)>100000000 or phase>4:
        raise ValueError("invalid DVS capture dimensions")
    fields=read('<f4',n*27,(n,27));flags=read('<i4',n*2,(n,2));topology=read('<i4',n*7,(n,7))
    faces=read('<i4',nf*4,(nf,4));neighbor_offsets=read('<i4',n+1);neighbors=read('<i4',en)
    incident_offsets=read('<i4',n+1)
    # Observer writes each original face row followed by that same row's corners.
    incident_faces=np.empty(ei,np.int32);incident_corners=np.empty(ei,np.uint8)
    for i in range(n):
        start,end=map(int,incident_offsets[i:i+2]);count=end-start
        if start<0 or end>ei or count<0:raise ValueError("invalid DVS incident CSR")
        incident_faces[start:end]=read('<i4',count)
        corners=read('<i4',count)
        if corners.size and (corners.min()<0 or corners.max()>2):raise ValueError("invalid original triangle corner")
        incident_corners[start:end]=corners
    selection=dict(defect_vertices=read('<i4',k),convex_hull_vertices=read('<i4',h),
                   border_vertices=read('<i4',b),vertex_translation=read('<i4',nmap))
    if offset!=len(raw):raise ValueError("unexpected DVS capture trailing data")
    state=dict(nfaces=nf,faces=faces[:,:3].copy(),face_rip=faces[:,3].copy(),
        original_vertices=fields[:,:3].copy(),current_vertices=fields[:,3:6].copy(),
        flattened_vertices=fields[:,6:9].copy(),vertex_normals=fields[:,9:12].copy(),
        tangent_e1=fields[:,12:15].copy(),tangent_e2=fields[:,15:18].copy(),
        k1=fields[:,18].copy(),k2=fields[:,19].copy(),vertex_area=fields[:,20].copy(),
        val=fields[:,21].copy(),val2=fields[:,22].copy(),val2bak=fields[:,23].copy(),
        curvbak=fields[:,24].copy(),undefval=fields[:,25].copy(),old_undefval=fields[:,26].copy(),
        vertex_marks=flags[:,0].copy(),vertex_rip=flags[:,1].copy(),vertex_topology=topology,
        neighbor_storage=_rows(neighbor_offsets,neighbors,n,'neighbors'),
        incident_faces=_rows(incident_offsets,incident_faces,n,'incident faces'),
        incident_corners=_rows(incident_offsets,incident_corners,n,'incident corners'))
    return state,selection,dict(phase=phase,vertices=n,faces=nf,neighbor_storage_values=en,
                               incident_values=ei,capture_sha256=_sha(path))


def load_state_checkpoint(path):
    """只读无pickle NPZ → 显式表面state；CSR必须是完整存储，不是vtotal截断。"""
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    n = len(values["original_vertices"])
    values["nfaces"] = int(values["nfaces"].item())
    values["neighbor_storage"] = _rows(values.pop("neighbor_offsets"), values["neighbor_storage"], n, "neighbor")
    offsets = values.pop("incident_offsets")
    values["incident_faces"] = _rows(offsets, values["incident_faces"], n, "incident face")
    values["incident_corners"] = _rows(offsets, values["incident_corners"], n, "incident corner")
    return values


def dump_state_checkpoint(path, state):
    """私有诊断写出显式CSR；保留零有效长度时仍存在的storage。"""
    values = {key: value for key, value in state.items()
              if key not in ("neighbor_storage", "incident_faces", "incident_corners")}
    for key, offsets_key, dtype in (("neighbor_storage", "neighbor_offsets", np.int32),
                                    ("incident_faces", "incident_offsets", np.int32),
                                    ("incident_corners", "incident_offsets", np.uint8)):
        rows = state[key]
        offsets = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64)
        if offsets_key in values and not np.array_equal(offsets, values[offsets_key]):
            raise ValueError("face/corner storage lengths differ; cannot serialize shared offsets")
        values[offsets_key] = offsets
        values[key] = np.concatenate([np.asarray(row, dtype=dtype) for row in rows]) if rows else np.empty(0, dtype)
    np.savez_compressed(path, **values)


def _compare_value(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(shape_equal=a.shape==b.shape, dtype_equal=a.dtype==b.dtype, byte_equal=False)
    if a.dtype.kind == "f":
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        errors = np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
    else:
        equal = a == b
        errors = np.abs(a.astype(np.float64)-b.astype(np.float64)).reshape(-1)
    return dict(shape_equal=True, dtype_equal=True,
        byte_equal=a.tobytes()==b.tobytes(), differing_values=int(np.count_nonzero(~equal)),
        differing_bytes=int(np.count_nonzero(np.frombuffer(a.tobytes(),np.uint8)!=np.frombuffer(b.tobytes(),np.uint8))),
        max_abs=float(errors.max(initial=0)), p99_abs=float(np.percentile(errors,99)) if errors.size else 0.)


def compare_states(candidate, reference):
    result = {}
    for key in reference:
        if key not in candidate:
            result[key] = dict(missing=True, byte_equal=False)
        elif key in ("faces","face_rip"):
            # The source only truncates nfaces; inactive capacity is not an output mesh.
            result[key] = _compare_value(candidate[key][:candidate["nfaces"]], reference[key][:reference["nfaces"]])
        elif key in ("neighbor_storage", "incident_faces", "incident_corners"):
            def logical_length(state, vertex):
                t = state["vertex_topology"][vertex]
                if key != "neighbor_storage": return int(t[6])
                maximum, total = int(t[1]), int(t[2])
                return max((0, int(t[3]), int(t[4]), int(t[5]))[maximum], total)
            rows = [_compare_value(a[:logical_length(candidate,i)],b[:logical_length(reference,i)])
                    for i,(a,b) in enumerate(zip(candidate[key],reference[key]))]
            result[key] = dict(rows_equal=len(candidate[key])==len(reference[key]),
                byte_equal=len(candidate[key])==len(reference[key]) and all(row["byte_equal"] for row in rows),
                differing_rows=sum(not row["byte_equal"] for row in rows),
                differing_values=sum(row.get("differing_values",0) for row in rows),
                differing_bytes=sum(row.get("differing_bytes",0) for row in rows))
        else:
            result[key] = _compare_value(candidate[key],reference[key])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("record-state", "candidate-state", "reference-state", "selection", "source-manifest", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--reference-repeat", type=Path)
    p.add_argument("--face-candidate-state", type=Path)
    p.add_argument("--face-reference-state", type=Path)
    p.add_argument("--profile", choices=("chull","border"), default="chull")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("fresh DVS benchmark output required")
    if bool(args.face_candidate_state) != bool(args.face_reference_state):
        raise ValueError("face-only before/after must be supplied together")
    from fnit.recon_all.thread_budget import thread_budget
    started = time.perf_counter()
    with thread_budget(threads=args.threads) as budget:
        baseline = load_state_checkpoint(args.record_state)
        candidate = load_state_checkpoint(args.candidate_state)
        selection = np.load(args.selection,allow_pickle=False)
        loaded = time.perf_counter()-started
        snapshot = record_candidate_state(surface_state=baseline,
            defect_vertices=selection["defect_vertices"], convex_hull_vertices=selection["convex_hull_vertices"],
            border_vertices=selection["border_vertices"], vertex_translation=selection["vertex_translation"],
            profile=args.profile)
        selection.close()
        restored, full_info = restore_candidate_state(surface_state=candidate,snapshot=snapshot)
        repeat, repeat_info = restore_candidate_state(surface_state=candidate,snapshot=snapshot)
        # Source after-state cannot influence state selection or restoration.
        reference = load_state_checkpoint(args.reference_state)
        comparison = compare_states(restored,reference)
        self_repeat = compare_states(restored,repeat)
        ref_repeat = None
        if args.reference_repeat:
            ref_repeat = compare_states(reference,load_state_checkpoint(args.reference_repeat))
        faces = None
        if args.face_candidate_state:
            input_faces = load_state_checkpoint(args.face_candidate_state)
            face_state, face_info = restore_candidate_state(surface_state=input_faces,snapshot=snapshot,mode="faces")
            faces = dict(comparisons=compare_states(face_state,load_state_checkpoint(args.face_reference_state)),info=face_info)
        args.output.mkdir(parents=True)
        write_started=time.perf_counter()
        dump_state_checkpoint(args.output/"candidate_state.private.npz",restored)
        writing=time.perf_counter()-write_started
    strict=all(item["byte_equal"] for item in comparison.values())
    if faces:strict &= all(item["byte_equal"] for item in faces["comparisons"].values())
    report=dict(scope="frozen same-input DVS transaction, not GA/end-to-end MRI",execution_complete=True,
        strict_dvs_restoration=strict,full_comparisons=comparison,face_only=faces,
        reference_repeat=ref_repeat,self_repeat=self_repeat,
        selected_records=len(snapshot.vertex_numbers),mapped_records=snapshot.vertex_numbers.tolist(),
        saved_neighbor_values=sum(len(row) for row in snapshot.neighbor_storage),
        skipped_records=int(np.count_nonzero(~snapshot.restore_vertex)),
        vsize_exceeds_vtotal=int(np.count_nonzero(snapshot.vsize_exceeds_vtotal)),
        input_sha256={key:_sha(getattr(args,key.replace('-','_'))) for key in
                      ("record-state","candidate-state","reference-state","selection","source-manifest")},
        full_restore=full_info,repeat_restore=repeat_info,
        seconds=dict(loading=loaded,record=snapshot.seconds,first_restore=full_info["seconds"],
                     repeat_restore=repeat_info["seconds"],candidate_checkpoint_writing=writing,
                     script_before_report=time.perf_counter()-started),
        environment=dict(python=platform.python_version(),numpy=np.__version__,thread_budget=budget),
        whole_ga=False,mesh_quality="not_assessed",equivalence="not_assessed")
    (args.output/"report.private.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return 0 if strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
