"""Private CPU comparison of two completed, independently reconstructed subjects.
No reconstruction, GPU, license-content reads, or mutation of either original namespace.
"""
import hashlib
import struct
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from datetime import datetime, timezone

nib = np = None

def scientific_modules():
    """Import CPU readers only when scientific data is actually available."""
    global nib, np
    if nib is None:
        import nibabel as nib_module
        import numpy as np_module
        nib, np = nib_module, np_module
    return nib, np

OBSERVED_INTERNAL_LINKS = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('x') as stream:
        stream.write(json.dumps(content, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(array):
    scientific_modules()
    array = np.asarray(array)
    return np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'), copy=False))


def array_sha(array):
    return hashlib.sha256(canonical(array).tobytes()).hexdigest()


def describe_array(array):
    scientific_modules()
    array = np.asarray(array)
    return {'dtype': array.dtype.str, 'shape': [int(n) for n in array.shape],
            'elements': int(array.size), 'canonical_scalar_bytes_sha256': array_sha(array),
            'canonical_scalar_byte_order': 'little-endian, C order; dtype/shape reported separately',
            'finite': bool(np.isfinite(array).all())}


def compare_arrays(left, right):
    scientific_modules()
    left, right = np.asarray(left), np.asarray(right)
    result = {'baseline': describe_array(left), 'candidate': describe_array(right),
              'dtype_equal': left.dtype == right.dtype, 'shape_equal': left.shape == right.shape}
    check(result['baseline']['finite'] and result['candidate']['finite'], 'nonfinite scientific array')
    if left.shape != right.shape:
        result.update(numeric_neq=None, raw_scalar_bits_neq=None, rmse=None, max_abs_error=None,
                      exact_scientific_array_equal=False)
        return result
    result['numeric_neq'] = int(np.count_nonzero(left != right))
    if left.dtype.kind == right.dtype.kind and left.dtype.itemsize == right.dtype.itemsize:
        dtype = np.dtype(f'<u{left.dtype.itemsize}')
        result['raw_scalar_bits_neq'] = int(np.count_nonzero(canonical(left).view(dtype) != canonical(right).view(dtype)))
    else:
        result['raw_scalar_bits_neq'] = None
    squared_error, maximum = 0.0, 0.0
    a, b = left.reshape(-1), right.reshape(-1)
    for start in range(0, a.size, 1_000_000):
        delta = a[start:start + 1_000_000].astype(np.float64) - b[start:start + 1_000_000].astype(np.float64)
        squared_error += float(np.dot(delta, delta))
        if delta.size:
            maximum = max(maximum, float(np.abs(delta).max()))
    result['rmse'] = math.sqrt(squared_error / max(1, a.size))
    result['max_abs_error'] = maximum
    result['exact_scientific_array_equal'] = bool(result['dtype_equal'] and result['numeric_neq'] == 0 and result['raw_scalar_bits_neq'] == 0)
    return result


def read_record(path, touched):
    path = Path(path)
    check(path.is_file() and not path.is_symlink(), f'report absent or linked: {path}')
    digest = sha(path)
    touched[str(path)] = digest
    return json.loads(path.read_text()), {'path': str(path), 'sha256': digest}


def input_t1(case, report):
    expected = next(entry for entry in case['input_files'] if entry['kind'] == 'raw_t1w')
    check(report['raw_input_provenance'] == case['input_files'], 'raw input provenance changed')
    matches = [entry for entry in report['input_verification'] if entry['kind'] == 'raw_t1w']
    check(len(matches) == 1 and matches[0]['path'] == case['t1w'] and
          matches[0]['sha256'] == expected['sha256'] and matches[0]['actual_sha256'] == expected['sha256'],
          'original raw T1 verification incomplete')
    actual = sha(case['t1w'])
    check(actual == expected['sha256'], 'raw T1 current SHA differs from original run')
    return {'path': case['t1w'], 'sha256': actual, 'bytes': Path(case['t1w']).stat().st_size}


def validate_run(root, case, version, manifest_case, touched, *, validation_path=None):
    """Verify an actual official raw-T1 run; no license contents are read."""
    job = Path(root) / version / case
    report, identity = read_record(job / 'recon_report.json', touched)
    check(report['case_id'] == case and report['subject'] == manifest_case['subject'] and
          report['version'] == version and report['action'] == 'recon' and report['exit_code'] == 0,
          'official reconstruction exit/case identity invalid')
    cmd = report['command']
    check(len(cmd) == 10 and cmd[1] == '-sd' and cmd[3] == '-s' and cmd[5] == '-i' and
          cmd[6] == manifest_case['t1w'] and cmd[7:] == ['-all','-openmp','8'] and report['cpu_threads'] == 8,
          'not the declared raw T1 -i -all -openmp8 official run')
    subject = Path(cmd[2]) / cmd[4]
    check(subject.resolve().is_relative_to(job.resolve()) and not subject.is_symlink(),
          'subject escaped fresh independent namespace')
    check(report['freesurfer_version']['returncode'] == 0 and
          'freesurfer' in report['freesurfer_version']['stdout'].lower(),
          'wrong or unverified official version')
    check(sha(cmd[0]) == report['executable_sha256'], 'official executable changed')
    setup = Path(report['environment']['FREESURFER_HOME']) / 'SetUpFreeSurfer.sh'
    check(sha(setup) == report['setup_script_sha256'], 'official setup changed')
    t1 = input_t1(manifest_case, report)
    touched[t1['path']] = t1['sha256']
    if version == 'baseline':
        check(report['status'] == 'failed' and report['error']['type'] == 'RuntimeError' and
              report['error']['message'].startswith('actual nibabel anatomy validation failed:') and
              report['error']['message'].rstrip().endswith('TypeError: Object of type int32 is not JSON serializable'),
              'baseline is not the known official-success validation failure')
        path = Path(validation_path) if validation_path else job / 'recon_report.revalidated.json'
        valid, validation_identity = read_record(path, touched)
        check(valid['status'] == 'completed' and valid['action'] in ('revalidate_official_anatomy', 'bind_same_round_official_anatomy') and
              valid['case_id'] == case and valid['version'] == version and
              valid['original_report'] == identity and valid.get('raw_t1_sha256_after', t1['sha256']) == t1['sha256'] and
              valid['recon_all_rerun'] is False and valid['anatomy'] == report['anatomy'] and bool(valid['anatomy_geometry']),
              'baseline independent real revalidation is invalid')
        for key in ('input_verification', 'input_verification_after'):
            verified = valid[key]
            check(len(verified) == len(manifest_case['input_files']) and all(
                a['path'] == b['path'] and a['kind'] == b['kind'] and a['sha256'] == b['sha256'] and a['actual_sha256'] == b['sha256']
                for a,b in zip(verified,manifest_case['input_files'])), 'revalidation input ledger differs from actual raw run')
    else:
        check(report['status'] == 'completed' and report['raw_t1_sha256_after'] == t1['sha256'] and bool(report['anatomy_geometry']),
              'candidate official reconstruction did not complete actual validation')
        valid, validation_identity = read_record(job / 'anatomy_prep_report.json', touched)
        check(valid['status'] == 'completed' and valid['action'] == 'prepare_official_anatomy' and
              valid['candidate_source'] == 'unknown' and valid['gpu_started'] is False and valid['recon_all_reused'] is False and
              valid['reconstruction_report'] == identity and valid['reconstruction_result'] == report,
              'candidate original preparation report is incomplete or reused')
    for relative, record in report['anatomy'].items():
        path = subject / relative
        check(not Path(relative).is_absolute() and '..' not in Path(relative).parts and
              record['path'] == str(path) and sha(path) == record['sha256'],
              'official anatomy changed after reconstruction/revalidation')
        touched[str(path)] = record['sha256']
    check('scripts/recon-all.done' in report['anatomy'] and (subject/'scripts/recon-all.done').stat().st_size > 0,
          'official completion marker is absent')
    record = {'reconstruction_report': identity, 'validation_report': validation_identity,
              'original_report_status': report['status'], 'official_exit_code': report['exit_code'],
              'baseline_original_failure': report.get('error'),
              'actual_validation_status': valid['status'], 'subject_directory': str(subject),
              'raw_T1': t1, 'command': cmd, 'cpu_threads': report['cpu_threads'],
              'official_version': report['freesurfer_version'], 'official_host': report['identity'],
              'official_executable_sha256': report['executable_sha256'], 'setup_script_sha256': report['setup_script_sha256'],
              'recon_done': report['anatomy']['scripts/recon-all.done'],
              'original_execution_timing': {'start_utc': report['start_utc'], 'end_utc': report['end_utc'],
                  'recon_command_seconds': report['recon_command_seconds'], 'worker_wall_seconds': report['worker_wall_seconds'],
                  'timer_source':'actual original CPU worker monotonic command/worker timers; UTC timestamps from the original official host',
                  'scope':'official recon-all only for recon_command_seconds; worker wall includes setup, I/O and validation (failed JSON validation on original baseline)'},
              'validation_timing': {key: valid[key] for key in ('start_utc','end_utc','revalidation_wall_seconds','preparation_worker_wall_seconds') if key in valid}}
    return subject, record


def compared_files(left, right, reader, touched):
    check(left.is_file() and right.is_file(), f'missing scientific file: {left} or {right}')
    for path in (left,right):
        check(path.resolve().is_relative_to(path.parent.parent.resolve()), f'scientific file escapes its own subject: {path}')
        if path.is_symlink():
            OBSERVED_INTERNAL_LINKS[str(path)] = {'link_target':os.readlink(path),'resolved':str(path.resolve())}
    check((left.stat().st_dev, left.stat().st_ino) != (right.stat().st_dev, right.stat().st_ino), 'scientific files share an inode across the independent namespaces')
    before = [sha(left), sha(right)]
    touched[str(left)], touched[str(right)] = before
    started = time.perf_counter()
    result = reader(left, right)
    check([sha(left), sha(right)] == before, 'scientific file changed during actual read')
    result.update(baseline_file={'path':str(left),'bytes':left.stat().st_size,'sha256':before[0],'internal_link':OBSERVED_INTERNAL_LINKS.get(str(left))},
                  candidate_file={'path':str(right),'bytes':right.stat().st_size,'sha256':before[1],'internal_link':OBSERVED_INTERNAL_LINKS.get(str(right))},
                  file_bytes_equal=before[0] == before[1], comparison_cpu_wall_seconds=time.perf_counter()-started,
                  file_bytes_equality_is_scientific_gate=False)
    return result


def volume(left, right):
    scientific_modules()
    a, b = nib.load(left), nib.load(right)
    data = compare_arrays(np.asarray(a.dataobj),np.asarray(b.dataobj))
    affine = compare_arrays(a.affine,b.affine)
    tkr = compare_arrays(a.header.get_vox2ras_tkr(),b.header.get_vox2ras_tkr())
    zooms = compare_arrays(np.asarray(a.header.get_zooms()),np.asarray(b.header.get_zooms()))
    orientation = {'baseline':list(nib.aff2axcodes(a.affine)), 'candidate':list(nib.aff2axcodes(b.affine))}
    orientation['equal'] = orientation['baseline'] == orientation['candidate']
    return {'type':'MGH volume','reader':'nibabel.load/dataobj; no interpolation or float conversion before comparison',
            'space':'FreeSurfer conformed volume; affine=scanner RAS millimetres, vox2ras_tkr=surface/tkregister RAS millimetres',
            'baseline_image_class':type(a).__name__,'candidate_image_class':type(b).__name__,
            'voxel_data':data,'scanner_RAS_affine':affine,'surface_RAS_vox2ras_tkr':tkr,'zooms':zooms,'orientation':orientation,
            'strict_scientific_equal':all(item['exact_scientific_array_equal'] for item in (data,affine,tkr,zooms)) and orientation['equal']}



def stored_surface_payload(path):
    """Read actual triangle coordinates/faces bytes, excluding creation metadata."""
    scientific_modules()
    with Path(path).open("rb") as stream:
        magic = stream.read(3)
        check(int.from_bytes(magic, "big") == 16777214, "official surface is not a supported triangular file")
        creation = stream.readline()
        blank = stream.readline()
        counts = stream.read(8)
        check(len(counts) == 8, "surface vertex/face counts are truncated")
        vertices, faces = struct.unpack(">ii", counts)
        check(vertices >= 0 and faces >= 0, "negative surface dimensions")
        coordinate_bytes = stream.read(vertices * 3 * 4)
        face_bytes = stream.read(faces * 3 * 4)
        check(len(coordinate_bytes) == vertices * 3 * 4 and len(face_bytes) == faces * 3 * 4, "stored surface payload is truncated")
    return {"vertices": vertices, "faces": faces, "coordinates": coordinate_bytes, "face_indices": face_bytes,
            "creation_comment_sha256": hashlib.sha256(creation).hexdigest(), "blank_line_sha256": hashlib.sha256(blank).hexdigest()}

def surface(left,right):
    scientific_modules()
    a, fa, va = nib.freesurfer.read_geometry(left,read_metadata=True)
    b, fb, vb = nib.freesurfer.read_geometry(right,read_metadata=True)
    coords, faces = compare_arrays(a,b), compare_arrays(fa,fb)
    metadata={}
    for key in ('head','valid','volume','voxelsize','xras','yras','zras','cras'):
        check((key in va) == (key in vb), 'surface scientific geometry metadata keys differ')
        if key not in va: continue
        if isinstance(va[key],str):
            metadata[key]={'baseline':va[key],'candidate':vb[key],'equal':va[key] == vb[key]}
        else:
            metadata[key]=compare_arrays(np.asarray(va[key]),np.asarray(vb[key]))
    pa, pb = stored_surface_payload(left), stored_surface_payload(right)
    stored = {"coordinate_storage_dtype": ">f4", "face_storage_dtype": ">i4",
              "coordinates_payload_equal": pa["coordinates"] == pb["coordinates"],
              "faces_payload_equal": pa["face_indices"] == pb["face_indices"],
              "coordinate_uint32_bits_neq": int(np.count_nonzero(np.frombuffer(pa["coordinates"], dtype=">u4") != np.frombuffer(pb["coordinates"], dtype=">u4"))) if len(pa["coordinates"]) == len(pb["coordinates"]) else None,
              "face_uint32_bits_neq": int(np.count_nonzero(np.frombuffer(pa["face_indices"], dtype=">u4") != np.frombuffer(pb["face_indices"], dtype=">u4"))) if len(pa["face_indices"]) == len(pb["face_indices"]) else None,
              "baseline_coordinate_payload_sha256": hashlib.sha256(pa["coordinates"]).hexdigest(),
              "candidate_coordinate_payload_sha256": hashlib.sha256(pb["coordinates"]).hexdigest(),
              "baseline_face_payload_sha256": hashlib.sha256(pa["face_indices"]).hexdigest(),
              "candidate_face_payload_sha256": hashlib.sha256(pb["face_indices"]).hexdigest(),
              "baseline_creation_comment_sha256": pa["creation_comment_sha256"],
              "candidate_creation_comment_sha256": pb["creation_comment_sha256"]}
    equal = stored["coordinates_payload_equal"] and stored["faces_payload_equal"] and coords['exact_scientific_array_equal'] and faces['exact_scientific_array_equal'] and all(
        item.get('equal',item.get('exact_scientific_array_equal',False)) for item in metadata.values())
    return {'type':'FreeSurfer triangular surface','reader':'nibabel.freesurfer.read_geometry(read_metadata=True)',
            'coordinate_space':'FreeSurfer surface RAS millimetres; no coordinate transformation',
            'coordinates':coords,'faces':faces,'stored_payload':stored,'volume_geometry_metadata':metadata,
            'ignored_non_scientific_metadata':['file creation comment','volume_info filename'],
            'strict_scientific_equal':bool(equal)}


def annotation(left,right):
    scientific_modules()
    la, ca, na = nib.freesurfer.read_annot(left,orig_ids=False)
    lb, cb, nb = nib.freesurfer.read_annot(right,orig_ids=False)
    ia, _, _ = nib.freesurfer.read_annot(left,orig_ids=True)
    ib, _, _ = nib.freesurfer.read_annot(right,orig_ids=True)
    labels, ids, ctab = compare_arrays(la,lb),compare_arrays(ia,ib),compare_arrays(ca,cb)
    names_a=[item.hex() for item in na];names_b=[item.hex() for item in nb]
    names={'baseline_utf8':[item.decode('utf8') for item in na], 'candidate_utf8':[item.decode('utf8') for item in nb],
           'baseline_raw_bytes_hex':names_a,'candidate_raw_bytes_hex':names_b,
           'equal':names_a == names_b, 'count_baseline':len(na),'count_candidate':len(nb),
           'unequal_positions':sum(a != b for a,b in zip(names_a,names_b))+abs(len(names_a)-len(names_b)),
           'baseline_names_sha256':hashlib.sha256(json.dumps(names_a,separators=(',',':')).encode()).hexdigest(),
           'candidate_names_sha256':hashlib.sha256(json.dumps(names_b,separators=(',',':')).encode()).hexdigest()}
    return {'type':'FreeSurfer surface annotation','reader':'nibabel.freesurfer.read_annot with both orig_ids=False and True',
            'vertex_index_labels':labels,'vertex_original_annotation_IDs':ids,'color_table_RGB_T_ID':ctab,'region_names':names,
            'strict_scientific_equal':all(item['exact_scientific_array_equal'] for item in (labels,ids,ctab)) and names['equal']}



SCIENTIFIC_GROUPS = (
    (volume, ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz")),
    (surface, tuple(f"surf/{hemisphere}.{name}" for hemisphere in ("lh", "rh") for name in ("white", "pial", "sphere.reg"))),
    (annotation, tuple(f"label/{hemisphere}.{name}.annot" for hemisphere in ("lh", "rh") for name in ("aparc", "aparc.a2009s"))),
)


def verify_unchanged(touched):
    for path, digest in touched.items():
        check(sha(path) == digest, "original file changed during comparison: " + path)
    for path, record in OBSERVED_INTERNAL_LINKS.items():
        check(Path(path).is_symlink() and os.readlink(path) == record["link_target"] and
              str(Path(path).resolve()) == record["resolved"], "official internal link changed during comparison")


def compare_subjects(left, right, touched=None):
    """Read thirteen actual FS outputs. Whole-file hashes are provenance only."""
    scientific_modules()
    left, right = Path(left), Path(right)
    check(left.resolve() != right.resolve() and not left.is_symlink() and not right.is_symlink(),
          "official subjects must be in two independent real directories")
    touched = {} if touched is None else touched
    started = time.perf_counter()
    files = {}
    for reader, names in SCIENTIFIC_GROUPS:
        for name in names:
            files[name] = compared_files(left / name, right / name, reader, touched)
    verify_unchanged(touched)
    return {"status": "completed", "files": files,
            "all_requested_scientific_data_equal": all(item["strict_scientific_equal"] for item in files.values()),
            "scientific_differences": [name for name, item in files.items() if not item["strict_scientific_equal"]],
            "file_byte_differences": [name for name, item in files.items() if not item["file_bytes_equal"]],
            "comparison_cpu_wall_seconds": time.perf_counter() - started,
            "scope": "actual CPU reads of thirteen official FS outputs; file creation comments, paths and compressed/footer bytes do not decide scientific equality",
            "GPU_used": False, "original_namespaces_modified": False}


def compare_fresh_case(case, baseline_root, candidate_root, *, baseline_validation_path=None):
    touched = {}
    left, baseline = validate_run(baseline_root, case["case_id"], "baseline", case, touched,
                                  validation_path=baseline_validation_path)
    right, candidate = validate_run(candidate_root, case["case_id"], "candidate", case, touched)
    check(baseline["raw_T1"] == candidate["raw_T1"] and baseline["official_version"] == candidate["official_version"] and
          baseline["official_executable_sha256"] == candidate["official_executable_sha256"] and
          baseline["setup_script_sha256"] == candidate["setup_script_sha256"], "independent official runs did not use identical raw T1/version/executable/setup")
    compared = compare_subjects(left, right, touched)
    compared.update(case_id=case["case_id"], baseline=baseline, candidate=candidate)
    verify_unchanged(touched)
    return compared


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-prep-root", type=Path, required=True)
    parser.add_argument("--baseline-validation-report", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.report.exists() or options.report.is_symlink():
        raise FileExistsError("comparison report is fresh; previous failed/complete reports are never overwritten")
    options.report.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(options.manifest.read_bytes())["cases"]
    case = next(item for item in cases if item["case_id"] == options.case_id)
    report = {"schema_version": 1, "start_utc": utc(), "private_input_paths": True,
              "tool_sha256": sha(__file__), "manifest_sha256": sha(options.manifest)}
    try:
        report.update(compare_fresh_case(case, options.baseline_root, options.candidate_prep_root,
                                        baseline_validation_path=options.baseline_validation_report))
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report["end_utc"] = utc()
    atomic_json(options.report, report)
    print(json.dumps({"status": report["status"], "case_id": options.case_id,
                      "all_requested_scientific_data_equal": report.get("all_requested_scientific_data_equal"),
                      "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
