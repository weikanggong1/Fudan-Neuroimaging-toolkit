"""只读复核已自产校准快照和固定源码三时点，不重新计算或修补候选。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    out = dict(candidate_shape=list(a.shape), reference_shape=list(b.shape),
               candidate_dtype=str(a.dtype), reference_dtype=str(b.dtype),
               shape_equal=a.shape == b.shape)
    if a.shape != b.shape:
        return dict(out, bitwise_equal=False, different_elements=None,
                    max_abs=None, p99_abs=None)
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(out, bitwise_equal=a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                different_elements=int(np.count_nonzero(a != b)),
                max_abs=float(delta.max(initial=0)),
                p99_abs=float(np.percentile(delta, 99)) if delta.size else 0.0)


def run(*, candidate, candidate_report, surface, capture_parser,
        before, raw, smooth, before_repeat, raw_repeat, smooth_repeat,
        priors_creation, priors_parser, observer_build, observer_receipt, output):
    """候选已完成快照→采样/凸壳/邻域/选点/原PDF/平滑PDF逐字段报告。

    所有参数均为显式路径。candidate 是先于参考读取保存的自产NPZ，
    candidate_report 是其完成报告；surface 为同输入原nofix。其余捕获
    只作参考比较，两个parser分别解析FNITHST1和FNITPDF1；build/receipt
    绑定实际原程序及两次运行。output必须不存在。落盘标准JSON及
    COMPLETE.json，保留dtype/shape/字节和数值差异、实际原选点顺序。
    不改候选、参考、线程、生产目录；不是新MRI计算或完整GA。
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError('new readonly comparison output required')
    report = json.loads(candidate_report.read_bytes())
    if (report['status'] != 'complete_frozen_nofix_mri_calibration_priors_regression'
            or sha(candidate) != report['candidate_checkpoint_sha256']
            or sha(surface) != report['candidate_input_sha256']['orig_nofix']):
        raise ValueError('actual completed candidate and unchanged nofix required')
    # Read the already independent candidate first; no reference field is fed
    # to a computation or written back into this snapshot.
    with np.load(candidate, allow_pickle=False) as archive:
        c = {key: archive[key].copy() for key in archive.files}
    _, faces = fsio.read_geometry(str(surface))
    faces = np.asarray(faces, np.int32)
    def parser(path, name):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return m
    loader = parser(capture_parser, 'calibration_parser').load_calibration_capture
    paths = [before, raw, smooth, before_repeat, raw_repeat, smooth_repeat]
    refs = [loader(capture_path=p) for p in paths]
    if [int(f['header'][6]) for f in refs] != [0, 1, 2, 0, 1, 2]:
        raise ValueError('two complete before/raw/smooth reference groups required')
    prior = parser(priors_parser, 'priors_parser').load_volume_preparation_capture(
        capture_path=priors_creation)
    fields = dict(original_order_smoothed_xyz=difference(c['smoothed_original'], refs[0]['vertex_fields'][:, :3]),
                  original_order_faces=difference(faces, refs[0]['faces']),
                  original_order_white_after_two_medians=difference(c['white_values'], refs[0]['vertex_fields'][:, 7]),
                  original_order_gray_after_two_medians=difference(c['gray_values'], refs[0]['vertex_fields'][:, 8]),
                  full_ordered_chull=difference(c['hull'], refs[0]['hull']))
    for tissue in ['white', 'gray']:
        fields[tissue+'_raw_pdf'] = difference(c['calibration_'+tissue+'_raw'], refs[1][tissue+'_counts'])
        fields[tissue+'_smoothed_pdf'] = difference(c['calibration_'+tissue+'_smooth'], refs[2][tissue+'_counts'])
        fields[tissue+'_raw_bins'] = difference(c['calibration_bins'], refs[1][tissue+'_bins'])
        fields[tissue+'_smoothed_bins'] = difference(c['calibration_bins'], refs[2][tissue+'_bins'])
        fields[tissue+'_raw_bin_size'] = difference(np.zeros(1, np.float32), refs[1][tissue+'_bin_size'])
        fields[tissue+'_smoothed_bin_size'] = difference(np.zeros(1, np.float32), refs[2][tissue+'_bin_size'])
        fields[tissue+'_same_prior_source_pdf'] = difference(refs[2][tissue+'_counts'], prior[tissue+'_pdf'])
    selected = dict(candidate_count=len(c['calibration_selected']),
                    source_count=len(refs[1]['selected']),
                    candidate_sorted_unique=bool(np.array_equal(c['calibration_selected'], np.unique(c['calibration_selected']))),
                    source_unique=bool(len(np.unique(refs[1]['selected'])) == len(refs[1]['selected'])),
                    set_equal=bool(np.array_equal(np.sort(c['calibration_selected']), np.sort(refs[1]['selected']))),
                    ordered_equal=bool(np.array_equal(c['calibration_selected'], refs[1]['selected'])),
                    raw_smooth_order_equal=bool(np.array_equal(refs[1]['selected'], refs[2]['selected'])),
                    source_actual_order_is_preserved_in_private_capture=True,
                    operation='exact integer bin increments before F32 normalization; output selection is sorted')
    # Source vtotal CSR includes two rings. Test the existing one-ring ordering
    # and the two-ring set; do not invent a new production neighborhood recipe.
    from fnit.recon_all.smooth_surface_python import ordered_neighbors
    import inspect
    neighbors = ordered_neighbors(faces, len(c['white_values']))
    off, source_neighbors, top = (refs[0][key] for key in ('neighbor_offsets', 'neighbors', 'topology'))
    one_bad = 0; two_bad = 0
    for v, row in enumerate(neighbors):
        stored = source_neighbors[off[v]:off[v+1]]
        if row != stored[:int(top[v, 0])].tolist():
            one_bad += 1
        wanted = set(row)
        for adjacent in row:
            wanted.update(neighbors[adjacent])
        wanted.discard(v)
        if wanted != set(stored.tolist()):
            two_bad += 1
    repeats = {label:dict(capture_bytes_equal=sha(paths[i]) == sha(paths[i+3]),
                         every_field_bitwise_equal=all(refs[i][key].dtype == refs[i+3][key].dtype
                             and refs[i][key].shape == refs[i+3][key].shape
                             and refs[i][key].tobytes() == refs[i+3][key].tobytes() for key in refs[i]))
               for i, label in enumerate(['before', 'raw', 'smooth'])}
    identical_input_fields = {key:all(refs[0][key].tobytes() == item[key].tobytes() for item in refs[1:])
        for key in ['vertex_fields', 'topology', 'faces', 'neighbor_offsets', 'neighbors', 'hull', 'inside_status', 'border']}
    pdf_passed = all(value['bitwise_equal'] for name, value in fields.items() if 'pdf' in name or 'bin' in name)
    ordered_pass = all(value['bitwise_equal'] for name, value in fields.items() if 'pdf' not in name and 'bin' not in name)
    result = dict(status='complete_readonly_raw_calibration_comparison',
        scope='supplement to frozen nofix/MRI calibration, no new MRI or GA execution',
        candidate_report_sha256=sha(candidate_report),candidate_checkpoint_sha256=sha(candidate),
        surface_sha256=sha(surface),capture_parser_sha256=sha(capture_parser),
        priors_parser_sha256=sha(priors_parser),comparison_script_sha256=sha(__file__),
        ordered_neighbor_module_sha256=sha(inspect.getfile(ordered_neighbors)),
        observer_build_receipt_sha256=sha(observer_build),observer_run_receipt_sha256=sha(observer_receipt),
        reference_capture_sha256={label:sha(path) for label,path in zip(
            ['before','raw','smooth','before_repeat','raw_repeat','smooth_repeat'],paths)},
        prior_creation_capture_sha256=sha(priors_creation),comparisons=fields,
        selected=selected,neighborhood=dict(original_order_vertices=len(neighbors),
            source_vtotal_entries=len(source_neighbors),one_ring_order_different_rows=one_bad,
            two_ring_set_different_rows=two_bad,one_ring_order_equal=one_bad == 0,
            two_ring_sets_equal=two_bad == 0),
        source_repeats=repeats,source_input_fields_stable_across_phases=identical_input_fields,
        source_marked_changes_are_controller_state_not_candidate_output=True,
        original_normals_comparison='not captured in independent candidate checkpoint; no claim',
        complete_pdf_bits_equal=pdf_passed,complete_ordered_captured_inputs_bits_equal=ordered_pass,
        calibration_field_gate_passed=bool(pdf_passed and ordered_pass and selected['set_equal']
            and selected['raw_smooth_order_equal'] and one_bad == 0 and two_bad == 0
            and all(x['every_field_bitwise_equal'] and x['capture_bytes_equal'] for x in repeats.values())),
        internal_matrix_raw_strict_equal=report['strict_full_internal_fields_passed'],
        production_changed=False,full_ga=False,overall_equivalence='not_assessed',
        comparison_wall_seconds=time.perf_counter()-started)
    output.mkdir(parents=True)
    p = output/'report.private.json';p.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=result['status'],
        report_sha256=sha(p)),indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','candidate-report','surface','capture-parser','before','raw','smooth',
                 'before-repeat','raw-repeat','smooth-repeat','priors-creation','priors-parser',
                 'observer-build','observer-receipt','output']:
        p.add_argument('--'+name, type=Path, required=True)
    run(**vars(p.parse_args()))
