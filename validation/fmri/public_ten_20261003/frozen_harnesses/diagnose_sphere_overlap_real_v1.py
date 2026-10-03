"""只读检查真实 CON07 球面残留负面，并在同例 graymid 上标出位置。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def surface(path):
    if str(path).endswith('.gii'):
        image = nib.load(path)
        points = image.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
        triangles = image.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
        if len(points) != 1 or len(triangles) != 1:
            raise ValueError('the saved GIFTI must contain one pointset and one triangle array')
        return np.asarray(points[0].data, np.float64), np.asarray(triangles[0].data, np.int64)
    points, faces = fsio.read_geometry(str(path))
    return np.asarray(points, np.float64), np.asarray(faces, np.int64)


def geometric_metrics(points, faces, native_area):
    triangle = points[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    determinant = (cross * triangle[:, 0]).sum(1)
    area = np.linalg.norm(cross, axis=1) * .5
    bad = np.flatnonzero(determinant < 0)
    signed_area = np.where(determinant < 0, -area, area)
    edges = np.linalg.norm(np.roll(triangle[bad], -1, axis=1) - triangle[bad], axis=2)
    count32 = int(np.count_nonzero(native_area < 0))
    return {
        'vertices': len(points), 'faces': len(faces), 'finite': bool(np.isfinite(points).all()),
        'strict_FP64_outward_negative_faces': int(len(bad)),
        'strict_FP64_zero_determinant_faces': int(np.count_nonzero(determinant == 0)),
        'native_FP32_signed_area_negative_faces': count32,
        'native_FP32_zero_area_faces': int(np.count_nonzero(native_area == 0)),
        'native_FP32_and_FP64_negative_face_sets_exact': bool(np.array_equal(
            np.flatnonzero(native_area < 0), bad)),
        'negative_face_indices': bad.tolist(),
        'negative_vertex_indices': np.unique(faces[bad]).tolist(),
        'minimum_signed_determinant_mm3': float(determinant.min()),
        'negative_area_sum_mm2': float(-signed_area[signed_area < 0].sum()),
        'unsigned_total_area_mm2': float(area.sum()),
        'negative_area_fraction_of_unsigned_area': float(-signed_area[signed_area < 0].sum() / area.sum()),
        'native_FP32_negative_area_sum_mm2': float(-native_area[native_area < 0].astype(np.float64).sum()),
        'negative_face_unsigned_area_mm2_range': [float(area[bad].min()), float(area[bad].max())] if len(bad) else None,
        'negative_face_edge_length_mm_range': [float(edges.min()), float(edges.max())] if len(bad) else None,
        'sphere_radius_mm_range': [float(np.linalg.norm(points, axis=1).min()), float(np.linalg.norm(points, axis=1).max())],
    }


def cleanup_metrics(value, saved_count):
    history = value['negative_counts']
    return {'update_count_before_cleanup': len(value['updates']),
            'cleanup_pre_step_history_count': len(history),
            'first_pre_step_negative_count': history[0] if history else 0,
            'minimum_pre_step_negative_count': min(history) if history else 0,
            'last_pre_step_negative_count': history[-1] if history else 0,
            'saved_final_negative_count': saved_count,
            'cleanup_history_includes_final_count': False,
            'cleanup_finish_seconds': value.get('finish_seconds'),
            'scope': 'History is measured before each update; saved final counts are checked independently. A nonzero last history entry also occurs when the final update succeeds.'}


def draw_geometry(axes, points, faces, bad, *, zoom):
    # Direct same-subject surface-RAS projection onto anatomical y/z axes.
    # Every triangle and highlighted vertex comes from the actual ordered graymid.
    xy = points[:, [1, 2]]
    ids = np.unique(faces[bad])
    center = xy[ids].mean(0)
    if zoom:
        half = np.maximum(np.ptp(xy[ids], axis=0) * .7, 6.)
        low, high = center - half, center + half
        selected = ((xy[faces].max(axis=1) >= low) & (xy[faces].min(axis=1) <= high)).all(1)
        shown = faces[selected]
    else:
        low, high = xy.min(0) - 3., xy.max(0) + 3.
        shown = faces
    axes.add_collection(PolyCollection(xy[shown], facecolors='#d8dde1', edgecolors='#acb3b9', linewidths=.025,
                                       alpha=.45, rasterized=True))
    axes.add_collection(PolyCollection(xy[faces[bad]], facecolors='#c82127', edgecolors='#a00000',
                                       linewidths=.7 if zoom else .2, alpha=.95, rasterized=True))
    axes.scatter(xy[ids, 0], xy[ids, 1], s=12 if zoom else 4, c='#f4a21c', edgecolors='#712700',
                 linewidths=.25, zorder=5)
    axes.set_xlim(low[0], high[0]); axes.set_ylim(low[1], high[1]); axes.set_aspect('equal')
    axes.set_xlabel('Surface RAS y (mm)'); axes.set_ylabel('Surface RAS z (mm)')
    axes.grid(alpha=.15)
    if not zoom:
        from matplotlib.patches import Rectangle
        half = np.maximum(np.ptp(xy[ids], axis=0) * .7, 6.)
        axes.add_patch(Rectangle(center-half, *(2*half), fill=False, edgecolor='#252525', linewidth=.8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--upstream-source', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise RuntimeError('this read-only diagnostic requires hidden CUDA and CPU execution')
    torch.set_num_threads(1)
    manifest = json.loads(args.data_manifest.read_text())
    case = next(value for value in manifest['subjects'] if value['subject'] == 'CON07')
    if manifest['license'] != 'CC0' or case['complete_original_frames'] != 180:
        raise ValueError('only the verified public CC0 complete CON07 case is permitted')
    run_report = args.case_root/'report/report.public.json'
    report = json.loads(run_report.read_text())
    revision = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
    if report['status'] != 'complete' or report['source_revision'] != revision:
        raise ValueError('requires the saved complete frozen1128 formal case')
    for name, key in [('T1w','t1w'),('BOLD','bold')]:
        if report['input_sha256'][key] != case[name]['sha256']:
            raise ValueError('the actual formal raw identity differs from the public manifest')
    args.output_root.mkdir(mode=0o700, exist_ok=False)
    files = json.loads((args.case_root/'report/files.private.json').read_text())
    subject = Path(files['recon_all'])
    paths = {'diagnostic_source':Path(__file__),'source_manifest':args.data_manifest,
             'formal_report':run_report,'formal_files':args.case_root/'report/files.private.json',
             'native_run_report':subject/'fnit-native-free-run.json','production_QC':Path(files['qc_report']),
             'upstream_smoothing_source':args.upstream_source/'utils/mrisurf_timeStep.cpp'}
    for module in ['mris_register_overlap.py','mris_register_nonlinear.py','sphere_standard_finish.py',
                   'sphere_standard_run.py','sphere_standard_python.py','sphere_python.py','mris_register_kernels.py']:
        paths['frozen_source/'+module] = args.source_root/'src/fnit/recon_all'/module
    paths['frozen_source/msmsulc.py'] = args.source_root/'src/fnit/msm/msmsulc.py'
    for hemi in ['lh','rh']:
        for key in ['sphere','sphere.reg','graymid']:
            paths[hemi+'/'+key] = subject/'surf'/f'{hemi}.{key}'
    for hemi,name in [('L','lh'),('R','rh')]:
        paths[name+'/MSM'] = Path(files['left' if hemi=='L' else 'right']).with_name(
            Path(files['left' if hemi=='L' else 'right']).name.split('_space-fsLR_')[0]
            +'_space-fsLR_desc-preprocReg_sphere.surf.gii')
    before = {name:sha256(path) for name,path in paths.items()}
    tick = time.perf_counter()
    sys.path.insert(0, str(args.source_root/'src'))
    from fnit.recon_all.mris_register_nonlinear import face_area_normals
    if Path(sys.modules[face_area_normals.__module__].__file__).resolve() != paths['frozen_source/mris_register_nonlinear.py'].resolve():
        raise ValueError('the actual native signed-area predicate was imported from another source')
    meshes, metrics = {}, {}
    for hemi in ['lh','rh']:
        display, display_faces = surface(paths[hemi+'/graymid'])
        for name in ['sphere','sphere.reg','MSM']:
            points, faces = surface(paths[hemi+'/'+name])
            if not np.array_equal(faces, display_faces) or len(points) != len(display):
                raise ValueError('same-case graymid and saved sphere ordered topology differ')
            native_area,_ = face_area_normals(torch.from_numpy(points.astype(np.float32)),
                torch.from_numpy(faces),signed_sphere=True)
            metrics[hemi+'/'+name] = geometric_metrics(points,faces,native_area.numpy())
            meshes[hemi+'/'+name] = (display,faces,np.asarray(metrics[hemi+'/'+name]['negative_face_indices']))
    native = json.loads(paths['native_run_report'].read_text())
    cleanup = {}
    for hemi in ['lh','rh']:
        standard = native['hemisphere_scheduling']['groups'][0]['values'][hemi]['result']['standard_sphere_report']
        cleanup[hemi+'/sphere'] = cleanup_metrics(standard,metrics[hemi+'/sphere']['native_FP32_signed_area_negative_faces'])
        smoothwm = native['sphere_registration']['reports'][hemi]['smoothwm_pass']
        cleanup[hemi+'/sphere.reg'] = cleanup_metrics(smoothwm,metrics[hemi+'/sphere.reg']['native_FP32_signed_area_negative_faces'])
    figure, axes = plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    for row,(name,label) in enumerate([('sphere','Native sphere'),('sphere.reg','FreeSurfer-compatible registered sphere'),
                                      ('MSM','Saved MSMSulc sphere')]):
        points,faces,bad = meshes['lh/'+name]
        if not len(bad): raise ValueError('the requested diagnostic anomaly is absent')
        for column in [0,1]:
            draw_geometry(axes[row,column],points,faces,bad,zoom=bool(column))
            axes[row,column].set_title(f'{label}: {len(bad)} strictly negative faces\n'+
                ('Same-CON07 graymid, whole LH projection' if column==0 else 'Direct same-face/vertex local view'))
    figure.suptitle('Public CC0 CON07, frozen1128: actual residual negative faces on its own graymid\n'
        'Red = actual negative faces; orange = their actual vertices; no alignment or MRI recomputation',fontsize=12)
    image = args.output_root/'CON07_native_sphere_negative_faces.png'
    figure.savefig(image,dpi=160);plt.close(figure)
    after = {name:sha256(path) for name,path in paths.items()}
    if before != after:raise ValueError('readonly input, source, formal result or upstream implementation changed')
    qc = json.loads(paths['production_QC'].read_text())['MSM']['Report']
    result = {'status':'complete','scope':'Independent CPU posthoc diagnostic and display; no reconstruction/MSM rerun, fitting, normalization, original mesh modification or MRI timing change.',
        'dataset':'OpenNeuro ds001226 v5.0.1','public_dataset_url':'https://openneuro.org/datasets/ds001226/versions/5.0.1',
        'license':'CC0','subject':'CON07','scientific_execution_revision':revision,
        'raw_sha256':{k:case[k]['sha256'] for k in ['T1w','BOLD']},'metrics':metrics,'cleanup':cleanup,
        'production_relative_MSM_QC':{h:{k:v for k,v in qc[h].items() if k in ['folded_output_faces','folded_solver_faces',
            'minimum_output_orientation_ratio','minimum_solver_orientation_ratio','degenerate_input_faces']} for h in ['L','R']},
        'count_definitions':{'strict_outward':'cross(b-a,c-a) dot a < 0 in float64 on actual saved coordinates; absolute orientation.',
            'native_predicate':'Actual frozen FNIT face_area_normals signed_sphere=True on saved float32 coordinates, CPU.',
            'production_relative':'Production MSM divides output signed determinant by normalized rotated native input determinant. This is a different baseline, not the absolute outward face count.',
            'cleanup_history':'Pre-update counts, excludes independently checked saved final state.'},
        'upstream_contract':{'source_file':'utils/mrisurf_timeStep.cpp','source_sha256':before['upstream_smoothing_source'],
            'initialization_lines':[2417,2424],'progress_limit_lines':[2470,2472],'iteration_limit_lines':[2486,2487],
            'residual_return_lines':[2493,2501],
            'conclusion':'The fixed original implementation uses min_neg_iter=0 and returns NO_ERROR after limited cleanup with remaining negative triangles. No source-compatible numerical bug is established by this residual alone.'},
        'same_case_graymid_faces_exact':True,'input_sha256_before':before,'input_sha256_after':after,'input_guards_equal':True,
        'actual_native_predicate_module_bound':True,'CUDA_visible_devices':'','actual_execution_device':'CPU',
        'diagnostic_and_plot_seconds':time.perf_counter()-tick,'figure_sha256':sha256(image),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'nibabel_version':nib.__version__,'torch_version':torch.__version__}
    (args.output_root/'report.public.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_root/'files.private.json').write_text(json.dumps({'paths':{k:str(v) for k,v in paths.items()},'image':str(image)},indent=2)+'\n')
    print('READONLY_CON07_SPHERE_DIAGNOSTIC_COMPLETE',result['diagnostic_and_plot_seconds'],flush=True)


if __name__ == '__main__':
    main()
