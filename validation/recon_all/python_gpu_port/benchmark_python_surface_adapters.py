"""真实冻结自产输入的生产放置适配器回归；参考输出只用于诊断比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import nibabel as nib
import numpy as np
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(a: Path, b: Path) -> dict:
    x, f = nib.freesurfer.read_geometry(str(a)); y, g = nib.freesurfer.read_geometry(str(b))
    ordered = x.shape == y.shape and np.array_equal(f, g)
    distances = np.linalg.norm(x-y, axis=1) if ordered else None
    return dict(ordered_faces=ordered, reference_vertices=len(x), candidate_vertices=len(y),
        different_coordinate_elements=int(np.count_nonzero(x!=y)) if ordered else None,
        mean_mm=float(distances.mean()) if ordered else None,
        p99_mm=float(np.percentile(distances,99)) if ordered else None,
        max_mm=float(distances.max()) if ordered else None,
        reference_sha256=sha(a),candidate_sha256=sha(b))


def volume(a: Path, b: Path) -> dict:
    x,y=nib.load(str(a)),nib.load(str(b));u,v=np.asarray(x.dataobj),np.asarray(y.dataobj)
    same=u.shape==v.shape;delta=np.abs(u.astype(np.float64)-v.astype(np.float64)) if same else None
    return dict(shape_equal=same,dtype_equal=u.dtype==v.dtype,affine_equal=np.array_equal(x.affine,y.affine),
        different_voxels=int(np.count_nonzero(delta)) if same else None,
        max_error=float(delta.max()) if same else None,p99_error=float(np.percentile(delta,99)) if same else None,
        reference_sha256=sha(a),candidate_sha256=sha(b))


def publish(path: Path, report: dict) -> None:
    temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-subject',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--code-base-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--reference-binary',type=Path,required=True,
                        help='仅在隔离参考目录运行的固定源码构建 mris_place_surface')
    parser.add_argument('--reference-assets',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    from fnit.recon_all import surface_placement_python as api
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.white_preaparc_conda import run_white_preaparc
    from fnit.recon_all.final_white_conda import run_final_white
    from fnit.recon_all.pial_t1_conda import run_pial_t1
    native_runners={'preaparc':run_white_preaparc,'white':run_final_white,'pial':run_pial_t1}
    root=Path(api.__file__).parents[1]
    report=dict(scope='frozen own full-stage inputs; not a continuous chain or raw T1 run',
        code_base_commit=args.code_base_commit,source_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')},
        script_sha256=sha(Path(__file__)),host=platform.node(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads=args.threads,device=args.device,precision=dict(dtype='float32 plus declared float64 geometry',autocast=False),
        reference_program_sha256=sha(args.reference_binary),reference_repeats=1,
        stages={},status='running',strict_reproduction='not_assessed',overall_metric_equivalence='not_assessed')
    receipt=args.output_directory/'report.json';publish(receipt,report)
    tick=perf_counter()
    try:
        with thread_budget(threads=args.threads):
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            report['precision'].update(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
            for hemi in ('lh','rh'):
                for stage, runner, surface, diagnostic in (
                    ('preaparc',api.run_preaparc_python,'white.preaparc','mrisps.wpa.mgz'),
                    ('white',api.run_final_white_python,'white','mrisps.white.mgz'),
                    ('pial',api.run_pial_python,'pial.T1',None)):
                    dest=args.output_directory/f'{hemi}-{stage}'
                    for name in ('mri','surf','label','scripts'):(dest/name).mkdir(parents=True)
                    inputs=['mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz',
                            f'surf/{hemi}.orig',f'surf/{hemi}.orig.premesh'] if stage=='preaparc' else [
                            'mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz',f'surf/autodet.gw.stats.{hemi}.dat',
                            f'surf/{hemi}.white.preaparc',f'surf/{hemi}.white',
                            f'label/{hemi}.cortex.label',f'label/{hemi}.cortex+hipamyg.label',f'label/{hemi}.aparc.annot']
                    hashes={name:sha(args.source_subject/name) for name in inputs}
                    for name in inputs:shutil.copy2(args.source_subject/name,dest/name)
                    # 每侧/阶段有独立MRI，避免将整例RH末次覆盖的mrisps体积当作LH参考。
                    reference=args.output_directory/f'{hemi}-{stage}-reference'
                    for name in ('mri','surf','label','scripts'):(reference/name).mkdir(parents=True)
                    for name in inputs:shutil.copy2(args.source_subject/name,reference/name)
                    reference_started=perf_counter()
                    reference_result=native_runners[stage](subject_dir=reference,hemi=hemi,
                        binary=args.reference_binary,assets_dir=args.reference_assets,threads=args.threads)
                    reference_seconds=perf_counter()-reference_started
                    attempts=[];popen=subprocess.Popen
                    def denied(*items,**kwargs):
                        attempts.append(repr(items[0]) if items else 'unknown')
                        raise RuntimeError('production adapter attempted an external process')
                    if torch.device(args.device).type=='cuda':torch.cuda.synchronize(args.device)
                    if torch.device(args.device).type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
                    started=perf_counter();subprocess.Popen=denied
                    try:result=runner(subject_dir=dest,hemi=hemi,device=args.device)
                    finally:subprocess.Popen=popen
                    if torch.device(args.device).type=='cuda':torch.cuda.synchronize(args.device)
                    elapsed=perf_counter()-started
                    assert hashes=={name:sha(args.source_subject/name) for name in inputs}
                    row=dict(input_sha256=hashes,input_unchanged=True,external_process_attempts=attempts,
                        wall_seconds_including_write_and_synchronization=elapsed,result=result,
                        reference_result=reference_result,reference_wall_seconds_including_write=reference_seconds,
                        surface=geometry(reference/f'surf/{hemi}.{surface}',dest/f'surf/{hemi}.{surface}'))
                    if torch.device(args.device).type=='cuda':
                        row['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
                        row['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
                        row['process_tree_gpu_memory']='not_measured; Torch memory is not total process memory'
                    if diagnostic:row['volume']=volume(reference/f'mri/{diagnostic}',dest/f'mri/{diagnostic}')
                    if stage=='preaparc':row['threshold_table_bytes_equal']=sha(reference/f'surf/autodet.gw.stats.{hemi}.dat')==sha(dest/f'surf/autodet.gw.stats.{hemi}.dat')
                    report['stages'][f'{hemi}_{stage}']=row;publish(receipt,report)
        report['status']='complete'
        report['same_input_exact']=all(v['surface']['different_coordinate_elements']==0 and v['surface']['ordered_faces'] and ('volume' not in v or v['volume']['different_voxels']==0 and v['volume']['affine_equal'] and v['volume']['dtype_equal']) for v in report['stages'].values())
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['total_diagnostic_seconds']=perf_counter()-tick;publish(receipt,report)


if __name__=='__main__':main()
