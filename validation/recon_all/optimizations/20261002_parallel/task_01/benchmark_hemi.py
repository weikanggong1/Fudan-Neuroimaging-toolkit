"""真实冻结检查点的配对阶段回归；输出不包含真实影像。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import torch
from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
from fnit.recon_all.native_free import _hemisphere_operation, _run_surface_metrics
from fnit.recon_all.profiling import ProcessTreeDeviceSampler, configure_cuda_allocator
from fnit.recon_all.thread_budget import thread_budget


def metrics(subject, hemi, device, threads, operation, *, assets, binaries, **ignored):
    from fnit.recon_all.surface_area_gpu import mid_area_map
    from fnit.recon_all.surface_roi_gpu import vertex_volume_map
    subject = Path(subject)
    result = _run_surface_metrics(Path(binaries['metrics']), subject, hemi, Path(assets), device=device)
    mid_area_map(subject / f'surf/{hemi}.area', subject / f'surf/{hemi}.area.pial',
                 subject / f'surf/{hemi}.area.mid', device=device)
    vertex_volume_map(subject / f'surf/{hemi}.white', subject / f'surf/{hemi}.pial',
                      subject / f'label/{hemi}.cortex.label', subject / f'surf/{hemi}.volume', device=device)
    return {'result': result}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_array(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'shape_equal': a.shape == b.shape, 'dtype_equal': a.dtype == b.dtype,
            'different_elements': int(np.count_nonzero(a != b)),
            'max_abs': float(delta.max(initial=0)), 'p99_abs': float(np.quantile(delta,.99)) if delta.size else 0.}


def compare(a, b, paths):
    rows = []
    for relative in sorted(set(paths)):
        left, right = a / relative, b / relative
        row = {'file': relative, 'present': left.is_file() and right.is_file()}
        if not row['present']:
            rows.append(row); continue
        row.update(serial_sha256=sha(left), parallel_sha256=sha(right))
        row['bytes_equal'] = row['serial_sha256'] == row['parallel_sha256']
        name = left.name
        if relative.startswith('scripts/'):
            row['kind'] = 'execution_log'
        else:
            try:
                if name.endswith(('.mgz','.mgh')):
                    ai, bi = nib.load(left), nib.load(right)
                    row.update(compare_array(np.asarray(ai.dataobj), np.asarray(bi.dataobj)),
                               affine_max_abs=float(np.max(np.abs(ai.affine-bi.affine))), kind='volume')
                elif name.endswith('.annot'):
                    ai, ac, an = fs.read_annot(left); bi, bc, bn = fs.read_annot(right)
                    row.update(compare_array(ai,bi), color_table_equal=bool(np.array_equal(ac,bc)),
                               names_equal=an==bn, kind='annotation',
                               per_label_dice={str(code): float(2*np.count_nonzero((ai==code)&(bi==code))/
                                  (np.count_nonzero(ai==code)+np.count_nonzero(bi==code)))
                                  for code in np.union1d(ai,bi)})
                else:
                    try:
                        ai, af = fs.read_geometry(left); bi, bf = fs.read_geometry(right)
                        row.update(ordered_faces_equal=bool(np.array_equal(af,bf)), kind='mesh')
                        if not np.array_equal(af,bf) or ai.shape != bi.shape:
                            row['ordered_comparable'] = False
                            row['numeric_status'] = 'not_assessed_topology_differs'
                        else:
                            row.update(compare_array(ai,bi), ordered_comparable=True)
                    except (ValueError, UnicodeDecodeError):
                        ai, bi = fs.read_morph_data(left), fs.read_morph_data(right)
                        row.update(compare_array(ai,bi), kind='vertex_map')
            except Exception as error:
                row.update(kind='other', numeric_status='not_assessed', diagnostic=type(error).__name__)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--operation', choices=('metrics','annotation','register','surface','finish_surface'), default='metrics')
    parser.add_argument('--order', choices=('AB','BA'), default='AB')
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # 严格复现门槛预先声明为0；沿用项目正式数值容差，不修改整体等效门槛。
    tolerance = Path(__file__).resolve().parents[5] / 'tests/recon_all/tolerances_numeric.json'
    report = {'commit': args.commit, 'measured_source_diff': 'see measured_source_delta.patch when present', 'checkpoint': str(args.checkpoint), 'operation': args.operation,
              'order': args.order, 'total_threads': 4, 'strict_tolerance': 0,
              'numeric_tolerances': json.loads(tolerance.read_text()), 'overall_equivalence':'not_assessed',
              'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[5])):sha(p)
                 for p in (Path(__file__).resolve().parents[5]/'src/fnit').rglob('*.py')},
              'input_sha256': {str(p.relative_to(args.checkpoint)):sha(p)
                 for folder in ('mri','surf','label') for p in (args.checkpoint/folder).rglob('*') if p.is_file()},
              'binary_sha256': {p.name:sha(p) for p in args.binaries.iterdir() if p.is_file()},
              'results': {}}
    # 素材授权及实际 SHA 在独立审计报告记录；仅使用既有文件，不下载。
    binaries = {k: str(args.binaries/name) for k,name in
        (('metrics','mris_place_surface'),('topology','mris_fix_topology_fnit'),
         ('inflate','mris_inflate'),('intersection','mris_remove_intersection'),
         ('defect','mri_label2vol'),('paint','mrisp_paint'))}
    common = {'assets': str(args.assets), 'binaries': binaries,
              'registration_atlases': {h:str(args.assets/'average'/f'{h}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif') for h in ('lh','rh')}}
    configure_cuda_allocator(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with thread_budget(threads=4):
        # 显式初始化父 CUDA API，验证 exec 不继承其上下文。
        parent_tensor = torch.zeros(1,device=args.device)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        for mode in (('serial','parallel') if args.order == 'AB' else ('parallel','serial')):
            subject = args.output/mode/'subject'
            subject.parent.mkdir(exist_ok=True)
            shutil.copytree(args.checkpoint,subject)
            sampler = ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid())
            stop = threading.Event()
            def sample():
                while not stop.is_set():
                    sampler.sample_if_due()
                    stop.wait(.5)
            monitor = threading.Thread(target=sample,daemon=True)
            monitor.start()
            started = time.monotonic()
            try:
                if mode == 'serial':
                    values={}
                    for hemi in ('lh','rh'):
                        function=metrics if args.operation=='metrics' else _hemisphere_operation
                        values[hemi]=function(subject=str(subject),hemi=hemi,device=args.device,
                                              threads=4,operation=args.operation,**common)
                    result={'values':values,'group_wall_seconds':time.monotonic()-started}
                    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
                else:
                    result=run_hemisphere_group(subject,args.operation,device=args.device,threads=4,
                            workers=2,profile_stages=True,kwargs=common,
                            callable_path='benchmark_hemi:metrics' if args.operation=='metrics'
                              else 'fnit.recon_all.native_free:_hemisphere_operation')
                result['command_wall_seconds']=time.monotonic()-started
                if args.operation=='surface':
                    from fnit.recon_all.native_free import _run_defects_volume
                    for h in ('lh','rh'):_run_defects_volume(Path(binaries['defect']),subject,h,args.assets)
                report['results'][mode]=result
            finally:
                stop.set();monitor.join();sampler.sample_if_due(force=True)
                report['results'].setdefault(mode,{})['external_monitor']=sampler.report()
                (args.output/'result.json').write_text(json.dumps(report,indent=2))
    paths=report['results']['parallel'].get('published',[])
    report['output_differences']=compare(args.output/'serial/subject',args.output/'parallel/subject',paths)
    report['strict_reproduction']='passed' if all(row.get('bytes_equal',False) for row in report['output_differences'] if row.get('kind')!='execution_log') else 'failed'
    report['speedup']=report['results']['serial']['command_wall_seconds']/report['results']['parallel']['command_wall_seconds']
    report['checkpoint_unchanged']=all(sha(args.checkpoint/p)==value for p,value in report['input_sha256'].items())
    (args.output/'result.json').write_text(json.dumps(report,indent=2))
    with (args.output/'differences.csv').open('w') as stream:
        columns=['file','kind','present','bytes_equal','different_elements','max_abs','p99_abs','ordered_faces_equal','affine_max_abs']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(report['output_differences'])
    print(json.dumps({k:report[k] for k in ('operation','speedup','strict_reproduction','checkpoint_unchanged')}))


if __name__=='__main__':main()
