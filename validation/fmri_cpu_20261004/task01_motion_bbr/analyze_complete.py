"""完整 MCFLIRT/BBR 原生进程证明和全体素精度；独立计时外分析。"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def publish(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def array_precision(actual,expected,mask=None):
    import numpy as np
    if actual.shape!=expected.shape:raise ValueError('Complete output shapes differ')
    slices=range(actual.shape[3]) if actual.ndim==4 else (None,)
    rows=[]
    for frame in slices:
        a=np.asarray(actual if frame is None else actual[...,frame],dtype=np.float64)
        b=np.asarray(expected if frame is None else expected[...,frame],dtype=np.float64)
        if mask is not None:a,b=a[mask],b[mask]
        finite=np.isfinite(a)&np.isfinite(b)
        if not finite.all():raise ValueError('Nonfinite full real outputs')
        delta=a-b
        rows.append({'count':a.size,'different':int(np.count_nonzero(delta)),
                     'sum_absolute':float(np.abs(delta).sum(dtype=np.float64)),
                     'sum_squared':float(np.square(delta).sum(dtype=np.float64)),
                     'reference_sum_squared':float(np.square(b).sum(dtype=np.float64)),
                     'maximum_absolute':float(np.abs(delta).max(initial=0))})
    count=sum(row['count'] for row in rows);squared=math.fsum(row['sum_squared'] for row in rows)
    reference=math.fsum(row['reference_sum_squared'] for row in rows)
    return {'values':count,'different_values':sum(r['different'] for r in rows),
            'max_absolute':max(r['maximum_absolute'] for r in rows),
            'mean_absolute':math.fsum(r['sum_absolute'] for r in rows)/count,
            'rmse':math.sqrt(squared/count),'normalized_rmse_by_reference_rms':math.sqrt(squared/reference) if reference else None,
            'all_values_bit_equal':all(r['different']==0 for r in rows),'finite_complete_arrays':True}


def compare_images(actual_path,expected_path,mask_path=None):
    import nibabel as nib
    import numpy as np
    actual=nib.load(actual_path);expected=nib.load(expected_path)
    mask=None
    if mask_path:
        image=nib.load(mask_path)
        if image.shape!=actual.shape[:3] or not np.allclose(image.affine,actual.affine,atol=1e-5,rtol=0):raise ValueError('Held mask differs from output grid')
        mask=np.asanyarray(image.dataobj)>0.5
    # 一次顺序解压完整文件，避免逐帧重新打开 gzip 造成 O(T²) 读取。
    a=np.asanyarray(actual.dataobj);b=np.asanyarray(expected.dataobj)
    result={'shape':list(actual.shape),'complete_shape_equal':actual.shape==expected.shape,
            'actual_stored_dtype':str(actual.get_data_dtype()),'expected_stored_dtype':str(expected.get_data_dtype()),
            'stored_dtype_equal':actual.get_data_dtype()==expected.get_data_dtype(),
            'affine_max_absolute':float(np.max(np.abs(actual.affine-expected.affine))),
            'affine_equal_1e5':bool(np.allclose(actual.affine,expected.affine,atol=1e-5,rtol=0)),
            'zooms_actual':list(map(float,actual.header.get_zooms())),
            'zooms_expected':list(map(float,expected.header.get_zooms())),
            'units_actual':actual.header.get_xyzt_units(),'units_expected':expected.header.get_xyzt_units(),
            'qform_codes':[int(actual.header['qform_code']),int(expected.header['qform_code'])],
            'sform_codes':[int(actual.header['sform_code']),int(expected.header['sform_code'])],
            'header_binary_equal':actual.header.binaryblock==expected.header.binaryblock,
            'whole_grid':array_precision(a,b),
            'actual_file_sha256':sha256(actual_path),'expected_file_sha256':sha256(expected_path)}
    if mask is not None:result['held_real_mask']=array_precision(a,b,mask)
    return result


def trace_proof(trace_path,returncode,source):
    sys.path.insert(0,str(Path(__file__).parent))
    from native_exec import trace_exit_evidence
    parsed=trace_exit_evidence(trace_path,returncode)
    # trace_parser 验证所有 exec/exit/SIGCHLD 链；再单独验证实际 C++ ELF，不凭 wrapper255 或输出存在推断成功。
    paths={};pending={};exit_codes={}
    for line in Path(trace_path).read_text(errors='replace').splitlines():
        match=re.match(r'^(\d+)\s+(.*)$',line)
        if not match:continue
        pid=int(match[1]);event=match[2]
        if event.startswith('execve('):
            name=re.match(r'execve\(("(?:\\.|[^"\\])*")',event)
            if name:
                path=json.loads(name[1])
                if event.endswith('<unfinished ...>'):pending[pid]=path
                elif re.search(r'\)\s+= 0$',event):paths[pid]=path
        elif event.startswith('<... execve resumed>') and re.search(r'\)\s+= 0$',event):
            if pid in pending:paths[pid]=pending.pop(pid)
        elif event.startswith('exit_group('):
            code=re.match(r'exit_group\((-?\d+)',event)
            if code:exit_codes[pid]=int(code[1])%256
    root=parsed['command_root_pid']
    parents=set()
    for line in Path(trace_path).read_text(errors='replace').splitlines():
        match=re.match(r'^(\d+)\s+--- SIGCHLD ',line)
        if match:parents.add(int(match[1]))
    leaves=[]
    for pid,path in paths.items():
        if pid==root or pid in parents:continue
        executable=Path(path)
        leaf={'pid':pid,'exit_code':exit_codes.get(pid),'successful_exec_recorded':True,
              'official_requested_program':Path(paths.get(root,'')).name}
        try:
            with executable.open('rb') as stream:magic=stream.read(4)
            leaf.update(elf_magic_currently_verifiable=magic==b'\x7fELF',binary_sha256=sha256(executable),binary_bytes=executable.stat().st_size)
        except FileNotFoundError:
            leaf.update(elf_magic_currently_verifiable=None,binary_sha256=None,
                        executable_removed_after_exit=True)
        leaves.append(leaf)
    parsed['official_payload_leaves']=leaves
    parsed['complete_payload_exit_zero']=bool(leaves) and all(row['exit_code']==0 for row in leaves)
    parsed['accepted_official_payload_and_complete_chain']=parsed['original_process_accepted'] and parsed['complete_payload_exit_zero']
    parsed['payload_identity_scope']='Successful temporary payload exec/exit and full launcher SIGCHLD chain of the installed official FSL program; cleaned leaf file cannot be hashed after execution. Installed launcher/version/source binding is recorded separately, without rewriting exit255.'
    parsed['trace_sha256']=sha256(trace_path)
    return parsed


def matrices(prefix,function,unrounded=False):
    import numpy as np
    if unrounded and Path(str(prefix)+'.matrices.npy').is_file():return np.load(str(prefix)+'.matrices.npy')
    if function=='bbr':return np.loadtxt(str(prefix)+'.mat')
    return np.stack([np.loadtxt(path) for path in sorted(Path(str(prefix)+'.mat').glob('MAT_*'))])


def compare_prefixes(actual,expected,function,mask=None):
    import numpy as np
    a=matrices(actual,function,unrounded=True);b=matrices(expected,function,unrounded=True)
    result={'matrices':array_precision(a,b),'matrix_count':int(a.shape[0]) if function!='bbr' else 1,
            'images':compare_images(str(actual)+'.nii.gz',str(expected)+'.nii.gz',mask)}
    if function!='bbr':
        p=Path(str(actual)+'.parameters.npy')
        actual_parameters=np.load(p) if p.is_file() else np.loadtxt(str(actual)+'.par')
        p=Path(str(expected)+'.parameters.npy')
        expected_parameters=np.load(p) if p.is_file() else np.loadtxt(str(expected)+'.par')
        result['parameters']=array_precision(actual_parameters,expected_parameters)
        result['parameter_column_max_absolute']=np.max(np.abs(actual_parameters-expected_parameters),axis=0).tolist()
        result['normal_outputs']={'all_mat_frames':a.shape[0]==actual_parameters.shape[0],
                                   'parameters_shape':list(actual_parameters.shape),'corrected_frames':result['images']['shape'][-1]}
        result['rms']={}
        for suffix in ('_abs.rms','_rel.rms','_abs_mean.rms','_rel_mean.rms'):
            ap=Path(str(actual)+suffix);bp=Path(str(expected)+suffix)
            if ap.exists() and bp.exists():result['rms'][suffix]=array_precision(np.atleast_1d(np.loadtxt(ap)),np.atleast_1d(np.loadtxt(bp)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--wait-baseline',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());manifest=plan['manifest'];out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    publish(out/'status.safe.json',{'status':'waiting_for_baseline','analysis_pid':os.getpid()})
    if args.wait_baseline:
        while True:
            status=json.loads(Path(plan['controller_status']).read_text())
            if status.get('status')=='completed':break
            if status.get('status')=='failed':raise RuntimeError('Baseline controller failed; inspect actual outputs before analysis')
            time.sleep(15)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,{int(manifest['cpu_group'][0])})
    with Path(manifest['cpu_lock']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        inputs=json.loads(Path(plan['inputs_file']).read_text());records=[];native=[]
        publish(out/'status.safe.json',{'status':'analyzing_complete_outputs','analysis_pid':os.getpid()})
        for job in manifest['jobs']:
            report=json.loads((Path(job['output_dir'])/'report.safe.json').read_text())
            if job['name'].find('official')<0:continue
            prefix=Path(job['output_dir'])/'repeat_0/result'
            proof=trace_proof(prefix.parent/'native_exec.private.trace',report['native_exit_code'],plan['source_root'])
            native.append({'job':job['name'],'evidence':proof})
            publish(out/'native_exit.safe.json',native)
            if not proof['accepted_official_payload_and_complete_chain']:raise RuntimeError('Native process lacks full successful official payload and launcher chain')
        for job in manifest['jobs']:
            if job['name'].find('fnit')<0:continue
            report=json.loads((Path(job['output_dir'])/'report.safe.json').read_text())
            official=next(value for value in manifest['jobs'] if value['name']==job['name'].replace('fnit','official'))
            function=report['function'];case=report['case'];mask=inputs[case].get('brain_mask') if function!='bbr' else None
            native_prefix=Path(official['output_dir'])/'repeat_0/result'
            for repeat in range(len(report['repeat_records'])):
                prefix=Path(job['output_dir'])/f'repeat_{repeat}/result'
                row={'job':job['name'],'function':function,'case':case,'threads':report['cpu_threads'],'repeat':repeat,
                     'comparison':'frozen_fnit_vs_official_same_budget','precision':compare_prefixes(prefix,native_prefix,function,mask)}
                if repeat>0:row['warm_vs_first']=compare_prefixes(prefix,Path(job['output_dir'])/'repeat_0/result',function,mask)
                records.append(row);publish(out/'precision.safe.json',records)
        publish(out/'status.safe.json',{'status':'completed','analysis_pid':os.getpid(),'native_jobs':len(native),'comparisons':len(records),'full_real_frames':True})
        print(json.dumps({'status':'completed','native_jobs':len(native),'comparisons':len(records)}))

if __name__=='__main__':main()
