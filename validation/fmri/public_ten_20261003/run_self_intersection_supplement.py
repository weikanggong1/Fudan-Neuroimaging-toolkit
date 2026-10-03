#!/usr/bin/env python3
"""以新独立预算补测已完成正式病例的自交；复用冻结原生 predicate，不改 MRI。"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posthoc-case', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--waiter-pid', type=int, required=True)
    parser.add_argument('--pair-budget', type=int, default=200000000)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    original_path = args.posthoc_case / 'report.public.json'
    original = json.loads(original_path.read_text())
    bindings = json.loads((args.posthoc_case / 'files.private.json').read_text())
    if original['status'] not in ('measured', 'partially_measured') or not original['inputs_unchanged'] or not original['code_unchanged']:
        raise ValueError('Original posthoc evidence was not completed with intact guards')
    if original['candidate_source_revision'] != '1128bc52c7a0233266e5b8a8d7dc0b382994e676':
        raise ValueError('Only formal v4 source is accepted')
    if sha(args.comparator) != original['code_sha256']['wrapper']:
        raise ValueError('Self worker is not the exact frozen original comparator')
    paths = {'original_posthoc': original_path, 'original_binding': args.posthoc_case/'files.private.json',
             'comparator': args.comparator, 'own_helper': Path(__file__)}
    for key in ('candidate_report', 'reference_report', 'raw_t1w'):
        path = Path(bindings[key])
        if sha(path) != original['input_sha256'][key]:
            raise ValueError('Original input binding changed')
        paths[key] = path
    source_manifest = json.loads((paths['candidate_report'].parent/'source.private.json').read_text())
    for rel, expected in source_manifest.items():
        path = args.source / rel
        if sha(path) != expected:
            raise ValueError('Actual completed candidate source manifest mismatch')
        paths['fnit_source/'+rel] = path
    meshes = []
    for chain in ('reference', 'candidate'):
        subject = Path(bindings[chain+'_subject'])
        for hemi in ('lh','rh'):
            for kind in ('white','pial'):
                key = chain+'/'+hemi+'.'+kind
                path = subject/'surf'/(hemi+'.'+kind)
                expected = original['quality'][chain][hemi]['input_sha256'][kind]
                if sha(path) != expected:
                    raise ValueError('Current saved mesh differs from original posthoc')
                vertices, faces = fs.read_geometry(str(path))
                paths[key] = path
                meshes.append((key, path, len(vertices), len(faces), hashlib.sha256(np.asarray(faces,dtype='<i4').tobytes()).hexdigest()))
    before = {key:sha(path) for key,path in paths.items()}
    state = dict(case_id=original['case_id'], status='waiting_for_CPU4_gate',
                 original_posthoc_sha256=before['original_posthoc'], original_posthoc_status=original['status'],
                 candidate_source_revision=original['candidate_source_revision'], helper_sha256=before['own_helper'],
                 comparator_sha256=before['comparator'], threads=4, device='cpu',
                 budget=dict(maximum_raw_bounding_sphere_pairs=args.pair_budget, hard_timeout_seconds_per_worker=args.timeout),
                 input_sha256=before, results={},
                 timing_scope='Independent saved-mesh self-intersection supplement; excludes wait and is not production MRI whole wall',
                 scope='Exact frozen comparator worker and native triangle predicate; ordered original saved faces; shared-vertex pairs excluded; no geometry repair and no equivalence threshold changed')
    save(args.output_root/'report.public.json',state)
    proc = Path('/proc')/str(args.waiter_pid)
    def validate_waiter():
        cmd = (proc/'cmdline').read_bytes()
        if b'posthoc_reconstruction_v7_code/run_reconstruction_qc.py' not in cmd or b'candidate_v4' not in cmd:
            raise ValueError('Refuse to signal a different process')
    def children():
        return (proc/'task'/str(args.waiter_pid)/'children').read_text().strip()
    # 暂停只读等待父进程时必须无 worker，以免改变已有比较进程的记录边界。
    while True:
        validate_waiter()
        if children():
            time.sleep(20)
            continue
        os.kill(args.waiter_pid,signal.SIGSTOP)
        if children():
            os.kill(args.waiter_pid,signal.SIGCONT)
            time.sleep(20)
            continue
        break
    state.update(waiter_paused_only_without_children=True, queue_paused_utc=datetime.now(timezone.utc).isoformat(), status='running')
    save(args.output_root/'report.public.json',state)
    run = None
    def abort(signum, frame):
        raise KeyboardInterrupt('Supplement interrupted')
    signal.signal(signal.SIGTERM,abort)
    signal.signal(signal.SIGINT,abort)
    start = time.perf_counter()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
            env[key]='4'
        env['NUMBA_CACHE_DIR']=str(args.output_root/'numba_cache')
        for key,path,nv,nf,faces_sha in meshes:
            rowpath=args.output_root/(key.replace('/','-')+'.public.json')
            cmd=[sys.executable,str(args.comparator),'--self-intersection-worker',str(path),
                 '--source',str(args.source),'--threads','4','--pair-budget',str(args.pair_budget),'--output',str(rowpath)]
            tick=time.perf_counter()
            with (args.output_root/(key.replace('/','-')+'.private.log')).open('wb') as log:
                run=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                try:
                    code=run.wait(timeout=args.timeout)
                    row=json.loads(rowpath.read_text()) if code==0 and rowpath.exists() else dict(status='failed_worker',exit_code=code)
                except subprocess.TimeoutExpired:
                    run.kill();run.wait()
                    row=dict(status='incomplete_time_budget',timeout_seconds=args.timeout)
                finally:
                    run=None
            row.update(process_wall_seconds=time.perf_counter()-tick, mesh_sha256=before[key],
                       vertices=nv, faces=nf, ordered_faces_sha256=faces_sha)
            if row.get('source_sha256') and row['source_sha256'] != sha(args.source/'src/fnit/recon_all/mris_remove_intersection_python.py'):
                raise ValueError('Actual native predicate source changed')
            state['results'][key]=row
            save(args.output_root/'report.public.json',state)
        after={key:sha(path) for key,path in paths.items()}
        state.update(input_after_sha256=after,inputs_unchanged=before==after,
                     diagnostic_active_wall_seconds=time.perf_counter()-start,
                     status='measured' if all(v['status']=='measured' for v in state['results'].values()) and before==after else 'partially_measured_or_guard_failed')
    except BaseException:
        state.update(status='failed_interrupted_or_guard',diagnostic_active_wall_seconds=time.perf_counter()-start)
        raise
    finally:
        if run is not None and run.poll() is None:
            run.kill();run.wait()
        os.kill(args.waiter_pid,signal.SIGCONT)
        state.update(waiter_resumed=True,queue_resumed_utc=datetime.now(timezone.utc).isoformat())
        save(args.output_root/'report.public.json',state)
    print(json.dumps(dict(status=state['status'],sha256=sha(args.output_root/'report.public.json'),seconds=state['diagnostic_active_wall_seconds'])))


if __name__=='__main__':
    main()
