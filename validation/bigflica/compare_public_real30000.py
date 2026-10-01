"""Compare independent 30k CPU/GPU stages, including scientific gate failures.

Usage: python compare_public_real30000.py EXPERIMENT_DIR
Only comparison.json, after reviewing any error fields, is for publication.
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


def correlation_matrix(left, right):
    a = np.asarray(left, dtype=np.float64).copy()
    b = np.asarray(right, dtype=np.float64).copy()
    a -= a.mean(0); b -= b.mean(0)
    denominator = np.linalg.norm(a,axis=0)[:,None] * np.linalg.norm(b,axis=0)[None]
    return np.divide(a.T@b,denominator,out=np.zeros_like(denominator),where=denominator>0)


def relative(left, right):
    return float(np.linalg.norm(left-right)/np.linalg.norm(left))


def main():
    root=Path(sys.argv[1])
    config=json.loads((root/'input.json').read_text())
    runs={}
    for name in ('cpu','gpu'):
        aggregate=root/name/'aggregate.json'
        progress=root/name/'progress.json'
        if aggregate.exists() or progress.exists():
            runs[name]=json.loads((aggregate if aggregate.exists() else progress).read_text())
        else:
            runs[name]={'status':'initialization_failed','stage_timings_s':{},'wall_s':None,'peak_rss_gib':None}
    report={'subjects':30000,'modalities':['vbm','fa','md'],
            'source_commit':config['source_commit'],'subjects_sha256':config['subjects_sha256'],
            'actual_execution_order':config.get('actual_execution_order',config['execution_order']),
            'backend_status':{name:run['status'] for name,run in runs.items()},
            'stage_timings_s':{name:run['stage_timings_s'] for name,run in runs.items()},
            'wall_s':{name:run['wall_s'] for name,run in runs.items()},
            'peak_rss_gib':{name:run['peak_rss_gib'] for name,run in runs.items()},
            'peak_gpu_allocated_gib':runs['gpu'].get('peak_gpu_allocated_gib'),
            'scientific_gate':{name:run.get('flica_diagnostics') for name,run in runs.items()},
            'same_input_signature':runs['cpu'].get('input_signature')==runs['gpu'].get('input_signature') and runs['cpu'].get('input_signature') is not None,
            'conditions':'Same gpucw1, eight CPU threads, sequential execution (order recorded separately), independent raw-image caches; CPU float64 and GPU float32 mMIGP; uncontrolled OS page cache; shared GPU load. No controlled speedup claim.'}
    left_dir=root/'cpu/output'; right_dir=root/'gpu/output'
    left_u=left_dir/'mmigp_100/U.npy'; right_u=right_dir/'mmigp_100/U.npy'
    if left_u.exists() and right_u.exists():
        a=np.load(left_u).astype(np.float64); b=np.load(right_u).astype(np.float64)
        signs=np.where(np.sum(a*b,axis=0)<0,-1.,1.)
        report['mmigp']={'u_shape':list(a.shape),'raw_relative_error':relative(a,b),
                         'sign_aligned_relative_error':relative(a,b*signs),
                         'same_index_absolute_correlations':np.abs(np.diag(correlation_matrix(a,b))).tolist(),
                         'projected':{}}
        report['dicl']={}
        for name in config['modalities']:
            pa=left_dir/'mmigp_100'/f'{name}_projected.h5'; pb=right_dir/'mmigp_100'/f'{name}_projected.h5'
            if pa.exists() and pb.exists():
                numerator=denominator=0.
                with h5py.File(pa) as fa,h5py.File(pb) as fb:
                    for start in range(0,fa['data'].shape[0],2048):
                        x=fa['data'][start:start+2048].astype(np.float64)
                        y=fb['data'][start:start+2048].astype(np.float64)*signs
                        numerator+=float(np.square(x-y).sum()); denominator+=float(np.square(x).sum())
                report['mmigp']['projected'][name]={'sign_aligned_relative_error':float(np.sqrt(numerator/denominator))}
            da=list(left_dir.glob(f'dicl_*/*{name}_dictionary.npy')); db=list(right_dir.glob(f'dicl_*/*{name}_dictionary.npy'))
            if len(da)==len(db)==1:
                x=np.load(da[0]); y=np.load(db[0])*signs
                cosines=(x/np.linalg.norm(x,axis=1)[:,None])@(y/np.linalg.norm(y,axis=1)[:,None]).T
                rows,cols=linear_sum_assignment(-np.abs(cosines))
                matched=y[cols]*np.sign(cosines[rows,cols])[:,None]
                report['dicl'][name]={'matched_relative_error':relative(x,matched),
                                      'matched_absolute_cosine_min':float(np.abs(cosines[rows,cols]).min()),
                                      'matched_absolute_cosine_median':float(np.median(np.abs(cosines[rows,cols])))}
    if all(run['status']=='complete' for run in runs.values()):
        a=left_dir/'components_20_lambda_R'; b=right_dir/'components_20_lambda_R'
        x=np.load(a/'subj_course.npy'); y=np.load(b/'subj_course.npy')
        corr=correlation_matrix(x,y)
        rows,cols=linear_sum_assignment(-np.abs(corr))
        signs=np.sign(corr[rows,cols])
        report['course']={'matched_absolute_correlations':np.abs(corr[rows,cols]).tolist()}
        report['maps']={}
        for name in config['modalities']:
            x=np.load(a/f'{name}_zstat.npy'); y=np.load(b/f'{name}_zstat.npy')[:,cols]*signs
            report['maps'][name]={'matched_signed_correlations':np.diag(correlation_matrix(x,y)).tolist(),
                                  'relative_error':relative(x,y)}
        report['status']='complete'
    else:
        report['status']='scientific_acceptance_incomplete'
        report['maps']='C20 course/map comparison requires both rank gates to pass; no silent reduction of C.'
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
