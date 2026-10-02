"""Check actual producer TCK/metrics against frozen-PT baseline replay (CPU only)."""
import argparse, hashlib, json
from pathlib import Path
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root-checkpoints',type=Path,required=True);p.add_argument('--component',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def diff(left,right):
 if left.shape!=right.shape or left.dtype!=right.dtype:return {'shape_dtype_exact':False,'left_shape':list(left.shape),'right_shape':list(right.shape),'left_dtype':str(left.dtype),'right_dtype':str(right.dtype)}
 delta=np.abs(left.astype(np.float64)-right.astype(np.float64))
 return {'shape_dtype_exact':True,'neq':int(np.count_nonzero(left!=right)),'max_abs':float(delta.max(initial=0))}
paths=list(nib.streamlines.load(str(a.root_checkpoints/'tracks.tck'),lazy_load=True).streamlines)
counts=np.array([len(path) for path in paths],dtype=np.int64)
offsets=np.concatenate((np.zeros(1,dtype=np.int64),counts.cumsum()))
points=np.concatenate(paths,axis=0)
metrics=np.load(a.root_checkpoints/'track_metrics.npz',allow_pickle=False)
checks={'points':diff(points,np.load(a.component/'points.npy',allow_pickle=False)), 'offsets':diff(offsets,np.load(a.component/'offsets.npy',allow_pickle=False)), 'lengths':diff(metrics['lengths'],np.load(a.component/'lengths_mm.npy',allow_pickle=False)), 'endpoints':diff(metrics['endpoints'],np.load(a.component/'endpoints.npy',allow_pickle=False))}
report={'scope':'Actual root producer trajectory vs frozen-PT baseline replay; not paired candidate speedup or official software precision benchmark','root_paths':len(paths),'root_points':len(points),'checks':checks,'inputs_sha256':{str(path):digest(path) for path in [a.root_checkpoints/'tracks.tck',a.root_checkpoints/'track_metrics.npz',a.component/'report.json']}}
report['strict_passed']=all(item.get('shape_dtype_exact') and item.get('neq')==0 for item in checks.values())
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['strict_passed'] else 1)
