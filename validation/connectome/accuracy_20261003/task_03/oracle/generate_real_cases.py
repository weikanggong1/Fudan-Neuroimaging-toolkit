from pathlib import Path
import argparse, hashlib, json, numpy as np, nibabel as nib
p=argparse.ArgumentParser(); p.add_argument('--images',type=Path,required=True);p.add_argument('--tracks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
image=nib.load(args.images/'gmwmi.nii.gz'); weights=np.asarray(image.dataobj,dtype=np.float32); positive=np.flatnonzero(weights.ravel()>0); rng=np.random.default_rng(20261003)
flat=rng.choice(positive,size=10000,p=weights.ravel()[positive].astype(np.float64)/weights.ravel()[positive].sum(dtype=np.float64)); voxel=np.array(np.unravel_index(flat,weights.shape)).T+rng.uniform(-.5,.5,(len(flat),3)); seeds=(voxel @ image.affine[:3,:3].T+image.affine[:3,3]).astype(np.float32)
tck=nib.streamlines.load(args.tracks,lazy_load=True); segments=[]
for i,track in enumerate(tck.streamlines):
 if i>=1000:break
 if len(track)<4:continue
 for j in (1,len(track)//2,len(track)-2):
  direction=track[j+1]-track[j];direction/=np.linalg.norm(direction)
  tangent=rng.normal(size=3); tangent-=tangent.dot(direction)*direction;tangent/=np.linalg.norm(tangent)
  angle=rng.uniform(0,np.pi/4); end_dir=direction*np.cos(angle)+tangent*np.sin(angle)
  segments.append(np.r_[track[j],direction,end_dir].astype(np.float32))
segments=np.array(segments,dtype=np.float32)
with (args.output/'cases.txt').open('w') as f:
 for q in seeds:f.write('G '+' '.join(format(float(x),'.9g') for x in q)+'\n')
 for q in segments:f.write('A '+' '.join(format(float(x),'.9g') for x in q)+'\n')
np.savez(args.output/'cases.npz',gmwmi_candidates=seeds,arc_cases=segments)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(args.output/'provenance.json').write_text(json.dumps({'dataset':'OpenNeuro ds001226 CON03 real official fixed-input images and tracks','rng':{'numpy':20261003},'gmwmi_candidates':len(seeds),'arc_cases':len(segments),'source_sha256':sha(Path(__file__)),'case_sha256':sha(args.output/'cases.txt'),'input_sha256':{str(x):sha(x) for x in (args.images/'gmwmi.nii.gz',args.tracks)},'definition':'Fixed random voxel candidates from real GMWMI weights; real streamline positions with fixed cone arcs. Diagnostic only, not tractography benchmark.'},indent=2)+'\n')
