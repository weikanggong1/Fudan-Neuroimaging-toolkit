import argparse,hashlib,json,pathlib
import numpy as np
import nibabel.freesurfer.io as fsio
parser=argparse.ArgumentParser()
for name in ('candidate','reference','output'):parser.add_argument('--'+name,type=pathlib.Path,required=True)
args=parser.parse_args()
a,af=fsio.read_geometry(str(args.candidate));b,bf=fsio.read_geometry(str(args.reference))
assert np.array_equal(af,bf)
u,_,vt=np.linalg.svd(a.T@b);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt);rotation=u@correction@vt
raw=np.linalg.norm(a-b,axis=1);aligned=np.linalg.norm(a@rotation-b,axis=1)
def summary(x):return {'mean_mm':float(x.mean()),'p99_mm':float(np.percentile(x,99)),'max_mm':float(x.max())}
d={'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'input_sha256':{name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ('candidate','reference')},'purpose':'orientation diagnostic only; raw comparison remains unchanged and no acceptance threshold added','operation':'one proper orthogonal rotation, no translation or scaling, all corresponding vertices','raw':summary(raw),'after_rotation':summary(aligned),'rotation_matrix':rotation.tolist(),'rotation_angle_degrees':float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),'overall_equivalence':'not_assessed'}
args.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
