"""真实表面差异脑图仅写指定私有目录；影像/表面/脑图不进入公开报告。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,nibabel.freesurfer.io as fs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
reference,faces=fs.read_geometry(str(a.reference));candidate,cf=fs.read_geometry(str(a.candidate));np.testing.assert_array_equal(faces,cf);assert candidate.shape==reference.shape
error=np.linalg.norm(reference-candidate,axis=1);face_error=error[faces].mean(axis=1)
fig=plt.figure(figsize=(10,5))
for index,azimuth in enumerate((-90,90),1):
 ax=fig.add_subplot(1,2,index,projection='3d');norm=plt.Normalize(0,max(.001,float(np.quantile(error,.99))))
 mesh=Poly3DCollection(candidate[faces],linewidths=0);mesh.set_facecolor(plt.cm.viridis(norm(face_error)));ax.add_collection3d(mesh)
 low=candidate.min(axis=0);high=candidate.max(axis=0);center=(low+high)/2;radius=(high-low).max()/2
 ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius);ax.view_init(elev=0,azim=azimuth);ax.set_axis_off();ax.set_title('surface displacement / mm')
fig.suptitle(f'Ordered correspondence; max={error.max():.6g} mm, P99={np.quantile(error,.99):.6g} mm');fig.tight_layout();fig.savefig(a.output/'brain_difference_private.png',dpi=160);plt.close(fig)
(a.output/'metadata.json').write_text(json.dumps(dict(scope='private real brain visualization; do not commit image or input surfaces',reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),vertices=len(candidate),faces=len(faces),max_mm=float(error.max()),p99_mm=float(np.quantile(error,.99))),indent=2)+'\n')
