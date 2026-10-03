"""绘制真实remesh几何对照；PNG仅保存在受控私有输出，不提交影像。"""
import argparse,hashlib,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel.freesurfer.io as fsio
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 av,af=fsio.read_geometry(str(a.baseline));bv,bf=fsio.read_geometry(str(a.candidate))
 if av.shape!=bv.shape or not np.array_equal(af,bf):raise ValueError('visual displacement requires corresponding ordered topology')
 d=np.linalg.norm(av-bv,axis=1);faces=bf[::12];fig=plt.figure(figsize=(12,4),layout='constrained')
 extent=np.ptp(bv,axis=0);center=(bv.max(axis=0)+bv.min(axis=0))/2;radius=max(extent)/2
 for idx,(vertices,title) in enumerate([(av,'Frozen FNIT baseline'),(bv,'Optimized FNIT'),(bv,'Vertex displacement (mm)')]):
  ax=fig.add_subplot(1,3,idx+1,projection='3d');poly=Poly3DCollection(vertices[faces],linewidths=0)
  if idx==2:
   colors=plt.cm.viridis(d[faces].mean(axis=1)/max(float(d.max()),1e-6));poly.set_facecolor(colors)
  else:poly.set_facecolor('#cccccc')
  ax.add_collection3d(poly);ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.set_box_aspect((1,1,1));ax.view_init(elev=5,azim=180);ax.set_axis_off();ax.set_title(title)
 fig.suptitle(f'Ordered faces identical; max displacement {d.max():.3g} mm')
 a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);plt.close(fig)
 meta={'source_commit':a.commit,'coordinate_and_ordered_faces_equal':bool(np.array_equal(av,bv)),'max_displacement_mm':float(d.max()),'png_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'privacy':'PNG is a private real-data derived-brain artifact; never redistribute it in repository','display_sampling':'every twelfth face for private display only; metrics use every vertex and ordered face'}
 a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
