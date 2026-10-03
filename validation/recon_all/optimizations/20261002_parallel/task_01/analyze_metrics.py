import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
import nibabel.freesurfer.io as fs
from fnit.recon_all.compare_subject import _numeric
parser=argparse.ArgumentParser(description='真实maps配对/串行重复容差控制和脑图；不运行影像pipeline')
parser.add_argument('--pair-ab',type=Path,required=True)
parser.add_argument('--pair-ba',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
a=json.loads((args.pair_ab/'result.json').read_text());b=json.loads((args.pair_ba/'result.json').read_text())

checks={}
for comparison,left,right in [('serial_vs_parallel_AB',args.pair_ab/'serial/subject',args.pair_ab/'parallel/subject'),('serial_vs_parallel_BA',args.pair_ba/'serial/subject',args.pair_ba/'parallel/subject'),('serial_repeat_AB_BA',args.pair_ab/'serial/subject',args.pair_ba/'serial/subject')]:
 checks[comparison]={}
 for hemi in ('lh','rh'):
  for metric in ('thickness','area','area.pial','area.mid','curv','curv.pial','volume'):
   x=fs.read_morph_data(left/f'surf/{hemi}.{metric}');y=fs.read_morph_data(right/f'surf/{hemi}.{metric}')
   checks[comparison][f'{hemi}.{metric}']=_numeric(x,y,a['numeric_tolerances'][metric],50)
report={'scope':'existing operator tolerances; maps on frozen matching ordered meshes only','source_commit':a['commit'],'strict_reproduction':'failed','numeric_regression':'passed' if all(c['status']=='passed' for rows in checks.values() for c in rows.values()) else 'failed','new_degradation':'not_assessed_globally','overall_equivalence':'not_assessed','checks':checks,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(args.output/'metrics_numeric_control.json').write_text(json.dumps(report,indent=2))
print(report['numeric_regression'])
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import Normalize
 fig=plt.figure(figsize=(10,4))
 for index,hemi in enumerate(('lh','rh')):
  xyz,faces=fs.read_geometry(args.pair_ab/f'serial/subject/surf/{hemi}.pial')
  x=fs.read_morph_data(args.pair_ab/f'serial/subject/surf/{hemi}.curv.pial')
  y=fs.read_morph_data(args.pair_ab/f'parallel/subject/surf/{hemi}.curv.pial')
  ax=fig.add_subplot(1,2,index+1,projection='3d')
  take=np.arange(len(xyz))[::3]
  sc=ax.scatter(xyz[take,0],xyz[take,1],xyz[take,2],c=np.abs(x-y)[take],s=.1,cmap='inferno',norm=Normalize(0,.00012),rasterized=True)
  ax.view_init(elev=0,azim=180 if hemi=='lh' else 0);ax.set_axis_off();ax.set_box_aspect(np.ptp(xyz,axis=0));ax.set_title(hemi+' pial curvature absolute difference')
 fig.colorbar(sc,ax=fig.axes,shrink=.55,label='mm^-1');fig.savefig(args.output/'metrics_difference.png',dpi=200,bbox_inches='tight');plt.close(fig)
 print('figure generated')
except ImportError:
 print('matplotlib unavailable; no brain visualization generated')
