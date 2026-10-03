"""只读对照同输入放置报告；证明有序网格来源后比较坐标及逐脑区局部误差。"""
import argparse, json
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True); a=p.parse_args()
summary={'overall_equivalence':'not_assessed','scope':'same_frozen_input_operator_diagnostic','hemispheres':{}}
for h in ('lh','rh'):
 hemi={}
 for kind in ('prewhite','white','pial'):
  reports={}
  for backend in ('official','conda','python'):
   path=a.root/f'{h}_{kind}_{backend}/report.json'
   if path.is_file(): reports[backend]=json.loads(path.read_text())
  if 'official' not in reports: hemi[kind]={'status':'pending_official','available':list(reports)}; continue
  ref=reports['official']; refxyz,reffaces=fs.read_geometry(ref['output']['path'])
  result={'status':'partial' if len(reports)<(3 if kind=='pial' else 2) else 'complete',
          'python_full_white':'unavailable' if kind!='pial' else 'available','comparisons':{}}
  for backend,report in reports.items():
   if ref['input_sha256']!=report['input_sha256']: raise ValueError(f'input hashes differ: {h}/{kind}/{backend}')
   xyz,faces=fs.read_geometry(report['output']['path'])
   if xyz.shape!=refxyz.shape or not np.array_equal(faces,reffaces): raise ValueError('same-index comparison refused: ordered topology mismatch')
   if not report['output']['ordered_faces_unchanged']: raise ValueError('output no longer corresponds to copied original mesh')
   d=np.linalg.norm(xyz-refxyz,axis=1)
   labels,_,names=fs.read_annot(str(Path(report['output']['path']).parents[1]/f'label/{h}.aparc.annot'))
   regions={}
   for i,name in enumerate(names):
    mask=labels==i
    if mask.any(): regions[name.decode()]={'count':int(mask.sum()),'mean_mm':float(d[mask].mean()),'p99_mm':float(np.quantile(d[mask],.99)),'max_mm':float(d[mask].max())}
   edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])); edges.sort(axis=1)
   unique,counts=np.unique(edges,axis=0,return_counts=True)
   graph=coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(len(xyz),len(xyz)))
   components,_=connected_components(graph,directed=False)
   result['comparisons'][backend]={'correspondence_proof':'identical complete input hashes + unchanged ordered faces + placement-only operators',
    'different_coordinate_components':int(np.count_nonzero(xyz!=refxyz)),'mean_mm':float(d.mean()),'p99_mm':float(np.quantile(d,.99)),
    'max_mm':float(d.max()),'regional_displacement':regions,'largest_error_vertex_ids':np.argsort(d)[-20:][::-1].tolist(),
    'mesh_quality':{'connected_components':int(components),'boundary_edges':int(np.count_nonzero(counts==1)),
     'nonmanifold_edges':int(np.count_nonzero(counts>2)), 'self_intersection':'not_assessed_by_this_read_only_report',
     'white_pial_triangle_crossing':'not_assessed_by_this_read_only_report'},'seconds':report['seconds_including_io'],
    'rounds':report.get('native_rounds',report.get('stage',{}).get('pass_ends'))}
  hemi[kind]=result
 summary['hemispheres'][h]=hemi
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2)+'\n')
