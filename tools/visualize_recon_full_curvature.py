"""真实同索引smoothwm投影：参考H、旧BE误差及修复后BE误差，不改变比较范围。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface',type=Path,required=True)
    p.add_argument('--reference-h',type=Path,required=True)
    p.add_argument('--reference-be',type=Path,required=True)
    p.add_argument('--previous-be',type=Path,required=True)
    p.add_argument('--candidate-be',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--title',required=True)
    a=p.parse_args()
    xyz,faces=fsio.read_geometry(str(a.surface))
    h,reference,previous,candidate=[fsio.read_morph_data(str(f)).astype(np.float64)
      for f in (a.reference_h,a.reference_be,a.previous_be,a.candidate_be)]
    if any(v.shape!=(len(xyz),) for v in (h,reference,previous,candidate)):raise ValueError('vertex correspondence differs')
    before,after=abs(previous-reference),abs(candidate-reference)
    maximum=int(before.argmax());vmax=float(before.max()) or 1.0
    order=np.argsort(xyz[:,0])[::-1]
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.3),constrained_layout=True)
    plots=[(h,'Reference H (display +/-1 mm^-1)','coolwarm',-1,1),
      (before,f'Previous BE error: max {before.max():.6g} mm^-2','magma',0,vmax),
      (after,f'Own complete BE: max {after.max():.6g} mm^-2','magma',0,vmax)]
    for axis,(values,title,cmap,low,high) in zip(axes,plots):
      display_order=order if title.startswith('Reference') else np.argsort(values,kind='stable')
      display=axis.scatter(xyz[display_order,1],xyz[display_order,2],c=values[display_order],s=.24,cmap=cmap,vmin=low,vmax=high,rasterized=True)
      if not title.startswith('Reference'):
        axis.scatter(xyz[maximum,1],xyz[maximum,2],c=[values[maximum]],s=11,cmap=cmap,vmin=low,vmax=high)
      axis.scatter(xyz[maximum,1],xyz[maximum,2],s=70,facecolor='none',edgecolor='lime',linewidth=1.1)
      axis.set_aspect('equal');axis.set_title(title,fontsize=10)
      axis.set_xlabel('surface RAS y (mm)');axis.set_ylabel('surface RAS z (mm)')
      fig.colorbar(display,ax=axis,shrink=.8)
    fig.suptitle(a.title+' | same-index projection, no excluded vertices',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);plt.close(fig)
    receipt={'vertices':len(xyz),'faces':len(faces),'before_max_abs_BE':float(before.max()),
      'after_max_abs_BE':float(after.max()),'highlight_vertex':maximum,'surface_RAS_mm':xyz[maximum].tolist(),
      'display_H_clipped_range':[-1,1],'error_display_range':[0,vmax],
      'comparison_scope':'all corresponding vertices; projection only, not T1 overlay or mesh-quality acceptance',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'inputs_sha256':{role:hashlib.sha256(f.read_bytes()).hexdigest() for role,f in
          zip(('surface','reference_H','reference_BE','previous_BE','candidate_BE'),
              (a.surface,a.reference_h,a.reference_be,a.previous_be,a.candidate_be))}}
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
