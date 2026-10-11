"""从两条自产prefix生成私有真实脑图，不导出影像或原始/派生数值数组。"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
fig,axes=plt.subplots(4,9,figsize=(24,11),layout='constrained')
for row,name in enumerate(('tmp/nu0','brain','wm','filled')):
 images=[nib.load(str(root/'mri'/f'{name}.mgz')) for root in (a.reference,a.candidate)]
 first,second=[np.asarray(image.dataobj) for image in images]
 if first.shape!=second.shape or not np.array_equal(images[0].affine,images[1].affine):
  raise ValueError('same-grid visualization required; no implicit resampling')
 for axis in range(3):
  index=first.shape[axis]//2
  slices=[np.take(data,index,axis=axis).T for data in (first,second)]
  delta=np.abs(slices[1].astype(float)-slices[0].astype(float))
  for col,(values,title) in enumerate(zip((*slices,delta),('strict reference','Torch N4 candidate','absolute difference'))):
   panel=axes[row,axis*3+col]
   panel.imshow(values,origin='lower',cmap='magma' if col==2 else 'gray',vmin=0,vmax=max(1,float(delta.max())) if col==2 else 255)
   panel.set_title(f'{name} axis{axis} {title}',fontsize=8);panel.axis('off')
fig.suptitle('Private real-data Python prefix comparison; quantitative gates are recorded separately')
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=140);plt.close(fig)
(a.output.with_suffix('.receipt.public.json')).write_text(json.dumps({'scope':'private_real_brain_visualization_only','image_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'image_size':a.output.stat().st_size},indent=2)+'\n')
