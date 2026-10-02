"""从真实私有结果画三平面脑图；只保存到用户指定私有目录，不发布影像。

--original-check/--candidate-check 为同目标网格的uint8最近邻结果，
--original-forward/--candidate-forward 为FS RAS毫米位移；--output为PNG。
参考只供诊断。展示固定中间切片，位移误差色限1e-4mm，不排除脑外。
"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['original-check','candidate-check','original-forward','candidate-forward','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ref=nib.load(a.original_check);cand=nib.load(a.candidate_check)
    if ref.shape!=cand.shape or not np.allclose(ref.affine,cand.affine,rtol=0,atol=1e-6):raise ValueError('check grids differ')
    r,c=np.asarray(ref.dataobj),np.asarray(cand.dataobj)
    f,g=nib.load(a.original_forward),nib.load(a.candidate_forward)
    if f.shape!=g.shape or f.shape[:3]!=ref.shape:raise ValueError('field grids differ')
    err=np.sqrt(np.sum((np.asarray(f.dataobj,dtype=np.float64)[:,:,:,0,:]-np.asarray(g.dataobj,dtype=np.float64)[:,:,:,0,:])**2,axis=-1))
    fig,axes=plt.subplots(4,3,figsize=(10,11),constrained_layout=True)
    arrays=[r,c,(r!=c).astype(np.uint8),err]
    labels=['Frozen check','Candidate check','Different voxels','RAS error (mm)']
    for axis in range(3):
        index=ref.shape[axis]//2
        for row,data in enumerate(arrays):
            image=axes[row,axis].imshow(np.take(data,index,axis=axis).T,origin='lower',cmap='gray' if row<2 else 'magma',vmin=0,vmax=255 if row<2 else (1 if row==2 else 1e-4))
            axes[row,axis].set_title(f'{labels[row]}\nvoxel axis {axis}, slice {index}');axes[row,axis].axis('off')
            if row==3:fig.colorbar(image,ax=axes[row,axis],shrink=.7)
    fig.suptitle('Real MNI outputs: full-grid comparison, no masking')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);plt.close(fig)
if __name__=='__main__':main()
