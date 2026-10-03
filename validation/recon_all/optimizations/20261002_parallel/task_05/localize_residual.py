"""非计时CPU诊断：全域及脑内逆场残差最大点与私有三平面图。

--forward/--inverse 为相同输入的FS位移，--brainmask 为同源网格brainmask>0；
--original 只给私有图灰度背景，--output为JSON，--figure为私有PNG。
与GPU residual相同矩阵/插值顺序；不修改生产结果、不屏蔽全域统计。
"""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from benchmark import sha
from fnit.recon_all.ca_register_inverse import read_warp_geometries,_inverse_4x4_native
from fnit.recon_all.mni_warp_sampling import _grid,_affine,_linear_absolute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['forward','inverse','brainmask','original','output','figure']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    f,i=nib.load(a.forward),nib.load(a.inverse);source,atlas,shape=read_warp_geometries(f)
    delta=torch.as_tensor(np.ascontiguousarray(np.asarray(f.dataobj,np.float32)[:,:,:,0,:].transpose(3,0,1,2)))
    coords=torch.empty_like(delta)
    for lo in range(0,f.shape[0],16):
        hi=min(lo+16,f.shape[0]);g=_grid(f.shape[:3],lo,hi,'cpu')
        coords[:,lo:hi]=_affine(_inverse_4x4_native(source),(_affine(atlas,g).double()+delta[:,lo:hi].double()).float())
    inv=np.asarray(i.dataobj,np.float32)[:,:,:,0,:];errors=np.empty(shape,np.float32);outside=0
    for lo in range(0,shape[0],16):
        hi=min(lo+16,shape[0]);g=_grid(shape,lo,hi,'cpu');d=torch.as_tensor(np.ascontiguousarray(inv[lo:hi].transpose(3,0,1,2)))
        points=_affine(_inverse_4x4_native(atlas),(_affine(source,g).double()+d.double()).float());mapped,valid=_linear_absolute(coords,points)
        errors[lo:hi]=(_affine(source,mapped).double()-_affine(source,g).double()).square().sum(0).sqrt().numpy();outside+=int((~valid).sum())
    mask_image=nib.load(a.brainmask);mask=np.asarray(mask_image.dataobj)>0
    original=nib.load(a.original)
    if mask.shape!=shape or original.shape!=shape or not np.allclose(original.affine,mask_image.affine,atol=1e-4,rtol=0):raise ValueError('mask/original grids differ')
    def describe(values,index):
        return {'maximum_mm':float(values.max()),'p99_mm':float(np.quantile(values,.99)),'mean_mm':float(values.mean()),'voxels':int(values.size),'over_0_1_mm_diagnostic':int((values>.1).sum()),'maximum_source_voxel':list(map(int,index)),'maximum_scanner_RAS_mm':(source@np.array([*index,1.],np.float64))[:3].tolist()}
    full=np.unravel_index(errors.argmax(),shape);brain=np.unravel_index(np.where(mask,errors,-1).argmax(),shape)
    report={'device':'cpu','threads':1,'not_a_performance_measurement':True,'script_sha256':sha(__file__),'input_sha256':{str(x):sha(x) for x in [a.forward,a.inverse,a.brainmask,a.original]},'full_grid':describe(errors,full),'brain':describe(errors[mask],brain),'inverse_sample_outside_forward_grid':outside,'excluded_voxels':0,'diagnostic_bin_not_acceptance_threshold':True}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=np.asarray(original.dataobj);fig,axes=plt.subplots(3,3,figsize=(10,10),constrained_layout=True)
    for axis in range(3):
        plane=int(brain[axis]);remaining=[d for d in range(3) if d!=axis]
        for row,data in enumerate([values,errors,np.where(mask,errors,np.nan)]):
            im=axes[row,axis].imshow(np.take(data,plane,axis=axis).T,origin='lower',cmap='gray' if row==0 else 'magma',vmin=0,vmax=255 if row==0 else (250 if row==1 else .3))
            axes[row,axis].plot(brain[remaining[0]],brain[remaining[1]],'c+',markersize=9)
            axes[row,axis].set_title(f'{["Original","Full-grid error mm","Brain view mm (scale 0–0.3)"][row]}\naxis {axis}, slice {plane}');axes[row,axis].axis('off')
            if row>0:fig.colorbar(im,ax=axes[row,axis],shrink=.7)
    fig.suptitle(f'Frozen inverse residual: brain maximum {errors[brain]:.3f} mm at {tuple(map(int,brain))}\nFull-grid statistics exclude no voxels; brain color saturation is display only')
    a.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.figure,dpi=170);plt.close(fig)
if __name__=='__main__':main()
