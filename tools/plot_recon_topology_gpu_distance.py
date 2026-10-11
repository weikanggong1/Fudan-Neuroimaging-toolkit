"""真实局部MRI、自产CPU距离及GPU差图；不替代完整脑表面质量QC。"""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ['inputs','schema','cpu-distance','gpu-distance','output']:p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    schema=json.loads(args.schema.read_text())['mapping']
    key=schema['volume_inputs']['mapping']['defect_intensity_volume']['array']
    with np.load(args.inputs,allow_pickle=False)as z:volume=z[key].copy()
    cpu=np.load(args.cpu_distance,allow_pickle=False);gpu=np.load(args.gpu_distance,allow_pickle=False)
    if not(volume.shape==cpu.shape==gpu.shape):raise ValueError('same XYZ volume grid required')
    fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
    for axis in range(3):
        take=lambda a:np.take(a,a.shape[axis]//2,axis=axis).T
        m=take(volume);d=take(cpu);delta=take(gpu-cpu)
        for column in range(3):axes[axis,column].imshow(m,cmap='gray',origin='lower');axes[axis,column].set_xticks([]);axes[axis,column].set_yticks([])
        overlay=axes[axis,1].imshow(np.ma.masked_where(cpu.take(cpu.shape[axis]//2,axis=axis).T==1000,d),cmap='coolwarm',vmin=-3,vmax=3,alpha=.72,origin='lower')
        error=axes[axis,2].imshow(delta,cmap='coolwarm',vmin=-1e-5,vmax=1e-5,origin='lower')
        axes[axis,0].set_ylabel(['X midpoint','Y midpoint','Z midpoint'][axis])
    for ax,title in zip(axes[0],['Real local MRI','CPU signed distance (mm)','GPU minus CPU (mm)']):ax.set_title(title)
    fig.colorbar(overlay,ax=axes[:,1],shrink=.65);fig.colorbar(error,ax=axes[:,2],shrink=.65)
    fig.suptitle('Real fourth topology defect: local MRI and complete signed distance\nOpenNeuro ds000114 (CC0); same voxel grid; no whole-surface QC claim',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160);plt.close(fig)
if __name__=='__main__':main()
