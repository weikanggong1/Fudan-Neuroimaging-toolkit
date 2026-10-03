"""真实forward的散射边界诊断；非性能测量，禁止据此推断求逆一致。

--config沿用benchmark配置，--output为元数据JSON；逐例读取同一冻结forward，
统计native初次夹取后rint边界拒绝、旧size-1夹取影响节点数。CPU单线程。
"""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np
from benchmark import sha
from fnit.recon_all.ca_register_inverse import read_warp_geometries,warp_to_source_voxels


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());report={'code_commit':cfg['code_commit'],'script_sha256':sha(__file__),'not_a_performance_measurement':True,'cases':{}}
    for case in cfg['cases']:
        path=Path(case['subject'])/'mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.nii.gz'
        image=nib.load(path);source,atlas,shape=read_warp_geometries(image)
        coords=warp_to_source_voxels(np.asarray(image.dataobj,np.float32)[:,:,:,0,:],atlas,source)
        reject=np.zeros(coords.shape[:3],bool);changed=np.zeros_like(reject)
        for axis,n in enumerate(shape):
            q=coords[...,axis].astype(np.float64)
            changed|=(q>n-1)&(q<n)
            native=np.where(q<0,0,np.where(q>=n,n-1,q))
            reject|=np.rint(native)>=n
        report['cases'][case['id']]={'input_sha256':sha(path),'source_shape':shape,'nodes':int(reject.size),'native_rint_rejected_nodes':int(reject.sum()),'nodes_with_old_early_clamp':int(changed.sum()),'all_finite':bool(np.isfinite(coords).all())}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
