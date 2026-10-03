"""保存残差诊断所用冻结脑掩膜的哈希和空间；不读取影像内容参与生产。

--config同benchmark.py；--output元数据JSON。只核查orig和brainmask几何，
记录brainmask>0节点数，与残差的脑内范围定义相同。非性能测量。
"""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np
from benchmark import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());report={'script_sha256':sha(__file__),'not_a_performance_measurement':True,'definition':'frozen brainmask.mgz > 0; full-grid residual never masked','cases':{}}
    for case in cfg['cases']:
        mri=Path(case['subject'])/'mri';path=mri/'brainmask.mgz';image=nib.load(path);original=nib.load(mri/'orig.mgz')
        error=float(np.abs(image.affine-original.affine).max())
        if image.shape!=original.shape or error>1e-4:raise ValueError('brainmask/orig grids differ')
        report['cases'][case['id']]={'brainmask_sha256':sha(path),'bytes':path.stat().st_size,'shape':list(map(int,image.shape)),'affine':image.affine.tolist(),'affine_maximum_vs_orig':error,'brain_nodes':int(np.count_nonzero(np.asarray(image.dataobj)>0))}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
