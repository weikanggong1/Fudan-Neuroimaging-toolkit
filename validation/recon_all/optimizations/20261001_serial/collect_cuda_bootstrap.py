"""只读收集CUDA初始化对照的报告，逐元素比较原网格影像和LTA矩阵。

--root为实际运行根目录，含cuda_bootstrap_diag和bootstrap_0..5.json。
--output为新的tar.gz；不写生产输出，不打包任何影像、权重或许可证。
在root/cuda_bootstrap_summary.json写六次耗时、实际前向、数组/几何/dtype
差异及脚本/配置/程序版本。对照采用同case首个current，无整体等效阈值。
MGZ按完整数组，RAS LTA按4×4矩阵比较，不把路径元信息差异当坐标差异。
缺失或失败输入直接报错，保留已有报告。无独立官方等价CLI。
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import nibabel as nib
import numpy as np


def matrix(path):
    lines=path.read_text().splitlines();start=lines.index('1 4 4')+1
    return np.asarray([[float(v) for v in line.split()] for line in lines[start:start+4]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=sorted((a.root/'cuda_bootstrap_diag').glob('*/diagnostic.json'))
    if len(reports)!=6:raise ValueError('six completed diagnostic reports required')
    base={};rows=[]
    for file in reports:
        x=json.loads(file.read_text())
        if x['status']!='complete':raise ValueError(file)
        case=Path(x['config']['input']).name;subject=file.parent
        if case not in base:base[case]=subject
        comparisons={}
        for name in ['orig.mgz','synthstrip.mgz']:
            ref=nib.load(base[case]/'mri'/name);cand=nib.load(subject/'mri'/name)
            ra=np.asarray(ref.dataobj);ca=np.asarray(cand.dataobj)
            d=np.abs(ca.astype(np.float64)-ra.astype(np.float64))
            comparisons[name]={'different_voxels':int(np.count_nonzero(d)),
                'maximum':float(d.max()),'same_geometry':bool(np.array_equal(ref.affine,cand.affine)),
                'same_dtype':str(ra.dtype)==str(ca.dtype),'shape':list(ca.shape),'dtype':str(ca.dtype)}
        name='mri/transforms/synthmorph.mni305/aff.lta'
        ra=matrix(base[case]/name);ca=matrix(subject/name)
        comparisons['aff.lta']={'different_elements':int(np.count_nonzero(ra!=ca)),
                               'maximum':float(np.abs(ra-ca).max())}
        rows.append({'name':subject.name,'seconds':x['seconds'],'code_commit':x['config']['code_commit'],
            'script_sha256':x['script_sha256'],'prime_child':x['config']['prime_child'],
            'comparisons':comparisons,'child_gpu':x['result']['talairach_child_gpu']})
    summary={'scope':'six monitored raw-T1 input-chain replays; not complete recon-all',
        'rows':rows,'conclusion':'both original and explicitly primed entries completed; no controlled reproduction or proven fix',
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary_path=a.root/'cuda_bootstrap_summary.json'
    if summary_path.exists() or a.output.exists():raise FileExistsError('existing report')
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    with tarfile.open(a.output,'x:gz') as archive:
        archive.add(summary_path,arcname='summary.json')
        for file in reports:
            archive.add(file,arcname=file.parent.name+'/diagnostic.json')
            monitor=Path(str(file.parent)+'_monitor')
            for name in ['monitor.json','gpu_samples.csv','command.log']:
                archive.add(monitor/name,arcname=file.parent.name+'/'+name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
