"""绘制已完成同输入inflation的公开表面与误差图；仅用于结果展示。

--data为公开smoothwm清单，--pair为完整配对结果，--output为新目录。
候选算法不读取本脚本/参考图。图使用完整顶点的矢状投影，无表面重采样；
输入、配对报告、绘图代码和PNG记录SHA。图不能代替三维自相交检查。
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    pair=json.loads((args.pair/'summary.json').read_text())
    if pair['status']!='complete_stage_pair':raise ValueError('stage pair must be complete')
    args.output.mkdir(parents=True)
    entries=json.loads((args.data/'manifest.json').read_text())['cases']
    figure,axes=plt.subplots(len(entries),3,figsize=(11,3*len(entries)),squeeze=False)
    report={'scope':'full_vertex_sagittal_projection_not_intersection_QC',
        'pair_summary_sha256':sha(args.pair/'summary.json'),'source_sha256':sha(__file__),'rows':[]}
    for row,entry in enumerate(entries):
        source=args.data/entry['surface']
        if sha(source)!=entry['sha256']:raise ValueError('input hash changed')
        hemi=entry['hemisphere'];directory=args.pair/entry['case']/hemi
        original,faces=fsio.read_geometry(str(source))
        reference,reference_faces=fsio.read_geometry(str(directory/'native_1'/(hemi+'.inflated')))
        candidate,candidate_faces=fsio.read_geometry(str(directory/'torch_2'/(hemi+'.inflated')))
        if not np.array_equal(faces,reference_faces) or not np.array_equal(faces,candidate_faces):
            raise ValueError('same vertex correspondence must be established')
        error=np.linalg.norm(candidate-reference,axis=1)
        for column,(xyz,title) in enumerate(((original,'Input smoothwm'),(reference,'Native inflated'))):
            order=np.argsort(xyz[:,0]);axis=axes[row,column]
            axis.scatter(xyz[order,1],xyz[order,2],c=xyz[order,0],cmap='gray',s=.2,rasterized=True)
            axis.set_title(f'{entry["case"]} {hemi}: {title}',fontsize=9)
        axis=axes[row,2]
        image=axis.scatter(candidate[:,1],candidate[:,2],c=error,cmap='magma',vmin=0,vmax=.001,s=.2,rasterized=True)
        axis.set_title(f'Torch vs native: max {error.max():.6g} mm',fontsize=9)
        figure.colorbar(image,ax=axis,label='Same-index distance (mm)',shrink=.7)
        for axis in axes[row]:axis.set_aspect('equal');axis.set_xlabel('surface RAS Y (mm)');axis.set_ylabel('Z (mm)')
        report['rows'].append({'case':entry['case'],'hemisphere':hemi,'input_sha256':entry['sha256'],
            'vertices':len(original),'faces':len(faces),'max_vertex_distance_mm':float(error.max())})
    figure.tight_layout();destination=args.output/'smoothwm_inflated_error.png'
    figure.savefig(destination,dpi=160,metadata={'Software':'FNIT fixed-input inflation diagnostic'});plt.close(figure)
    report['figure_sha256']=sha(destination)
    (args.output/'figure_receipt.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
