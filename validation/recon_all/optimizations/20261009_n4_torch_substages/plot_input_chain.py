"""原始输入链完成后的公开N4/nu脑图；只作诊断，不修改任何影像。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-pair',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    report=json.loads((a.raw_pair/'summary.json').read_text())
    if report['status']!='complete_raw_input_to_nu_pair':raise ValueError('raw pair must be complete')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    figure,axes=plt.subplots(len(report['rows']),4,figsize=(13,3.4*len(report['rows'])),squeeze=False)
    receipt={'scope':'RAS axial diagnostic, maximum brain difference slice, not final segmentation metrics',
        'raw_summary_sha256':sha(a.raw_pair/'summary.json'),'source_sha256':sha(__file__),'rows':[]}
    for index,row in enumerate(report['rows']):
        native=a.raw_pair/row['case']/'native/subject/mri';candidate=a.raw_pair/row['case']/'torch/subject/mri'
        paths=[native/'orig.mgz',native/'nu.mgz',candidate/'nu.mgz',native/'synthstrip.mgz']
        images=[nib.as_closest_canonical(nib.load(str(p))) for p in paths]
        arrays=[np.asarray(im.dataobj) for im in images]
        if not all(np.array_equal(im.affine,images[0].affine) for im in images):raise ValueError('same geometry required')
        orig,reference,output,strip=arrays;brain=strip>0
        delta=output.astype(np.int16)-reference.astype(np.int16)
        slices=np.count_nonzero((delta!=0)&brain,axis=(0,1));plane=int(slices.argmax())
        affine=images[0].affine;step=affine.diagonal()[:3]
        extent=(affine[0,3]-.5*step[0],affine[0,3]+(orig.shape[0]-.5)*step[0],
                affine[1,3]-.5*step[1],affine[1,3]+(orig.shape[1]-.5)*step[1])
        z=float(affine[2,3]+plane*step[2])
        for column,(volume,title) in enumerate(((orig,'Conformed T1'),(reference,'Native nu'),(output,'Complete Torch nu'))):
            axis=axes[index,column];axis.imshow(volume[:,:,plane].T,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=255)
            axis.set_title(f'{row["case"]}: {title}\nRAS Z={z:.1f} mm',fontsize=9)
        axis=axes[index,3]
        drawing=axis.imshow(delta[:,:,plane].T,origin='lower',extent=extent,cmap='coolwarm',vmin=-3,vmax=3)
        axis.contour(brain[:,:,plane].T,levels=[.5],colors=['black'],linewidths=.45,origin='lower',extent=extent)
        axis.set_title('Torch - native nu\nself SynthStrip boundary',fontsize=9)
        figure.colorbar(drawing,ax=axis,label='MRI intensity difference',shrink=.8)
        for axis in axes[index]:axis.set_xlabel('RAS X (mm)');axis.set_ylabel('RAS Y (mm)')
        receipt['rows'].append({'case':row['case'],'selected_axial_index_canonical':plane,'RAS_Z_mm':z,
            'brain_different_voxels_on_slice':int(slices[plane]),'input_files_sha256':{str(p):sha(p) for p in paths}})
    figure.tight_layout();p=a.output/'raw_chain_nu_error.png'
    figure.savefig(p,dpi=160,metadata={'Software':'FNIT complete N4 diagnostic'});plt.close(figure)
    receipt['figure_sha256']=sha(p)
    (a.output/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
