"""只读取已完成 GPU tensor 和官方结果，定位同一掩膜内的真实尾部。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def checked(info):
    path = Path(info['path'])
    if sha(path) != info['sha256']: raise ValueError(f'bytes changed: {path}')
    return path


def eigenvalues(coefficients):
    # Saved MRtrix six-coefficient order: xx yy zz xy xz yz.
    xx, yy, zz, xy, xz, yz = map(float, coefficients)
    return np.linalg.eigvalsh([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['CON07','CON11'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('fresh output required')
    report_sha=sha(args.report); report=json.loads(args.report.read_text())
    output={'scope':'CPU read-only localization of already saved full GPU tensors; no solver rerun',
        'report':{'path':str(args.report),'sha256':report_sha},'harness_sha256':sha(__file__),
        'eigenvalue_scope':'Double eigvalsh of saved Float32 tensor coefficients; not a trace of native unchecked LLT',
        'mask_policy':'all original brain-mask rows retained; positive/nonpositive reporting does not filter solver',
        'cases':{}}
    for case in args.cases:
        record=report['cases'][case]; paths={k:checked(v) for k,v in record['inputs'].items()}
        signal=nib.load(paths['official_corrected_dwi']).get_fdata(dtype=np.float32)
        mask=np.asarray(nib.load(paths['brain_mask']).dataobj)>0
        reference={k:nib.load(checked(record['reference_geometry'][k])).get_fdata(dtype=np.float32) for k in ['fa','direction','tensor']}
        saved={}
        for version in ['baseline','candidate']:
            path=args.report.parent/case/(version+'.npz')
            if sha(path)!=record['accuracy'][version]['output_sha256']: raise ValueError('saved full output bytes mismatch')
            saved[version]=dict(np.load(path))
        indices=np.flatnonzero(mask.reshape(-1)); index_of={int(index):row for row,index in enumerate(indices)}
        selected=saved['candidate']; finite=mask&np.isfinite(selected['fa'])&np.isfinite(reference['fa'])
        errors=np.where(finite,np.abs(selected['fa'].astype(np.float64)-reference['fa']),-np.inf)
        direction_finite=mask&np.isfinite(selected['direction']).all(-1)&np.isfinite(reference['direction']).all(-1)
        left=selected['direction'].astype(np.float64);right=reference['direction'].astype(np.float64)
        denominator=np.linalg.norm(left,axis=-1)*np.linalg.norm(right,axis=-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            angle=np.degrees(np.arccos(np.clip(np.abs(np.sum(left*right,axis=-1)/denominator),0,1)))
        angle=np.where(direction_finite&np.isfinite(angle),angle,-np.inf)
        ranked=np.unique(np.r_[np.argsort(errors.reshape(-1))[-5:],np.argsort(angle.reshape(-1))[-5:]])
        rows=[]
        for index in ranked:
            ijk=np.unravel_index(int(index),mask.shape); row=index_of[int(index)]; s=signal[ijk]
            rows.append({'ijk':list(map(int,ijk)),'mask_flat_row':row,'batch4096':row//4096,'row_in_batch4096':row%4096,
                'measurement_count':len(s),'positive_count':int((s>0).sum()),'zero_count':int((s==0).sum()),
                'negative_count':int((s<0).sum()),'signal_min':float(s.min()),'signal_max':float(s.max()),
                'candidate_fa_abs_error':float(errors[ijk]),'candidate_antipodal_degrees':float(angle[ijk]),
                'official':{'fa':float(reference['fa'][ijk]),'direction':reference['direction'][ijk].tolist(),
                    'tensor':reference['tensor'][ijk].tolist(),'saved_tensor_eigenvalues':eigenvalues(reference['tensor'][ijk])},
                **{version:{'fa':float(values['fa'][ijk]),'direction':values['direction'][ijk].tolist(),
                    'tensor':values['tensor'][ijk].tolist(),'saved_tensor_eigenvalues':eigenvalues(values['tensor'][ijk])}
                   for version,values in saved.items()}})
        output['cases'][case]={'mask_voxels':int(mask.sum()),'fa_finite_mask_voxels':int(finite.sum()),
            'worst_fa_ijk':list(map(int,np.unravel_index(errors.argmax(),mask.shape))),
            'worst_direction_ijk':list(map(int,np.unravel_index(angle.argmax(),mask.shape))), 'selected_rows':rows}
        for info in record['inputs'].values(): checked(info)
    if sha(args.report)!=report_sha: raise ValueError('source report changed during read')
    args.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__': main()
