"""只读扫描显式公开检查点清单的四字节 ROI 可及范围；不执行 WM 编辑。"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np


def scan(*, fnit_root, input_manifest, output):
    """具名路径参数；读取cases的case/relative_input，输出真实存储/ROI/SHA JSON。"""
    manifest=json.loads(input_manifest.read_text());rows=[]
    for entry in manifest['cases']:
        relative=Path(entry['relative_input'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('explicit safe relative image path required')
        path=fnit_root/relative;image=nib.load(str(path));a=np.asarray(image.dataobj)
        if a.ndim!=3 or min(a.shape)<3 or image.get_data_dtype().itemsize!=4:
            raise ValueError('matching original four-byte 3D storage required')
        sides=[]
        for side,labels in [('lh',[17,18,5]),('rh',[53,54,44])]:
            mask=np.isin(a,labels);xs=np.flatnonzero(mask.any(axis=(1,2)))
            minx=int(xs.min()) if len(xs) else None
            dilated_min=max(0,minx-5) if minx is not None else None
            byte_max=(a.shape[0]-1)//4
            sides.append({'hemisphere':side,'labels':labels,'seed_voxel_count':int(mask.sum()),
                'minimum_seed_x':minx,'minimum_five_dilation_x':dilated_min,
                'byte_access_max_numerical_x':byte_max,
                'proof_unreachable':dilated_min is None or dilated_min>byte_max})
        rows.append({'case':entry['case'],'relative_input':str(relative),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
            'shape':list(a.shape),'dtype':str(image.get_data_dtype()),'hemispheres':sides,
            'actual_path_branch_required':any(not s['proof_unreachable'] for s in sides)})
    report={'schema':1,'scope':'readonly original-storage ROI reachability scan; no WM/reference execution',
        'candidate_input_producer':manifest['candidate_input_producer'],'cases':rows,
        'expected_cases':len(manifest['cases']),'found_cases':len(rows),
        'active_cases':[r['case'] for r in rows if r['actual_path_branch_required']],
        'scan_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_manifest_sha256':hashlib.sha256(input_manifest.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    scan(**vars(p.parse_args()))
