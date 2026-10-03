"""Read-only CON11 origin gate for a future explicit subset; does not launch models."""
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def validate_actual_CON11_packing_route(route, frozen_configuration, expected_baseline_lineage):
    """Expected baseline lineage must come from the root's verified frozen execution record."""
    if route.get('state') != 'actual_CON11_packing_verified':
        return None
    if route.get('case_id') != 'sub-CON11' or route.get('root_actual_origin_verified') is not True:
        raise ValueError('Actual root-verified CON11 origin required')
    if set(expected_baseline_lineage) != {'source','configuration','driver'} or route.get('baseline_lineage') != expected_baseline_lineage:
        raise ValueError('CON11 baseline source/configuration/driver lineage differs')
    for record in expected_baseline_lineage.values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Frozen baseline lineage file changed')
    if route.get('manifest_sha256') != frozen_configuration['manifest_sha256'] or sha(frozen_configuration['manifest']) != frozen_configuration['manifest_sha256']:
        raise ValueError('Canonical manifest differs')
    manifest=json.loads(Path(frozen_configuration['manifest']).read_text())
    cases=[case for case in manifest['cases'] if case['subject'].removeprefix('sub-')=='CON11']
    if len(cases)!=1:raise ValueError('Unique canonical CON11 required')
    canonical={str(Path(frozen_configuration['raw_root'])/relative):digest for relative,digest in cases[0]['input_sha256'].items()}
    if len(canonical)!=10:raise ValueError('Canonical ten raw files required')
    for path,digest in canonical.items():
        if sha(path)!=digest:raise ValueError('Canonical raw acquisition changed')
    old=Path(frozen_configuration['formal_FNIT_packing_root'])/'sub-CON11/connectome/preproc/topup/B0_AP_PA.nii.gz'
    packing=Path(route['packing']['path'])
    if packing.resolve()==old.resolve() or packing.is_symlink():
        raise ValueError('No old CON11 packing path or symlink alias')
    if sha(packing)!=route['packing']['sha256']:raise ValueError('Actual new CON11 packing SHA differs')
    pair=nib.load(packing);data=np.asarray(pair.dataobj)
    if data.ndim!=4 or data.shape[-1]!=2:raise ValueError('Actual AP/PA pair format required')
    indices=route['actual_raw_frame_indices']
    if set(indices)!={'ap_index','pa_index'} or any(type(x) is not int or x<0 for x in indices.values()):
        raise ValueError('Actual frame indices required, never guessed')
    for stem,key,frame in [('AP','ap_index',0),('PA','pa_index',1)]:
        paths=route['canonical_inputs'][stem]
        for kind in ('dwi','bvals'):
            record=paths[kind]
            if canonical.get(record['path'])!=record['sha256']:
                raise ValueError('CON11 raw route differs from canonical acquisition')
        raw=nib.load(paths['dwi']['path']);bvals=np.loadtxt(paths['bvals']['path']).reshape(-1);index=indices[key]
        if raw.shape[:3]!=data.shape[:3] or index>=raw.shape[3] or index>=bvals.size or not bvals[index]<100:
            raise ValueError('CON11 selected raw b0 frame invalid')
        if stem=='AP' and not np.array_equal(raw.affine,pair.affine):raise ValueError('CON11 packing must retain original AP header')
        if not np.array_equal(np.asarray(raw.dataobj)[...,index],data[...,frame]):
            raise ValueError('CON11 actual packing/raw frame voxels differ')
    return {'case_id':'sub-CON11','actual_packing':dict(route['packing']),
            'actual_raw_frame_indices':indices,'baseline_lineage':expected_baseline_lineage,
            'canonical_raw_input_sha256':canonical,'root_actual_origin_verified':True,
            'modeling_ready':False,'scope':'packing origin only; require independent completed verified CPU rawprep before subset modeling'}
