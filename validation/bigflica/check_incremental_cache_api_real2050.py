"""Check public outputs using validated 2,050-person stage caches.

python check_incremental_cache_api_real2050.py OLD_PUBLIC_DIR STAGE_OUTPUT_DIR NEW_OUTPUT_DIR

The two upstream cache directories are reused. The dictionaries already fitted
by the production stage benchmark are imported with their verified input hashes.
This is a cache/output check, not a cold-start or end-to-end speed benchmark.
Only cache_api_aggregate.json may be published; subject data stay private.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.bigflica import apply_model, run_bigflica
from fnit.bigflica.pipeline import _file_record, _save_manifest, _signature


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old, stage, output = map(Path, sys.argv[1:])
    old_model = json.loads((old/'components_3_lambda_R/model.json').read_text())
    report_stage = json.loads((stage/'aggregate.json').read_text())
    root = Path(old_model['subjects_root'])
    ids = old_model['subjects']
    specs = old_model['source_modalities']
    images = {name: [_file_record(root/subject/spec['image']) for subject in ids]
              for name, spec in specs.items()}
    masks = {name:_file_record(Path(spec['mask'])) for name,spec in specs.items()}
    signature = _signature({'ids':ids,'specs':specs,'masks':masks,'images':images})
    normalized_signature = _signature({
        'subjects':ids,'normalized_dtype':'float32-v2-stats64',
        'modalities':{name:{'image':spec['image'],'mask':masks[name]} for name,spec in specs.items()},
        'images':images})
    if signature != old_model['input_signature'] or normalized_signature != json.loads(
            (old/'normalized_f32/manifest.json').read_text())['signature']:
        raise ValueError('Original input metadata changed; caches cannot be reused')
    for name in specs:
        path=old/'mmigp_100'/f'{name}_projected.h5'
        if sha(path) != report_stage['modalities'][name]['projected_sha256']:
            raise ValueError('Stage dictionary projected input mismatch')
    model=output/'components_3_lambda_R'
    resumed=(model/'model.json').is_file()
    mmigp_signature=json.loads((old/'mmigp_100/manifest.json').read_text())['signature']
    dicl_signature=_signature([mmigp_signature,200,1000,0,32,120,'rsvd3bpdn'])
    dictionary_dir=output/'dicl_100_200_1000_0_32_120_rsvd3bpdn_cuda'
    started=time.perf_counter()
    if not resumed:
        output.mkdir(parents=True,exist_ok=False)
        for name in ('normalized_f32','mmigp_100'):
            (output/name).symlink_to((old/name).resolve(),target_is_directory=True)
        dictionary_dir.mkdir()
        for name in specs:
            shutil.copyfile(stage/f'{name}_gpu_dictionary.npy',dictionary_dir/f'{name}_dictionary.npy')
        _save_manifest(dictionary_dir,{'signature':dicl_signature,'mmigp_signature':mmigp_signature})
        model=run_bigflica(root,specs,output,3,migp_dim=100,dicl_dim=200,
                          subjects=ids,device='cuda:0',max_gpu_gb=19,
                          dicl_max_iter=1000,flica_max_iter=100,top_voxels=1000,
                          flica_lambda_dims='R',feature_block=2048)
    dictionary_hashes={name:sha(dictionary_dir/f'{name}_dictionary.npy') for name in specs}
    for name in specs:
        if dictionary_hashes[name] != sha(stage/f'{name}_gpu_dictionary.npy'):
            raise ValueError('Imported dictionary changed')
    metadata=json.loads((model/'model.json').read_text())
    if metadata['flica_signature'] != _signature([dicl_signature,3,100,'R']):
        raise ValueError('Output uses a different dictionary cache')
    if not metadata['timings'].get('dicl_reused') or not metadata['timings'].get('mmigp_reused'):
        raise ValueError('Expected verified caches to be reused')
    course=np.load(model/'subj_course.npy')
    if course.shape != (2050,3) or not np.isfinite(course).all():
        raise ValueError('Subject course invalid')
    maps={}
    for name in specs:
        mask=np.asarray(nib.load(str(model/f'{name}_mask.nii.gz')).dataobj)>0
        z=np.load(model/f'{name}_zstat.npy')
        for component in range(3):
            prefix=model/'maps'/name/f'component-{component+1:03d}'
            values=np.asarray(nib.load(str(prefix)+'_zstat.nii.gz').dataobj)[mask]
            if not np.array_equal(values,z[:,component]):
                raise ValueError('NIfTI z-stat differs from its stored array')
            for suffix in ('_top-1000.nii.gz','_top-1000.png'):
                if not Path(str(prefix)+suffix).is_file():
                    raise ValueError('Thresholded output missing')
        maps[name]={'voxels':int(mask.sum()),'components':3,'nifti_array_max_abs_error':0.0}
    training_ids=set(ids)
    held_out=next(directory for directory in sorted(root.iterdir())
                  if directory.is_dir() and directory.name not in training_ids
                  and all((directory/spec['image']).is_file() for spec in specs.values()))
    scores=apply_model(model,held_out,device='cuda:0')
    if scores.shape != (3,) or not np.isfinite(scores).all():
        raise ValueError('Held-out projection invalid')
    report={'dataset':'2050 real subjects; full-mask VBM/FA/MD; task excluded',
            'scope':'Public API output and imported stage-cache verification; no new cold-start timing',
            'stage_report_sha256':sha(stage/'aggregate.json'),'dictionary_sha256':dictionary_hashes,
            'course_shape':list(course.shape),'maps':maps,'held_out_finite':True,
            'stage_timings_s':metadata['timings'],'verification_resumed':resumed,'verification_wall_s':time.perf_counter()-started,
            'status':'complete'}
    (output/'cache_api_aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
