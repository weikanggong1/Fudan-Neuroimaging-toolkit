"""对真实完整候选的标量输入类型做旧/新合成回放，不运行几何或 GA。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from fnit.recon_all.topology_fitness_search import compose_patch_fitness


def run(*, reference_json: Path, baseline_module: Path, output_json: Path):
    if output_json.exists():
        raise FileExistsError("new input-type diagnostic required")
    reference=json.loads(reference_json.read_text())
    if reference['schema']!='fnit-topology-fitness-reference-v1':
        raise ValueError('complete frozen same-call reference required')
    fields=reference['tp_fields'];weights=reference['requested_weights']
    if weights!={'l_mri':1.,'l_unmri':10.,'l_qcurv':1.,'l_curv':1.}:
        raise ValueError('earlier fixed-weight helper only accepts this pinned default recipe')
    spec=importlib.util.spec_from_file_location('frozen_fitness_baseline',baseline_module)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    names=['face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll']
    rows={}
    for name,args in [('python_float',[float(fields[k]) for k in names]),
                      ('numpy_float32',list(np.array([fields[k] for k in names],np.float32)))]:
        baseline=old.compose_patch_fitness(*args,reference['face_check']>=0)
        candidate=compose_patch_fitness(*args,reference['face_check']>=0)
        rows[name]={'baseline_fitness':float(baseline),'candidate_fitness':candidate,
            'reference_fitness':reference['fitness'],
            'baseline_minus_reference':float(baseline)-reference['fitness'],
            'candidate_minus_reference':candidate-reference['fitness'],
            'candidate_double_bytes_equal':np.float64(candidate).tobytes()==np.float64(reference['fitness']).tobytes()}
    result={'scope':'real same-call scalar replay under Python float and NumPy float32 containers; no component regeneration/GA/whole-pipeline claim',
        'reference_json_sha256':hashlib.sha256(reference_json.read_bytes()).hexdigest(),
        'baseline_module_sha256':hashlib.sha256(baseline_module.read_bytes()).hexdigest(),
        'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_default_weights_compatible':True,'rows':rows,'no_gpu_or_precision_change':True}
    output_json.parent.mkdir(parents=True,exist_ok=True)
    output_json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not all(row['candidate_double_bytes_equal'] for row in rows.values()):
        raise AssertionError('input container affected new source-order composition')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-json','baseline-module','output-json']:
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
