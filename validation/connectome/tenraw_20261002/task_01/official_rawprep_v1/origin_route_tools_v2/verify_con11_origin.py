#!/usr/bin/env python3
"""Read an explicit root-provided actual CON11 packing origin; no solver/GPU dispatch."""
import argparse,json,hashlib
from pathlib import Path
import nibabel as nib
import numpy as np

BASELINE_COMMIT='da427c21a5a7619f4c3457c4406ab48a187ce390'
BASELINE_CORE_SHA='981b61d822a4bacb2aba05637f1001cdd3114d5d70139d1f31f859bb3011aba9'
FLAGS=['--flm=quadratic','--resamp=jac','--slm=linear','--niter=8','--fwhm=10,8,4,2,0,0,0,0','--ff=10','--sep_offs_move','--nvoxhp=1000','--repol','--rms','--initrand=12345']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def artifact(value):
    path=Path(value['path'])
    if not path.is_absolute() or not path.is_file() or sha(path)!=value['sha256']:raise ValueError('missing/changed explicit origin artifact')
    return path

def select_con11_case(manifest):
    matches=[case for case in manifest["cases"] if case.get("case_id")=="sub-CON11"]
    if len(matches)!=1:raise ValueError("unique canonical case_id sub-CON11 required")
    case=matches[0]
    if not isinstance(case.get("subject"),str) or case["subject"].removeprefix("sub-")!="CON11":
        raise ValueError("canonical subject disagrees with case_id")
    return case


def verify(origin_path,manifest_path,source):
    o=json.loads(origin_path.read_text());manifest=json.loads(manifest_path.read_text())
    if o['schema_version']!=1 or o['case_id']!='sub-CON11' or o['version']!='baseline' or o['baseline_commit']!=BASELINE_COMMIT:raise ValueError('not frozen baseline CON11 origin')
    if o['canonical_manifest_sha256']!=sha(manifest_path):raise ValueError('canonical manifest differs')
    if sha(source/'src/fnit/flirt/core.py')!=BASELINE_CORE_SHA:raise ValueError('frozen baseline core differs')
    frozen=json.loads(Path(__file__).with_name('frozen_scientific_source_sha256.json').read_text())
    for relative,digest in frozen['source_files'].items():
        if sha(source/relative)!=digest:raise ValueError('frozen scientific source differs: '+relative)
    scientific=o['scientific_parameters']
    if scientific['seed']!=0 or scientific['eddy_gp_seed']!=12345 or scientific['tf32'] is not True or scientific['pair_geometry']!='fslmerge-first' or scientific['official_EDDY_flags']!=FLAGS:raise ValueError('scientific parameter receipt differs')
    evidence={name:str(artifact(o[name])) for name in ('packing','acqparams','config','driver_lineage_receipt','baseline_source_receipt')}
    source_receipt=json.loads(Path(evidence['baseline_source_receipt']).read_text())
    if source_receipt['code_commit']!=BASELINE_COMMIT or source_receipt['core_sha256']!=BASELINE_CORE_SHA:raise ValueError('baseline source receipt differs')
    if not o['source_files']:raise ValueError('actual frozen source file receipt required')
    for value in o['source_files']:
        path=artifact(value)
        if path.resolve().is_relative_to(source.resolve()) is False:raise ValueError('origin source outside explicit frozen baseline source')
    # These are declared explicit by root, never derived from an old baseline namespace.
    fnit_case=Path(o['actual_fnit_case_directory']);packing=Path(evidence['packing']);acq=Path(evidence['acqparams'])
    if not fnit_case.is_absolute() or packing!=fnit_case/'connectome/preproc/topup/B0_AP_PA.nii.gz' or acq!=fnit_case/'connectome/preproc/topup/acqparams.txt':raise ValueError('actual packing not bound to explicit formal case directory')
    case=select_con11_case(manifest);pair_image=nib.load(packing);pair=np.asarray(pair_image.dataobj);frames=[]
    for stem,j in [('AP',0),('PA',1)]:
        raw=Path(case['dwi'] if stem=='AP' else case['reverse_dwi']);expected=next(v for k,v in case['input_sha256'].items() if k.endswith(raw.name))
        if sha(raw)!=expected:raise ValueError('canonical raw changed')
        image=nib.load(raw);data=np.asarray(image.dataobj);bvals=np.loadtxt(raw.with_name(raw.name.replace('.nii.gz','.bval'))).reshape(-1)
        if pair.shape!=(*image.shape[:3],2):raise ValueError('packing shape differs')
        matches=[int(i) for i in np.where(bvals<100)[0] if np.array_equal(data[...,i],pair[...,j])]
        if len(matches)!=1:raise ValueError('unique actual canonical b0 match required')
        index=matches[0];selected=np.ascontiguousarray(data[...,index]);frames.append({'direction':stem,'index':index,'raw_image_sha256':expected,'selected_voxel_byte_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'dtype':selected.dtype.str,'shape':list(selected.shape)})
        if stem=='AP' and not np.array_equal(image.affine,pair_image.affine):raise ValueError('first-header packing affine differs')
    datain=np.loadtxt(acq)
    if datain.shape!=(2,4) or not np.isfinite(datain).all() or not np.allclose(datain[:,3],.0266,rtol=0,atol=1e-12):raise ValueError('effective readout differs')
    if not np.array_equal(datain[:,:3],np.asarray([[0,-1,0],[0,1,0]])):raise ValueError('canonical phase-encoding rows differ')
    return {'state':'actual_CON11_origin_verified_metadata_only','root_origin_path':str(origin_path),'root_origin_sha256':sha(origin_path),'canonical_manifest_sha256':sha(manifest_path),'baseline_commit':BASELINE_COMMIT,'frames':frames,'actual_fnit_case_directory':str(fnit_case),'artifacts':o,'science_dispatched':False,'GPU_initialized':False,'scope':'explicit actual common original input choice only; no FNIT field/mask/DWI reused; future official CON11 needs its own fresh CPU stages/output namespace'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('origin','manifest','baseline-source','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh verification receipt required')
    result=verify(a.origin,a.manifest,a.baseline_source);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
