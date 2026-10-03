"""Real frozen output writer AB/BA regression without rerunning the network."""
import fcntl, hashlib, importlib.util, json, platform, time
from pathlib import Path
import nibabel as nib
import numpy as np
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
spec=importlib.util.spec_from_file_location('fnit.recon_all.input_talairach_chain_candidate',ROOT/'input_talairach_chain.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
report={'host':platform.node(),'validation':'same frozen network output, writer only','code_sha256':hashlib.sha256((ROOT/'input_talairach_chain.py').read_bytes()).hexdigest(),'runs':[]}
subjects=[('sub01',BASE/'parallel_20261002/whole_sub01_candidate_8d750e2',BASE.parent/'reconall_benchmark_pair_ac_20260924/official_subjects/a_official'),('sub02',BASE/'parallel_20261002/whole_sub02_candidate_8d750e2_retry_v3',BASE/'reference_sub02_official/subjects/b_official')]
with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for label,subject,official in subjects:
        image=nib.load(str(subject/'mri/synthstrip.mgz'));expected=nib.load(str(official/'mri/synthstrip.mgz'))
        source=subject/'mri/orig.mgz'
        for index,backend in enumerate(('baseline','candidate','candidate','baseline')):
            output=ROOT/f'{label}_{backend}_{index}.mgz';start=time.perf_counter()
            if backend=='baseline': nib.save(image,str(output))
            else: module.save_synthstrip_mgh(source_file=source,stripped_image=image,output_file=output)
            elapsed=time.perf_counter()-start; actual=nib.load(str(output));a=np.asanyarray(actual.dataobj);b=np.asanyarray(expected.dataobj)
            report['runs'].append({'subject':label,'backend':backend,'seconds':elapsed,'output_size_bytes':output.stat().st_size,'dtype':str(a.dtype),'official_dtype':str(b.dtype),'different_voxels_vs_official':int(np.count_nonzero(a!=b)),'different_vs_frozen_network':int(np.count_nonzero(a!=np.asanyarray(image.dataobj))),'geometry_max_abs':float(np.max(np.abs(actual.affine-expected.affine))),'input_sha256':hashlib.sha256((subject/'mri/synthstrip.mgz').read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    report['fractional_rejection']=False
    bad=np.asanyarray(image.dataobj).copy();bad[0,0,0]=.5
    try: module.save_synthstrip_mgh(source_file=source,stripped_image=nib.Nifti1Image(bad,image.affine),output_file=ROOT/'invalid.mgz')
    except ValueError: report['fractional_rejection']=True
(ROOT/'synthstrip_writer.json').write_text(json.dumps(report,indent=2))
