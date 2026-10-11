"""在独立冻结源码重放真实第四缺陷；分类计时并导出自产准备后输入。"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
from collections.abc import Mapping


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import fnit
    from fnit.recon_all import topology_patch_fitness as patch
    from fnit.recon_all import topology_candidate_pipeline as pipeline
    from fnit.recon_all import topology_genetic_search as genetic
    records={};captured=False
    def wrap(module,name,label,capture=False):
        original=getattr(module,name)
        def observed(*a,**kw):
            nonlocal captured
            if capture and not captured:
                arrays={}
                def encode(value):
                    if isinstance(value,np.ndarray):
                        key='array_'+str(len(arrays));arrays[key]=value.copy();return {'array':key}
                    if isinstance(value,np.generic):return {'scalar_dtype':str(value.dtype),'value':value.item()}
                    if isinstance(value,Mapping):return {'mapping':{k:encode(v)for k,v in value.items()}}
                    if isinstance(value,(list,tuple)):return {'tuple'if isinstance(value,tuple)else'list':[encode(v)for v in value]}
                    if value is None or isinstance(value,(str,int,float,bool)):return value
                    raise TypeError(type(value).__name__)
                schema=encode(kw);np.savez_compressed(args.output/'prepared_inputs.private.npz',**arrays)
                (args.output/'prepared_inputs.schema.private.json').write_text(json.dumps(schema)+'\n');captured=True
            started=time.perf_counter()
            try:return original(*a,**kw)
            finally:records.setdefault(label,[]).append(time.perf_counter()-started)
        setattr(module,name,observed)
        return observed
    wrap(pipeline,'candidate_geometry','candidate_retess_match_smooth_normals')
    wrap(pipeline,'candidate_second_fundamental_form','surface_curvature_sff')
    wrap(pipeline,'candidate_face_check','candidate_face_check')
    observed=wrap(patch,'evaluate_prepared_patch_fitness','prepared_five_total',True)
    pipeline.evaluate_prepared_patch_fitness=observed
    for name,label in [('patch_mri_likelihood','mri_intensity_and_sampling'),('patch_normal_dot_likelihood','normal_dot_likelihood'),('patch_curvature_likelihood','curvature_likelihood'),('defect_signed_distance','distance'),('defect_volume_likelihood','volume_likelihood_and_feedback'),('compose_patch_fitness','fitness_composition')]:wrap(patch,name,label)
    wrap(genetic,'prepare_genetic_defect_evaluator','evaluator_preparation')
    spec=importlib.util.spec_from_file_location('frozen_fourth_driver',args.driver);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    sys.path.insert(0,str(args.driver.parent))
    command=json.loads(args.request.read_text())['command'];keys={}
    for index,token in enumerate(command):
        if token.startswith('--'):keys[token[2:].replace('-','_')]=command[index+1]
    for key in ['input_surface','input_qsphere','brain','state_checkpoint','producer_progress']:keys[key]=Path(keys[key])
    keys.update(frozen_source_manifest=args.manifest,expected_source_manifest_sha256=sha(args.manifest),output=args.output/'candidate',defect_number=3,threads=4)
    started=time.perf_counter();report=driver.run(**keys)
    (args.output/'classified_profile.private.json').write_text(json.dumps({'status':'complete','source_parent_commit':'8b14cf4b74bd7f08e6348e28634730d115e0a662','fnit_location_private':fnit.__file__,'driver_sha256':sha(args.driver),'wrapper_sha256':sha(__file__),'source_manifest_sha256':sha(args.manifest),'complete_wrapper_run_seconds':time.perf_counter()-started,'categorized_calls_seconds':records,'prepared_capture_sha256':sha(args.output/'prepared_inputs.private.npz'),'prepared_schema_sha256':sha(args.output/'prepared_inputs.schema.private.json'),'actual_rng_draws':report['actual_rng_draws'],'repeated_all_fields_equal':report['all_repeated_fields_byte_equal'],'threads':4,'cpu_affinity':sorted(os.sched_getaffinity(0)),'scope':'fresh fourth first candidate, no full GA, nested timers must not be added'},indent=2)+'\n')
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
