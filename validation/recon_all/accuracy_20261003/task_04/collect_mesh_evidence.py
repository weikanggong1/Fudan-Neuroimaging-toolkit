import hashlib,json,pathlib,sys
import nibabel.freesurfer.io as fsio
import numpy as np
root=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_04');base=root.parent.parent;source=root.parent/'baseline_runtime_816e5610';sys.path.insert(0,str(source/'src'))
from fnit.recon_all.compare_subject import _topology

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def metadata(p):
 v,f,m=fsio.read_geometry(str(p),read_metadata=True);return {'sha256':sha(p),'vertices':len(v),'faces':len(f),'metadata':{k:np.asarray(value).tolist() if isinstance(value,np.ndarray) else value for k,value in m.items()},'topology':_topology(v,f),'sphere_radial_fold':folds(v,f)}
def folds(v,f):
 t=v[f];sgn=np.einsum('ij,ij->i',np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),t.mean(1)-v.mean(0));return {'negative_faces':int((sgn<0).sum()),'zero_faces':int((sgn==0).sum())}
checkpoint=base/'parallel_20261002/whole_sub01_candidate_8d750e2';evidence={'status':'collected_completed_only','baseline_commit':'816e5610417a4c587caf321049438a9554139016','overall_equivalence':'not_assessed','sphere_seed_contract_matched':False,'sphere_seed_contract_evidence':{'fnit_seed':1234,'official_argv_contains_seed':False,'official_log_randomSeed':0,'official_log_seed_not_set_warning':True,'official_log_final_seed':-1791003502,'seed_fix_replay':'pending; isolated mris_sphere -seed 1234 only; current queue unchanged'},'stages':{},'sphere_context_input_comparison':{},'upstream_source':{},'file_access_trace':'not_captured; default dependency list comes from inspected pinned source, not syscall trace'}
for stage in ('remesh','sphere','register'):
 result={}
 for kind in ('official','fnit'):
  path=root/'frozen_v2/sub01/lh'/stage/kind/'report.json';r=json.loads(path.read_text());result[kind]=r
  if kind=='official':result[kind]['program_sha256_live']=sha(r['command'][0]);result[kind]['program_hash_still_matches']=result[kind]['program_sha256_live']==r['program_sha256']
  else:
   failures=[]
   for rel,expected in r['source_sha256'].items():
    if sha(source/rel)!=expected:failures.append(rel)
   result[kind]['source_sha256_live_mismatch']=failures
  geom=root/'frozen_v2/sub01/lh'/stage/kind/('lh.'+stage);result[kind+'_output_geometry']=metadata(geom)
 evidence['stages'][stage]=result
copied=root/'frozen_v2/sub01/lh/sphere/official/subject/surf'
for path in sorted(copied.glob('lh.*')):
 other=checkpoint/'surf'/path.name
 if other.is_file():evidence['sphere_context_input_comparison'][path.name]={'checkpoint_sha256':sha(other),'private_copy_sha256':sha(path),'bytes_equal':sha(path)==sha(other)}
for name in ('inflated','smoothwm'):
 evidence['sphere_context_input_comparison']['geometry_'+name]=metadata(checkpoint/'surf'/('lh.'+name))
upstream=base/'source_codeload_probe/source'
for relative in ('mris_sphere/mris_sphere.cpp','utils/utils.cpp','utils/mrisurf_integrate.cpp','utils/mrisurf_vals.cpp'):
 path=upstream/relative;text=path.read_text();evidence['upstream_source'][relative]={'sha256':sha(path),'matching_line_numbers':{term:[i for i,line in enumerate(text.splitlines(),1) if term in line] for term in ('orig_name','setRandomSeed','idum = -1L','MRISsampleDistances','randomNumber','MRISreadCurvature','MRISreadOriginalProperties','MRISreadVertexPositions')}}
(root/'completed_stage_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({'collected_stages':list(evidence['stages']),'source_mismatch':{k:r['fnit']['source_sha256_live_mismatch'] for k,r in evidence['stages'].items()},'program_mismatch':{k:not r['official']['program_hash_still_matches'] for k,r in evidence['stages'].items()},'private_copy_mismatch':[k for k,r in evidence['sphere_context_input_comparison'].items() if r.get('bytes_equal') is False],'sphere_quality':{k:evidence['stages']['sphere'][k+'_output_geometry']['sphere_radial_fold'] for k in ('official','fnit')}},indent=2))
