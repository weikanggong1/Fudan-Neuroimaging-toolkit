import pathlib,importlib.util,sys,hashlib,json,time,platform
import numpy as np,torch,numba
from nibabel.freesurfer.io import read_geometry
from fnit.recon_all import place_surface_normals as new
d=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930")
spec=importlib.util.spec_from_file_location('original_normals_279',d/'original_normals_279.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
torch.set_num_threads(4)
rows={}
for sid in ('01','02'):
 for hemi in ('lh','rh'):
  for surf in ('inflated','sphere'):
   f=d/f'full_sub{sid}_279e09f/surf/{hemi}.{surf}';v,faces=read_geometry(str(f))
   outputs=[];seconds=[]
   for implementation in (old,new):
    start=time.perf_counter();outputs.append(implementation.initial_vertex_normals(v,faces));seconds.append(time.perf_counter()-start)
   delta=np.abs(outputs[0].astype(np.float64)-outputs[1])
   key=f'sub{sid}/{hemi}/{surf}'
   rows[key]={'input_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'vertices':len(v),'faces':len(faces),'old_seconds':seconds[0],'new_seconds':seconds[1],'different_elements':int(np.count_nonzero(delta)),'max_abs':float(delta.max())}
report={'base_commit':'279e09f0d2a166237871b3d683a6be75bd5e99b4','candidate_state':'5668338 plus stable face incidence index','old_source_sha256':hashlib.sha256(pathlib.Path(old.__file__).read_bytes()).hexdigest(),'new_source_sha256':hashlib.sha256(pathlib.Path(new.__file__).read_bytes()).hexdigest(),'host':platform.node(),'torch_threads':4,'numba_threads':numba.get_num_threads(),'scope':'8 real frozen meshes; per-function timing includes JIT on first call','results':rows}
(d/'normals_stage_pair_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
assert all(r['different_elements']==0 for r in rows.values())
