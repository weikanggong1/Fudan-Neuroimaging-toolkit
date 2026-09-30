import cProfile,pstats,pathlib,json,hashlib,platform,time
import torch, numba, numpy as np
from nibabel.freesurfer.io import read_geometry
from fnit.recon_all import sphere_standard_run
p=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930")
base=p/'full_sub01_279e09f';out=p/'sub01/sphere_profile_279'
out.mkdir(exist_ok=False)
torch.set_num_threads(4)
profile=cProfile.Profile()
report=profile.runcall(sphere_standard_run.run_standard_sphere,
 inflated=base/'surf/lh.inflated',smoothwm=base/'surf/lh.smoothwm',
 output=out/'lh.sphere',finish_device='cpu')
profile.dump_stats(str(out/'sphere.prof'))
with (out/'profile.txt').open('w') as stream:
 pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(30)
a,fa=read_geometry(str(base/'surf/lh.sphere'));b,fb=read_geometry(str(out/'lh.sphere'))
report.update(code_commit='279e09f0d2a166237871b3d683a6be75bd5e99b4',scope='frozen FNIT stage with cProfile overhead; not a before-after speed comparison',host=platform.node(),torch_threads=torch.get_num_threads(),numba_threads=numba.get_num_threads(),ordered_faces_equal=bool(np.array_equal(fa,fb)),different_coordinate_elements=int(np.count_nonzero(a!=b)),maximum_coordinate_difference_mm=float(np.abs(a-b).max()),source_sha256=hashlib.sha256(pathlib.Path(sphere_standard_run.__file__).read_bytes()).hexdigest(),input_sha256={n:hashlib.sha256((base/'surf'/('lh.'+n)).read_bytes()).hexdigest() for n in ('inflated','smoothwm')},profile_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
assert np.array_equal(fa,fb) and np.array_equal(a,b)
