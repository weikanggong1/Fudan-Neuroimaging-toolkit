from pathlib import Path
import os,sys,json,hashlib
import numpy as np
R=Path(os.environ['FNIT_CACHE_ROOT']);B=R/'workspaces/topology_candidate_cycle_20261010_v2';D=R/'runs/topology_ga_finalize_20261010_v2'
plot=B/'tools/plot_recon_topology_candidate_geometry.py'
if hashlib.sha256(plot.read_bytes()).hexdigest()!='7db770f74ec66aa10d2c927ff34e403c8f0c1265728a9e44b02c326e4c39d4c2':raise RuntimeError('mature plot helper hash changed')
sys.path[:0]=[str(B/'src'),str(B/'tools')]
from plot_recon_topology_candidate_geometry import run
with np.load(D/'candidate/A.final.private.npz',allow_pickle=False)as z:
 data={name:z[key].copy()for name,key in [('matched','final_original_vertices'),('selected_vertices','final_selected_vertices'),('faces','final_faces')]}
input_npz=D/'final_plot_input.private.npz'
if input_npz.exists():raise FileExistsError('new figure input required')
np.savez_compressed(input_npz,**data)
q=run(brain=R/'workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject/mri/brain.mgz',candidate=input_npz,before=R/'runs/topology_source_retessellation_observer_20261010_v3/repeat0/retess.before.bin',reference_after=R/'runs/topology_source_match_observer_20261010_v2/repeat0/match.after.bin',output=D/'final_patch.png')
q['scope']='fixed-order final geometry; source figure reference is first-candidate phase, not a complete GA winner or full surface QC'
q['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
q['mature_plot_helper_sha256']=hashlib.sha256(plot.read_bytes()).hexdigest()
q['actual_final_input_npz_sha256']=hashlib.sha256((D/'candidate/A.final.private.npz').read_bytes()).hexdigest()
q['finalization_report_sha256']=hashlib.sha256((D/'candidate/finalization.report.json').read_bytes()).hexdigest()
(D/'figure.public.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
print(json.dumps(q,allow_nan=False))
