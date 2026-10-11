from pathlib import Path
import json,hashlib,time,os,platform,importlib.util,unittest
from benchmark_recon_topology_mri_likelihood import run,json_safe_diagnostics
B=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
D=B/'runs/topology_zero_area_mri_20261010_v1'
R=B/'runs/topology_source_fitness_observer_20261010_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tick=time.perf_counter()
manifest=json.loads((W/'SOURCE_MANIFEST.json').read_text())
assert all(sha(W/x['path'])==x['sha256'] for x in manifest['payloads'])
reference_files=['statistics.bin','patch_statistics.bin','fitness.json','volume.bin','input/subject/mri/brain.mgz']
reference_hashes={n:sha(R/'repeat0'/n) for n in reference_files}
assert all(reference_hashes[n]==sha(R/'repeat1'/n) for n in reference_files)
spec=importlib.util.spec_from_file_location('tests_mri',W/'tests/recon_all/test_topology_mri_likelihood.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
with (D/'contracts.log').open('w') as stream:
 outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
contracts=dict(tests=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),passed=outcome.wasSuccessful())
(D/'contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
assert outcome.wasSuccessful()
results={}
for label,capture,role in [('initial','statistics.bin','initial-statistics'),('patch','patch_statistics.bin','retessellated-patch')]:
 results[label]=run(capture=R/'repeat0'/capture,brain=R/'repeat0/input/subject/mri/brain.mgz',output=D/label,threads=4,capture_role=role,joint_pdf_input=None if label=='initial' else D/'initial/candidate.npz')
assert all(reference_hashes[n]==sha(R/'repeat0'/n) for n in reference_files)
report=dict(status='complete_two_real_frozen_MRI_regressions',scope='initial MRI and first retessellated patch MRI only; not whole fitness, signed distance, GA or full recon-all',git_head=manifest['git_head'],source_manifest_sha256=sha(W/'SOURCE_MANIFEST.json'),runtime_sha256=sha(W/'src/fnit/recon_all/topology_mri_likelihood.py'),benchmark_sha256=sha(W/'tools/benchmark_recon_topology_mri_likelihood.py'),reference_repeat_exact=True,reference_files_sha256=reference_hashes,observer_receipt_sha256=sha(R/'receipt.private.json'),observer_build_sha256=sha(B/'workspaces/topology_source_fitness_observer_20261010_v2/build.private.json'),contracts=contracts,results=results,wall_seconds_including_contracts_validation_load_io=time.perf_counter()-tick,host=platform.node(),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=4,gpu='not_used',overall_equivalence='not_assessed')
(D/'COMPLETE.private.json').write_text(json.dumps(json_safe_diagnostics(report),indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'contracts':contracts,'gates':{k:v['strict_mri_component_passed'] for k,v in results.items()}}))
