from pathlib import Path
import hashlib,json,importlib.util,unittest,time
from benchmark_recon_topology_pseudonormals import run
B=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;D=B/'runs/topology_angle_pseudo_20261010_v1';R=B/'runs/topology_source_distance_observer_20261010_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((W/'SOURCE_MANIFEST.json').read_text());assert all(sha(W/v['path'])==v['sha256'] for v in manifest['payloads'])
spec=importlib.util.spec_from_file_location('tests_angle',W/'tests/recon_all/test_topology_angle_pseudonormals.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tick=time.perf_counter()
with (D/'contracts.log').open('w') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
contracts=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),seconds=time.perf_counter()-tick)
(D/'contracts.json').write_text(json.dumps(contracts,indent=2)+'\n');assert result.wasSuccessful()
report=run(capture=R/'repeat0/distance.bin',repeated_capture=R/'repeat1/distance.bin',observer_receipt=R/'receipt.private.json',output=D/'regression',threads=4)
report['actual_snapshot_git_head']=manifest['git_head'];report['source_manifest_sha256']=sha(W/'SOURCE_MANIFEST.json');report['contracts']=contracts;report['observer_build_sha256']=sha(B/'workspaces/topology_source_distance_observer_20261010_v2/build.private.json')
(D/'COMPLETE.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'gate':report['strict_component_passed'],'comparisons':report['comparisons'],'seconds':report['seconds'],'raw_reference_repeat_equal':report['native_raw_bytes_equal']}))
