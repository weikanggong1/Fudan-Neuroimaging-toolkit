from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
root=Path(__file__).parent
launch=json.loads((root/'launch.json').read_text())
start=time.monotonic()
result=subprocess.run([sys.executable,'-m','pytest',*launch['pytest_arguments']],cwd=root/'source',capture_output=True,text=True)
(root/'stdout.log').write_text(result.stdout)
(root/'stderr.log').write_text(result.stderr)
report={'state':'related_CPU_tests_passed' if result.returncode==0 else 'related_CPU_tests_failed','returncode':result.returncode,'wall_s':time.monotonic()-start,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':launch['source_commit'],'archive_sha256':launch['archive_sha256'],'launch_sha256':hashlib.sha256((root/'launch.json').read_bytes()).hexdigest(),'command':[sys.executable,'-m','pytest',*launch['pytest_arguments']],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'stdout_sha256':hashlib.sha256((root/'stdout.log').read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256((root/'stderr.log').read_bytes()).hexdigest(),'output_summary':result.stdout[-3000:],'scope':'scoped CPU regression tests of connectome and directly affected BIDS/EDDY/geometry APIs; GPU tests skip with CUDA invisible; not an MRI benchmark or whole repository test claim'}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(result.returncode)
