"""安装本轮冻结wheel并验证真实入口；不运行影像或原版阶段程序。"""
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,required=True)
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    run=args.run_directory.resolve()
    run.mkdir(parents=True,exist_ok=False)
    expected=json.loads((workspace/'expected.json').read_text())
    wheel=workspace/expected['wheel']['name']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert wheel.stat().st_size==expected['wheel']['bytes'] and sha(wheel)==expected['wheel']['sha256']
    checks=[]
    with zipfile.ZipFile(wheel) as archive:
        for item in expected['package_checks']:
            actual=hashlib.sha256(archive.read(item['path'])).hexdigest()
            assert actual==item['source_sha256']
            checks.append(dict(path=item['path'],source_sha256=item['source_sha256'],wheel_sha256=actual))
    start=time.perf_counter()
    target=run/'installed'
    tick=time.perf_counter()
    environment=dict(os.environ,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--no-compile','--target',str(target),str(wheel)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=environment)
    (run/'install.private.log').write_text(result.stdout)
    install_seconds=time.perf_counter()-tick
    assert result.returncode==0
    for item in checks:
        item['installed_sha256']=sha(target/item['path'])
        assert item['installed_sha256']==item['source_sha256']
    tick=time.perf_counter()
    result=subprocess.run([sys.executable,str(workspace/'verify_entry.private.py'),str(target),str(run/'entry.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=environment,timeout=120)
    (run/'entry.private.log').write_text(result.stdout)
    entry_seconds=time.perf_counter()-tick
    assert result.returncode==0
    entry=json.loads((run/'entry.json').read_text())
    assert entry['status']=='passed'
    receipt=dict(status='passed',scope='accelerator-b existing dependencies; independent wheel target/import/fresh CLI only',git_base=expected['git_base'],source_overlay=expected['source_overlay'],source_manifest_sha256=expected['source_manifest_sha256'],wheel=expected['wheel'],package_checks=checks,installation=dict(exit_code=0,seconds_including_io=install_seconds),entry=dict(exit_code=0,seconds_including_io=entry_seconds,checks=entry),total_seconds_including_install_and_entry=time.perf_counter()-start,verification_script_sha256=sha(Path(__file__)),entry_script_sha256=sha(workspace/'verify_entry.private.py'),python_version=sys.version.split()[0],cpu_affinity=sorted(os.sched_getaffinity(0)),runtime_gpu_precision='not_assessed',gpu_memory_bytes=None,whole_recon_all='not_assessed',clean_conda_full_reconstruction='not_assessed')
    data=json.dumps(receipt,indent=2)+'\n'
    (run/'report.json').write_text(data)
    (run/'COMPLETE.json').write_text(json.dumps(dict(status='complete',report_sha256=hashlib.sha256(data.encode()).hexdigest()))+'\n')
if __name__=='__main__':main()
