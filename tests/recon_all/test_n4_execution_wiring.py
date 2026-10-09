"""N4显式隔离路由与生产/批次/benchmark参数契约；真实影像另测。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fnit.recon_all import input_n4_chain as chain, native_free, n4_torch_worker
from fnit.recon_all.batch import run_recon_all_python_batch


@pytest.mark.parametrize("backend,execution,device", [
    ("invalid", "in-process", "cuda:0"), ("torch", "invalid", "cuda:0"),
    ("native", "isolated", "cuda:0"), ("torch", "isolated", "cpu"),
    ("torch", "isolated", "cuda"), ("torch", "isolated", "invalid-device"),
])
def test_input_and_production_reject_before_creating_outputs(monkeypatch,tmp_path,backend,execution,device):
    def forbidden(*args,**kwargs):raise AssertionError("upstream must not run")
    monkeypatch.setattr(chain,"run_input_talairach_chain",forbidden)
    for function,extra in ((chain.run_input_n4_chain,{"n4_binary":"own-n4"}),
                           (native_free.run_recon_all_python,{})):
        output=tmp_path/function.__name__
        with pytest.raises(ValueError):
            function(t1="raw.nii.gz",subject_dir=output,weights_dir="weights",assets_dir="assets",
                     device=device,n4_backend=backend,n4_execution=execution,**extra)
        assert not output.exists()


def test_input_chain_isolation_reuses_complete_worker_and_own_orig(monkeypatch,tmp_path):
    subject=tmp_path/"subject";seen={}
    def upstream(*args,**kwargs):
        (subject/"mri").mkdir(parents=True)
        return {"talairach_xfm":"self.xfm"}
    def worker(**kwargs):
        seen.update(kwargs)
        return {"api":{"iterations":200},"allocated_peak_bytes":123,"source_sha256":{"full_n4":"frozen"}}
    monkeypatch.setattr(chain,"run_input_talairach_chain",upstream)
    monkeypatch.setattr(n4_torch_worker,"run_isolated_n4",worker)
    monkeypatch.setattr(chain,"make_nu",lambda *args:(1.,(2,3)))
    result=chain.run_input_n4_chain(t1="raw.nii.gz",subject_dir=subject,weights_dir="weights",assets_dir="assets",
        device="cuda:1",threads=2,n4_backend="torch",n4_execution="isolated",profile=True)
    assert seen=={"input_path":subject/"mri/orig.mgz","output_path":subject/"mri/tmp/nu0.mgz",
                  "report_path":subject/"scripts/n4-isolated.json","device":"cuda:1","threads":2,"profile":True}
    assert result["n4_details"]["api"]["iterations"]==200
    assert result["n4_details"]["allocated_peak_bytes"]==123
    assert result["n4_execution"]=="isolated" and result["production_default_changed"] is False


def test_public_wrapper_and_cli_forward_nondefault_n4(monkeypatch,tmp_path):
    subject=tmp_path/"subject";subject.mkdir();seen={}
    def internal(**kwargs):seen.update(kwargs);return {"total_seconds":0.}
    monkeypatch.setattr(native_free,"_run_recon_all_python",internal)
    native_free.run_recon_all_python(t1="raw.nii.gz",subject_dir=subject,weights_dir="weights",assets_dir="assets",
                                     n4_backend="torch",n4_execution="isolated")
    assert seen["n4_backend"]=="torch" and seen["n4_execution"]=="isolated"
    def api(*args,**kwargs):seen.clear();seen.update(kwargs);return {"status":"complete"}
    monkeypatch.setattr(native_free,"run_recon_all_python",api)
    with contextlib.redirect_stdout(io.StringIO()):
        native_free.main(["raw.nii.gz","unused","--weights-dir","weights","--assets-dir","assets",
                          "--n4-backend","torch","--n4-execution","isolated"])
    assert seen["n4_backend"]=="torch" and seen["n4_execution"]=="isolated"
    with contextlib.redirect_stdout(io.StringIO()):
        native_free.main(["raw.nii.gz","unused","--weights-dir","weights","--assets-dir","assets"])
    assert seen["n4_backend"]=="native" and seen["n4_execution"]=="in-process"


def test_batch_n4_options_reach_each_cli_and_invalid_combo_never_dispatches(monkeypatch,tmp_path):
    weights,assets=tmp_path/"weights",tmp_path/"assets";weights.mkdir();assets.mkdir()
    raw=tmp_path/"raw.nii.gz";raw.write_bytes(b"contract only")
    subject=tmp_path/"subject";commands=[]
    def run(command,**kwargs):
        commands.append(command);subject.mkdir()
        (subject/"fnit-native-free-run.json").write_text(json.dumps({"status":"complete"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("fnit.recon_all.batch.subprocess.run",run)
    run_recon_all_python_batch([{"t1":raw,"subject_dir":subject}],weights,assets,
                              n4_backend="torch",n4_execution="isolated")
    command=commands[0]
    assert command[command.index("--n4-backend")+1]=="torch"
    assert command[command.index("--n4-execution")+1]=="isolated"
    with pytest.raises(ValueError,match="isolated N4"):
        run_recon_all_python_batch([],weights,assets,n4_backend="native",n4_execution="isolated")
    assert len(commands)==1


def test_benchmark_n4_flags_and_preflight(monkeypatch,tmp_path):
    script=Path(__file__).resolve().parents[2]/"tools/benchmark_recon_torch_end_to_end.py"
    spec=importlib.util.spec_from_file_location("n4_benchmark_contract",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ("weights","assets","native"):(tmp_path/name).mkdir()
    raw=tmp_path/"raw.nii.gz";raw.write_bytes(b"contract only")
    output=tmp_path/"output";seen={}
    arguments=[str(script),"--t1",str(raw),"--output-root",str(output),"--weights-dir",str(tmp_path/"weights"),
        "--assets-dir",str(tmp_path/"assets"),"--native-bin-dir",str(tmp_path/"native"),"--code-version","contract",
        "--n4-execution","isolated"]
    monkeypatch.setattr(sys,"argv",arguments)
    with pytest.raises(ValueError,match="isolated N4"):module.main()
    assert not output.exists()
    class Sampler:
        def __init__(self,**kwargs):pass
        def report(self):return {}
    monkeypatch.setattr(module,"ProcessTreeDeviceSampler",Sampler)
    monkeypatch.setattr(module,"sha",lambda path:"contract-hash")
    def launch(command,**kwargs):
        seen["command"]=command
        return SimpleNamespace(pid=123,returncode=0,poll=lambda:0)
    monkeypatch.setattr(module.subprocess,"Popen",launch)
    monkeypatch.setattr(sys,"argv",arguments+["--n4-backend","torch"])
    assert module.main()==0
    command=seen["command"]
    assert command[command.index("--n4-backend")+1]=="torch"
    assert command[command.index("--n4-execution")+1]=="isolated"
