"""离线构建、独立安装并核对 GA 配方入口；不运行影像重建。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


PACKAGE_FILES = tuple("fnit/recon_all/" + name for name in (
    "native_free.py", "python_profile.py", "batch.py", "hemisphere_parallel.py",
    "topology_genetic_files.py", "topology_genetic_repair.py",
    "topology_genetic_search.py", "topology_retessellation_python.py",
    "topology_retessellation_dynamic.py", "topology_edge_blocker_scan.py",
    "input_n4_chain.py", "n4_simpleitk.py", "execution_recipe.py",
    "python_runtime.py", "mni_nonlinear_chain.py", "surface_stats_cache.py",
    "mri_em_register.py", "mri_em_register_python.py", "mri_em_register_complete.py",
    "mri_em_register_em_candidate.py", "topology_gpu_distance.py",
    "topology_gpu_distance_kernel.py", "topology_patch_fitness.py",
    "topology_candidate_pipeline.py", "gca_torch_worker.py", "wm_torch_worker.py",
    "talairach_synthmorph.py", "n4_torch_worker.py", "worker_program_guard.py",
    "place_pial_python.py", "place_surface_final_cleanup.py",
    "place_surface_quality.py", "place_surface_observer.py",
    "place_surface_benchmark_memory.py"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(*, source_directory: Path, source_manifest: Path,
           output_directory: Path, python_executable: Path) -> dict:
    """核对冻结清单，离线安装 wheel 并执行真实 import 和 fresh CLI help。

    source_directory是含pyproject/README/licenses/src的已冻结源码目录；
    source_manifest是JSON，files为相对路径→bytes/sha256，git_base及
    source_overlay记录来源。output_directory须不存在，python_executable
    为已装项目依赖及setuptools/wheel的解释器。无MRI或坐标输入，不下载
    依赖、不运行原软件或recon-all。返回构建/安装/入口的wall、哈希和
    精度状态dict，并写report.json；清单、构建或入口失败抛异常。
    入口复核完整EM、有序阻挡扫描、可选GPU距离、程序边界、MNI与统计模块及库参数，
    另核对pial源rip默认和五个放置/质量/观测模块的安装字节，
    按具名选项验证配方，不把字典键数当作算法完整性或阶段计数；
    不执行JIT或影像计算。
    另复用所选SimpleITK N4和完整Python配方的生产依赖预检，
    核对固定版本和扩展库SHA，并调用实际安装的依赖检查console入口；
    NVML保持可选，不读资产、不查询驱动、不运行N4或初始化CUDA。详细参数、中文示例
    和实测范围见PYTHON_GENETIC_PROFILE.md。
    """
    source_directory = source_directory.resolve()
    source_manifest = source_manifest.resolve()
    output_directory = output_directory.resolve()
    python_executable = python_executable.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if output_directory.is_relative_to(source_directory):
        raise ValueError("installation output must be outside frozen source")
    manifest = json.loads(source_manifest.read_text())
    files = manifest["files"]
    for relative, expected in files.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("manifest requires source-relative paths: " + relative)
        source = (source_directory / relative).resolve()
        if not source.is_relative_to(source_directory) or not source.is_file():
            raise ValueError("manifest path is missing or outside source: " + relative)
        if source.stat().st_size != expected["bytes"] or _sha(source) != expected["sha256"]:
            raise ValueError("source size/SHA differs: " + relative)
    output_directory.mkdir(parents=True)
    # pip的构建元数据仅写在本次新目录，不改变科学运行的冻结源码。
    build_source = output_directory / "build-source"
    for relative in files:
        destination = build_source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_directory / relative, destination)

    def run(arguments, name):
        tick = time.perf_counter()
        result = subprocess.run([str(python_executable), *arguments], cwd=output_directory,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output_directory / (name + ".private.log")).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(name + " failed; inspect its private log")
        return dict(exit_code=result.returncode, seconds_including_io=time.perf_counter() - tick)

    build = run(["-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
                 "--wheel-dir", str(output_directory / "dist"), str(build_source)], "build")
    wheels = list((output_directory / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("one newly built FNIT wheel required")
    wheel = wheels[0]
    checks = []
    with zipfile.ZipFile(wheel) as archive:
        for relative in PACKAGE_FILES:
            expected = files["src/" + relative]["sha256"]
            actual = hashlib.sha256(archive.read(relative)).hexdigest()
            if actual != expected:
                raise RuntimeError("wheel source differs: " + relative)
            checks.append(dict(path=relative, source_sha256=expected, wheel_sha256=actual))
    target = output_directory / "installed"
    install = run(["-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
                   "--target", str(target), str(wheel)], "install")
    for check in checks:
        installed = (target / check["path"]).resolve()
        if not installed.is_relative_to(target.resolve()) or not installed.is_file():
            raise RuntimeError("installed source is missing or outside target: " + check["path"])
        actual = _sha(installed)
        if actual != check["source_sha256"]:
            raise RuntimeError("installed source differs: " + check["path"])
        check["installed_sha256"] = actual
    worker = output_directory / "verify_entry.private.py"
    worker.write_text('''import sys,json,subprocess,time,hashlib,inspect,os
from pathlib import Path
target=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(target))
import torch
def precision():
    return dict(cuda_initialized=torch.cuda.is_initialized(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
before=precision()
import fnit.recon_all.native_free as pipeline
import fnit.recon_all.topology_genetic_files as ga
import fnit.recon_all.topology_retessellation_dynamic as retessellation
import fnit.recon_all.topology_edge_blocker_scan as blocker_scan
from fnit.recon_all.python_profile import python_profile_options
from fnit.recon_all.execution_recipe import python_runtime_selection, execution_dag
from fnit.recon_all.mri_em_register_complete import align_em
from fnit.recon_all.topology_gpu_distance import gpu_defect_signed_distance
from fnit.recon_all.worker_program_guard import inherit_original_stage_guard
from fnit.recon_all.input_n4_chain import require_n4_runtime_dependencies
from fnit.recon_all.python_runtime import check_python_runtime
from fnit.recon_all.place_pial_python import place_pial_t1
assert Path(pipeline.__file__).resolve().is_relative_to(target)
assert Path(ga.__file__).resolve().is_relative_to(target)
assert Path(retessellation.__file__).resolve().is_relative_to(target)
assert Path(blocker_scan.__file__).resolve().is_relative_to(target)
blocker_default=inspect.signature(ga.run_genetic_topology_files).parameters['blocker_scan_backend'].default
assert blocker_default=='python'
profile=python_profile_options(device='cuda:0',threads=4,topology_backend='python-ga')
assert profile['topology_backend']=='python-ga' and 'native' not in profile.values()
assert profile['input_execution']=='serial' and profile['native_optimizations']=='torch'
assert profile['surface_placement_backend']=='python' and profile['mni_execution']=='parallel-late'
runtime_selection=python_runtime_selection()
assert runtime_selection['em_algorithm_completion']=='complete_stable_and_all_sample_protocol'
assert runtime_selection['white_binary'] is None and runtime_selection['pial_binary'] is None
assert callable(align_em) and callable(gpu_defect_signed_distance) and callable(inherit_original_stage_guard)
pial_cleanup_default=inspect.signature(place_pial_t1).parameters['cleanup_constraint_mode'].default
assert pial_cleanup_default=='source_rip'
dag=execution_dag(threads=4,input_execution='serial',mni_execution='parallel-late',hemisphere_workers=2,backend='python')
assert not dag['public_default_is_python'] and dag['overall_equivalence']=='not_assessed'
runtime_dependencies=require_n4_runtime_dependencies(n4_backend=profile['n4_backend'])
python_runtime=check_python_runtime(require_nvml=False,assets_dir=None)
assert python_runtime['status']=='passed',python_runtime
after=precision()
assert before==after
cli=[]
for module,required in [('fnit.recon_all.native_free',['python-ga','--topology-backend','--backend']),
                       ('fnit.recon_all.topology_genetic_files',['--final-orig','--remesh-scalar-storage','--threads','--blocker-scan-backend']),
                       ('fnit.recon_all.gca_torch_worker',['--nu','--mask','--atlas','--code-version']),
                       ('fnit.recon_all.wm_torch_worker',['--source','--histogram-batch-size','--code-version']),
                       ('fnit.recon_all.talairach_synthmorph',['--weights','--memory-report']),
                       ('fnit.recon_all.n4_torch_worker',['--matmul-tf32','--cudnn-tf32','--code-version'])]:
    tick=time.perf_counter()
    command=[sys.executable,'-c',"import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name='__main__')",str(target),module,'--help']
    cli_environment=dict(os.environ)
    worker_guard=module in {'fnit.recon_all.gca_torch_worker','fnit.recon_all.wm_torch_worker',
        'fnit.recon_all.talairach_synthmorph','fnit.recon_all.n4_torch_worker'}
    if worker_guard:cli_environment['FNIT_RECON_FORBID_ORIGINAL_STAGES']='1'
    completed=subprocess.run(command,env=cli_environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    assert completed.returncode==0 and all(option in completed.stdout for option in required)
    cli.append(dict(module=module,exit_code=completed.returncode,
        fresh_cli_seconds_including_imports=time.perf_counter()-tick,
        output_sha256=hashlib.sha256(completed.stdout.encode()).hexdigest(),verified_options=required,
        inherited_program_guard_enabled=worker_guard))
environment=dict(os.environ)
environment['PYTHONPATH']=str(target)+os.pathsep+environment.get('PYTHONPATH','')
dependency_output=target.parent/'installed_dependency_cli.json'
tick=time.perf_counter()
completed=subprocess.run([sys.executable,str(target/'bin/fnit-verify-recon-all-python'),
    '--output',str(dependency_output)],env=environment,stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,text=True,timeout=60)
assert completed.returncode==0,completed.stdout
dependency_cli=json.loads(dependency_output.read_text())
assert dependency_cli['status']=='passed'
cli.append(dict(entry='fnit-verify-recon-all-python',exit_code=completed.returncode,
    fresh_cli_seconds_including_imports=time.perf_counter()-tick,
    report_sha256=hashlib.sha256(dependency_output.read_bytes()).hexdigest(),
    installed_target_first_in_pythonpath=True,report=dependency_cli))
assert precision()==before
report=dict(status='passed',python_version=sys.version.split()[0],torch_version=torch.__version__,
    ga_library_blocker_scan_backend_default=blocker_default,
    pial_library_cleanup_constraint_mode_default=pial_cleanup_default,
    selected_profile=profile,selected_profile_key_count=len(profile),
    selected_runtime=runtime_selection,declared_execution_dag=dag,
    recipe_validation='named backend and dependency-DAG options; no fixed key-count gate',
    n4_runtime_dependencies=runtime_dependencies,python_runtime_dependencies=python_runtime,
    precision_and_cuda_at_entry=before,
    precision_and_cuda_after_import_and_profile_resolution=after,cli=cli)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\\n')
''')
    entry = run([str(worker), str(target), str(output_directory / "entry.json")], "entry")
    report = dict(status="passed", scope="offline wheel/import/CLI; existing dependencies",
                  source_manifest_sha256=_sha(source_manifest), source_file_count=len(files),
                  git_base=manifest.get("git_base"), source_overlay=manifest.get("source_overlay", []),
                  wheel=dict(name=wheel.name, bytes=wheel.stat().st_size, sha256=_sha(wheel)),
                  package_checks=checks, build=build, install=install, entry=entry,
                  entry_checks=json.loads((output_directory / "entry.json").read_text()),
                  verification_script_sha256=_sha(__file__),
                  dependency_installation="existing environment; --no-deps --no-index",
                  whole_recon_all="not_assessed", clean_conda_full_reconstruction="not_assessed",
                  runtime_gpu_precision="not_assessed", gpu_memory_bytes=None)
    (output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_directory", "source_manifest", "output_directory", "python_executable"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    print(json.dumps(verify(**vars(parser.parse_args(argv))), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
