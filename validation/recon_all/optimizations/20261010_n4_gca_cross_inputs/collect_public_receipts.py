"""显式收集本次完成诊断的JSON与误差PNG，复用现有脱敏器公开数值收据。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_public_receipts(*, run_directory: Path, output_directory: Path,
                           exporter_file: Path, replacements_file: Path,
                           failed_initialization_directory: Path,
                           plot_subdirectory: str = 'plots-v2') -> dict:
    """输入本工具run、已有JSON脱敏器和私有替换表，输出公开reports/figures。

    五个路径必填无默认；plot_subdirectory默认plots-v2且只能是目录名。
    只选择两例report/controls/plot-source/controller
    JSON与两张已授权公开T1派生PNG，不复制MRI、权重、许可证或原生程序。
    失败初始化只记录真实退出码与日志SHA。原件SHA、公开SHA及数字类型
    不变合同沿用已有exporter；任何未完成/覆盖/图片哈希失败拒绝。
    """
    if output_directory.exists():raise FileExistsError(output_directory)
    if Path(plot_subdirectory).name!=plot_subdirectory or plot_subdirectory in ('.','..'):raise ValueError('plot directory name required')
    for sentinel in ('controller.exit','analysis.exit',plot_subdirectory+'.exit'):
        if (run_directory/sentinel).read_text().strip()!='0':raise ValueError('successful task required: '+sentinel)
    staging=run_directory/'publication-staging-v1'
    if staging.exists():raise FileExistsError(staging)
    staging.mkdir()
    devices=set();unmapped=True
    for subject in ('sub06','sub07'):
        source=run_directory/subject
        report=json.loads((source/'report.json').read_text());controls=json.loads((source/'controls-analysis-v1.json').read_text())
        plot=json.loads((source/plot_subdirectory/'source.json').read_text())
        if report['status']!='complete' or controls['status']!='diagnostic_complete' or plot['status']!='diagnostic_complete':raise ValueError('incomplete '+subject)
        if controls['diagnostic_report_sha256']!=sha(source/'report.json'):raise ValueError('analysis/report binding changed')
        if plot['figure_sha256']!=sha(source/plot_subdirectory/'error_slices.png'):raise ValueError('plot hash changed')
        devices.add(report['device'])
        unmapped=unmapped and all(item['registration']['cuda_visible_devices'] is None
                                 for item in report['backends']['torch'].values())
        for local,name in (('report.json','report'),('controls-analysis-v1.json','controls'),(plot_subdirectory+'/source.json','plot')):
            shutil.copyfile(source/local,staging/(subject+'.'+name+'.json'))
    shutil.copyfile(run_directory/'controller.json',staging/'controller.json')
    # 只追加完成后的硬件身份观测，不伪装成阶段起点/连续显存采样。
    if len(devices)!=1:raise ValueError('two-case target device mismatch')
    device=next(iter(devices));index=int(device.removeprefix('cuda:')) if unmapped else None
    gpu=subprocess.run(['nvidia-smi','--id='+str(index),'--query-gpu=index,uuid,name,pci.bus_id,driver_version',
                        '--format=csv,noheader'],text=True,capture_output=True,check=False) if index is not None else None
    cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')),None)
    hardware={'scope':'hardware observation after completed diagnostic; not start snapshot or memory budget proof',
        'observed_utc':datetime.now(timezone.utc).isoformat(),'cpu_model':cpu_model,
        'logical_device':device,'target_nvidia_index':index,'gpu_identity_query_exit_code':gpu.returncode if gpu else None,
        'gpu_identity_csv':gpu.stdout.strip() if gpu else None,'gpu_identity_stderr':gpu.stderr.strip() if gpu else None,
        'process_tree_simultaneous_gpu_memory_bytes':None,
        'gpu_memory_scope':'existing individual cached worker allocated/reserved peaks only; total process-tree memory not measured'}
    (staging/'hardware_after_diagnostic.json').write_text(json.dumps(hardware,indent=2)+'\n')
    logfile=failed_initialization_directory/'controller.log'
    failure={'scope':'v1 initialization only; no registration execution',
        'exit_code':int((failed_initialization_directory/'controller.exit').read_text()),
        'controller_log_sha256':sha(logfile),
        'failure_line':[line for line in logfile.read_text().splitlines() if 'FileNotFoundError:' in line]}
    (staging/'failed_initialization_v1.json').write_text(json.dumps(failure,indent=2)+'\n')
    initial_plot_exit=run_directory/'plots.exit'
    if initial_plot_exit.exists() and initial_plot_exit.read_text().strip()!='0':
        plotlog=run_directory/'plots.log'
        plot_failure={'scope':'initial plot metadata failure; numeric diagnostics unaffected',
            'exit_code':int(initial_plot_exit.read_text()),'plot_log_sha256':sha(plotlog),
            'error_line':[line for line in plotlog.read_text().splitlines() if 'TypeError:' in line],
            'successful_plot_directory':plot_subdirectory}
        (staging/'failed_plot_metadata_v1.json').write_text(json.dumps(plot_failure,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('existing_receipt_exporter',exporter_file)
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    output_directory.mkdir(parents=True)
    manifest=exporter.export_receipts(input_directory=staging,output_directory=output_directory/'reports',replacements_file=replacements_file)
    figures=output_directory/'figures';figures.mkdir()
    for subject in ('sub06','sub07'):
        shutil.copyfile(run_directory/subject/plot_subdirectory/'error_slices.png',figures/(subject+'_error_slices.png'))
    result={'status':'public_export_complete','scope':'explicit completed diagnostic receipts and public MRI-derived figures only',
        'collector_sha256':sha(Path(__file__)),'exporter_sha256':sha(exporter_file),
        'files_sha256':{str(path.relative_to(output_directory)):sha(path) for path in sorted(output_directory.rglob('*')) if path.is_file()},
        'numeric_boolean_null_values_unchanged':manifest['numeric_boolean_null_values_unchanged']}
    (output_directory/'EXPORT_PROVENANCE.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run_directory','output_directory','exporter_file','replacements_file','failed_initialization_directory'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--plot-subdirectory',default='plots-v2')
    collect_public_receipts(**vars(parser.parse_args()))
