"""同一固定原生程序的低详细度诊断；只用于独立参考，不进入生产。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

from benchmark_native_ga_repeat import EXPECTED_INPUTS, EXPECTED_PREMESH, preflight, sha, write


def parse_control(log: Path) -> dict:
    """日志原打印值与完成代数；不推测未打印的总评分数或精确double。"""
    blocks = []; current = None
    for line in log.read_text().splitlines():
        match = re.fullmatch(r"CORRECTING DEFECT (\d+) \(vertices=(\d+), convex hull=(\d+), v0=(\d+)\)", line)
        if match:
            current = dict(defect_number=int(match[1]), initial_population_fitness_prints=[],
                generation_rows=[], improving_event_lines=[], deletion_event_lines=[])
            blocks.append(current)
        if current is None:
            continue
        match = re.search(r"for the patch #(\d+), we have fitness = ([-+0-9.eE]+)", line)
        if match:
            current['initial_population_fitness_prints'].append(dict(slot=int(match[1]), text=match[2], value=float(match[2])))
        match = re.search(r"generation (\d+) complete, optimal fitness = ([-+0-9.eE]+)", line)
        if match:
            text=match[2]
            mantissa, exponent=text.lower().split('e')
            decimals=len(mantissa.split('.')[1])
            centre=float(text);half_step=.5*10.0**(int(exponent)-decimals)
            current['generation_rows'].append(dict(completed_generation_number=int(match[1]),
                best_fitness_print=text, printed_value=centre,
                rounding_interval_lower=centre-half_step,rounding_interval_upper=centre+half_step))
        if 'new optimal fitness found at' in line:
            current['improving_event_lines'].append(line.strip())
        if any(s in line for s in ('Deleting worst vertices', 'vertices have been deleted',
                                    'Worst vertex has been deleted','Worst ')):
            current['deletion_event_lines'].append(line.strip())
    for row in blocks:
        numbers=[g['completed_generation_number']for g in row['generation_rows']]
        row['completed_generations_printed']=len(numbers)
        row['generation_prints_contiguous']=numbers==list(range(1,len(numbers)+1))
        row['last_generation_best_fitness_print']=row['generation_rows'][-1]if numbers else None
        row['exact_candidate_evaluations']=None
        row['exact_final_double_fitness']=None
        row['exact_termination_code']=None
        row['scope']='actual low-verbosity prints; source branch audit separate'
    return dict(defects=blocks, exact_total_evaluations=None,
        missing_fields_reason='VERBOSE_MODE_LOW does not print number_of_patches, exact final double or explicit termination code',
        log_sha256=sha(log))


def classify_surface(surface: dict, *, reference: Path, candidate: Path) -> dict:
    return dict(strict_coordinates=surface['coordinates_mm']['exact']==surface['coordinates_mm']['elements'],
        strict_ordered_faces=surface['ordered_faces']['exact']==surface['ordered_faces']['elements'],
        strict_geometric_fields=bool(surface['volume_geometry_data_exact']),
        strict_header_filename=bool(surface['volume_geometry_filename_equal']),
        strict_file_bytes=sha(reference)==sha(candidate))


def run(*,source,subject,benchmark,runtime,reference_log,binary,assets,comparison_module,
        baseline_repeat,output):
    started=time.perf_counter()
    check=preflight(source=source,subject=subject,benchmark=benchmark,runtime=runtime,
        reference_log=reference_log,binary=binary,assets=assets,comparison_module=comparison_module)
    baseline_report_bytes=(baseline_repeat/'report.json').read_bytes()
    baseline_report=json.loads(baseline_report_bytes)
    complete=json.loads((baseline_repeat/'COMPLETE.json').read_bytes())
    assert (baseline_repeat/'EXIT_CODE').read_text().strip()=='0'
    assert complete['status']=='complete'and baseline_report['status']=='complete'
    assert complete['report_sha256']==sha(baseline_repeat/'report.json')
    baseline_surface=baseline_repeat/'candidate/subject/surf/lh.orig.premesh'
    assert sha(baseline_surface)==baseline_report['candidate_premesh_sha256']
    if any((output/name).exists()for name in ('candidate','report.json','COMPLETE.json')):
        raise FileExistsError('fresh independent benchmark output required')
    output.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(output,0o700)
    candidate=output/'candidate/subject';candidate.mkdir(mode=0o700,parents=True)
    for name,expected in EXPECTED_INPUTS.items():
        destination=candidate/name;destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        shutil.copyfile(subject/name,destination);os.chmod(destination,0o600);assert sha(destination)==expected
    sys.path.insert(0,str(source/'src'))
    from fnit.recon_all.topology_preflight_python import write_centered_topology_sphere
    centred=candidate/'surf/lh.topology-centered.sphere'
    centering=write_centered_topology_sphere(candidate/'surf/lh.qsphere.nofix',centred)
    scripts=candidate/'scripts';scripts.mkdir(mode=0o700)
    log=scripts/'lh.topology-ga-fnit.log'
    env=dict(os.environ,SUBJECTS_DIR=str(candidate.parent),FREESURFER_HOME=str(assets),
        FNIT_CENTERED_COORDS=str(centred),OMP_NUM_THREADS='1')
    arguments=['-threads','1','-mgz','-sphere','qsphere.nofix','-inflated','inflated.nofix',
        '-orig','orig.nofix','-out','orig.premesh','-ga','-seed','1234','-threads','1',
        '-verbose_low','subject','lh']
    native_started=time.perf_counter()
    with log.open('w')as stream:
        process=subprocess.run([str(binary),*arguments],cwd=scripts,env=env,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    native_seconds=time.perf_counter()-native_started
    assert 'FNIT_CENTERED_SUBSTITUTION_USED'in log.read_text()
    result=candidate/'surf/lh.orig.premesh';assert result.is_file()
    spec=importlib.util.spec_from_file_location('_existing_surface_comparison',comparison_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    comparisons={}
    for label,reference in (('original_79',subject/'surf/lh.orig.premesh'),('default_repeat',baseline_surface)):
        surface=module._surface(reference,result)
        comparisons[label]=dict(surface=surface,classifications=classify_surface(surface,reference=reference,candidate=result),
            reference_sha256=sha(reference),candidate_sha256=sha(result))
    for name,expected in EXPECTED_INPUTS.items():assert sha(candidate/name)==expected==sha(subject/name)
    report=dict(status='complete',scope='isolated full native GA; sole additional argument -verbose_low',
        full_workflow_complete=False,preflight=check,baseline_repeat_report_sha256=sha(baseline_repeat/'report.json'),
        baseline_repeat_complete_sha256=sha(baseline_repeat/'COMPLETE.json'),
        original_premesh_sha256=EXPECTED_PREMESH,script_sha256=sha(Path(__file__)),
        shared_benchmark_tool_sha256=sha(Path(__file__).with_name('benchmark_native_ga_repeat.py')),
        command_arguments=arguments,native_seconds=native_seconds,
        comparisons=comparisons,control=parse_control(log),centering_sha256=sha(centred),
        inputs_after_unchanged=True,cpu_affinity=list(os.sched_getaffinity(0)),total_cpu_budget=4,
        actual_native_threads=1,process_exit_code=process.returncode,
        child_max_rss_bytes=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024),
        cuda_memory_bytes=None,shared_load_average=list(os.getloadavg()),
        shared_load_scope='same host, different time; verbosity overhead; not a speed pair',
        overall_metric_equivalence='not_assessed',full_diagnostic_wall_seconds=time.perf_counter()-started)
    write(output/'report.json',report)
    write(output/'COMPLETE.json',dict(status='complete',report_sha256=sha(output/'report.json'),
        full_workflow_complete=False,geometric_reproduction=all(all(c['classifications'][k]for k in
            ('strict_coordinates','strict_ordered_faces','strict_geometric_fields'))for c in comparisons.values())))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','subject','benchmark','runtime','reference-log','binary','assets',
                 'comparison-module','baseline-repeat','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({'status':result['status'],'native_seconds':result['native_seconds']}))
