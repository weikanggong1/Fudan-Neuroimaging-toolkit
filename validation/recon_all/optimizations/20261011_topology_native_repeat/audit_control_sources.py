"""原生默认/原日志与已有自有首缺陷控制记录的只读对齐；不执行GA。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*,repository,report_directory,fixed_defect_source,fixed_main_source,fixed_utils_source):
    report_directory.mkdir(parents=True,exist_ok=True)
    scope=repository/'validation/recon_all/optimizations/20261011_topology_native_repeat'
    previous=repository/'validation/recon_all/optimizations/20261010_topology_genetic_search/reports'
    original=json.loads((scope/'reports/default_report.json').read_bytes())
    verbose=json.loads((scope/'reports/verbose_report.json').read_bytes())
    own=json.loads((previous/'v2_search_control.json').read_bytes())
    own_stage=json.loads((previous/'v2_stage.json').read_bytes())
    own_source=json.loads((previous/'v2_runtime_source.json').read_bytes())
    modules={}
    for name in ('topology_ga_population.py','topology_ga_selection.py','topology_genetic_search.py','gcsa_permutation.py'):
        relative='src/fnit/recon_all/'+name;path=repository/relative
        modules[relative]=dict(sha256=digest(path),bytes=path.stat().st_size,
            tested_first_ga_sha256=own_source['payload'][relative]['sha256'],
            bytes_match_tested_first_ga=digest(path)==own_source['payload'][relative]['sha256'])
    sources={label:dict(sha256=digest(path),bytes=path.stat().st_size)for label,path in
        (('utils/mrisurf_defect.cpp',fixed_defect_source),('mris_fix_topology/mris_fix_topology.cpp',fixed_main_source),
         ('utils/utils.cpp',fixed_utils_source))}
    expected_sources={
        'utils/mrisurf_defect.cpp':'55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187',
        'mris_fix_topology/mris_fix_topology.cpp':'55cca5a4f8a82baee768a94bd545955bd805a4ad446fa19995511240a5e09be1',
        'utils/utils.cpp':'c19ae738cd9d22fada40c2f5f89ff790512fcbeab1d7c08059b249134db2eac5',
    }
    for name,expected in expected_sources.items():
        if sources[name]['sha256']!=expected:raise ValueError('fixed source SHA mismatch: '+name)
    rows=[
        ('population_size','10','mris_fix_topology/mris_fix_topology.cpp',112,'topology_ga_population.py',169),
        ('max_unchanged','10, not fixed generations','mris_fix_topology/mris_fix_topology.cpp',113,'topology_ga_population.py',170),
        ('niters','-1, no finite cap','mris_fix_topology/mris_fix_topology.cpp',127,'topology_ga_population.py',171),
        ('initial_selection','True via -ga','mris_fix_topology/mris_fix_topology.cpp',544,'topology_ga_population.py',225),
        ('vertex_eliminate','True via -ga','mris_fix_topology/mris_fix_topology.cpp',542,'topology_ga_population.py',275),
        ('max_unchanged_runtime_assignment','parms overrides old static default3','utils/mrisurf_defect.cpp',12200,'topology_ga_population.py',170),
        ('initial_ordering','slot0 identity; 11 ordered shuffle passes','utils/mrisurf_defect.cpp',4768,'topology_ga_population.py',96),
        ('generation_condition','nunchanged < max_unchanged','utils/mrisurf_defect.cpp',12801,'topology_ga_population.py',231),
        ('strict_improvement','reset unchanged and save RP only on >','utils/mrisurf_defect.cpp',12864,'topology_ga_population.py',215),
        ('generation_end_FEQUAL','unchanged++ else reset/euthanasia=-1','utils/mrisurf_defect.cpp',13261,'topology_ga_population.py',271),
        ('euthanasia_wait','5 completed generations; no reported deletion also stops','utils/mrisurf_defect.cpp',13281,'topology_ga_population.py',277),
        ('periodic_elimination','generation>=10 and modulo3==0','utils/mrisurf_defect.cpp',13307,'topology_ga_population.py',284),
        ('seed_priming','one OpenRan1 on fresh seed1234','utils/utils.cpp',77,'topology_genetic_search.py',30),
        ('random_interval','double low + (high-low)*OpenRan1','utils/utils.cpp',106,'topology_genetic_search.py',45),
        ('native_mutation_counter','strict best improvements, not attempts','utils/mrisurf_defect.cpp',12907,'topology_ga_population.py',240),
        ('native_crossover_counter','includes improved crossover_mutation','utils/mrisurf_defect.cpp',13216,'topology_ga_population.py',262),
    ]
    audit=[dict(item=k,source_semantics=value,original_file=of,original_line=ol,
        own_file='src/fnit/recon_all/'+pf,own_line=pl,scope='static source comparison')for k,value,of,ol,pf,pl in rows]
    native_first=verbose['control']['defects'][0]
    own_initial=[x for x in own['evaluation_trace']if x['phase']=='initial']
    initial=[]
    for actual,observed in zip(own_initial,native_first['initial_population_fitness_prints']):
        initial.append(dict(slot=actual['slot'],own_fitness=actual['fitness'],
            own_six_decimal_print=format(actual['fitness'],'.6f'),native_six_decimal_print=observed['text'],
            printed_exact=format(actual['fitness'],'.6f')==observed['text']))
    generations=[]
    for actual,observed in zip(own['generation_trace'],native_first['generation_rows']):
        generations.append(dict(own_zero_based_generation=actual['generation'],
            native_completed_generation_number=observed['completed_generation_number'],
            own_best_fitness=actual['best_fitness'],native_best_print=observed['best_fitness_print'],
            printed_exact=format(actual['best_fitness'],'.4e')==observed['best_fitness_print']))
    own_deletions=[x['reported_deleted']for x in own['generation_trace']if x['elimination_fraction']is not None]
    native_deletions=[]
    for line in native_first['deletion_event_lines']:
        match=re.search(r'(\d+) vertices have been deleted|Worst (\d+) vertices have been deleted',line)
        if match:native_deletions.append(int(match[1]or match[2]))
        elif 'Worst vertex has been deleted'in line:native_deletions.append(1)
    improvements=[x for x in own['evaluation_trace']if x['strict_improvement']and x['phase']!='initial']
    native_improvements=[]
    for line in native_first['improving_event_lines']:
        match=re.search(r'new optimal fitness found at (\d+): ([-+0-9.eE]+)',line)
        if match and ('CROSSOVER'in line or 'MUTATION'in line):
            phase='crossover_mutation'if'CROSSOVER'in line and 'MUTATION'in line else'crossover'if'CROSSOVER'in line else'replacement'
            native_improvements.append(dict(phase=phase,slot=int(match[1]),fitness_print=match[2]))
    improvement_comparison=[dict(own_generation=actual['generation'],own_phase=actual['phase'],
        native_phase=observed['phase'],own_slot=actual['slot'],native_slot=observed['slot'],
        own_fitness=actual['fitness'],native_fitness_print=observed['fitness_print'],
        phase_slot_printed_exact=actual['phase']==observed['phase']and actual['slot']==observed['slot']
            and format(actual['fitness'],'.4e')==observed['fitness_print'])for actual,observed in zip(improvements,native_improvements)]
    final=native_first['last_generation_best_fitness_print']
    classified=dict(strict_coordinates=original['surface_comparison']['coordinates_mm']['exact']==original['surface_comparison']['coordinates_mm']['elements'],
        strict_ordered_faces=original['surface_comparison']['ordered_faces']['exact']==original['surface_comparison']['ordered_faces']['elements'],
        strict_geometric_fields=original['surface_comparison']['volume_geometry_data_exact'],
        strict_header_filename=original['surface_comparison']['volume_geometry_filename_equal'],
        strict_file_bytes=original['strict_file_reproduction'],
        original_legacy_strict_decoded_field_preserved=original['strict_decoded_reproduction'])
    result=dict(scope='native same-input full GA repeat and printed first-defect controls; not own full-GA geometry comparison',
        script_sha256=digest(Path(__file__)),fixed_original_commit='d932c45b7941662ea380a05efef580568b98d41a',
        fixed_source=sources,own_modules=modules,static_comparisons=audit,
        default_report_sha256=digest(scope/'reports/default_report.json'),
        verbose_report_sha256=digest(scope/'reports/verbose_report.json'),
        own_first_ga_control_sha256=digest(previous/'v2_search_control.json'),
        own_first_ga_report_sha256=digest(previous/'v2_stage.json'),
        own_first_ga_runtime_manifest_sha256=digest(previous/'v2_runtime_source.json'),
        own_first_ga_git_commit_at_bundle=own_source['git_commit_at_bundle'],
        geometric_reproduction_classifications=classified,
        native_first=dict(completed_generations_printed=len(native_first['generation_rows']),
            exact_total_evaluations=None,exact_final_double_fitness=None,exact_rng_initial_state=None,
            exact_termination_code=None,last_printed_best_fitness=final),
        own_first=dict(reported_generations=own_stage['search_generations'],completed_generation_trace_rows=len(own['generation_trace']),
            counts=own['counts'],best_fitness=own_stage['best_fitness'],termination=own_stage['search_termination']),
        initial_fitness_comparison=initial,printed_initial_fitness_equal_count=sum(x['printed_exact']for x in initial),
        generation_comparison=generations,printed_generation_best_equal_count=sum(x['printed_exact']for x in generations),
        own_deletion_counts=own_deletions,native_deletion_counts=native_deletions,ordered_deletion_counts_exact=own_deletions==native_deletions,
        improvement_event_comparison=improvement_comparison,
        printed_improvement_event_sequence_exact=len(improvements)==len(native_improvements)and all(x['phase_slot_printed_exact']for x in improvement_comparison),
        own_best_in_native_last_print_rounding_interval=final['rounding_interval_lower']<=own_stage['best_fitness']<=final['rounding_interval_upper'],
        candidate_orderings_or_rng_trace_same_input_gate='not_assessed',
        control_mismatch_explains_runtime='not_supported_by_current_evidence',
        global_native_mutation_count_scope='all nine defects strict improvements; cannot compare to first-defect mutation_attempts',
        diagnostic_counter_difference='own crossover_mutation improvement counts only mutation_improvements; original global increments both nmut and ncross',
        production_changed=False,overall_metric_equivalence='not_assessed')
    out=report_directory/'control_audit.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','report-directory','fixed-defect-source','fixed-main-source','fixed-utils-source'):
        p.add_argument('--'+name,type=Path,required=True)
    v=run(**vars(p.parse_args()));print(json.dumps({k:v[k]for k in('printed_initial_fitness_equal_count','printed_generation_best_equal_count','ordered_deletion_counts_exact','printed_improvement_event_sequence_exact')}))
