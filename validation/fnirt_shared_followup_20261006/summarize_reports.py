"""Read-only scalar gates and one metadata summary; no MRI or solver execution."""
import hashlib
import json
from pathlib import Path


def bound(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    root=Path(__file__).resolve().parent
    def read(relative):return json.loads((root/relative).read_text())
    dynamic=read('reports/v3/arithmetic/summary.public.json')
    contracts=read('reports/v3/arithmetic/contracts.public.json')
    assembly=read('reports/v2/assembly/summary.public.json')
    local=read('reports/v2/arithmetic/contracts.public.json')
    rows=dynamic['rows']
    actual={
      'contracts_93_all_exact':len(contracts['rows'])==93 and all(row['exact'] for row in contracts['rows']),
      'dynamic_all_69_vector_bits_exact':len(rows)==69 and all(all(v['different_bits']==0 for v in row['native_vector_comparisons'].values()) for row in rows),
      'dynamic_all_69_scalar_fields_exact':len(rows)==69 and all(all(row['native_scalar_exact'].values()) for row in rows),
      'dynamic_solution_all_1177_bits_exact':dynamic['solution_vs_native']['different_bits']==0 and dynamic['solution_vs_native']['shape']==[1177],
      'natural_stop_unchanged':dynamic['iterations']==69 and dynamic['converged'] and dynamic['tolerance']==1e-3 and dynamic['maximum_iterations']==500,
      'assembly_finite_full_H':assembly['H_current_vs_native']['shape']==[1177,1177] and assembly['H_current_vs_native']['all_finite'],
      'assembly_CPU_no_CUDA':not assembly['cuda_initialized'] and assembly['coefficient_dtype']=='torch.float64',
      'all503_bindings_each_attempt':all(read(f'reports/v{k}/preflight.public.json')['all_bindings_match'] for k in (1,2,3))}
    if not all(actual.values()):raise RuntimeError('a scalar acceptance contract failed')
    g=assembly['rhs_current_lm_vs_native']['relative_l2'];ordered=assembly['rhs_current_fsl_order_vs_native']['relative_l2']
    summary={'scope':'bounded shared-system diagnostic recovery; no production patch/full nonlinear registration',
             'frozen_current_source_head':'6f62404023fb566fcfa9e2c20d77c2c7e79796e5',
             'gates':actual,'production_changed':False,'full_registration_tested':False,
             'stock_cache_restored':False,'GPU_tested':False,
             'arithmetic':{key:dynamic[key] for key in ('iterations','relative_residual','true_relative_residual','solution_vs_native','wall_seconds')},
             'native_context_contracts':{'rows':93,'all_exact':True,'identity':contracts['blas_symbol_bindings']['identity'],
                                          'success_level':'whole independently linked observer context; individual cause not isolated'},
             'LOCAL_nonbitexact_fields':{key:sum(row['absolute_difference'][key]!=0 for row in local['rows']) for key in local['rows'][0]['absolute_difference']},
             'assembly':{key:assembly[key] for key in ('fixed_native_effective_lambda','lm_damping','rhs_current_lm_vs_native',
                            'rhs_current_fsl_order_vs_native','H_current_vs_native','diagonal_current_vs_native_H','actual_dtypes','wall_seconds')},
             'Jte_RHS_relative_l2_reduction_percent':100*(1-ordered/g),
             'failures_preserved':['head synchronous compile25s timeout','DEEPBIND CDLL load segfault with/without NumPy',
                                   'LOCAL bridge93 contract failed despite same BLAS identity'],
             'bindings':{name:bound(root/name) for name in ('reports/v3/arithmetic/summary.public.json',
                          'reports/v3/arithmetic/contracts.public.json','reports/v2/assembly/summary.public.json',
                          'reports/build/native_cli.binding.public.json','reports/build/symbol_context.public.json',
                          'source/pcg_arithmetic_cli.py','source/assembly_shared_v2.py')},
             'next_CPU_scope':'Self/Conda-buildable arithmetic with93+69 gates; remaining current assembly RHS residual unresolved.',
             'reference_prohibition':'Production must not depend on installed FSL, benchmark executable or copied native solutions.'}
    (root/'summary.public.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scalar_gates_pass':True,'Jte_improvement_percent':summary['Jte_RHS_relative_l2_reduction_percent']}))


if __name__=='__main__':main()
