"""初始三项前缀/连续反馈/随机流合同；固定小输入及schema夹具不是MRI benchmark。"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import benchmark_recon_topology_edge_predicate_population_prefix as api


def population_arguments(*, groups=True, overlap=False):
    n, k = 6, 4
    return dict(edge_count=n, initial_selection=groups,
        segment_edge_numbers=np.arange(n,dtype=np.int32) if groups else None,
        segment_rows=[[0,2],[],[1,4],[3,5]] if groups else None,
        use_overlap=overlap, overlap_offsets=np.arange(n+1,dtype=np.int64) if overlap else np.zeros(n+1,np.int64),
        overlapping_edges=np.asarray([1,2,3,4,5,0],np.int32) if overlap else np.empty(0,np.int32),
        overlap_list_incomplete=np.zeros(n,bool), translated_defect_vertices=np.arange(k,dtype=np.int32),
        discarded_defect_vertices=np.asarray([False,True,False,False]), vertex_rip=np.zeros(k+2,np.int32),
        use_counts=np.asarray([2,3,1,0],np.int32), vertex_fitness=np.asarray([.2,.3,.4,.5],np.float32),
        vertex_eliminate=True, deletion_count=1, elimination_threshold=np.float32(4))


class Feedback:
    """非恒定反馈让下一candidate的误复位能被发现；不声称MRI评分正确。"""
    def __init__(self):
        self.calls=[]; self.persistent=np.asarray([.125,-.25,.5],np.float32)

    def __call__(self, **inputs):
        self.calls.append({k:v.copy() for k,v in inputs.items()})
        step=len(self.calls); self.persistent=np.float32(self.persistent+np.float32(step/8))
        count=inputs['use_counts'].copy(); count[~inputs['discarded_defect_vertices']]+=step
        rip=inputs['vertex_rip'].copy(); rip[(step-1)%len(rip)]=-step
        fit=inputs['vertex_fitness'].copy()
        fit[:]=np.float32(fit+np.float32(self.persistent.sum()/32))
        result=dict(fitness=float(np.dot(inputs['ordering'],np.arange(6)))+float(self.persistent.sum()),
                    vertex_rip=rip,use_counts=count,vertex_fitness=fit)
        self.calls[-1]['feedback']={k:np.asarray(v).copy() for k,v in result.items()}
        return result


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('groups,overlap', [(True,False),(False,False),(True,True)])
def test_three_prefix_matches_normal_ten_order_rng_and_real_feedback(groups,overlap):
    from fnit.recon_all.topology_genetic_search import TopologyRandomStream
    from fnit.recon_all.topology_ga_population import run_population_search
    args=population_arguments(groups=groups,overlap=overlap)
    untouched={k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
    captured=[]
    for normal in (False,True):
        rng=api.RandomTrace(TopologyRandomStream(seed=1234)); callback=Feedback()
        with api.observe_orderings(rng) as ordering:
            if normal:
                out=run_population_search(**args,population_size=10,max_unchanged=10,niters=0,
                    draw_random_number=rng,evaluate_patch=callback)
            else:
                out=api.run_initial_prefix(search_arguments=args,draw_random_number=rng,evaluate_patch=callback)
        captured.append((out,callback,ordering,rng))
    p,n=captured
    assert len(p[1].calls)==3 and len(n[1].calls)==10
    assert p[0]['generation_trace']==[] and p[0]['generations']==0 and p[0]['counts']['evaluated']==3
    for slot in range(3):
        for key,value in p[2][slot]['result'].items():
            other=n[2][slot]['result'][key]
            assert exact(value,other) if isinstance(value,np.ndarray) else value==other
        assert p[2][slot]['rng_call_count_before']==n[2][slot]['rng_call_count_before']
        assert p[2][slot]['rng_call_count_after']==n[2][slot]['rng_call_count_after']
        for key,value in p[1].calls[slot].items():
            if key=='feedback':
                assert all(exact(v,n[1].calls[slot][key][k]) for k,v in value.items())
            else: assert exact(value,n[1].calls[slot][key])
        assert p[0]['evaluation_trace'][slot]==n[0]['evaluation_trace'][slot]
        if slot:
            previous=p[1].calls[slot-1]['feedback']
            assert exact(p[1].calls[slot]['use_counts'],previous['use_counts'])
            assert exact(p[1].calls[slot]['vertex_fitness'],previous['vertex_fitness'])
            assert exact(p[1].calls[slot]['vertex_rip'],previous['vertex_rip'])
    end=p[2][-1]['rng_call_count_after']
    assert exact(p[3].array(),n[3].array()[:end])
    assert p[3].source.draws==end and end>0
    assert p[2][0]['result']['random_draws'].shape==(0,3)
    assert p[2][0]['result']['mutated'] is False
    assert np.array_equal(p[2][0]['result']['ordering'],np.arange(6))
    for key,value in untouched.items(): assert exact(value,args[key])


def test_order_observer_restores_after_source_evaluation_failure():
    from fnit.recon_all import topology_ga_population as population
    from fnit.recon_all.topology_genetic_search import TopologyRandomStream
    original=population.generate_initial_ordering; trace=api.RandomTrace(TopologyRandomStream(seed=1234))
    def fail(**inputs): raise RuntimeError('intentional source callback failure')
    with pytest.raises(RuntimeError,match='intentional'):
        with api.observe_orderings(trace):
            api.run_initial_prefix(search_arguments=population_arguments(),draw_random_number=trace,evaluate_patch=fail)
    assert population.generate_initial_ordering is original


def test_three_prefix_matches_unbounded_normal_policy_before_fourth_callback():
    from fnit.recon_all.topology_genetic_search import TopologyRandomStream
    from fnit.recon_all.topology_ga_population import run_population_search
    args=population_arguments(); d=Feedback(); n=Feedback()
    rd=api.RandomTrace(TopologyRandomStream(seed=1234));rn=api.RandomTrace(TopologyRandomStream(seed=1234))
    with api.observe_orderings(rd) as od:
        api.run_initial_prefix(search_arguments=args,draw_random_number=rd,evaluate_patch=d)
    def fourth_guard(**inputs):
        if len(n.calls)==3:raise RuntimeError('test observer stops before fourth callback')
        return n(**inputs)
    with api.observe_orderings(rn) as on:
        with pytest.raises(RuntimeError,match='before fourth'):
            run_population_search(**args,population_size=10,max_unchanged=10,niters=-1,
                draw_random_number=rn,evaluate_patch=fourth_guard)
    assert len(n.calls)==len(d.calls)==3 and len(on)==4
    for slot in range(3):
        assert exact(od[slot]['result']['ordering'],on[slot]['result']['ordering'])
        assert od[slot]['result']['mutated']==on[slot]['result']['mutated']
        for key in ('ordering','vertex_rip','use_counts','vertex_fitness','discarded_defect_vertices'):
            assert exact(d.calls[slot][key],n.calls[slot][key])
    assert exact(rd.array(),rn.array()[:od[-1]['rng_call_count_after']])


def test_prefix_rejects_implicit_policy_and_normal_population_without_third_slot():
    args=population_arguments()
    with pytest.raises(ValueError,match='policy'):
        api.run_initial_prefix(search_arguments=dict(args,niters=-1),draw_random_number=lambda l,h:l,evaluate_patch=Feedback())
    with pytest.raises(ValueError,match='fewer than three'):
        api.run_initial_prefix(search_arguments=args,draw_random_number=lambda l,h:l,evaluate_patch=Feedback(),normal_population_size=2)


def test_random_trace_preserves_direct_vnl_values_bounds_and_signed_zero():
    from fnit.recon_all.topology_genetic_search import TopologyRandomStream
    a,b=TopologyRandomStream(seed=1234),TopologyRandomStream(seed=1234);trace=api.RandomTrace(a)
    bounds=[(-0.,0.),(0.,1.),(-2.,3.),(5.,5.)]
    for low,high in bounds:
        assert np.asarray(trace(low,high),np.float64).tobytes()==np.asarray(b(low,high),np.float64).tobytes()
    z=trace.array(); assert z.dtype==api.RNG_DTYPE and z['ordinal'].tolist()==[0,1,2,3]
    assert z['low'][0].tobytes()==np.asarray(-0.,np.float64).tobytes()
    assert a.draws==b.draws==4


def fake_binding(root):
    """完整来源schema夹具，只供离线比较器负向门，不是实际导入证据。"""
    files={api.TOOL_RELATIVE:dict(size=1,sha256='f'*64),api.identity.TOOL_RELATIVE:dict(size=1,sha256='e'*64)}
    names=api.identity.REQUIRED_FNIT
    def relative(name): return 'src/'+name.replace('.','/')+('/__init__.py' if name in ('fnit','fnit.recon_all') else '.py')
    files.update({relative(n):dict(size=1,sha256='d'*64) for n in names})
    helpers=api.identity.REQUIRED_HELPERS|{'benchmark_recon_topology_edge_predicate_cache'}
    files.update({'tools/'+n+'.py':dict(size=1,sha256=('e' if n.endswith('edge_predicate_cache') else '1')*64) for n in helpers})
    manifest=root/'manifest.json';protocol=root/'protocol.json'
    api.write(manifest,dict(git_base='a'*40,files=files,expected_fnit_module_sha256={n:'d'*64 for n in names}))
    api.write(protocol,dict(git_base='a'*40,source_manifest_sha256=api.sha(manifest),seed=1234,hemisphere='lh',
        prefix_slots=list(api.SLOTS),arm_run_order=list(api.RUNS),
        diagnostic_policy=dict(population_size=3,max_unchanged=0,niters=0),normal_policy=dict(population_size=10,max_unchanged=10,niters=-1),
        cache=dict(maximum_entries=250000), input_sha256={'same_input':'b'*64},
        resources=dict(requested_threads=2,candidate_cpu_affinity=[1,2]),strict_gates=dict(
            candidate_npz_fields=sorted(api.identity.CANDIDATE_FIELDS),candidate_npz_field_count=36,
            state_npz_fields=sorted(api.identity.STATE_FIELDS),state_npz_field_count=40,
            full_control_fields=sorted(api.identity.CONTROL_FIELDS),ordering_npz_fields=sorted(api.ORDER_FIELDS),
            prefix_npz_fields=sorted(api.PREFIX_FIELDS),prefix_control_fields=sorted(api.PREFIX_CONTROL_FIELDS),
            prefix_count_fields=sorted(api.COUNT_FIELDS),artifact_names=sorted(api.ARTIFACT_NAMES),
            comparison_count=114,rng_comparison='ordinal_low_high_value_bytes')))
    return dict(source_manifest=manifest,protocol=protocol),relative,helpers,files


def fake_pair(root):
    """小schema收据，绝不用于benchmark精度或速度结论。"""
    binding,relative,helpers,files=fake_binding(root)
    for arm in ('baseline','memo'):
        p=root/arm;p.mkdir();records=[]
        for run in api.RUNS:
            rng=np.asarray([(0,0.,1.,.25),(1,0.,1.,.5),(2,0.,1.,.75),(3,0.,1.,1.)],dtype=api.RNG_DTYPE)
            np.save(p/f'{run}.rng.private.npy',rng,allow_pickle=False);slots=[]
            def cache(logical,hits):
                misses=logical-hits
                return dict(canonical_sha256='0'*64,maximum_entries=250000,entries=misses,
                    logical_predicate_calls=logical,cache_hits=hits,actual_predicate_evaluations=misses,
                    inserted_unique_keys=misses,capacity_bypass_evaluations=0,geometry_only=True,
                    concurrent_writes_allowed=False,complete_unique_key_count=misses)
            cumulative=cache(0,0)
            for slot in api.SLOTS:
                for kind,fields in (('candidate',api.identity.CANDIDATE_FIELDS),('state',api.identity.STATE_FIELDS)):
                    values={k:np.asarray([1,-0.],np.float32) for k in fields}
                    if kind=='candidate':
                        values.update(feedback_vertex_rip=np.zeros(6,np.int32),
                            feedback_use_counts=np.full(4,slot+1,np.int32),
                            feedback_vertex_fitness=np.ones(4,np.float32),fitness=np.asarray(1.+slot,np.float64))
                    np.savez_compressed(p/f'{run}.slot{slot}.{kind}.private.npz',**values)
                before,after=(0,0) if slot==0 else (2*(slot-1),2*slot)
                order=dict(ordering=np.arange(6,dtype=np.int32),segment_order=np.arange(2,dtype=np.int32),
                    selected_segment_index=np.asarray(-1 if slot==0 else slot,np.int32),mutated=np.asarray(slot!=0,np.bool_),
                    ordering_random_draws=np.column_stack([rng[k][before:after] for k in ('low','high','value')]),
                    input_vertex_rip=np.zeros(6,np.int32),input_use_counts=np.full(4,slot,np.int32),
                    input_vertex_fitness=np.ones(4,np.float32),input_discarded_defect_vertices=np.zeros(4,bool),
                    population_slot=np.asarray(slot,np.int32),rng_call_count_before=np.asarray(before,np.int64),
                    rng_call_count_after=np.asarray(after,np.int64),canonical_sha256=np.zeros(32,np.uint8),
                    base_context_state_sha256=np.ones(32,np.uint8))
                np.savez_compressed(p/f'{run}.slot{slot}.ordering.private.npz',**order)
                api.write(p/f'{run}.slot{slot}.control.private.json',{k:[] for k in api.identity.CONTROL_FIELDS})
                before=cumulative;cumulative=cache(3*(slot+1),sum(range(slot+1)))
                slots.append(dict(slot=slot,logical_predicate_calls=3,sequence_sha256='a'*64 if run=='audit' else None,
                    cache_before=before if arm=='memo' else None,cache_after=cumulative if arm=='memo' else None))
            prefix=dict(best_ordering=np.arange(6,dtype=np.int32),best_discarded=np.zeros(4,bool),best_fitness=np.asarray(3.,np.float64),
                population_orderings=np.tile(np.arange(6,dtype=np.int32),(3,1)),population_fitness=np.asarray([1.,2.,3.],np.float64),
                population_ranks=np.asarray([2,1,0],np.int32),best_population_index=np.asarray(2,np.int32),
                discarded_defect_vertices=np.zeros(4,bool),vertex_rip=np.zeros(6,np.int32),use_counts=np.full(4,3,np.int32),
                vertex_fitness=np.ones(4,np.float32),deletion_count=np.asarray(1,np.int32),elimination_threshold=np.asarray(4,np.float32),
                generations=np.asarray(0,np.int32))
            np.savez_compressed(p/f'{run}.prefix.private.npz',**prefix)
            control=dict(evaluation_trace=[dict(generation=0,phase='initial',slot=s,fitness=1.+s,strict_improvement=True) for s in api.SLOTS],generation_trace=[],termination='unchanged_limit',
                counts={k:3 if k=='evaluated' else 0 for k in api.COUNT_FIELDS},final_tessellation_complete=False,complete_ga=False,overall_metric_equivalence='not_assessed')
            api.write(p/f'{run}.prefix_control.private.json',control)
            records.append(dict(run=run,slots=slots,actual_rng_draws=4,
                full_rng_stream_sha256=api.hashlib.sha256(rng.tobytes()).hexdigest(),canonical_domain_constant=True,
                hooks_restored_observed=True,memo_shared_across_slots=arm=='memo',canonical_sha256='0'*64))
        report=dict(status='complete_fixed_initial3_prefix_diagnostic',arm=arm,records=records,seed=1234,hemisphere='lh',
            source_manifest_sha256=api.sha(binding['source_manifest']),protocol_sha256=api.sha(binding['protocol']),
            actual_tool_sha256='f'*64,all_fnit_modules_inside_frozen_root=True,
            actual_loaded_fnit_sha256={n:'d'*64 for n in api.identity.REQUIRED_FNIT},
            actual_loaded_fnit_sources={n:dict(relative_path=relative(n),origin_relative_path=relative(n),size=1,sha256='d'*64) for n in api.identity.REQUIRED_FNIT},
            actual_loaded_benchmark_tool_sources={n:dict(relative_path='tools/'+n+'.py',origin_relative_path='tools/'+n+'.py',**files['tools/'+n+'.py']) for n in helpers},
            artifact_sha256={n:api.sha(p/n) for n in api.ARTIFACT_NAMES},input_sha256={'same_input':'b'*64},threads=2,cpu_affinity=[1,2],
            cuda_hidden=True,global_function_replacement=True,memo_shared_between_repeats=False,production_wiring=False,
            actual_thread_policy=dict(requested_threads=2,restoration_complete=True,
                torch=dict(before=2,effective=2,restored=2),numba=dict(before=2,effective=2,restored=2)),
            fresh_thread_environment={n:'2' for n in api.THREAD_NAMES})
        api.write(p/'report.private.json',report);api.write(p/'COMPLETE.json',dict(status=report['status'],report_sha256=api.sha(p/'report.private.json')))
    return binding


def refresh(path, mutation=None):
    r=json.loads((path/'report.private.json').read_text())
    if mutation:mutation(r)
    r['artifact_sha256']={n:api.sha(path/n) for n in r['artifact_sha256']}
    api.write(path/'report.private.json',r)
    api.write(path/'COMPLETE.json',dict(status=r['status'],report_sha256=api.sha(path/'report.private.json')))


def compare(root,binding):
    return api.compare_completed(baseline=root/'baseline',candidate=root/'memo',output=root/'compared.json',**binding)


def test_closed_all_prefix_state_order_rng_control_and_repeat_gate(tmp_path):
    r=compare(tmp_path,fake_pair(tmp_path));assert len(r['checks'])==114
    assert r['status']=='passed_fixed_initial3_prefix_pair' and all(x['equal'] for x in r['checks'])


@pytest.mark.parametrize('kind,field', [('candidate','distance_mm'),('state','canonical_vertices'),('ordering','mutated'),('prefix','best_ordering')])
def test_joint_missing_array_fields_cannot_pass(tmp_path,kind,field):
    binding=fake_pair(tmp_path)
    for arm in ('baseline','memo'):
        p=tmp_path/arm/(f'cold.{kind}.private.npz' if kind=='prefix' else f'cold.slot1.{kind}.private.npz')
        with np.load(p,allow_pickle=False) as z:fields={k:z[k] for k in z.files if k!=field}
        np.savez_compressed(p,**fields);refresh(p.parent)
    with pytest.raises(ValueError,match='field schema'):compare(tmp_path,binding)


@pytest.mark.parametrize('wrong', ['count_only','ordinal','slot_boundary','canonical','control','artifact','source','origin','entry',
    'feedback_reset','feedback_shape','hooks','threads','audit_digest','cross_repeat','counter_delta','memo_reset','prefix_feedback','trace_metadata','hemisphere'])
def test_joint_rng_metadata_artifact_control_source_failures_rejected(tmp_path,wrong):
    binding=fake_pair(tmp_path)
    for arm in ('baseline','memo'):
        p=tmp_path/arm
        if wrong in ('count_only','ordinal'):
            f=p/'cold.rng.private.npy';z=np.load(f,allow_pickle=False)
            if wrong=='ordinal':z['ordinal'][1]=9
            else:z['value'][1]=.75
            np.save(f,z,allow_pickle=False)
            refresh(p,lambda r:r['records'][0].update(full_rng_stream_sha256=api.hashlib.sha256(z.tobytes()).hexdigest()))
        elif wrong in ('slot_boundary','canonical'):
            f=p/'cold.slot2.ordering.private.npz'
            with np.load(f,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
            if wrong=='slot_boundary':fields['rng_call_count_before']=np.asarray(0,np.int64)
            else:fields['canonical_sha256'][0]=1
            np.savez_compressed(f,**fields);refresh(p)
        elif wrong=='control':
            f=p/'cold.slot1.control.private.json';v=json.loads(f.read_text());v.pop('face_check');api.write(f,v);refresh(p)
        elif wrong=='artifact':refresh(p,lambda r:r.update(artifact_sha256={}))
        elif wrong=='entry':refresh(p,lambda r:r.update(actual_tool_sha256='0'*64))
        elif wrong=='source':refresh(p,lambda r:r.update(actual_loaded_fnit_sha256={},actual_loaded_fnit_sources={}))
        elif wrong=='origin':
            refresh(p,lambda r:r['actual_loaded_benchmark_tool_sources']['benchmark_recon_topology_edge_predicate_cache'].update(origin_relative_path='outside/helper.py'))
        elif wrong in ('feedback_reset','feedback_shape'):
            f=p/'cold.slot1.ordering.private.npz'
            with np.load(f,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
            fields['input_use_counts']=np.zeros((4,1) if wrong=='feedback_shape' else 4,np.int32)
            np.savez_compressed(f,**fields);refresh(p)
        elif wrong=='hooks':refresh(p,lambda r:r['records'][0].update(hooks_restored_observed=False))
        elif wrong=='threads':refresh(p,lambda r:r['actual_thread_policy']['numba'].update(restored=1))
        elif wrong=='audit_digest':refresh(p,lambda r:[s.update(sequence_sha256=None) for s in r['records'][2]['slots']])
        elif wrong=='cross_repeat':refresh(p,lambda r:r.update(memo_shared_between_repeats=True))
        elif wrong=='counter_delta':refresh(p,lambda r:r['records'][0]['slots'][1].update(logical_predicate_calls=4))
        elif wrong=='memo_reset':
            refresh(p,lambda r:r['records'][0]['slots'][1].update(cache_before=r['records'][0]['slots'][0]['cache_before']))
        elif wrong=='prefix_feedback':
            f=p/'cold.prefix.private.npz'
            with np.load(f,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
            fields['use_counts'][:]=0;np.savez_compressed(f,**fields);refresh(p)
        elif wrong=='trace_metadata':
            f=p/'cold.prefix_control.private.json';v=json.loads(f.read_text());v['evaluation_trace'][1]['strict_improvement']=False
            api.write(f,v);refresh(p)
        elif wrong=='hemisphere':refresh(p,lambda r:r.update(hemisphere='rh'))
    with pytest.raises(ValueError):compare(tmp_path,binding)


def test_full_random_values_mismatch_and_self_state_mismatch_preserve_failed_receipt(tmp_path):
    binding=fake_pair(tmp_path);p=tmp_path/'memo'/'warm.slot2.state.private.npz'
    with np.load(p,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
    fields['K'][0]=2;np.savez_compressed(p,**fields);refresh(p.parent)
    with pytest.raises(RuntimeError,match='mismatch'):compare(tmp_path,binding)
    assert json.loads((tmp_path/'compared.json').read_text())['status']=='failed_fixed_initial3_prefix_pair'


def test_canonical_contract_requires_same_immutable_prepared_domain():
    from types import SimpleNamespace
    from fnit.recon_all.topology_retessellation_dynamic import prepare_retessellation_base
    points=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
    base=prepare_retessellation_base(canonical=points,base_faces=np.asarray([[0,1,2]],np.int32),neighbors=[[1,2],[0,2],[0,1],[]])
    e=SimpleNamespace(retessellation_inputs=dict(canonical=base.canonical,base_context=base))
    a=api.canonical_contract(e);b=api.canonical_contract(e)
    assert a[0] is b[0] and a[2] is b[2] and a[3:]==b[3:]
    e.retessellation_inputs['canonical']=base.canonical.copy()
    with pytest.raises(ValueError,match='same prepared'):api.canonical_contract(e)
