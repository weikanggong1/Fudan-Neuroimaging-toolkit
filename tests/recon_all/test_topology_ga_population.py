"""源种群依赖/随机消费/反馈合同；控制值不能替代真实MRI benchmark。"""
import random
import unittest
import numpy as np

from fnit.recon_all.topology_ga_population import generate_initial_ordering, run_population_search


class PopulationContracts(unittest.TestCase):
    def initial(self, n=4):
        return dict(edge_count=n,population_index=0,initial_selection=True,
            segment_edge_numbers=np.arange(n,dtype=np.int32),segment_rows=[[0,1],[],[2,3]],
            use_overlap=False,overlap_offsets=np.zeros(n+1,np.int32),
            overlapping_edges=np.empty(0,np.int32),overlap_list_incomplete=np.zeros(n,bool),
            draw_random_number=lambda lo,hi:.75*hi)

    def search(self, p=5, k=10, score=None):
        rng=random.Random(23);calls=[]
        def draw(lo,hi):
            calls.append(('draw',lo,hi));return lo+(.11+.78*rng.random())*(hi-lo)
        def evaluate(**inputs):
            calls.append(('evaluate',inputs['ordering'].copy(),inputs['discarded_defect_vertices'].copy()))
            counts=inputs['use_counts'].copy();counts[~inputs['discarded_defect_vertices']]+=1
            number=sum(row[0]=='evaluate' for row in calls)
            return dict(fitness=0. if score is None else score(number),
                vertex_rip=inputs['vertex_rip'],use_counts=counts,vertex_fitness=np.ones(k,np.float32))
        return dict(edge_count=2,population_size=p,max_unchanged=1,niters=1,
            initial_selection=False,segment_edge_numbers=None,segment_rows=None,
            use_overlap=False,overlap_offsets=np.zeros(3,np.int32),overlapping_edges=np.empty(0,np.int32),
            overlap_list_incomplete=np.zeros(2,bool),draw_random_number=draw,evaluate_patch=evaluate,
            translated_defect_vertices=np.arange(k,dtype=np.int32),discarded_defect_vertices=np.zeros(k,bool),
            vertex_rip=np.zeros(k+2,np.int32),use_counts=np.zeros(k,np.int32),vertex_fitness=np.zeros(k,np.float32),
            vertex_eliminate=False,deletion_count=1,elimination_threshold=np.float32(4)),calls

    def test_first_population_preserves_identity_without_random_draws(self):
        args=self.initial();calls=[]
        args['draw_random_number']=lambda lo,hi:calls.append((lo,hi))
        out=generate_initial_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],np.arange(4))
        self.assertEqual(calls,[]);self.assertFalse(out['mutated'])

    def test_segment_holes_source_r_controls_mutation_and_full_edge_fallback(self):
        args=self.initial();args['population_index']=1
        out=generate_initial_ordering(**args)
        # nsegments=2，但max_segments=3；r选择第2非空槽2，shuffle仅前2槽。
        self.assertEqual(out['selected_segment_index'],2)
        np.testing.assert_array_equal(out['ordering'],[2,3,0,1])
        self.assertEqual(out['random_draws'].shape,(26,3))  # 22 shuffle + 4 mutation decision
        self.assertTrue(out['mutated'])  # 最后r为1，不是i+1=2。
        np.testing.assert_array_equal(out['segment_order'],[1,0])  # 11轮奇数次交换。

    def test_shuffle_can_skip_mutation_based_on_last_draw_instead_of_segment(self):
        args=self.initial(n=3);args.update(population_index=1,
            segment_rows=[[0],[1],[2]],segment_edge_numbers=np.arange(3,dtype=np.int32))
        args['draw_random_number']=lambda lo,hi:hi
        out=generate_initial_ordering(**args)
        self.assertFalse(out['mutated'])  # r=2==i+1
        self.assertEqual(out['random_draws'].shape,(33,3))
        np.testing.assert_array_equal(out['ordering'],[1,2,0])

    def test_no_segmentation_reuses_ordered_mutation_without_shuffle(self):
        args=self.initial();args.update(population_index=2,segment_edge_numbers=None,segment_rows=None)
        draws=iter([0.,3.,1.,1.,1.]);args['draw_random_number']=lambda lo,hi:next(draws)
        out=generate_initial_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],[3,1,2,0])
        self.assertEqual(out['random_draws'].shape,(5,3))
        self.assertTrue(out['mutated'])

    def test_full_generation_elite_not_reevaluated_failed_cross_also_mutated(self):
        args,calls=self.search(p=10);out=run_population_search(**args)
        phases=[row['phase'] for row in out['evaluation_trace']]
        self.assertEqual(phases,['initial']*10+['replacement']+['crossover','crossover_mutation']*8)
        self.assertEqual(out['counts']['evaluated'],27)
        self.assertEqual(out['counts']['mutation_attempts'],9)
        self.assertEqual(out['counts']['crossover_attempts'],8)
        np.testing.assert_array_equal(out['use_counts'],np.full(10,27))
        np.testing.assert_array_equal(out['population_ranks'],np.arange(10))
        self.assertEqual(out['best_population_index'],9)  # 源>=末次同分索引。
        np.testing.assert_array_equal(out['best_ordering'],[0,1])
        self.assertEqual(out['termination'],'unchanged_limit')
        self.assertEqual(out['generations'],1)
        self.assertFalse(out['complete_ga'])
        # 上游RP初态未改变；不得将source累积写回调用者数组。
        self.assertFalse(args['use_counts'].any());self.assertFalse(args['vertex_fitness'].any())

    def test_parent_tail_draw_occurs_after_replacement_evaluation(self):
        args,calls=self.search(p=13,score=lambda n:-10000.+n if n<=13 else -10000.)
        out=run_population_search(**args)
        positions=[i for i,row in enumerate(calls) if row[0]=='evaluate']
        self.assertEqual(calls[positions[13]+1],('draw',0.,1.))
        self.assertEqual(out['evaluation_trace'][13]['phase'],'replacement')
        self.assertEqual(out['counts']['crossover_attempts'],11)

    def test_improved_cross_skips_extra_mutation_and_preserves_best_status(self):
        # 初始5、replacement1、第1cross1改善，另外2cross和2mutation失败。
        args,_=self.search(score=lambda n:0. if n<=6 else (2. if n==7 else 1.))
        out=run_population_search(**args)
        self.assertEqual([r['phase'] for r in out['evaluation_trace']],
            ['initial']*5+['replacement','crossover','crossover','crossover_mutation','crossover','crossover_mutation'])
        self.assertEqual(out['counts']['crossover_improvements'],1)
        self.assertEqual(out['counts']['mutation_attempts'],3)
        self.assertEqual(out['best_fitness'],2.)
        self.assertEqual(out['termination'],'iteration_limit')

    def test_stagnation_elimination_retains_best_status_and_persistent_state(self):
        args,_=self.search();args.update(vertex_eliminate=True,niters=2)
        out=run_population_search(**args)
        self.assertEqual(out['generation_trace'][0]['elimination_fraction'],.2)
        np.testing.assert_array_equal(out['generation_trace'][0]['fitness_deleted_entries'],[0,1])
        self.assertEqual(out['elimination_threshold'],np.float32(2))
        self.assertEqual(out['deletion_count'],2)
        self.assertTrue(out['discarded_defect_vertices'][[0,1]].all())
        self.assertFalse(out['best_discarded'].any())
        # 旧最佳status与当前剔除不混淆；后续评分跳过新丢弃点。
        self.assertGreater(out['use_counts'][2],out['use_counts'][0])
        self.assertEqual(out['termination'],'iteration_limit')

    def test_small_defect_zero_delete_break_keeps_source_side_effects(self):
        args,_=self.search(k=5);args.update(vertex_eliminate=True,niters=-1)
        out=run_population_search(**args)
        self.assertEqual(out['termination'],'no_vertices_reported_deleted')
        self.assertEqual(out['generations'],0)  # 源break先于ngenerations++。
        self.assertEqual(out['deletion_count'],1)
        self.assertEqual(out['elimination_threshold'],4)

    def test_periodic_elimination_after_generation_twelve_does_not_increment_count(self):
        args,_=self.search(k=20);args.update(max_unchanged=100,niters=13,vertex_eliminate=True)
        out=run_population_search(**args)
        rows=[row for row in out['generation_trace'] if row['elimination_fraction'] is not None]
        self.assertEqual([row['generation'] for row in rows],[12])
        self.assertEqual(rows[0]['elimination_fraction'],.1)
        self.assertEqual(out['deletion_count'],1)
        self.assertEqual(out['elimination_threshold'],np.float32(2))

    def test_one_patch_can_finish_initial_only_and_rejects_undefined_generation(self):
        args,_=self.search(p=1);args['niters']=0
        out=run_population_search(**args)
        self.assertEqual(out['counts']['evaluated'],1)
        self.assertEqual(out['termination'],'iteration_limit')
        args['niters']=-1
        with self.assertRaisesRegex(ValueError,'missing second ranked parent'):run_population_search(**args)

    def test_bad_phase_segmentation_or_callback_is_explicit_failure(self):
        args=self.initial();args['population_index']=1
        for changed in [dict(segment_rows=[[],[]]),dict(segment_rows=[[0],[0]]),
                dict(segment_edge_numbers=np.array([0,0],np.int32)),dict(initial_selection=1)]:
            with self.subTest(changed=list(changed)),self.assertRaises(ValueError):
                generate_initial_ordering(**{**args,**changed})
        args,_=self.search();before=args['use_counts'].copy()
        for bad in [lambda **kw:dict(fitness=0),lambda **kw:dict(fitness=np.nan,
                vertex_rip=kw['vertex_rip'],use_counts=kw['use_counts'],vertex_fitness=kw['vertex_fitness'])]:
            with self.assertRaises(ValueError):run_population_search(**{**args,'evaluate_patch':bad})
        np.testing.assert_array_equal(args['use_counts'],before)


if __name__=='__main__':unittest.main()
