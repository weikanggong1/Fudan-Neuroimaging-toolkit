"""GA父代选择源规则和边界合同；模拟值不能替代真实MRI回归。"""
import unittest
import numpy as np
from fnit.recon_all.topology_ga_selection import plan_generation_selection, choose_distinct_parent


class GaSelection(unittest.TestCase):
    def test_source_counts_order_statistics_and_no_unneeded_draw(self):
        calls = []
        values = np.arange(10, dtype=np.float64)
        result = plan_generation_selection(fitness=values,
            draw_random_number=lambda low, high: calls.append((low, high)))
        self.assertEqual(result['counts'],dict(population=10,elite=1,eligible=5,replacements=1,crossovers=8))
        np.testing.assert_array_equal(result['patch_by_rank'],np.arange(9,-1,-1))
        np.testing.assert_array_equal(result['selected_parents'],[9,9,8,8,7,7,6,5])
        self.assertEqual(result['mean'],4.5)
        self.assertEqual(result['sigma'],np.sqrt(8.25))
        self.assertEqual(result['two_sigma_sq'],4.)
        self.assertEqual(calls,[])

    def test_equal_fitness_source_index_ties_and_scale(self):
        result=plan_generation_selection(fitness=np.full(5,-10.),draw_random_number=lambda low,high:0.)
        np.testing.assert_array_equal(result['rank_by_patch'],np.arange(5))
        self.assertEqual(result['two_sigma_sq'],1.)
        self.assertEqual(result['sigma'],0.)
        np.testing.assert_array_equal(result['selected_parents'],[0,1,2])
        self.assertEqual(result['random_draws'].shape,(0,3))

    def test_tail_preserves_absolute_exponential_and_underflow_fallback(self):
        result=plan_generation_selection(fitness=-10000.+np.arange(13),draw_random_number=lambda low,high:.5)
        np.testing.assert_array_equal(result['selected_parents'],[12,12,11,11,10,10,9,8,7,6,6])
        self.assertEqual(result['random_draws'].tolist(),[[0.,1.,.5]])
        shifted=plan_generation_selection(fitness=np.arange(13),draw_random_number=lambda low,high:.5)
        # 前十个配额相同；补选源指数没有减best，平移改变最后一项。
        self.assertEqual(shifted['selected_parents'][-1],12)
        self.assertEqual(result['selected_parents'][-1],6)

    def test_small_population_does_not_invent_second_parent(self):
        result=plan_generation_selection(fitness=[0.,1.],draw_random_number=lambda low,high:.9)
        self.assertEqual(result['counts']['crossovers'],1)
        self.assertEqual(result['distinct_parent_count'],1)
        calls=[]
        with self.assertRaises(ValueError):
            choose_distinct_parent(selected_parents=result['selected_parents'],
                first_parent=result['selected_parents'][0],draw_random_number=lambda lo,hi:calls.append((lo,hi)))
        self.assertEqual(calls,[])

    def test_epsilon_nontransitive_ranking_rejects_undefined_inverse(self):
        epsilon=np.finfo(np.float32).eps
        with self.assertRaisesRegex(ValueError,'ranks collide'):
            plan_generation_selection(fitness=[0,.75*epsilon,1.5*epsilon],
                draw_random_number=lambda low,high:0.)

    def test_second_parent_retains_duplicate_slots_and_retry_draws(self):
        values=iter([.1,1.9,2.0]);calls=[]
        def draw(low,high):
            calls.append((low,high));return next(values)
        original=np.array([7,7,3,3],np.int32)
        result=choose_distinct_parent(selected_parents=original,first_parent=7,draw_random_number=draw)
        self.assertEqual(result['parent_index'],3)
        self.assertEqual(calls,[(0.,3.999)]*3)
        np.testing.assert_array_equal(original,[7,7,3,3])
        self.assertEqual(result['random_draws'].shape,(3,3))

    def test_validation_and_callback_failures_keep_inputs(self):
        for values in [[],[0.],[[0.,1.]],[False,True],[np.nan,1.],['0','1']]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                plan_generation_selection(fitness=values,draw_random_number=lambda low,high:0.)
        source=np.arange(13,dtype=np.float64)
        for bad in [True,np.nan,-.1,1.01]:
            with self.subTest(draw=bad),self.assertRaises(ValueError):
                plan_generation_selection(fitness=source,draw_random_number=lambda low,high:bad)
            np.testing.assert_array_equal(source,np.arange(13))
        with self.assertRaises(ValueError):
            choose_distinct_parent(selected_parents=[0.,1.],first_parent=0,draw_random_number=lambda lo,hi:0.)
        with self.assertRaises(ValueError):
            choose_distinct_parent(selected_parents=[0,1],first_parent=0,draw_random_number=lambda lo,hi:2.)

    def test_selection_inputs_preserved_and_statistics_overflow_explicit(self):
        values=np.array([3.,1.,2.])
        plan_generation_selection(fitness=values,draw_random_number=lambda lo,hi:0.)
        np.testing.assert_array_equal(values,[3.,1.,2.])
        with self.assertRaises(FloatingPointError):
            plan_generation_selection(fitness=[1e308,-1e308],draw_random_number=lambda lo,hi:0.)


if __name__=='__main__':unittest.main()
