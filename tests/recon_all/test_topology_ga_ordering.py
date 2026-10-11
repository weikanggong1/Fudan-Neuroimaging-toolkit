"""GA 有序控制、随机调用边界和失败事务契约；非真实MRI benchmark。"""
import unittest
import numpy as np
from fnit.recon_all.topology_ga_ordering import crossover_ordering, mutate_ordering


class GaOrdering(unittest.TestCase):
    def mutation_inputs(self,n=4):
        return dict(ordering=np.arange(n,dtype=np.int32),mutation_probability=.5,
            use_overlap=False,overlap_offsets=np.zeros(n+1,np.int32),
            overlapping_edges=np.empty(0,np.int32),
            overlap_list_incomplete=np.zeros(n,bool),draw_random_number=lambda lo,hi:.9)

    def stream(self,values):
        iterator=iter(values);calls=[]
        def draw(lo,hi):
            calls.append((lo,hi));return next(iterator)
        return draw,calls

    def test_source_nadded_pair_pattern_and_half_boundary(self):
        first=np.arange(6,dtype=np.int32);second=first[::-1].copy()
        args=dict(first_parent_ordering=first,second_parent_ordering=second)
        np.testing.assert_array_equal(crossover_ordering(**args,first_parent_draw=.49),[0,5,4,1,2,3])
        np.testing.assert_array_equal(crossover_ordering(**args,first_parent_draw=.5),[5,0,1,4,3,2])
        np.testing.assert_array_equal(first,np.arange(6));np.testing.assert_array_equal(second,np.arange(6)[::-1])

    def test_crossover_duplicate_candidates_and_empty_edge_domain(self):
        order=np.array([2,0,3,1],np.int32)
        np.testing.assert_array_equal(crossover_ordering(first_parent_ordering=order,
            second_parent_ordering=order,first_parent_draw=1.),order)
        empty=np.empty(0,np.int32)
        self.assertEqual(len(crossover_ordering(first_parent_ordering=empty,
            second_parent_ordering=empty,first_parent_draw=0.)),0)

    def test_invalid_permutation_or_draw_rejected(self):
        for value in [[0,0],[-1,0],[0.,1.],[True,False],[[0,1]]]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                crossover_ordering(first_parent_ordering=value,
                    second_parent_ordering=[0,1],first_parent_draw=.1)
        for draw in [True,np.nan,-.01,1.01]:
            with self.subTest(draw=draw),self.assertRaises(ValueError):
                crossover_ordering(first_parent_ordering=[0],second_parent_ordering=[0],first_parent_draw=draw)

    def test_every_edge_decision_draw_when_probability_zero(self):
        args=self.mutation_inputs();args['mutation_probability']=0
        draw,calls=self.stream([0,.1,.2,1.]);args['draw_random_number']=draw
        out=mutate_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],args['ordering'])
        self.assertEqual(calls,[(0.,1.)]*4)
        self.assertEqual(out['random_draws'].shape,(4,6))
        np.testing.assert_array_equal(out['random_draws'][:,2],np.zeros(4))

    def test_any_swap_keeps_source_position_map(self):
        args=self.mutation_inputs();draw,calls=self.stream([.1,2.2,.9,.9,.9])
        args['draw_random_number']=draw;before=args['ordering'].copy()
        out=mutate_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],[2,1,0,3])
        np.testing.assert_array_equal(out['source_position_map'],[0,1,2,3])
        np.testing.assert_array_equal(out['stale_position_edges'],[0,2])
        self.assertEqual(calls[1],(0.,3.9));np.testing.assert_array_equal(args['ordering'],before)

    def test_complete_overlap_uses_source_map_after_any_swap(self):
        args=self.mutation_inputs();args.update(use_overlap=True,
            overlap_offsets=np.array([0,0,1,1,1],np.int32),overlapping_edges=np.array([2],np.int32))
        draw,calls=self.stream([.1,2.2,.1,0.,.9,.9]);args['draw_random_number']=draw
        out=mutate_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],[2,0,1,3])
        np.testing.assert_array_equal(out['source_position_map'],[1,2,2,3])
        np.testing.assert_array_equal(out['stale_position_edges'],[2])
        self.assertEqual(calls[3],(0.,.9999))
        self.assertEqual(out['random_draws'][3,2],2)

    def test_incomplete_overlap_has_any_swap_and_no_ntry_draws(self):
        args=self.mutation_inputs();args.update(use_overlap=True,
            overlap_offsets=np.array([0,1,1,1,1],np.int32),overlapping_edges=np.array([3],np.int32),
            overlap_list_incomplete=np.array([True,False,False,False]))
        draw,calls=self.stream([.1,1.8,.9,.9,.9]);args['draw_random_number']=draw
        out=mutate_ordering(**args)
        np.testing.assert_array_equal(out['ordering'],[1,0,2,3])
        self.assertEqual(len(calls),5);self.assertEqual(calls[1],(0.,3.9))
        self.assertEqual(out['random_draws'][1,2],1)

    def test_invalid_csr_or_callback_keeps_input_and_empty_has_no_draws(self):
        args=self.mutation_inputs();before=args['ordering'].copy()
        for replacement in [dict(overlap_offsets=np.array([0,0,2,0,0])),
                dict(overlap_list_incomplete=np.zeros(4,np.int32)),
                dict(draw_random_number=lambda low,high:high+1),
                dict(mutation_probability=True),dict(use_overlap=1)]:
            with self.subTest(replacement=list(replacement)),self.assertRaises(ValueError):
                mutate_ordering(**{**args,**replacement})
            np.testing.assert_array_equal(args['ordering'],before)
        args=self.mutation_inputs(0);calls=[]
        args['draw_random_number']=lambda low,high:calls.append((low,high))
        out=mutate_ordering(**args)
        self.assertEqual(calls,[]);self.assertEqual(out['random_draws'].shape,(0,6))


if __name__=='__main__':unittest.main()
