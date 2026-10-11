"""统计权重、域/顺序和失败事务契约；真实参考另行验证。"""
import unittest

import numpy as np

from fnit.recon_all.topology_candidate_statistics import update_candidate_statistics


class CandidateStatistics(unittest.TestCase):
    def inputs(self):
        return dict(vertex_marks=np.array([7, 7, 7, 7, 2, 9], np.int32),
            vertex_curvbak=np.array([2, 4, 8, 16, 32, 64], np.float32),
            candidate_edges=np.array([[0, 1], [1, 2], [2, 3]], np.int32),
            edge_used=np.array([3, 4, 2], np.int32),
            translated_defect_vertices=np.array([0, 1, 2, -1, 4], np.int32),
            discarded_defect_vertices=np.array([False, False, False, True, False]),
            translated_border_vertices=np.array([1], np.int32),
            use_counts=np.array([0, 10, 2, 5, 3], np.int32),
            vertex_fitness=np.array([0, 1, 5, 2, 4], np.float32), candidate_fitness=-97.76)

    def test_source_weights_border_existing_marks_and_no_mutation(self):
        kwargs=self.inputs(); old={key:value.copy() for key,value in kwargs.items() if isinstance(value,np.ndarray)}
        a=update_candidate_statistics(**kwargs)
        b=update_candidate_statistics(**{**kwargs, 'candidate_fitness': 0.012})
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['updated_defect_entries'],[0,2,4])
        np.testing.assert_array_equal(a['use_counts'],[1,10,3,5,4])
        np.testing.assert_array_equal(a['vertex_fitness'],[2,1,6,2,11])
        np.testing.assert_array_equal(a['vertex_marks'],[0,0,0,7,0,9])
        for key,value in old.items(): np.testing.assert_array_equal(kwargs[key],value)

    def test_alias_obeys_ordered_clear(self):
        kwargs=self.inputs();kwargs['translated_defect_vertices']=np.array([0,0,2,-1,4],np.int32)
        kwargs['translated_border_vertices']=np.array([],np.int32)
        result=update_candidate_statistics(**kwargs)
        np.testing.assert_array_equal(result['updated_defect_entries'],[0,2,4])
        self.assertEqual(result['use_counts'][1],10)

    def test_discarded_endpoint_mark_and_rp_preserved(self):
        kwargs=self.inputs();kwargs['translated_defect_vertices'][3]=1
        kwargs['translated_border_vertices']=np.array([],np.int32)
        kwargs['discarded_defect_vertices'][1]=True
        result=update_candidate_statistics(**kwargs)
        self.assertEqual(result['vertex_marks'][1],2)
        self.assertEqual(result['use_counts'][1],10);self.assertEqual(result['vertex_fitness'][1],1)

    def test_types_and_active_mapping_rejected(self):
        kwargs=self.inputs()
        for key,value in [('candidate_edges',kwargs['candidate_edges'].astype(float)),
                ('discarded_defect_vertices',kwargs['discarded_defect_vertices'].astype(int)),
                ('use_counts',kwargs['use_counts'].astype(float)),
                ('candidate_fitness',True),('translated_defect_vertices',np.array([-1,1,2,-1,4]))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                update_candidate_statistics(**{**kwargs,key:value})

    def test_empty_edges_and_zero_feedback_still_increment_marked(self):
        kwargs=self.inputs();kwargs['candidate_edges']=np.empty((0,2),np.int32)
        kwargs['edge_used']=np.empty(0,np.int32);kwargs['vertex_curvbak'][:]=0
        result=update_candidate_statistics(**kwargs)
        np.testing.assert_array_equal(result['updated_defect_entries'],[4])
        self.assertEqual(result['use_counts'][4],4);self.assertEqual(result['vertex_fitness'][4],3)

    def test_counter_and_float_overflow_do_not_mutate_input(self):
        kwargs=self.inputs();kwargs['use_counts'][0]=np.iinfo(np.int32).max
        saved=kwargs['vertex_marks'].copy()
        with self.assertRaises(FloatingPointError): update_candidate_statistics(**kwargs)
        np.testing.assert_array_equal(kwargs['vertex_marks'],saved)
        kwargs=self.inputs();kwargs['vertex_fitness'][0]=np.finfo(np.float32).max;kwargs['use_counts'][0]=2
        with self.assertRaises(FloatingPointError): update_candidate_statistics(**kwargs)


if __name__=='__main__': unittest.main()
