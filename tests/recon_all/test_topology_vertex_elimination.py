"""GA剔除的状态、源舍入和副作用合同；非真实影像 benchmark。"""
import unittest
import numpy as np
from fnit.recon_all.topology_vertex_elimination import eliminate_worst_vertices


class EliminationContracts(unittest.TestCase):
    def inputs(self, k=10):
        return dict(discarded_defect_vertices=np.zeros(k, bool),
            translated_defect_vertices=np.arange(k, dtype=np.int32),
            vertex_rip=np.zeros(k+2, np.int32), use_counts=np.ones(k, np.int32),
            vertex_fitness=np.arange(1, k+1, dtype=np.float32),
            fraction=.1, deletion_count=1, threshold=np.float32(4))

    def test_small_defect_keeps_unused_side_effect_despite_zero_report(self):
        args=self.inputs(5);args['use_counts'][[1,3]]=0
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['source_reported_deleted'],0)
        np.testing.assert_array_equal(out['unused_deleted_entries'],[1,3])
        np.testing.assert_array_equal(out['vertex_rip'],[0,1,0,1,0,0,0])
        self.assertEqual(out['threshold'],4)
        self.assertFalse(args['discarded_defect_vertices'].any())

    def test_fraction_float32_point_one_halves_persistent_threshold(self):
        args=self.inputs();out=eliminate_worst_vertices(**args)
        self.assertEqual(out['threshold'],np.float32(2))
        self.assertEqual(out['threshold'].dtype,np.float32)
        np.testing.assert_array_equal(out['fitness_deleted_entries'],[9])
        args['threshold']=out['threshold']
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['threshold'],np.float32(1))
        args['threshold']=out['threshold']
        args['fraction']=0
        self.assertEqual(eliminate_worst_vertices(**args)['threshold'],np.float32(1))

    def test_quota_uses_initial_active_count_and_unused_deletes_are_additional(self):
        args=self.inputs();args['use_counts'][[0,1]]=0;args['fraction']=.2
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['source_reported_deleted'],4)
        np.testing.assert_array_equal(out['unused_deleted_entries'],[0,1])
        np.testing.assert_array_equal(out['fitness_deleted_entries'],[9,8])

    def test_tie_order_and_source_index_threshold_stopping(self):
        args=self.inputs();args['vertex_fitness']=np.array([5,4,3,2,1,0,0,0,0,0],np.float32)
        args.update(fraction=.6,threshold=np.float32(100))
        out=eliminate_worst_vertices(**args)
        # 源比较的是索引。剩余2/初始6时提前停止，即使仍有正反馈。
        np.testing.assert_array_equal(out['fitness_deleted_entries'],[0,1,2,3])
        self.assertEqual(out['source_reported_deleted'],4)
        args=self.inputs();args['vertex_fitness']=np.ones(10,np.float32)
        np.testing.assert_array_equal(eliminate_worst_vertices(**args)['fitness_deleted_entries'],[0])

    def test_nonpositive_feedback_and_signed_char_count_storage(self):
        args=self.inputs();args['vertex_fitness'][:]=0
        self.assertEqual(eliminate_worst_vertices(**args)['source_reported_deleted'],0)
        args['use_counts'][2]=0;args['deletion_count']=128
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['vertex_rip'][2],-128)
        args['deletion_count']=0
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['effective_deletion_count'],1)
        self.assertEqual(out['vertex_rip'][2],1)

    def test_discarded_negative_mapping_and_alias_order(self):
        args=self.inputs();args['discarded_defect_vertices'][0]=True
        args['translated_defect_vertices'][0]=-1;args['use_counts'][0]=0
        out=eliminate_worst_vertices(**args)
        self.assertEqual(out['source_reported_deleted'],0)  # 九个有效条目，不进入反馈删除。
        self.assertFalse(out['vertex_rip'].any())
        args=self.inputs(5);args['translated_defect_vertices'][3]=1
        args['use_counts'][[1,3]]=0
        np.testing.assert_array_equal(eliminate_worst_vertices(**args)['unused_deleted_entries'],[1,3])

    def test_invalid_fields_rejected_without_input_or_threshold_mutation(self):
        args=self.inputs();before={k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
        for changes in [dict(fraction=True),dict(fraction=np.nan),dict(fraction=2),
                dict(threshold=-1),dict(deletion_count=True),dict(deletion_count=2**31),
                dict(translated_defect_vertices=np.full(10,-1,np.int32)),
                dict(vertex_rip=np.full(12,128,np.int32)),
                dict(use_counts=np.full(10,-1,np.int32)),
                dict(vertex_fitness=np.full(10,np.inf,np.float32))]:
            with self.subTest(changes=list(changes)),self.assertRaises(ValueError):
                eliminate_worst_vertices(**{**args,**changes})
            for key,value in before.items():np.testing.assert_array_equal(args[key],value)


if __name__=='__main__':unittest.main()
