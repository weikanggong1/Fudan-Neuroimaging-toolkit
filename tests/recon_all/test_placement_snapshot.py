"""Ordered snapshot backend retains projections, accepted offsets and retries."""
import unittest
import numpy as np
from fnit.recon_all.place_surface_collision import asynchronous_first_step


class PlacementSnapshotTest(unittest.TestCase):
    def test_ordered_and_retained_retry_with_unrounded_offsets(self):
        xyz=np.array([[0,0,0],[.005,0,0],[0,1,0],[0,0,1],
                      [.2,.2,.2],[1.2,.2,.2],[.2,1.2,.2],[.2,.2,1.2]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3],
                        [4,6,5],[4,5,7],[4,7,6],[5,6,7]],np.int32)
        displacement=np.array([[.02,.01,.01],[-.01,.01,0],[.01,0,.01],[0,.01,0],
                               [.1,.1,.1],[-.02,0,0],[0,-.01,0],[0,0,-.01]],np.float32)
        proposal=np.float32(xyz+displacement)
        ripped=np.array([False]*7+[True])
        for stale in (None,proposal):
            accepted_tree=displacement.copy()
            args=dict(offsets=displacement,stale_mht_trial=stale)
            tree,tree_order=asynchronous_first_step(xyz,faces,proposal,ripped,
                        accepted_offsets=accepted_tree,candidate_backend='tree',**args)
            for backend in ('snapshot', 'torch_snapshot'):
                accepted_snapshot=displacement.copy()
                diagnostic={}
                snapshot,snapshot_order=asynchronous_first_step(xyz,faces,proposal,ripped,
                            accepted_offsets=accepted_snapshot,candidate_backend=backend,
                            candidate_device='cpu' if backend=='torch_snapshot' else None,
                            candidate_diagnostics=diagnostic,**args)
                np.testing.assert_array_equal(snapshot,tree)
                np.testing.assert_array_equal(snapshot_order,tree_order)
                np.testing.assert_array_equal(accepted_snapshot,accepted_tree)
                if backend=='torch_snapshot':
                    self.assertEqual(diagnostic['effective_candidate_backend'],
                                     'torch_snapshot' if stale is None else 'tree_retained_mht')

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            asynchronous_first_step(np.empty((0,3)),np.empty((0,3),np.int32),
                                    np.empty((0,3)),np.empty(0,bool),candidate_backend='invalid')

    def test_compiled_retained_live_bucket_matches_tree(self):
        # 两个相交三角面形成真实命中；改变保留状态可让原桶判据接受/拒绝。
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],
                      [.25,.25,-.3],[.25,.25,.3],[.75,.25,0]],np.float32)
        faces=np.array([[0,1,2],[3,4,5]],np.int32)
        displacement=np.array([[.02,0,.05],[0,0,.05],[0,0,.05],
                               [.01,0,.01],[.01,0,-.01],[.01,0,.01]],np.float32)
        proposed=np.float32(xyz+displacement)
        ripped=np.zeros(len(xyz),bool)
        for stale in (proposed, proposed+np.float32(8)):
            old_offsets=displacement.copy()
            expected,expected_order=asynchronous_first_step(xyz,faces,proposed,ripped,
                offsets=displacement,accepted_offsets=old_offsets,stale_mht_trial=stale)
            for backend in ('snapshot','torch_snapshot'):
                candidate_offsets=displacement.copy()
                diagnostic={}
                actual,actual_order=asynchronous_first_step(xyz,faces,proposed,ripped,
                    offsets=displacement,accepted_offsets=candidate_offsets,stale_mht_trial=stale,
                    candidate_backend=backend,candidate_device='cpu' if backend=='torch_snapshot' else None,
                    retained_mht_backend='compiled',candidate_diagnostics=diagnostic)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(actual_order,expected_order)
                np.testing.assert_array_equal(candidate_offsets,old_offsets)
                self.assertEqual(diagnostic['effective_candidate_backend'],'compiled_retained_mht')
                self.assertGreater(diagnostic['retained_mht_checks'],0)

    def test_compiled_retained_rejects_missing_snapshot(self):
        with self.assertRaisesRegex(ValueError,'requires fast snapshot'):
            asynchronous_first_step(np.empty((0,3)),np.empty((0,3),np.int32),
                np.empty((0,3)),np.empty(0,bool),retained_mht_backend='compiled')


if __name__=='__main__':unittest.main()
