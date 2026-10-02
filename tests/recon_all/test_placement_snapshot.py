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
            accepted_tree=displacement.copy();accepted_snapshot=displacement.copy()
            args=dict(offsets=displacement,stale_mht_trial=stale)
            tree,tree_order=asynchronous_first_step(xyz,faces,proposal,ripped,
                        accepted_offsets=accepted_tree,candidate_backend='tree',**args)
            snapshot,snapshot_order=asynchronous_first_step(xyz,faces,proposal,ripped,
                        accepted_offsets=accepted_snapshot,candidate_backend='snapshot',**args)
            np.testing.assert_array_equal(snapshot,tree)
            np.testing.assert_array_equal(snapshot_order,tree_order)
            np.testing.assert_array_equal(accepted_snapshot,accepted_tree)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            asynchronous_first_step(np.empty((0,3)),np.empty((0,3),np.int32),
                                    np.empty((0,3)),np.empty(0,bool),candidate_backend='invalid')


if __name__=='__main__':unittest.main()
