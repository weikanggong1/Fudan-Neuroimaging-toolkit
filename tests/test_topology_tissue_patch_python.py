"""组织代价patch接口契约；真实几何回归另存机器收据。"""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from fnit.recon_all.topology_tissue_patch_python import run_tissue_patch, _median_twice, _score, _ordered_collision_acceptor
from fnit.recon_all.topology_defect_mri_match import defect_mri_match
from fnit.recon_all.topology_collision_cache_python import FixedPatchCollisionCache
from fnit.recon_all.place_surface_collision import asynchronous_first_step


class TissuePatchContracts(unittest.TestCase):
    def test_invalid_prior_fails_before_reading_input_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'candidate'
            for kwargs in ({'log_shape_prior':np.nan},{'normal_offset_mm':0},
                           {'step_motion_limit_mm':.31},{'collision_acceptance':'approximate'}):
                with self.assertRaises(ValueError):
                    run_tissue_patch(input_surface='missing.orig',intensity_volume='missing.mgz',output_surface=output,**kwargs)
                self.assertFalse(output.exists())

    def test_median_is_self_plus_full_one_ring_and_exactly_two_passes(self):
        values=np.array([0,4,20,8],np.float32)
        neighbors=np.array([[1,2],[0,2],[0,1],[0,2]],np.int32);degree=np.full(4,2,np.int32)
        first=np.array([np.median(values[np.r_[v,neighbors[v]]]) for v in range(4)],np.float32)
        expected=np.array([np.median(first[np.r_[v,neighbors[v]]]) for v in range(4)],np.float32)
        np.testing.assert_array_equal(_median_twice(values,neighbors,degree),expected)

    def test_score_reuses_histogram_without_coordinate_label_selection(self):
        white=np.array([80,90,110],np.float32);gray=np.array([60,75,80],np.float32)
        neighbors=[[1,2],[0,2],[0,1]]
        histogram={'white_smooth':np.full(256,.01,np.float32),'gray_smooth':np.full(256,.02,np.float32),
                   'white_mean':np.float32(110),'gray_mean':np.float32(80)}
        with patch('fnit.recon_all.topology_tissue_patch_python.first_defect_intensity_histograms',return_value=histogram):
            score,returned=_score(white,gray,neighbors,0)
            np.testing.assert_allclose(score,np.log(2),rtol=0,atol=1e-12)
            shifted,_=_score(white,gray,neighbors,1)
            np.testing.assert_allclose(shifted,score-1,rtol=0,atol=1e-12)
            self.assertIs(returned,histogram)

    @staticmethod
    def mri_arguments():
        xyz=np.float32([[9,8,8],[7,8,8],[8,9,8],[8,7,8],[8,8,9],[8,8,7]])
        faces=np.int32([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
        neighbors=[[2,4,3,5],[2,4,3,5],[0,4,1,5],[0,4,1,5],[0,2,1,3],[0,2,1,3]]
        volume=np.indices((16,16,16)).sum(axis=0).astype(np.uint8)+np.uint8(55)
        return dict(xyz=xyz,faces=faces,inside_vertices=np.int32([0,1]),
                    runtime_neighbors=neighbors[:2],volume=volume,ras_to_voxel=np.eye(4),white_mean=90.,gray_mean=70.)

    def test_default_40_steps_equal_full_identity_acceptor_and_fixed_vertices(self):
        args=self.mri_arguments();trace=[]
        default=defect_mri_match(**args)
        split=defect_mri_match(**args,step_acceptor=lambda current,proposal,index:(trace.append(index) or proposal))
        np.testing.assert_array_equal(default,split)
        np.testing.assert_array_equal(default[2:],args['xyz'][2:])
        self.assertEqual(trace,list(range(40)))

    def test_acceptor_invalid_or_moving_fixed_vertices_is_rejected(self):
        args=self.mri_arguments()
        with self.assertRaises(ValueError): defect_mri_match(**args,iterations=-1)
        with self.assertRaises(ValueError): defect_mri_match(**args,step_acceptor=lambda current,proposal,index:proposal.astype(np.float64))
        def move_fixed(current,proposal,index):
            proposal[2,0]+=1
            return proposal
        with self.assertRaises(ValueError): defect_mri_match(**args,step_acceptor=move_fixed)

    def test_collision_acceptor_clips_and_restores_rejected_face_flip(self):
        args=self.mri_arguments();xyz=args['xyz'];trace=[]
        selected=np.array([True,False,False,False,False,False])
        proposal=xyz.copy();proposal[0,0]+=2
        def safe(vertices,faces,proposed,ripped,**kwargs):
            return proposed.copy(),np.flatnonzero(~ripped)
        with patch('fnit.recon_all.topology_tissue_patch_python.asynchronous_first_step',side_effect=safe):
            result=_ordered_collision_acceptor(args['faces'],selected,trace,.3)(xyz,proposal,0)
        self.assertLessEqual(np.linalg.norm(result[0].astype(np.float64)-xyz[0]),.300001)
        self.assertEqual(trace[0]['clipped_vertices'],1)
        np.testing.assert_array_equal(result[1:],xyz[1:])
        def flip(vertices,faces,proposed,ripped,**kwargs):
            candidate=vertices.copy();candidate[0]=vertices[1]
            return candidate,np.flatnonzero(~ripped)
        trace=[]
        with patch('fnit.recon_all.topology_tissue_patch_python.asynchronous_first_step',side_effect=flip):
            result=_ordered_collision_acceptor(args['faces'],selected,trace,.3)(xyz,proposal,0)
        np.testing.assert_array_equal(result,xyz)
        self.assertEqual(len(trace[0]['attempts']),13)

    def test_active_query_cache_equals_full_snapshot_and_rejects_topology_mutation(self):
        args=self.mri_arguments();xyz=args['xyz'];faces=args['faces']
        selected=np.array([True,False,False,False,False,False])
        cache=FixedPatchCollisionCache(faces=faces,selected=selected)
        proposal=xyz.copy();proposal[0]+=[-.15,.02,.03]
        full,full_order=asynchronous_first_step(vertices=xyz,faces=faces,proposed=proposal,
            ripped=~selected,candidate_backend='snapshot')
        active,active_order,diagnostics=cache.step(current=xyz,proposal=proposal,faces=faces)
        np.testing.assert_array_equal(active,full)
        np.testing.assert_array_equal(active_order,full_order)
        self.assertEqual(diagnostics['source_faces'],len(faces))
        self.assertEqual(diagnostics['query_faces'],4)
        self.assertTrue(diagnostics['all_target_faces_retained'])
        changed=faces.copy();changed[[0,1]]=changed[[1,0]]
        with self.assertRaises(ValueError): cache.step(current=xyz,proposal=proposal,faces=changed)
        moved_fixed=proposal.copy();moved_fixed[1,0]+=.01
        with self.assertRaises(ValueError): cache.step(current=xyz,proposal=moved_fixed,faces=faces)

    def test_inactive_target_triangle_still_rejects_a_collision(self):
        xyz=np.float32([[0,0,0],[1,0,0],[0,1,0],[.2,.2,.2],[.5,.2,.2],[.2,.5,.2]])
        faces=np.int32([[0,1,2],[3,4,5]])
        selected=np.array([True,False,False,False,False,False])
        proposal=xyz.copy();proposal[0,2]=.6
        full,order=asynchronous_first_step(vertices=xyz,faces=faces,proposed=proposal,
            ripped=~selected,candidate_backend='snapshot')
        active,active_order,diagnostics=FixedPatchCollisionCache(faces=faces,selected=selected).step(
            current=xyz,proposal=proposal,faces=faces)
        np.testing.assert_array_equal(full,xyz)
        np.testing.assert_array_equal(active,full)
        np.testing.assert_array_equal(active_order,order)
        self.assertEqual(diagnostics['query_faces'],1)
        self.assertEqual(diagnostics['source_faces'],2)


if __name__=='__main__': unittest.main()
