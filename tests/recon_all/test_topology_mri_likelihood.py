"""MRI likelihood 子项的组织PDF、空patch和非法几何契约。"""
import unittest
import numpy as np
from fnit.recon_all.topology_mri_likelihood import joint_gray_white_pdf, patch_mri_likelihood


class TopologyMRILikelihoodTests(unittest.TestCase):
    def arguments(self):
        return dict(original_vertices=np.array([[2,2,3],[5,2,3],[2,5,3]],np.float32),
            faces=np.array([[0,1,2]],np.int32),
            vertex_normals=np.array([[0,0,1]]*3,np.float32),
            face_normals=np.array([[0,0,1]],np.float32),
            vertex_areas_mm2=np.array([1.5]*3,np.float32),face_areas_mm2=np.array([4.5],np.float32),
            patch_vertices=np.arange(3,dtype=np.int32),patch_faces=np.array([0],np.int32),
            volume=np.full((9,9,9),100,np.uint8),ras_to_voxel=np.eye(4,dtype=np.float32),
            gray_white_pdf=np.full((256,256,1),1/65536,np.float32))

    def test_uniform_pdf_exact_unweighted_face_and_vertex_definition(self):
        result=patch_mri_likelihood(**self.arguments())
        expected=float(np.float32(np.log(1/65536)))
        self.assertEqual(result['face_ll'],expected)
        self.assertEqual(result['vertex_ll'],expected)
        self.assertEqual(result['mri_ll'],expected*2)

    def test_empty_patch_is_zero(self):
        args=self.arguments();args['patch_vertices']=np.empty(0,np.int32);args['patch_faces']=np.empty(0,np.int32)
        result=patch_mri_likelihood(**args)
        self.assertEqual(result['mri_ll'],0)

    def test_pdf_excludes_marked_vertex_without_changing_original_fields(self):
        white=np.array([90,90,130],np.float32);gray=np.array([60,60,20],np.float32)
        pdf=joint_gray_white_pdf(white_values=white,gray_values=gray,excluded_vertices=np.array([False,False,True]))
        self.assertEqual(pdf.dtype,np.float32)
        self.assertEqual(pdf.shape,(256,256,1))
        self.assertAlmostEqual(float(pdf[90,60,0]/pdf[130,20,0]),20,places=5)
        self.assertEqual(pdf[130,20,0],pdf[0,0,0])
        np.testing.assert_array_equal(white,np.array([90,90,130],np.float32))

    def test_tissue_pdf_prefers_matching_gray_white_samples(self):
        args=self.arguments();pdf=args['gray_white_pdf'].copy();pdf[100,100,0]=.25
        args['gray_white_pdf']=pdf;good=patch_mri_likelihood(**args)['mri_ll']
        args['volume']=np.full((9,9,9),60,np.uint8)
        self.assertGreater(good,patch_mri_likelihood(**args)['mri_ll'])

    def test_topology_sampling_uses_double_matrix_row_then_float_output(self):
        args=self.arguments()
        args['original_vertices'][:]=1
        args['vertex_normals'][:]=0;args['face_normals'][:]=0
        args['volume']=np.broadcast_to(np.arange(9,dtype=np.uint8)[:,None,None],(9,9,9)).copy()
        args['ras_to_voxel'][0]=[16777216,1,-16777216,.5]
        pdf=args['gray_white_pdf'].copy()
        for white in range(256):
            for gray in range(256):pdf[white,gray,0]=np.float32((white+gray+1)/100000)
        args['gray_white_pdf']=pdf
        result=patch_mri_likelihood(**args)
        # 正确采样得到white=gray=1.5；逐项F32归约会错误地得到.5。
        expected=float(np.float32(np.log(float(np.float64(pdf[1,1,0])*.25+
            np.float64(pdf[1,2,0])*.25+np.float64(pdf[2,1,0])*.25+np.float64(pdf[2,2,0])*.25))))
        self.assertEqual(result['vertex_ll'],expected)
        self.assertEqual(result['face_ll'],expected)

    def test_signed_source_face_areas_are_kept_in_weighted_diagnostic(self):
        args=self.arguments()
        args['faces']=np.array([[0,1,2],[0,2,1]],np.int32)
        args['face_normals']=np.array([[0,0,1],[0,0,1]],np.float32)
        args['face_areas_mm2']=np.array([4.5,-3.5],np.float32)
        args['patch_faces']=np.array([0,1],np.int32)
        result=patch_mri_likelihood(**args)
        expected=float(np.float32(np.log(1/65536)))
        self.assertEqual(result['face_ll'],expected)
        self.assertEqual(result['area_weighted_face_ll'],expected)

    def test_signed_vertex_area_is_source_metric_field(self):
        args=self.arguments()
        args['vertex_areas_mm2']=np.array([-3,1,1],np.float32)
        result=patch_mri_likelihood(**args)
        self.assertEqual(result['area_weighted_vertex_ll'],float(np.float32(np.log(1/65536))))

    def test_source_ordered_area_denominator_cannot_be_replaced_by_pairwise_sum(self):
        args=self.arguments()
        args['original_vertices']=np.tile(args['original_vertices'][0],(64,1))
        args['vertex_normals']=np.tile(args['vertex_normals'][0],(64,1))
        areas=np.ones(64,np.float32);areas[0]=np.float32(2**60);areas[-2]=-areas[0];areas[-1]=2
        args['vertex_areas_mm2']=areas;args['patch_vertices']=np.arange(64,dtype=np.int32)
        # 源有序double分母为2，numpy平衡归约却为0；不能错误拒绝有效定义。
        self.assertEqual(np.sum(areas,dtype=np.float64),0)
        result=patch_mri_likelihood(**args)
        self.assertTrue(np.isfinite(result['area_weighted_vertex_ll']))

    def test_zero_source_areas_preserve_undefined_diagnostics_and_finite_core(self):
        args=self.arguments()
        args['vertex_areas_mm2'][:]=0;args['face_areas_mm2'][:]=0
        before=[args[key].copy() for key in ('vertex_areas_mm2','face_areas_mm2')]
        result=patch_mri_likelihood(**args)
        expected=float(np.float32(np.log(1/65536)))
        self.assertEqual(result['face_ll'],expected)
        self.assertEqual(result['vertex_ll'],expected)
        for label in ('face','vertex'):
            self.assertTrue(np.isnan(result['area_weighted_'+label+'_ll']))
            self.assertFalse(result['area_weighted_'+label+'_defined'])
            self.assertEqual(result['area_weighted_'+label+'_bits'],0xffc00000)
        for key,value in zip(('vertex_areas_mm2','face_areas_mm2'),before):
            np.testing.assert_array_equal(args[key],value)

    def test_cancelling_signed_area_does_not_disable_finite_core(self):
        args=self.arguments()
        args['vertex_areas_mm2'][:]=[1,-2,1]
        result=patch_mri_likelihood(**args)
        self.assertTrue(np.isfinite(result['mri_ll']))
        self.assertFalse(result['area_weighted_vertex_defined'])
        self.assertTrue(result['area_weighted_face_defined'])

    def test_invalid_patch_does_not_drop_or_clip_indices(self):
        for name,value in [('patch_vertices',np.array([0,0],np.int32)),
            ('patch_faces',np.array([1],np.int32)),('face_areas_mm2',np.array([np.nan],np.float32)),
            ('gray_white_pdf',np.zeros((256,256,1),np.float32)),('ras_to_voxel',np.eye(3))]:
            args=self.arguments();args[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):patch_mri_likelihood(**args)


if __name__=='__main__':unittest.main()
