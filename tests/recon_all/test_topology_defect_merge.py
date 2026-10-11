"""源ordered merge及前后属性相位合同；不是影像benchmark。"""
import unittest
import numpy as np
from fnit.recon_all.topology_defect_merge import merge_neighboring_defects


class DefectMerge(unittest.TestCase):
    def grid(self,rows=3,columns=3):
        n=rows*columns
        faces=[]
        for row in range(rows-1):
            for column in range(columns-1):
                first=row*columns+column
                faces.extend([[first,first+1,first+columns+1],[first,first+columns+1,first+columns]])
        return dict(canonical_vertices=np.column_stack([np.arange(n),2*np.arange(n),3*np.arange(n)]).astype(np.float32),
            initial_vertex_normals=np.tile(np.array([1,0,0],np.float32),(n,1)),
            vertex_normals=np.tile(np.array([1,0,0],np.float32),(n,1)),
            original_vertex_area=np.arange(1,n+1,dtype=np.float32),faces=np.array(faces,np.int32),
            component_labels=np.zeros(n,np.int32))

    def test_threshold_one_contact_keeps_components_and_source_sum_area(self):
        args=self.grid();args['component_labels'][0]=1;args['component_labels'][8]=2
        out=merge_neighboring_defects(**args)
        self.assertEqual(len(out['regions']),2)
        self.assertEqual(out['merge_events'].shape,(0,3))
        self.assertEqual(out['regions'][0]['area'],1)
        self.assertEqual(out['regions'][1]['area'],9)
        np.testing.assert_array_equal(out['initial_vertex_val'],[1,0,0,0,0,0,0,0,9])

    def test_merge_order_mean_area_and_premerge_val_not_overwritten(self):
        args=self.grid();args['component_labels'][[0,1]]=1;args['component_labels'][[7,8]]=2
        args['vertex_normals'][[7,8]]=[0,1,0]
        saved={k:v.copy() for k,v in args.items()}
        out=merge_neighboring_defects(**args)
        self.assertEqual(len(out['regions']),1)
        region=out['regions'][0]
        np.testing.assert_array_equal(region['vertices'],[0,1,7,8])
        self.assertEqual(region['area'],5.)
        np.testing.assert_array_equal(region['normal'],[.5,.5,0])
        np.testing.assert_array_equal(region['centroid'],[4,8,12])
        np.testing.assert_array_equal(out['initial_vertex_val'][[0,1,7,8]],[3,3,17,17])
        np.testing.assert_array_equal(out['source_order'],[2,3,4,5,6,0,1,7,8])
        np.testing.assert_array_equal(out['source_order'][out['vertex_translation']],np.arange(9))
        for k,v in saved.items():np.testing.assert_array_equal(args[k],v)

    def test_recheck_current_group_can_merge_earlier_source_component(self):
        args=self.grid(rows=3,columns=6)
        args['component_labels'][[0,1]]=1;args['component_labels'][[3,4]]=2
        args['component_labels'][[15,16]]=3
        out=merge_neighboring_defects(**args)
        self.assertEqual(len(out['regions']),1)
        region=out['regions'][0]
        self.assertEqual(region['source_component'],1)
        np.testing.assert_array_equal(region['vertices'],[3,4,15,16,0,1])
        np.testing.assert_array_equal(out['merge_events'][:,:2],[[1,2],[1,0]])
        self.assertEqual(np.max(out['source_marks']),2)
        self.assertEqual(np.max(out['labels']),1)
        np.testing.assert_array_equal(out['initial_vertex_val'][[0,1,3,4,15,16]],[3,3,9,9,33,33])

    def test_float32_order_is_preserved_in_initial_area_accumulation(self):
        args=self.grid();args['component_labels'][[0,1,4]]=1
        args['original_vertex_area'][[0,1,4]]=[16777216,1,1]
        out=merge_neighboring_defects(**args)
        self.assertEqual(out['regions'][0]['area'],16777216.)
        self.assertEqual(out['initial_vertex_val'][0],16777216.)

    def test_empty_defects_and_zero_border_normal_have_complete_domain(self):
        args=self.grid();out=merge_neighboring_defects(**args)
        self.assertEqual(out['regions'],())
        np.testing.assert_array_equal(out['source_order'],np.arange(9))
        args['initial_vertex_normals'][:]=0;args['component_labels'][0]=1
        out=merge_neighboring_defects(**args)
        np.testing.assert_array_equal(out['regions'][0]['normal'],[0,0,0])

    def test_invalid_labels_or_metric_rejected_without_mutation(self):
        args=self.grid();before=args['component_labels'].copy()
        for change in [dict(component_labels=np.array([1,0,0,0,0,0,0,0,1],np.int32)),
                dict(component_labels=np.full(9,2,np.int32)),
                dict(original_vertex_area=np.full(9,-1,np.float32)),
                dict(faces=np.array([[0,1,9]],np.int32)),
                dict(initial_vertex_normals=np.zeros((9,3),np.float64)),
                dict(vertex_normals=np.zeros((9,3),np.float64))]:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):
                merge_neighboring_defects(**{**args,**change})
        np.testing.assert_array_equal(args['component_labels'],before)

    def test_initial_border_and_merged_inside_normals_use_separate_phases(self):
        args=self.grid();args['component_labels'][0]=1;args['component_labels'][8]=2
        args['initial_vertex_normals'][:]=[0,0,1]
        args['vertex_normals'][:]=[0,1,0]
        out=merge_neighboring_defects(**args)
        for region in out['regions']:
            np.testing.assert_array_equal(region['normal'],[0,0,1])
        args['component_labels'][:]=0
        args['component_labels'][[0,1]]=1;args['component_labels'][[7,8]]=2
        out=merge_neighboring_defects(**args)
        np.testing.assert_array_equal(out['regions'][0]['normal'],[0,1,0])


if __name__=='__main__':unittest.main()
