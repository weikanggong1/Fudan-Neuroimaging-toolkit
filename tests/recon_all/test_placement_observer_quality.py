"""观测透明性与严格几何诊断的小型单元测试；不充当真实benchmark。"""
import json
from types import SimpleNamespace
import numpy as np
import unittest
from pathlib import Path
import tempfile
from fnit.recon_all.place_surface_observer import PlacementObserver
from fnit.recon_all.place_surface_quality import mesh_structure, proper_pairs


class PlacementObserverQualityTests(unittest.TestCase):
    def test_observer_restores_on_error_and_returns_original_object(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);tmp_path=Path(temporary.name)
        xyz=np.array([[1.,2.,3.],[4.,5.,6.]],np.float32)
        gradient=np.ones_like(xyz);rip=np.array([False,True]);proposal=xyz+gradient
        result=(proposal,gradient)
        def original(*args,**kwargs):return result
        module=SimpleNamespace(**{name:original for name in ('compute_border_values_first_pass','intensity_gradient',
            'unconstrained_step_with_offsets','asynchronous_first_step','repair_intersections')})
        with self.assertRaises(RuntimeError):
            with PlacementObserver(stage_module=module,directory=tmp_path/'observation') as observer:
                assert module.unconstrained_step_with_offsets(xyz,gradient,rip,dt=.5) is result
                module.asynchronous_first_step(xyz,np.empty((0,3),np.int32),proposal,rip)
                observer.callback(1,0,proposal,{'stop':False})
                gradient[:]=99
                raise RuntimeError('caller failed after observation')
        assert module.unconstrained_step_with_offsets is original
        assert all(value is original for value in vars(module).values())
        events=json.loads((tmp_path/'observation/events.json').read_text())
        saved=np.load(tmp_path/'observation'/events[0]['file'])
        assert np.array_equal(saved['gradient'],np.ones_like(xyz))
        assert events[0]['arrays']['accepted']['dtype']=='float32'


    def test_strict_pair_chunks_complete_and_contact_not_proper(self):
        vertices=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],
                           [0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]],np.float32)
        faces=np.array([[0,1,2],[3,4,5]],np.int32)
        one,report=proper_pairs(first_vertices=vertices,faces=faces,chunk_faces=1)
        many,_=proper_pairs(first_vertices=vertices,faces=faces,chunk_faces=256)
        assert one==many=={(0,1)}
        assert report['candidate_truncation'] is False
        contacts=vertices.copy();contacts[3:]=[[-1.,-1.,0.],[1.,-1.,0.],[0.,-2.,0.]]
        assert proper_pairs(first_vertices=contacts,faces=faces)[0]==set()


    def test_cross_same_indices_are_not_exempted_and_inputs_untouched(self):
        first=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]],np.float32)
        second=np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]],np.float32)
        faces=np.array([[0,1,2]],np.int32);before=first.copy()
        assert proper_pairs(first_vertices=first,second_vertices=second,faces=faces)[0]=={(0,0)}
        assert np.array_equal(first,before)


    def test_structure_reports_closed_tetrahedron_and_nonmanifold_edge(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        report=mesh_structure(vertices=vertices,faces=faces)
        assert (report['euler'],report['connected_components'],report['boundary_edges'],report['nonmanifold_edges'])==(2,1,0,0)
        self.assertAlmostEqual(report['signed_volume_mm3'],1/6)
        broken=np.r_[faces,faces[:1]]
        assert mesh_structure(vertices=vertices,faces=broken)['nonmanifold_edges']==3

    def test_shared_one_corner_proper_crossing_is_retained(self):
        first=np.array([[0.,0.,0.],[1.,-1.,0.],[1.,1.,0.]],np.float32)
        second=np.array([[0.,0.,0.],[1.,0.,-1.],[1.,0.,1.]],np.float32)
        faces=np.array([[0,1,2]],np.int32)
        pairs,_=proper_pairs(first_vertices=first,second_vertices=second,faces=faces)
        self.assertEqual(pairs,{(0,0)})

    def test_self_shared_vertex_crossing_is_reported_separately(self):
        vertices=np.array([[0.,0.,0.],[1.,-1.,0.],[1.,1.,0.],[1.,0.,-1.],[1.,0.,1.]],np.float32)
        faces=np.array([[0,1,2],[0,3,4]],np.int32)
        pairs,report=proper_pairs(first_vertices=vertices,faces=faces)
        self.assertEqual(pairs,{(0,1)})
        self.assertEqual(report['shared_vertex_proper_pairs'],1)
        self.assertEqual(report['nonshared_vertex_proper_pairs'],0)
        self.assertEqual(proper_pairs(first_vertices=vertices,faces=faces,exclude_shared_vertices=True)[0],set())

    def test_certified_filter_retains_all_exact_proper_pairs(self):
        # 小型穷举单元测试；不是benchmark，也不用于精度门槛。
        from itertools import combinations
        from fnit.recon_all.place_surface_quality import _possible_open_crossing
        from fnit.recon_all.white_pial_crossing_python import _rational_triangle,_exact_relation
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[-1,-1,-1]],np.float64)
        triangles=[]
        for index in combinations(range(len(points)),3):
            tri=points[list(index)]
            if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])):triangles.append(tri)
        first=np.asarray([a for a in triangles for b in triangles]);second=np.asarray([b for a in triangles for b in triangles])
        possible=_possible_open_crossing(first,second)
        for index,(a,b) in enumerate(zip(first,second)):
            if _exact_relation(_rational_triangle(a),_rational_triangle(b))[0]==1:
                self.assertTrue(possible[index])

    def test_cleanup_pinned_union_preserves_labels_and_cannot_hide_crossing(self):
        from fnit.recon_all.place_surface_final_cleanup import pial_cleanup_fixed_vertices,repair_intersections
        ripped=np.array([False,False,False,True,False,False]);before=ripped.copy()
        cortex=np.array([],np.int64)
        fixed=pial_cleanup_fixed_vertices(ripped=ripped,cortex_vertices=cortex)
        self.assertTrue(fixed.all());self.assertTrue(np.array_equal(ripped,before))
        vertices=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]],np.float32)
        faces=np.array([[0,1,2],[3,4,5]],np.int32)
        after,diagnostics=repair_intersections(vertices,faces,fixed)
        self.assertTrue(np.array_equal(after,vertices))
        self.assertGreater(diagnostics['intersecting_faces_after'],0)
        self.assertEqual(proper_pairs(first_vertices=after,faces=faces)[0],{(0,1)})
        subset=pial_cleanup_fixed_vertices(ripped=ripped,cortex_vertices=np.array([0,3],np.int64))
        self.assertTrue(np.array_equal(subset,[False,True,True,True,True,True]))

    def test_cleanup_fixed_mask_rejects_invalid_labels(self):
        from fnit.recon_all.place_surface_final_cleanup import pial_cleanup_fixed_vertices
        for bad in (np.array([.5]),np.array([-1]),np.array([3])):
            with self.assertRaises(ValueError):pial_cleanup_fixed_vertices(ripped=np.zeros(3,bool),cortex_vertices=bad)

    def test_cleanup_default_keeps_nonrip_outside_cortex_neighbor_contribution(self):
        from inspect import signature
        from unittest.mock import patch
        from fnit.recon_all.place_surface_final_cleanup import pial_cleanup_mask, repair_intersections
        from fnit.recon_all.place_pial_python import place_pial_t1
        self.assertEqual(signature(place_pial_t1).parameters['cleanup_constraint_mode'].default, 'source_rip')
        ripped=np.array([False,False,True]); cortex=np.array([0,2],np.int64)
        vertices=np.array([[0.,0.,0.],[3.,0.,0.],[0.,3.,0.]],np.float32)
        faces=np.array([[0,1,2]],np.int32)
        original=ripped.copy()
        def one_cycle(mask):
            # 隔离SOAP的贡献合同：仅移动顶点0，一轮后marker停止。
            # 顶点1为非rip但cortex外，顶点2是原rip；不是精度benchmark。
            responses=[(np.array([True,False,False]),1),(np.zeros(3,bool),0)]
            with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=responses):
                return repair_intersections(vertices,faces,mask)[0]
        source_mask=pial_cleanup_mask(ripped=ripped,cortex_vertices=cortex)
        union_mask=pial_cleanup_mask(ripped=ripped,cortex_vertices=cortex,constraint_mode='cortex_pinned_union')
        source=one_cycle(source_mask); union=one_cycle(union_mask)
        self.assertTrue(np.array_equal(source_mask,original))
        self.assertTrue(np.array_equal(union_mask,[False,True,True]))
        self.assertGreater(source[0,0],2.99)  # 默认保留顶点1对平均的贡献。
        self.assertTrue(np.array_equal(union[0],vertices[0]))  # 显式并集会排除此邻居。
        self.assertTrue(np.array_equal(source[original],vertices[original]))
        self.assertTrue(np.array_equal(union[union_mask],vertices[union_mask]))
        self.assertTrue(np.array_equal(ripped,original))
        self.assertFalse(np.shares_memory(source_mask,ripped))

    def test_cleanup_mode_invalid_before_subject_io(self):
        from fnit.recon_all.place_pial_python import place_pial_t1
        with self.assertRaisesRegex(ValueError,'cleanup_constraint_mode'):
            place_pial_t1('/nonexistent/subject','lh',cleanup_constraint_mode='unknown')

    def test_nvml_equal_numeric_pid_and_single_nspid_do_not_prove_ownership(self):
        import os
        from unittest.mock import Mock,patch
        from fnit.recon_all.place_surface_benchmark_memory import PlacementMemorySampler
        identity=dict(local_pid=os.getpid(),starttime_ticks=123,pid_namespace='pid:[456]',nspid_chain=[os.getpid()],identity_error=None)
        sampler=PlacementMemorySampler(uuid='GPU-unit-test')
        sampler.pid_namespace_single=True
        sampler.query=Mock()
        sampler.query.snapshot.return_value=([dict(pid=os.getpid(),bytes=512)],4096,None)
        sampler.stop.wait=lambda seconds:sampler.stop.set()
        with patch('fnit.recon_all.profiling.ProcessTreeDeviceSampler._process_identity',return_value=identity):
            sampler._run()
        report=sampler.finish()
        self.assertIsNone(report['own_peak_bytes'])
        self.assertEqual(report['own_observed_samples'],0)
        self.assertEqual(report['device_total_peak_bytes'],4096)
        self.assertEqual(report['raw_samples'][0]['pid_mapping_status'],'unavailable')

    def test_nvml_trusted_mapping_rechecks_local_starttime(self):
        import os
        from unittest.mock import Mock,patch
        from fnit.recon_all.place_surface_benchmark_memory import PlacementMemorySampler
        identity=dict(local_pid=os.getpid(),starttime_ticks=123,pid_namespace='pid:[456]',nspid_chain=[os.getpid()],identity_error=None)
        trusted={987654:dict(local_pid=os.getpid(),starttime_ticks=123,pid_namespace='pid:[456]',source='unit-test independent host witness')}
        sampler=PlacementMemorySampler(uuid='GPU-unit-test',trusted_host_pid_map=trusted)
        sampler.query=Mock()
        sampler.query.snapshot.return_value=([dict(pid=987654,bytes=512)],4096,None)
        sampler.stop.wait=lambda seconds:sampler.stop.set() if len(sampler.samples)==2 else None
        stale=dict(identity,starttime_ticks=124)
        with patch('fnit.recon_all.profiling.ProcessTreeDeviceSampler._process_identity',side_effect=[identity,stale]):
            sampler._run()
        report=sampler.finish()
        self.assertEqual(report['own_peak_bytes'],512)
        self.assertEqual(report['own_observed_samples'],1)
        self.assertIsNone(report['raw_samples'][1]['own_bytes'])

    def test_structure_gate_rejects_no_cross_but_broken_surface(self):
        from fnit.recon_all.place_surface_quality import placement_structure_passed
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
        self.assertTrue(placement_structure_passed(mesh_structure(vertices=vertices,faces=faces)))
        # 无面对proper不能代替闭合或单组件；这里检查真实构造的结构。
        self.assertFalse(placement_structure_passed(mesh_structure(vertices=vertices,faces=faces[:-1])))
        double_vertices=np.vstack([vertices,vertices+10.])
        double_faces=np.vstack([faces,faces+4])
        self.assertEqual(proper_pairs(first_vertices=double_vertices,faces=double_faces)[1]['proper_pairs'],0)
        self.assertFalse(placement_structure_passed(mesh_structure(vertices=double_vertices,faces=double_faces)))
        collapsed=vertices.copy();collapsed[3]=collapsed[0]
        self.assertFalse(placement_structure_passed(mesh_structure(vertices=collapsed,faces=faces)))
        self.assertFalse(placement_structure_passed({}))

    def test_source_report_outside_cortex_move_is_not_selected_fixed_drift(self):
        import importlib.util
        script=Path(__file__).parents[2]/'tools/analyze_recon_white_pial_chain.py'
        spec=importlib.util.spec_from_file_location('fnit_own_chain_analysis_contract',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        white=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32)
        terminal=white.copy();terminal[1,0]=1.25
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
        kwargs=dict(terminal=terminal,white=white,faces=faces,ripped=np.array([True,False,False,False]),
            cortex_vertices=np.array([0,2,3]),terminal_pairs=np.empty((0,2),np.int64))
        mode,_=module.cleanup_mode_from_report({'pial_cleanup_constraint_mode':'source_rip'})
        record=module.cleanup_fixed_metrics(mode=mode,**kwargs)
        self.assertEqual(record['selected_mode_fixed']['different_elements_to_white'],0)
        self.assertEqual(record['union_pinned']['different_elements_to_white'],1)
        self.assertFalse(record['union_pinned_is_selected_constraint'])
        union=module.cleanup_fixed_metrics(mode='cortex_pinned_union',**kwargs)
        self.assertEqual(union['selected_mode_fixed']['different_elements_to_white'],1)
        # 历史commit名称不证明实际模式；新分析保持unknown而不误判漂移。
        mode,_=module.cleanup_mode_from_report({'candidate_commit':'b97a0027'})
        self.assertEqual(mode,'unknown')
        self.assertEqual(module.cleanup_fixed_metrics(mode=mode,**kwargs)['selected_mode_fixed']['assessment'],'not_assessed_unknown_cleanup_mode')
        self.assertIsNone(module.cleanup_fixed_metrics(mode=mode,**kwargs)['union_pinned_is_selected_constraint'])


if __name__=="__main__":unittest.main()
