"""联合候选族数学/API合同；小几何不替代真实pial benchmark。"""
import unittest
from unittest import mock
from types import SimpleNamespace
import json
import numpy as np

from fnit.recon_all import place_surface_residual_joint as joint
from tools.audit_white_pial_crossing_exact import exact_clip_relation


def example():
    xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[3,-1,-1],[3,-1,1],[.5,.5,0]],np.float32)
    faces=np.array([[0,1,2],[3,4,5]],np.int32)
    fixed=np.array([True,True,True,True,True,False])
    return xyz,faces,fixed


class JointResidualTests(unittest.TestCase):
    def test_mixed_fixed_normal_sides_use_existing_edge_family(self):
        xyz,faces,fixed=example()
        self.assertEqual(exact_clip_relation(xyz[faces[0]].astype(float),xyz[faces[1]].astype(float)),1)
        active,affected,matrix,rhs,report=joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=[(0,1)])
        np.testing.assert_array_equal(active,[5]);np.testing.assert_array_equal(affected,[1])
        selected=report['fixed_face_selections'][0]
        self.assertEqual(selected['selected_plane'],'edge_outside_0')
        for direction in selected['candidate_planes'][:2]:
            with self.subTest(direction=direction['plane']):
                self.assertEqual(direction['decision'],'fixed_corner_wrong_side')
                self.assertEqual(sorted(direction['fixed_corner_signs']),[-1,1])
        candidate,solve=joint.solve_joint_family(vertices=xyz,faces=faces,active=active,affected=affected,matrix=matrix,rhs=rhs)
        self.assertIsNotNone(candidate)
        self.assertEqual(solve['lp_calls'],1);self.assertEqual(solve['slsqp_calls'],1);self.assertEqual(solve['slsqp_dimensions'],3)
        np.testing.assert_array_equal(candidate[fixed],xyz[fixed]);np.testing.assert_array_equal(xyz,example()[0])
        self.assertLessEqual(candidate[5,1],-.002+1e-9)
        self.assertEqual(exact_clip_relation(candidate[faces[0]].astype(float),candidate[faces[1]].astype(float)),0)

    def test_lp_infeasible_scope_is_preselected_family_only(self):
        xyz,faces,fixed=example()
        active,affected,matrix,rhs,_=joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=[(0,1)])
        candidate,report=joint.solve_joint_family(vertices=xyz,faces=faces,active=active,affected=affected,matrix=matrix,rhs=rhs,max_displacement_mm=.1)
        self.assertIsNone(candidate);self.assertEqual(report['lp_calls'],1);self.assertEqual(report['slsqp_calls'],0)
        self.assertEqual(report['lp_infeasible_claim_scope'],'this preselected halfspace family under coordinate box only')

    def test_two_joint_linear_constraints_match_analytic_minimum(self):
        xyz,faces,fixed=example()
        candidate,report=joint.solve_joint_family(vertices=xyz,faces=faces,active=np.array([5]),affected=np.array([1]),matrix=np.array([[1.,0.,0.],[0.,1.,0.]]),rhs=np.array([.2,.3]))
        self.assertIsNotNone(candidate)
        np.testing.assert_allclose(candidate[5].astype(float)-xyz[5],[.2,.3,0.],rtol=0,atol=4e-8)
        self.assertEqual(report['lp_calls'],1);self.assertEqual(report['slsqp_calls'],1)
        np.testing.assert_array_equal(candidate[fixed],xyz[fixed])

    def test_nonlinear_constraint_jacobian_matches_independent_finite_difference(self):
        xyz,faces,_=example();original_minimize=joint.minimize;checked=[]
        def audit(fun,initial,**options):
            contract=options['constraints'][0];delta=np.array([.02,-.01,.03,-.02,.01,-.03])
            numeric=np.column_stack([(contract['fun'](delta+np.eye(6)[axis]*1e-7)-contract['fun'](delta-np.eye(6)[axis]*1e-7))/(2e-7) for axis in range(6)])
            np.testing.assert_allclose(contract['jac'](delta),numeric,rtol=2e-7,atol=2e-8);checked.append(True)
            return original_minimize(fun,initial,**options)
        with mock.patch.object(joint,'minimize',audit):
            trial,_=joint.solve_joint_family(vertices=xyz,faces=faces,active=np.array([4,5]),affected=np.array([1]),matrix=np.array([[0.,0.,0.,0.,1.,0.]]),rhs=np.array([.01]))
        self.assertEqual(checked,[True]);self.assertIsNotNone(trial)

    def test_fixed_exact_contact_receives_no_new_margin(self):
        xyz,faces,fixed=example();xyz[3]=[.1,0,0];xyz[4]=[.2,0,0]
        # 对方固定边恰在该三角面的边界；固定点不被clearance挪动。
        active,affected,matrix,rhs,report=joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=[(0,1)])
        self.assertFalse(report['existing_fixed_contact_margin_changed'])
        self.assertEqual(matrix.shape[1],3)
        candidate,solve=joint.solve_joint_family(vertices=xyz,faces=faces,active=active,affected=affected,matrix=matrix,rhs=rhs)
        self.assertIsNotNone(candidate);np.testing.assert_array_equal(candidate[fixed],xyz[fixed])

    def test_all_five_incompatible_reports_family_gap(self):
        xyz,faces,fixed=example();fixed[:]=True
        active,affected,matrix,rhs,report=joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=[(0,1)])
        self.assertEqual(len(active),0)
        candidate,solve=joint.solve_joint_family(vertices=xyz,faces=faces,active=active,affected=affected,matrix=matrix,rhs=rhs)
        self.assertIsNone(candidate);self.assertEqual(solve['slsqp_calls'],0)
        self.assertIn('not all collision-free geometries',report['family_completeness'])

    def test_pair_preflight_rejects_missing_domain_duplicate_and_components(self):
        xyz,faces,fixed=example()
        for pairs in ([(0,1),(0,1)],[(1,0)],[(0,2)],[(False,1)],[(0.,1)],[(0,0)],{(0,1)}):
            with self.subTest(pairs=pairs):
                with self.assertRaises(ValueError):joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=pairs)
        xyz2=np.concatenate((xyz,xyz+np.array([10,0,0],np.float32)))
        faces2=np.vstack((faces,faces+6));fixed2=np.tile(fixed,2)
        with self.assertRaises(ValueError):joint.build_fixed_face_family(vertices=xyz2,faces=faces2,fixed_vertices=fixed2,pairs=[(0,1),(2,3)])

    def test_solver_index_incidence_and_type_failures(self):
        xyz,faces,fixed=example()
        active,affected,matrix,rhs,_=joint.build_fixed_face_family(vertices=xyz,faces=faces,fixed_vertices=fixed,pairs=[(0,1)])
        valid=dict(vertices=xyz,faces=faces,active=active,affected=affected,matrix=matrix,rhs=rhs)
        for key,value in (('active',np.array([-1])),('active',np.array([6])),('active',np.array([True])),('active',np.array([5.])),('active',np.array([[5]])),('active',np.array([5,5])),('affected',np.array([0])),('affected',np.array([1,0])),('matrix',matrix.astype(bool)),('matrix',np.full_like(matrix,np.nan)),('rhs',np.array([[1.]])),('rhs',np.array([np.inf])),('optimizer_iterations',False),('minimum_area_ratio',0.),('max_displacement_mm',0.)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):joint.solve_joint_family(**(valid|{key:value}))

    def test_complete_tiny_geometry_accepts_once_no_new_white_gate(self):
        xyz,faces,fixed=example()
        result,trial,report=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
        self.assertIsNotNone(trial);np.testing.assert_array_equal(result,trial)
        self.assertEqual(report['decision'],'accepted_once');self.assertEqual(report['status'],'passed_zero_in_source_float64_and_affected_exact')
        self.assertEqual(report['source_directed_pairs_after'],0);self.assertEqual(report['float64_pairs_after'],0)
        self.assertTrue(all(report['trial_checks'].values()));self.assertEqual(report['fixed_different_elements'],0)
        self.assertFalse(report['white_pial_guard']['final_zero_proper'])
        self.assertEqual(report['optimizer']['slsqp_calls'],1)
        self.assertEqual(report['exact_affected_self_guard']['trial_proper_pairs'],0)

    def test_new_white_crossing_rejection_preserves_entry_and_trial(self):
        xyz,faces,fixed=example()
        class FakeIndex:
            def __init__(self,**kwargs):self.calls=0
            def query(self,values):
                self.calls+=1
                proper=set() if self.calls==1 else {(1,0)}
                return proper,set(),set(),{'proper_crossing_pairs':len(proper)}
        with mock.patch.object(joint,'WhitePialCrossingIndex',FakeIndex):
            result,trial,report=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
        self.assertIsNotNone(trial);self.assertFalse(np.array_equal(trial,xyz));np.testing.assert_array_equal(result,xyz)
        self.assertEqual(report['decision'],'rejected_by_complete_quality_gates');self.assertEqual(report['status'],'failed_residual')
        self.assertTrue(report['final_coordinates_equal_entry']);self.assertTrue(report['white_pial_guard']['final_zero_proper'])
        self.assertFalse(report['white_pial_guard']['trial_zero_proper'])
        self.assertFalse(report['trial_checks']['no_new_white_proper_pairs'])

    def test_nonproper_input_list_is_not_silently_waived(self):
        xyz,faces,fixed=example()
        with self.assertRaises(ValueError):joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=lambda a,b:2)

    def test_fixed_signed_zero_change_fails_bit_identity_gate(self):
        xyz,faces,fixed=example();original_solver=joint.solve_joint_family
        def changed_fixed_bits(**kwargs):
            trial,report=original_solver(**kwargs);trial[0,0]=-0.;return trial,report
        with mock.patch.object(joint,'solve_joint_family',changed_fixed_bits):
            result,trial,report=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
        self.assertTrue(np.array_equal(trial[0],xyz[0]))  # 普通浮点相等不足以证明身份不变。
        self.assertFalse(report['basic_trial_checks']['fixed_coordinates_equal']);self.assertFalse(report['basic_trial_checks']['only_group_movable_vertices_changed'])
        self.assertEqual(report['trial_fixed_different_elements'],1);self.assertEqual(report['decision'],'rejected_by_basic_geometry_before_intersection_qc')
        self.assertEqual(report['trial_intersection_qc'],'not_assessed')
        np.testing.assert_array_equal(result.view(np.uint32),xyz.view(np.uint32))

    def test_f64_feasible_sliver_can_collapse_after_float32_quantization(self):
        spacing=float(np.spacing(np.float32(100.)))
        xyz=np.array([[100.,100.,100.],[100.+spacing,100.,100.],[100.,100.+spacing,100.]],np.float32)
        delta=np.zeros(9);delta[3]=-.7*spacing
        answer=SimpleNamespace(x=delta,status=0,success=True,nit=1)
        with mock.patch.object(joint,'minimize',return_value=answer):
            trial,report=joint.solve_joint_family(vertices=xyz,faces=np.array([[0,1,2]],np.int32),active=np.arange(3),affected=np.array([0]),matrix=np.zeros((1,9)),rhs=np.array([-1.]))
        # F64面积比0.3满足预定0.2门；F32舍入令两个角点完全重合。
        p64=xyz.astype(float)+delta.reshape(3,3)
        self.assertGreater(np.linalg.norm(np.cross(p64[1]-p64[0],p64[2]-p64[0])),0.)
        self.assertGreaterEqual(report['minimum_optimizer_constraint'],0.)
        self.assertIsNotNone(trial);self.assertEqual(report['decision'],'finite_quantized_trial')
        self.assertEqual(np.linalg.norm(joint._face_normals(trial,np.array([[0,1,2]],np.int32))),0.)

    def test_finite_degenerate_trial_retained_without_intersection_qc(self):
        xyz,faces,fixed=example();xyz*=.1;original_solver=joint.solve_joint_family
        def degenerate_quantization(**kwargs):
            trial,report=original_solver(**kwargs);trial[5]=(xyz[3]+xyz[4])/2.;return trial,report
        original_pairs=joint._pair_sets
        with mock.patch.object(joint,'solve_joint_family',degenerate_quantization),mock.patch.object(joint,'exact_affected_self_pairs',side_effect=AssertionError('unmeasured exact QC')),mock.patch.object(joint,'WhitePialCrossingIndex',side_effect=AssertionError('unmeasured white QC')),mock.patch.object(joint,'_pair_sets',wraps=original_pairs) as pairs:
            result,trial,report=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
        self.assertEqual(pairs.call_count,1);self.assertIsNotNone(trial);self.assertTrue(np.isfinite(trial).all())
        self.assertEqual(report['trial_zero_area_faces'],1);self.assertEqual(report['trial_minimum_area_ratio'],0.)
        self.assertEqual(report['trial_evaluation'],'basic_geometry_rejected_before_intersection_qc');self.assertEqual(report['trial_intersection_qc'],'not_assessed')
        self.assertIsNone(report['exact_affected_self_guard']);self.assertIsNone(report['white_pial_guard'])
        self.assertNotIn('exact_trial',report);self.assertNotIn('source_directed_pairs_trial',report)
        self.assertEqual(report['status'],'failed_residual');np.testing.assert_array_equal(result.view(np.uint32),xyz.view(np.uint32))
        json.dumps(report,allow_nan=False)

    def test_invalid_optimizer_shape_and_nonfinite_values_never_enter_constraints(self):
        xyz,faces,_=example();valid=dict(vertices=xyz,faces=faces,active=np.array([5]),affected=np.array([1]),matrix=np.array([[1.,0.,0.]]),rhs=np.array([.1]))
        for values in (np.array([np.nan,0.,0.]),np.array([np.inf,0.,0.]),np.array([0.]),np.array([1.e308,0.,0.])):
            with self.subTest(values=values.tolist()):
                answer=SimpleNamespace(x=values,status=0,success=True,nit=1)
                with mock.patch.object(joint,'minimize',return_value=answer):trial,report=joint.solve_joint_family(**valid)
                self.assertIsNone(trial);self.assertIsNone(report['minimum_optimizer_constraint'])
                self.assertIn(report['decision'],{'selected_family_optimizer_invalid_or_nonfinite','selected_family_optimizer_constraints_nonfinite'})
                json.dumps(report,allow_nan=False)

    def test_nonfinite_quantized_candidate_not_retained(self):
        extreme=float(np.finfo(np.float32).max)
        xyz=np.array([[extreme,0.,0.],[extreme,1.,0.],[extreme,0.,1.]],np.float32)
        delta=np.array([1.e37,0.,0.]*3);answer=SimpleNamespace(x=delta,status=0,success=True,nit=1)
        # 大坐标/大合法API位移只检验finite失败门，不是本轮1mm实验配置。
        with mock.patch.object(joint,'minimize',return_value=answer):
            trial,report=joint.solve_joint_family(vertices=xyz,faces=np.array([[0,1,2]],np.int32),active=np.arange(3),affected=np.array([0]),matrix=np.zeros((1,9)),rhs=np.array([-1.]),max_displacement_mm=1.e39)
        self.assertIsNone(trial);self.assertEqual(report['decision'],'nonfinite_quantized_trial_not_retained')
        self.assertGreaterEqual(report['minimum_optimizer_constraint'],0.);json.dumps(report,allow_nan=False)

    def test_new_nonmedial_contact_also_rejects_trial(self):
        xyz,faces,fixed=example()
        class FakeIndex:
            def __init__(self,**kwargs):self.calls=0
            def query(self,values):
                self.calls+=1;contacts=set() if self.calls==1 else {(1,0)}
                return set(),contacts,set(),{'proper_crossing_pairs':0}
        with mock.patch.object(joint,'WhitePialCrossingIndex',FakeIndex):
            result,trial,report=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
        np.testing.assert_array_equal(result,xyz);self.assertIsNotNone(trial)
        self.assertFalse(report['trial_checks']['no_new_nonmedial_exact_contacts'])
        self.assertFalse(report['white_pial_guard']['trial_zero_nonmedial_contact'])
        self.assertTrue(report['white_pial_guard']['final_zero_nonmedial_contact'])

    def test_exact_local_guard_finds_shared_id_proper_missed_by_old_detectors(self):
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[1,.5,-1],[1,.5,1]],np.float32)
        faces=np.array([[0,1,2],[0,3,4]],np.int32)
        self.assertEqual(joint._pair_sets(xyz,faces),(set(),set()))
        proper,area,positive,report=joint.exact_affected_self_pairs(vertices=xyz,faces=faces,query_faces=np.array([1]),exact_relation=exact_clip_relation)
        self.assertEqual(proper,{(0,1)});self.assertEqual(positive,{(0,1)});self.assertFalse(area)
        self.assertEqual(report['complete_bbox_pairs'],1);self.assertEqual(report['positive_pairs_with_shared_vertex_ids'],1)

    def test_exact_local_guard_distinguishes_shared_edge_and_area(self):
        for vertex,area_count in (([0,-1,0],0),([.5,.5,0],1)):
            with self.subTest(vertex=vertex):
                xyz=np.array([[0,0,0],[2,0,0],[0,2,0],vertex],np.float32)
                faces=np.array([[0,1,2],[1,0,3]],np.int32)
                proper,area,positive,report=joint.exact_affected_self_pairs(vertices=xyz,faces=faces,query_faces=np.array([0,1]),exact_relation=exact_clip_relation)
                self.assertFalse(proper);self.assertEqual(len(area),area_count);self.assertEqual(positive,{(0,1)})
                self.assertEqual(report['exact_tested_pairs'],1)

    def test_exact_local_bbox_matches_complete_small_allpair_audit_and_tiny_crossing(self):
        rng=np.random.default_rng(726)
        xyz=np.vstack((np.array([[0,0,0],[2,0,0],[0,2,0],[1,.5,-1e-15],[1,.5,1e-15],[.5,.2,0]]),rng.normal(size=(6,3)))).astype(np.float32)
        faces=np.arange(12,dtype=np.int32).reshape(4,3);query=np.array([0,2,0],np.int64)
        proper,area,positive,report=joint.exact_affected_self_pairs(vertices=xyz,faces=faces,query_faces=query,exact_relation=exact_clip_relation)
        expected_proper=set();expected_positive=set()
        for a in range(4):
            for b in range(a+1,4):
                if a not in query and b not in query:continue
                relation=exact_clip_relation(xyz[faces[a]].astype(float),xyz[faces[b]].astype(float))
                if relation==1:expected_proper.add((a,b))
                if relation:expected_positive.add((a,b))
        self.assertEqual(proper,expected_proper);self.assertEqual(positive,expected_positive)
        self.assertIn((0,1),proper);self.assertEqual(report['query_faces'],2)


if __name__=='__main__':unittest.main()
