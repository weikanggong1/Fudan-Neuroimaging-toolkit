"""首阻挡顺序、未面支撑边和动态删边契约；真实MRI回归另列。"""
import hashlib
import itertools
import unittest
import numpy as np
from fnit.recon_all.topology_edge_blocker_scan import OrderedBlockerScan, _edge_intersects_query_half
from fnit.recon_all.topology_preflight_python import _edge_intersects
from fnit.recon_all.topology_retessellation_dynamic import retessellate_ordered_edges, prepare_retessellation_base


class OrderedBlockerScanTests(unittest.TestCase):
    def test_first_blocker_uses_source_scan_and_row_order(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[0,2,1],[0,-2,1]],np.float32)
        rows=[[],[],[3],[2],[5],[4]];scan=np.asarray([4,2],np.int32)
        context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=rows,audit=True)
        hit=context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(6,np.bool_))
        self.assertEqual(hit,(4,5));self.assertEqual(context.predicate_calls,1)
        expected=np.asarray([[0,1,4,5,1]],dtype='<i4')
        self.assertEqual(context.sequence_digest.hexdigest(),hashlib.sha256(expected.tobytes()).hexdigest())

    def test_mapping_validity_applies_to_neighbor_not_query(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
        rows=[[],[],[3],[2]];scan=np.asarray([2],np.int32)
        context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=rows)
        valid=np.asarray([1,1,0,1],np.bool_)
        self.assertEqual(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=valid),(2,3))
        valid[3]=False
        self.assertIsNone(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=valid))

    def test_buffer_copies_then_syncs_full_current_row(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
        rows=[[],[],[3],[2]];scan=np.asarray([2],np.int32)
        context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=rows)
        rows[2].clear()
        self.assertEqual(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_)),(2,3))
        context.sync_vertex(2,rows[2])
        self.assertIsNone(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_)))

    def test_duplicate_queries_are_not_dropped_from_predicate_sequence(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,2,1]],np.float32)
        rows=[[],[],[3],[2]];scan=np.asarray([2,2],np.int32)
        self.assertFalse(_edge_intersects(xyz,0,1,2,3))
        context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=rows,audit=True)
        self.assertIsNone(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_)))
        expected=np.asarray([[0,1,2,3,0],[0,1,2,3,0]],dtype='<i4')
        self.assertEqual(context.predicate_calls,2)
        self.assertEqual(context.sequence_digest.hexdigest(),hashlib.sha256(expected.tobytes()).hexdigest())

    def test_inexact_or_outside_rows_are_rejected_before_int32_cast(self):
        for bad in ([1.5],[2**32+1]):
            with self.assertRaises(ValueError):OrderedBlockerScan(scan_vertices=np.asarray([0]),neighbor_rows=[bad,[]])
        context=OrderedBlockerScan(scan_vertices=np.asarray([0],np.int32),neighbor_rows=[[],[]])
        with self.assertRaises(ValueError):context.sync_vertex(-1,[])
        with self.assertRaises(ValueError):context.first_blocker(xyz=np.ones((2,3),np.float32),a=0,b=1,
            scan_vertices=np.asarray([1],np.int32),valid=np.ones(2,np.bool_))

    def test_nearzero_collinear_and_shared_endpoint_rules_reuse_predicate(self):
        cases=[[[1,0,1],[-1,0,1],[0,1,1],[0,-1e-9,1]],
               [[1,0,1],[-1,0,1],[.5,0,1],[-.5,0,1]],
               [[1,0,1],[-1,0,1],[0,1,1],[0,2,1]]]
        for data in cases:
            xyz=np.asarray(data,np.float32);scan=np.asarray([2],np.int32)
            context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=[[],[],[0,3],[2]])
            actual=context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_))
            expected=(2,3) if _edge_intersects(xyz,0,1,2,3) else None
            self.assertEqual(actual,expected);self.assertEqual(context.predicate_calls,1)

    def test_query_half_preserves_directed_threshold_and_degenerate_predicates(self):
        threshold=np.float32(5e-8)
        for scale in (np.float32(1),np.float32(1e-30),np.float32(1e30)):
            xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],
                [.5,0,1],[-.5,0,1],[0,np.nextafter(threshold,np.float32(0)),1],
                [0,np.nextafter(threshold,np.float32(1)),1]],np.float32)*scale
            # 每个朝向都与成熟谓词比较，包含零叉积/共线/近零阈值两侧。
            for a,b,c,d in itertools.permutations(range(len(xyz)),4):
                u0=xyz[a].astype(np.float64);u1=xyz[b].astype(np.float64)
                actual=_edge_intersects_query_half(xyz,a,b,c,d,u0,u1,np.cross(u0,u1))
                self.assertEqual(actual,_edge_intersects(xyz,a,b,c,d),(scale,a,b,c,d))

    def test_query_half_reloads_coordinates_for_each_scan(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
        scan=np.asarray([2],np.int32)
        context=OrderedBlockerScan(scan_vertices=scan,neighbor_rows=[[],[],[3],[2]])
        self.assertEqual(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_)),(2,3))
        xyz[1]=[1,2,1]
        self.assertFalse(_edge_intersects(xyz,0,1,2,3))
        self.assertIsNone(context.first_blocker(xyz=xyz,a=0,b=1,scan_vertices=scan,valid=np.ones(4,np.bool_)))

    def patch(self,**change):
        kwargs=dict(canonical=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1]],np.float32),
            base_faces=np.empty((0,3),np.int32),neighbors=[[],[],[],[]],
            candidate_edges=np.asarray([[0,3],[0,1],[1,2],[2,0]],np.int32),edge_used=np.zeros(4,np.int32),ordering=np.arange(4),
            inside_vertices=np.arange(4,dtype=np.int32),inside_discarded=np.zeros(4,np.bool_),
            border_vertices=np.empty(0,np.int32),hull_vertices=np.empty(0,np.int32),mode='some',blocker_scan_audit=True)
        kwargs.update(change)
        result=[]
        for backend in ('python','numba'):result.append(retessellate_ordered_edges(**kwargs,blocker_scan_backend=backend))
        for key in ('faces','edge_used','marks','selected_edges','selected_vertices','removed_vertices','temporary_faces'):
            np.testing.assert_array_equal(result[0][key],result[1][key])
        for key in ('neighbors','retessellation_trace','rounds','ninside','ndiscarded','blocker_scan_calls',
                'blocker_predicate_calls','blocker_predicate_sequence_sha256'):
            self.assertEqual(result[0][key],result[1][key],key)
        return result

    def test_some_dynamic_classification_remove_and_all_acceptance(self):
        self.assertEqual(self.patch()[1]['removed_vertices'].tolist(),[3])
        result=self.patch(mode='all')[1]
        self.assertEqual(result['selected_vertices'].tolist(),[0,1,2,3])
        self.assertGreater(result['blocker_predicate_calls'],0)
        self.patch(mode='all',overlaps=[[],[0],[],[]],overlap_complete=np.ones(4,np.bool_))

    def test_accepted_edge_is_live_blocker_and_prepared_unfaced_edge_preserved(self):
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
        result=self.patch(canonical=xyz,candidate_edges=np.asarray([[2,3],[0,1]],np.int32),
            edge_used=np.zeros(2,np.int32),ordering=np.arange(2),mode='all')[1]
        self.assertEqual(result['retessellation_trace'][1]['blocker'],[2,3])
        rows=[[],[],[3],[2]]
        context=prepare_retessellation_base(canonical=xyz,base_faces=np.empty((0,3),np.int32),neighbors=rows)
        result=self.patch(canonical=context.canonical,base_faces=context.base_faces,neighbors=context.neighbors,
            base_context=context,candidate_edges=np.asarray([[0,1]],np.int32),edge_used=np.zeros(1,np.int32),
            ordering=np.arange(1),inside_vertices=np.asarray([0,1],np.int32),inside_discarded=np.zeros(2,np.bool_),
            hull_vertices=np.asarray([2,3],np.int32),mode='all')[1]
        self.assertEqual(result['retessellation_trace'][0]['blocker'],[2,3]);self.assertEqual(result['neighbors'],rows)

    def test_deleted_first_blocker_retries_then_accepts_in_second_round(self):
        xyz=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1],[.25,-.2,1],[-.25,.2,1]],np.float32)
        edges=np.asarray(list(itertools.combinations(range(6),2)),np.int32)
        result=self.patch(canonical=xyz,neighbors=[[]for _ in xyz],candidate_edges=edges,
            edge_used=np.zeros(15,np.int32),ordering=np.asarray([4,0,2,10,5,12,3,6,1,14,8,13,11,7,9]),
            inside_vertices=np.arange(6,dtype=np.int32),inside_discarded=np.zeros(6,np.bool_))[1]
        self.assertEqual(result['rounds'],2)
        np.testing.assert_array_equal(result['removed_vertices'],[5])
        records=[r for r in result['retessellation_trace']if r['edge']==6]
        self.assertEqual(records[0]['blocker'],[0,5]);self.assertEqual(records[1]['round'],2)
        self.assertEqual(records[1]['decision'],'accepted')


if __name__=='__main__':unittest.main()
