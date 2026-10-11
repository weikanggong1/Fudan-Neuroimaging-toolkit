"""固定结构障碍数学合同，模拟例不替代MRI验证。"""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

spec=importlib.util.spec_from_file_location("fixed_pair_audit",Path(__file__).resolve().parents[2]/"tools/audit_pial_fixed_crossings_exact.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FixedCrossingExactTests(unittest.TestCase):
    def test_open_triangle_crossing_and_closed_contact(self):
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        for segment,kind in (([[.5,.5,-1.],[.5,.5,1.]],"proper_interior"), ([[0.,0.,-1.],[0.,0.,1.]],"closed_contact"), ([[3.,0.,-1.],[3.,0.,1.]],"none"), ([[.5,.5,0.],[.5,.5,1.]],"closed_contact"), ([[.5,.5,0.],[1.,.5,0.]],"coplanar_contact"), ([[3.,0.,0.],[4.,0.,0.]],"none")):
            with self.subTest(kind=kind):self.assertEqual(audit.segment_triangle_relation(segment,triangle),kind)

    def test_rotated_translated_and_tiny_positive_kept(self):
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        rotation=np.array([[.8,0.,.6],[0.,1.,0.],[-.6,0.,.8]]);offset=np.array([73.331,12.717,-61.723])
        segment=np.array([[.5,.5,-1.],[.5,.5,1.]])
        self.assertEqual(audit.segment_triangle_relation(segment@rotation+offset,triangle@rotation+offset),"proper_interior")
        self.assertEqual(audit.segment_triangle_relation([[.5,.5,-1e-15],[.5,.5,1e-15]],triangle),"proper_interior")

    def test_three_dimensional_fixed_segment_relations(self):
        first=[[-1.,0.,0.],[1.,0.,0.]]
        for second,kind in (([[0.,-1.,0.],[0.,1.,0.]],"transverse_interiors"), ([[0.,-1.,1e-15],[0.,1.,1e-15]],"none"), ([[1.,0.,0.],[1.,1.,0.]],"closed_contact"), ([[-.5,0.,0.],[.5,0.,0.]],"collinear_overlap"), ([[2.,0.,0.],[3.,0.,0.]],"none")):
            with self.subTest(kind=kind):self.assertEqual(audit.segment_segment_relation(first,second),kind)

    def test_proper_face_edge_obstruction_persists_for_free_corner(self):
        from tools.audit_white_pial_crossing_exact import exact_clip_relation
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        edge=np.array([[.5,.5,-1.],[.5,.5,1.]])
        for corner in ([1.,.5,0.],[.5,1.,1.],[-5.,-3.,-4.],[100.,-70.,10.]):
            with self.subTest(corner=corner):self.assertEqual(exact_clip_relation(triangle,np.vstack((edge,corner))),1)

    def test_nonparallel_edge_contact_does_not_assert_triangle_proper(self):
        from tools.audit_white_pial_crossing_exact import exact_clip_relation
        a=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        b=np.array([[0.,-1.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertEqual(audit.segment_segment_relation(a[:2],b[:2]),"transverse_interiors")
        self.assertEqual(exact_clip_relation(a,b),2)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):audit.segment_triangle_relation([[0.,0.,0.],[0.,0.,0.]],np.eye(3))
        with self.assertRaises(ValueError):audit.segment_segment_relation([[0.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[1.,0.,0.]])


class FixedAuditPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual=json.loads((Path(__file__).resolve().parents[2]/"validation/recon_all/optimizations/20261011_white_pial_crossing/self_audit.safe.json").read_text())

    def test_actual_public_complete_rows_without_geometry_rerun(self):
        for record,name,count in zip(self.actual["states"],("entry","candidate"),(40,38)):
            with self.subTest(state=name):
                rows=audit.checked_pair_rows(record=record,state=name,face_count=self.actual["faces"])
                self.assertIs(rows,record["rows"])
                self.assertEqual(len(rows),count)

    def test_incomplete_duplicate_nonproper_and_invalid_types_rejected(self):
        def field(key,value):return lambda d:d.__setitem__(key,value)
        def row(key,value):return lambda d:d["rows"][0].__setitem__(key,value)
        changes=[
            ("removed row",lambda d:d["rows"].pop()),
            ("duplicate row",lambda d:d["rows"].__setitem__(1,copy.deepcopy(d["rows"][0]))),
            ("wrong state",field("state","candidate")),
            ("missing count",lambda d:d.pop("union_pairs")),
            ("bool count",field("source_directed_pairs",False)),
            ("float count",field("float64_pairs",40.)),
            ("negative count",field("union_pairs",-1)),
            ("wrong source count",field("source_directed_pairs",79)),
            ("wrong f64 count",field("float64_pairs",39)),
            ("wrong union count",field("union_pairs",39)),
            ("missing class",lambda d:d["exact_classes"].pop("none")),
            ("bool class",lambda d:d["exact_classes"].__setitem__("proper",True)),
            ("wrong class count",lambda d:d["exact_classes"].__setitem__("proper",39)),
            ("nonproper",row("exact_kind","edge_point_contact")),
            ("missing kind",lambda d:d["rows"][0].pop("exact_kind")),
            ("negative face",row("pair",[-1,1])),
            ("face too large",row("pair",[0,self.actual["faces"]])),
            ("bool face",row("pair",[False,1])),
            ("float face",row("pair",[0.,1.])),
            ("tuple faces",row("pair",(0,1))),
            ("unordered faces",row("pair",[1,0])),
            ("bool source",row("source_directed_orientations",True)),
            ("source too large",row("source_directed_orientations",3)),
            ("nonbool f64",row("float64_positive",1)),
            ("no membership",lambda d:d["rows"][0].update(source_directed_orientations=0,float64_positive=False)),
        ]
        for name,change in changes:
            with self.subTest(name=name):
                record=copy.deepcopy(self.actual["states"][0]);change(record)
                with self.assertRaises(ValueError):audit.checked_pair_rows(record=record,state="entry",face_count=self.actual["faces"])

    def test_fixed_domain_rejects_label_and_grid_errors(self):
        white=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces=np.array([[0,1,2],[0,3,1]],dtype=np.int64)
        valid=dict(white_vertices=white,faces=faces,cortex_indices=np.array([0,1,1,2]),hipamyg_indices=np.array([0,2]))
        got=audit.checked_fixed_domain(**valid)
        np.testing.assert_array_equal(got[0],white.astype(np.float32));np.testing.assert_array_equal(got[1],faces)
        np.testing.assert_array_equal(got[4],np.array([False,True,False,True]))
        for key,value in (
            ("cortex_indices",np.array([-1])),("hipamyg_indices",np.array([4])),
            ("cortex_indices",np.array([1.])),("hipamyg_indices",np.array([False])),
            ("cortex_indices",np.array([[1]])),("faces",np.array([[0,1,-1]])),
            ("faces",np.array([[0,1,4]])),("faces",faces.astype(float)),
            ("faces",np.array([[True,False,True]])),("faces",np.array([0,1,2])),
            ("white_vertices",np.full((4,3),np.nan)),("white_vertices",np.full((4,3),np.inf)),
            ("white_vertices",np.full((4,3),1e50)),("white_vertices",np.ones((4,3),dtype=int)),
            ("white_vertices",np.ones((4,2))),
        ):
            with self.subTest(field=key,value_type=str(value.dtype)):
                with self.assertRaises(ValueError):audit.checked_fixed_domain(**(valid|{key:value}))

    def test_run_rejects_bad_report_before_math_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);case=base/"case";case.mkdir();candidate=base/"candidate.npy";candidate.write_bytes(b"")
            terminal=case/"cleanup-result.private.npy";terminal.write_bytes(b"")
            hashes={}
            for name in audit.INPUTS:
                p=case/"subject"/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"");hashes[name]=audit.sha(p)
            hash_path=base/"hashes.json";hash_path.write_text(json.dumps(hashes))
            report=copy.deepcopy(self.actual);report.update(input_sha256=hashes,terminal_sha256=audit.sha(terminal),candidate_sha256=audit.sha(candidate),vertices=4,faces=2,fixed_vertices=0)
            report["states"][0]["rows"]=[report["states"][0]["rows"][0]|{"pair":[0,1]}]
            report_path=base/"report.json";report_path.write_text(json.dumps(report));output=base/"output.json"
            with mock.patch("nibabel.freesurfer.read_geometry",return_value=(np.eye(4,3),np.array([[0,1,2],[0,2,3]]))),mock.patch("nibabel.freesurfer.read_label",return_value=np.arange(4)),mock.patch("numpy.load") as load,mock.patch.object(audit,"segment_triangle_relation") as relation:
                with self.assertRaises(ValueError):audit.run(diagnostic_case=case,candidate=candidate,input_hashes=hash_path,terminal_sha256=audit.sha(terminal),candidate_sha256=audit.sha(candidate),exact_report=report_path,expected_exact_report_sha256=audit.sha(report_path),output=output,threads=4)
                load.assert_not_called();relation.assert_not_called()
            self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
