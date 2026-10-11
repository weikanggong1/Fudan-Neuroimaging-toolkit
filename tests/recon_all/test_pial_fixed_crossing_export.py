"""固定障碍导出schema人工fixture；不冒称真实MRI结果。"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("fixed_export",ROOT/"validation/recon_all/optimizations/20261011_white_pial_crossing/export_fixed_audit.py")
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)


def fixtures():
    row={"pair":[0,1],"fixed_corners":[{"vertex":0,"fixed_source":"rip","equals_own_white":True}],"face_edge_checks":[{"fixed_face":0,"fixed_edge":[3,4],"kind":"proper_interior"}],"edge_edge_checks":[],"immutable_proper_obstruction":True,"immutable_contact_obstruction":True}
    state={"state":"entry","pairs":1,"pairs_with_immutable_proper_obstruction":1,"pairs_with_any_immutable_contact":1,"face_edge_classes":{k:int(k=="proper_interior") for k in exporter.FACE},"edge_edge_classes":{k:0 for k in exporter.EDGE},"exact_checks_seconds":.1,"rows":[row]}
    candidate=copy.deepcopy(state);candidate["state"]="candidate"
    report={"schema_version":1,"execution_status":"complete_readonly_fixed_structure_audit","scope":"all fixed substructures of existing frozen exact selfpair lists; no joint optimizer or new quality gate","script_sha256":"a"*64,"exact_report_sha256":"b"*64,"input_sha256":{k:"c"*64 for k in exporter.INPUTS},"terminal_sha256":"d"*64,"candidate_sha256":"e"*64,"inputs_unchanged":True,"coordinate_space":"surface_RAS","coordinate_unit":"mm","vertices":6,"faces":2,"fixed_vertices":5,"threads":4,"cpu_affinity":[112,113,114,115],"states":[state,candidate],"wall_seconds_before_report_write":1.,"peak_process_cpu_rss_bytes":100,"gpu_computation":False,"gpu_memory_bytes":None,"original_stage_program_called":False,"quality_gate_changed":False,"overall_metric_equivalence":"not_assessed"}
    timer={"schema_version":1,"scope":"single frozen Python benchmark cold CLI; not end-to-end recon-all","command_sha256":"a"*64,"timer_script_sha256":"b"*64,"gpu_memory_bytes":None,"simultaneous_parent_child_peak_cpu_memory_bytes":None,"execution_status":"complete","child_exit_code":0,"cold_cli_wall_seconds":2.,"max_child_cpu_rss_bytes":100}
    return report,timer


class FixedExportTests(unittest.TestCase):
    def call(self,report,timer,root):
        (root/"raw").write_text(json.dumps(report));(root/"time").write_text(json.dumps(timer))
        return exporter.export(report_path=root/"raw",timer_path=root/"time",output=root/"safe")

    def test_complete_diagnostic_with_no_equivalence(self):
        report,timer=fixtures()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);result=self.call(report,timer,root)
            self.assertTrue(result["states"][0]["rows"][0]["immutable_proper_obstruction"])
            self.assertFalse(result["quality_gate_changed"])
            with self.assertRaises(FileExistsError):self.call(report,timer,root)

    def test_unknown_and_private_fields_fail_without_output(self):
        for mutate in (lambda d,t:d.update(private_path="/private/source"),lambda d,t:d["states"][0]["rows"][0]["fixed_corners"][0].update(fixed_source="/private/mask"),lambda d,t:t.update(child_exit_code=False),lambda d,t:d["states"][0].update(pairs=False)):
            d,t=fixtures();mutate(d,t)
            with self.subTest(mutation=True),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                with self.assertRaises(ValueError):self.call(d,t,root)
                self.assertFalse((root/"safe").exists())

    def test_proper_and_contact_obstructions_not_confused(self):
        d,t=fixtures()
        for state in d["states"]:
            state["rows"][0]["face_edge_checks"][0]["kind"]="closed_contact";state["rows"][0]["immutable_proper_obstruction"]=False;state["pairs_with_immutable_proper_obstruction"]=0
            state["face_edge_classes"]={k:int(k=="closed_contact") for k in exporter.FACE}
        with tempfile.TemporaryDirectory() as td:
            result=self.call(d,t,Path(td));self.assertEqual(result["states"][0]["pairs_with_immutable_proper_obstruction"],0)
        d["states"][0]["rows"][0]["immutable_proper_obstruction"]=True
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(ValueError):self.call(d,t,root)
            self.assertFalse((root/"safe").exists())


if __name__=="__main__":unittest.main()
