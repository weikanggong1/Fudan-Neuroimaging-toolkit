"""公共selfpair数值导出合同；人工schema fixture，不是MRI基准。"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("self_pair_export",ROOT/"validation/recon_all/optimizations/20261011_white_pial_crossing/export_self_audit.py")
exporter=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exporter)


def fixtures():
    row={"pair":[0,1],"exact_kind":"proper","source_directed_orientations":2,"float64_positive":True, "shared_vertex_ids":0,"shared_coordinate_corners":0,"unique_corner_vertices":6,"fixed_corner_vertices":0,"all_corners_fixed":False,"component":0}
    classes={k:int(k=="proper") for k in exporter.KINDS}
    state={"source_directed_pairs":2,"float64_pairs":1,"union_pairs":1,"exact_classes":classes, "float64_nonproper_pairs":0,"source_nonproper_unordered_pairs":0,"pairs_with_shared_vertex_ids":0,"pairs_with_fixed_corners":0, "rows":[row],"components":[{"component":0,"pairs":1,"unique_vertices":6,"fixed_vertices":0,"exact_classes":classes}], "state":"entry","enumeration_seconds":1.,"exact_audit_seconds":.1}
    candidate=copy.deepcopy(state);candidate["state"]="candidate"
    report={"schema_version":1,"execution_status":"complete_readonly_exact_pair_audit", "scope":"all existing self-detector positive pairs on frozen own entry and candidate; no whole or new optimizer", "script_sha256":"a"*64,"pair_helper_sha256":"b"*64,"exact_audit_sha256":"c"*64, "input_sha256":{k:"d"*64 for k in exporter.INPUTS},"terminal_sha256":"e"*64,"candidate_sha256":"f"*64, "inputs_unchanged":True,"coordinate_space":"surface_RAS","coordinate_unit":"mm","vertices":6,"faces":2,"fixed_vertices":0, "threads":4,"cpu_affinity":[112,113,114,115],"states":[state,candidate], "source_version_bindings":{"fnit.recon_all.place_surface_residual_cleanup":"b"*64}, "wall_seconds_before_report_write":2.5,"peak_process_cpu_rss_bytes":100, "gpu_computation":False,"gpu_memory_bytes":None,"original_stage_program_called":False, "overall_metric_equivalence":"not_assessed","quality_gate_changed":False}
    timer={"schema_version":1,"scope":"single frozen Python benchmark cold CLI; not end-to-end recon-all", "command_sha256":"a"*64,"timer_script_sha256":"b"*64,"gpu_memory_bytes":None, "simultaneous_parent_child_peak_cpu_memory_bytes":None,"execution_status":"complete","child_exit_code":0, "cold_cli_wall_seconds":3.,"max_child_cpu_rss_bytes":100}
    return report,timer


class SelfPairExactExportTests(unittest.TestCase):
    def export(self,report,timer,root):
        (root/"raw.json").write_text(json.dumps(report));(root/"timer.json").write_text(json.dumps(timer))
        return exporter.export(report_path=root/"raw.json",timer_path=root/"timer.json",output=root/"public.json")

    def test_normal_schema_and_no_overwrite(self):
        report,timer=fixtures()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);result=self.export(report,timer,root)
            self.assertEqual(result["states"][0]["exact_classes"]["proper"],1)
            self.assertEqual(result["overall_metric_equivalence"],"not_assessed")
            with self.assertRaises(FileExistsError):self.export(report,timer,root)

    def test_unknown_private_fields_and_false_exit_rejected_without_output(self):
        cases=[]
        for mutate in (lambda d,t:d.update(private_path="/private/source"), lambda d,t:d["states"][0]["rows"][0].update(coordinates=[1,2,3]), lambda d,t:t.update(child_exit_code=False), lambda d,t:d["source_version_bindings"].update({"/private/source":"a"*64})):
            report,timer=fixtures();mutate(report,timer);cases.append((report,timer))
        for report,timer in cases:
            with self.subTest(case=len(report)),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                with self.assertRaises(ValueError):self.export(report,timer,root)
                self.assertFalse((root/"public.json").exists())

    def test_inconsistent_counts_relations_and_boolean_count_rejected(self):
        for mutate in (lambda d:d["states"][0].update(float64_pairs=False), lambda d:d["states"][0]["exact_classes"].update(proper=2), lambda d:d["states"][0]["rows"][0].update(exact_kind="/private/relation"), lambda d:d["states"][0]["rows"][0].update(pair=[1,0]), lambda d:d["states"][0]["rows"][0].update(all_corners_fixed=True)):
            report,timer=fixtures();mutate(report)
            with self.subTest(mutation=True),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                with self.assertRaises(ValueError):self.export(report,timer,root)
                self.assertFalse((root/"public.json").exists())

    def test_nonproper_positive_is_preserved(self):
        report,timer=fixtures()
        for state in report["states"]:
            state["rows"][0]["exact_kind"]="none";state["float64_nonproper_pairs"]=state["source_nonproper_unordered_pairs"]=1
            state["exact_classes"]={k:int(k=="none") for k in exporter.KINDS};state["components"][0]["exact_classes"]=state["exact_classes"]
        with tempfile.TemporaryDirectory() as td:
            result=self.export(report,timer,Path(td));self.assertEqual(result["states"][0]["float64_nonproper_pairs"],1)


if __name__=="__main__":unittest.main()
