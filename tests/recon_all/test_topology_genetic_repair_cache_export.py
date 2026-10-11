"""完整多缺陷公开指标的拒绝门；这些小数据是调度测试，不是 MRI benchmark。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("repair_cache_export", TOOLS / "export_recon_topology_genetic_repair_cache.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class CacheExportGates(unittest.TestCase):
    def _arguments(self, root, **changes):
        run = root / "run"
        run.mkdir()
        source = root / "source"
        source.mkdir()
        manifest = source / 'SOURCE_MANIFEST.private.json'
        manifest.write_text(json.dumps(dict(driver_sha256='d' * 64)))
        source_sha = exporter.sha(manifest)
        policy = dict(threads=4, face_detection_cache=True, population_size=10)
        inputs = dict(nofix='b' * 64, qsphere='c' * 64, brain='e' * 64)
        protocol = source / 'protocol.private.json'
        protocol.write_text(json.dumps(dict(actual_policy=policy, input_sha256=inputs)))
        request = run / 'request.private.json'
        request.write_text(json.dumps(dict(source_manifest_sha256=source_sha,
            protocol_sha256=exporter.sha(protocol), actual_policy=policy, input_sha256=inputs)))
        process = run / 'process.private.json'
        process.write_text(json.dumps(dict(exit_code=0, source_manifest_sha256=source_sha,
            driver_sha256='d' * 64, unchanged_frozen_sources=True, complete_fresh_cli_seconds=7.25)))
        exit_record = run / "EXIT.json"
        exit_record.write_text(json.dumps(dict(exit_code=0, repair_cache_receipt_present=True)))
        result = dict(input_run=run, frozen_source=source, source_manifest=manifest,
            expected_source_manifest_sha256=source_sha, exit_record=exit_record,
            expected_exit_sha256=exporter.sha(exit_record), output_directory=root / "public",
            hemisphere="lh", expected_defects=9, process_record=process,
            expected_process_sha256=exporter.sha(process), request_record=request,
            expected_request_sha256=exporter.sha(request), protocol_record=protocol,
            expected_protocol_sha256=exporter.sha(protocol))
        result.update(changes)
        return result

    def test_incomplete_producer_never_creates_public_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            with self.assertRaises(FileNotFoundError):
                exporter.export_completed_premesh_cache(**args)
            self.assertFalse(args["output_directory"].exists())

    def test_failed_process_rejected_before_producer_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            args["exit_record"].write_text('{"exit_code":1}\n')
            args["expected_exit_sha256"] = exporter.sha(args["exit_record"])
            with mock.patch.object(exporter, "verified_premesh") as producer:
                with self.assertRaisesRegex(ValueError, "exit0"):
                    exporter.export_completed_premesh_cache(**args)
                producer.assert_not_called()
            self.assertFalse(args["output_directory"].exists())

    def test_other_successful_process_cannot_be_spliced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._arguments(root)
            unrelated = root / "unrelated.EXIT.json"
            unrelated.write_text('{"exit_code":0}\n')
            args.update(exit_record=unrelated, expected_exit_sha256=exporter.sha(unrelated))
            with self.assertRaisesRegex(ValueError, "actual run"):
                exporter.export_completed_premesh_cache(**args)
            self.assertFalse(args["output_directory"].exists())

    def test_process_actual_source_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            data = json.loads(args['process_record'].read_text())
            data['source_manifest_sha256'] = 'b' * 64
            args['process_record'].write_text(json.dumps(data))
            args["expected_process_sha256"] = exporter.sha(args["process_record"])
            with self.assertRaisesRegex(ValueError, "source/driver/request/protocol identity"):
                exporter.export_completed_premesh_cache(**args)
            self.assertFalse(args["output_directory"].exists())

    def test_missing_process_source_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            data = json.loads(args['process_record'].read_text())
            del data['source_manifest_sha256']
            args['process_record'].write_text(json.dumps(data))
            args["expected_process_sha256"] = exporter.sha(args["process_record"])
            with self.assertRaisesRegex(ValueError, "source/driver/request/protocol identity"):
                exporter.export_completed_premesh_cache(**args)
            self.assertFalse(args["output_directory"].exists())

    def _identity(self, args):
        return exporter.verify_execution_identity(**{key: value for key, value in args.items()
            if key not in ('output_directory', 'hemisphere', 'expected_defects')})

    def test_sparse_exit_requires_bound_process_request_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            process, request, protocol = self._identity(args)
            self.assertEqual(process['complete_fresh_cli_seconds'], 7.25)
            self.assertEqual(request['actual_policy'], protocol['actual_policy'])
            self.assertNotIn('source_manifest_sha256', json.loads(args['exit_record'].read_text()))
            for key in ('process_record', 'request_record', 'protocol_record'):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'explicit actual'):
                    self._identity(dict(args, **{key: None}))

    def test_foreign_process_and_mismatched_driver_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._arguments(root)
            foreign = root / 'process.private.json'
            foreign.write_bytes(args['process_record'].read_bytes())
            with self.assertRaisesRegex(ValueError, 'record must belong'):
                self._identity(dict(args, process_record=foreign))
            process = json.loads(args['process_record'].read_text())
            process['driver_sha256'] = 'f' * 64
            args['process_record'].write_text(json.dumps(process))
            args['expected_process_sha256'] = exporter.sha(args['process_record'])
            with self.assertRaisesRegex(ValueError, 'driver/request/protocol'):
                self._identity(args)

    def test_rehashed_wrong_protocol_and_bool_policy_cannot_be_spliced(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            protocol = json.loads(args['protocol_record'].read_text())
            protocol['actual_policy']['population_size'] = 5
            args['protocol_record'].write_text(json.dumps(protocol))
            args['expected_protocol_sha256'] = exporter.sha(args['protocol_record'])
            with self.assertRaisesRegex(ValueError, 'request/protocol'):
                self._identity(args)
            request = json.loads(args['request_record'].read_text())
            request['protocol_sha256'] = args['expected_protocol_sha256']
            request['actual_policy'] = dict(protocol['actual_policy'], face_detection_cache=1)
            args['request_record'].write_text(json.dumps(request))
            args['expected_request_sha256'] = exporter.sha(args['request_record'])
            with self.assertRaisesRegex(ValueError, 'request/protocol'):
                self._identity(args)

    def test_payload_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            file = source / "module.py"
            file.write_bytes(b"before\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(payload={"module.py":dict(sha256=exporter.sha(file), size=file.stat().st_size)})))
            digest = exporter.sha(manifest)
            file.write_bytes(b"after\n")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                exporter.verify_manifest(source_directory=source, manifest_path=manifest, expected_manifest_sha256=digest)

    def test_actual_module_and_numeric_privacy(self):
        with self.assertRaisesRegex(ValueError, "loaded module"):
            exporter.verify_modules(actual={"fnit.recon_all.example":"b" * 64},
                payload={"src/fnit/recon_all/example.py":dict(sha256="a" * 64)})
        values = dict(count=7, fraction=1.25, done=False, unknown=None,
                      private_path="/private/source", rows=[dict(step=3, message="private")])
        selected = exporter._numeric(values)
        data, count = exporter.serialize(selected)
        self.assertNotIn(b"/private", data)
        self.assertEqual(json.loads(data), dict(count=7, fraction=1.25, done=False, unknown=None, rows=[dict(step=3)]))
        self.assertIs(type(selected["count"]), int)
        self.assertIs(type(selected["done"]), bool)
        self.assertEqual(count, 5)


class OrigCacheExportGates(unittest.TestCase):
    """真实出口的来源/严格质量合同；小字节仅用于 guard，不是 MRI。"""
    def fixture(self, root):
        source=root/'source';run=root/'orig';parent=root/'parent';candidate=run/'candidate';pc=parent/'candidate'
        for p in (source,candidate,pc):p.mkdir(parents=True)
        def write(p,value):p.write_text(json.dumps(value))
        modules={}
        payload={}
        for name in ('mris_remesh_python','mris_remove_intersection_python','place_surface_final_cleanup'):
            relative='src/fnit/recon_all/'+name+'.py';p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# owned source fixture\n')
            digest=exporter.sha(p);modules['fnit.recon_all.'+name]=digest;payload[relative]=dict(sha256=digest,size=p.stat().st_size)
        tool=source/'strong_orig/benchmark_recon_topology_genetic_orig.py';tool.parent.mkdir();tool.write_text('# strong tool fixture\n')
        payload[str(tool.relative_to(source))]=dict(sha256=exporter.sha(tool),size=tool.stat().st_size)
        manifest=source/'SOURCE_MANIFEST.private.json';write(manifest,dict(payload=payload));mf=exporter.sha(manifest)
        for p in (pc/'lh.orig.premesh',pc/'lh.defect_labels'):p.write_bytes(b'fixture')
        parent_report=dict(completed_defects=9,input_sha256=dict(nofix='b'*64,qsphere='c'*64,brain='d'*64))
        write(pc/'report.json',parent_report);write(pc/'COMPLETE.json',dict(status='complete_premesh_stage'))
        cache=dict(status='complete_all_defects_premesh_cache',completed_defects=9,
            inherited_complete_sha256=exporter.sha(pc/'COMPLETE.json'),source_premesh_report_sha256=exporter.sha(pc/'report.json'),
            complete_execution_and_context_scope_passed=True,face_detection_cache=True)
        write(pc/'repair_cache.json',cache);write(pc/'REPAIR_CACHE_COMPLETE.json',dict(status=cache['status'],repair_cache_report_sha256=exporter.sha(pc/'repair_cache.json'),inherited_complete_sha256=exporter.sha(pc/'COMPLETE.json')))
        write(parent/'EXIT.json',dict(exit_code=0));write(parent/'wm_header_audit.private.json',dict(numeric_geometry_same=True))
        output=candidate/'lh.orig';output.write_bytes(b'owned orig fixture');digest=exporter.sha(output)
        quality=dict(status='passed',finite_coordinates=True,source_intersecting_faces=0,geometric_intersecting_faces=0,
            nonmanifold_vertices=0,edge_orientation_conflicts=0,zero_area_faces=0,intersection_counts_bound_to_output_sha256=digest,
            topology=dict(valid=True,euler=2,components=1,boundary_edges=0,nonmanifold_edges=0,degenerate_index_faces=0,duplicate_faces=0),
            geometry_footer_present=True,geometry=dict(valid=True),premesh_geometry_footer_bytes_preserved=True,premesh_footer_sha256='e'*64,orig_footer_sha256='e'*64)
        cleanup=dict(marking_backend='source_numba',require_zero=True,execution_status='complete',output_written=True,native_program_called=False,
            output_sha256=digest,source_sha256=modules['fnit.recon_all.mris_remove_intersection_python'],
            source_intersecting_faces_after_independent_recheck=0,geometric_intersecting_faces_after_independent_recheck=0,cleanup=dict(intersecting_faces_after=0))
        report=dict(status='complete_orig_stage',native_program_called=False,reference_inputs_used=False,remesh_iterations=3,remesh_scalar_storage='numpy',threads=4,cpu_affinity=[8,9,10,11],peak_rss_bytes=123,
            input_sha256=dict(premesh=exporter.sha(pc/'lh.orig.premesh'),premesh_report=exporter.sha(pc/'report.json'),premesh_complete=exporter.sha(pc/'COMPLETE.json')),
            parent_input_sha256=parent_report['input_sha256'],output_sha256={'lh.orig':digest},actual_module_sha256=modules,tool_sha256=exporter.sha(tool),quality=quality,cleanup=cleanup,stages=dict(remesh=2.),total_seconds_before_final_report=3.)
        write(candidate/'report.json',report);write(candidate/'COMPLETE.json',dict(status=report['status'],report_sha256=exporter.sha(candidate/'report.json')))
        process=run/'process.private.json';write(process,dict(exit_code=0,source_manifest_sha256=mf,strong_orig_tool_sha256=exporter.sha(tool),threads=4,cpu_affinity=[8,9,10,11],complete_fresh_cli_seconds=4.))
        request=run/'request.private.json';write(request,dict(source_manifest_sha256=mf,no_reference_inputs_used=True,
            actual_policy=dict(threads=4,remesh_iterations=3,remesh_scalar_storage='numpy',final_intersection_backend='source_numba',final_require_zero=True),
            parent_exit_sha256=exporter.sha(parent/'EXIT.json'),parent_cache_complete_sha256=exporter.sha(pc/'REPAIR_CACHE_COMPLETE.json'),parent_complete_sha256=exporter.sha(pc/'COMPLETE.json'),raw_domain_label_sha256=exporter.sha(pc/'lh.defect_labels'),wm_header_audit_sha256=exporter.sha(parent/'wm_header_audit.private.json')))
        exit_record=run/'EXIT.json';write(exit_record,dict(exit_code=0,orig_complete_present=True))
        args=dict(input_run=run,parent_run=parent,frozen_source=source,source_manifest=manifest,expected_source_manifest_sha256=mf,exit_record=exit_record,expected_exit_sha256=exporter.sha(exit_record),process_record=process,expected_process_sha256=exporter.sha(process),request_record=request,expected_request_sha256=exporter.sha(request),output_directory=root/'public',hemisphere='lh')
        return args,parent_report

    def test_complete_orig_export_keeps_numeric_types(self):
        with tempfile.TemporaryDirectory()as directory:
            args,pr=self.fixture(Path(directory))
            with mock.patch.object(exporter,'verified_premesh',return_value=(None,pr)):
                receipt=exporter.export_completed_orig_cache(**args)
            metrics=json.loads((args['output_directory']/'metrics.json').read_text())
            self.assertTrue(receipt['numeric_values_and_types_preserved'])
            self.assertIs(type(metrics['threads']),int)
            self.assertIs(type(metrics['stages']['remesh']),float)
            self.assertEqual(metrics['baseline_all_defects_comparison'],'not_assessed')

    def test_rehashed_request_policy_and_process_budget_rejected(self):
        for key in ('threads','remesh_iterations','remesh_scalar_storage','final_intersection_backend','final_require_zero','process_threads'):
            with self.subTest(key=key),tempfile.TemporaryDirectory()as directory:
                args,pr=self.fixture(Path(directory));path=args['process_record']if key=='process_threads'else args['request_record'];value=json.loads(path.read_text())
                if key=='process_threads':value['threads']=2
                else:value['actual_policy'][key]={'threads':2,'remesh_iterations':2,'remesh_scalar_storage':'other','final_intersection_backend':'legacy','final_require_zero':False}[key]
                path.write_text(json.dumps(value));args['expected_process_sha256'if key=='process_threads'else'expected_request_sha256']=exporter.sha(path)
                with mock.patch.object(exporter,'verified_premesh',return_value=(None,pr)),self.assertRaisesRegex(ValueError,'policy or execution budget'):
                    exporter.export_completed_orig_cache(**args)
                self.assertFalse(args['output_directory'].exists())

    def test_resealed_report_source_and_quality_defects_rejected(self):
        for key in ('modules','topology','footer','output','cleanup_status','written','require_zero','source_recheck','geometric_recheck','cleanup_source','tool'):
            with self.subTest(key=key),tempfile.TemporaryDirectory()as directory:
                args,pr=self.fixture(Path(directory));candidate=args['input_run']/'candidate';p=candidate/'report.json';v=json.loads(p.read_text())
                if key=='modules':v['actual_module_sha256']={}
                elif key=='topology':v['quality']['topology']['valid']=False
                elif key=='footer':v['quality'].update(premesh_footer_sha256=None,orig_footer_sha256=None)
                elif key=='output':(candidate/'lh.orig').write_bytes(b'changed')
                elif key=='cleanup_status':v['cleanup']['execution_status']='failed'
                elif key=='written':v['cleanup']['output_written']=False
                elif key=='require_zero':v['cleanup']['require_zero']=False
                elif key=='source_recheck':v['cleanup']['source_intersecting_faces_after_independent_recheck']=1
                elif key=='geometric_recheck':v['cleanup']['geometric_intersecting_faces_after_independent_recheck']=1
                elif key=='cleanup_source':v['cleanup']['source_sha256']='f'*64
                elif key=='tool':v['tool_sha256']='f'*64
                p.write_text(json.dumps(v));(candidate/'COMPLETE.json').write_text(json.dumps(dict(status=v['status'],report_sha256=exporter.sha(p))))
                with mock.patch.object(exporter,'verified_premesh',return_value=(None,pr)),self.assertRaises(ValueError):exporter.export_completed_orig_cache(**args)
                self.assertFalse(args['output_directory'].exists())

    def test_resealed_parent_and_false_exit_rejected(self):
        for key in ('status','face_detection_cache','inherited_complete_sha256','source_premesh_report_sha256','parent_false_exit'):
            with self.subTest(key=key),tempfile.TemporaryDirectory()as directory:
                args,pr=self.fixture(Path(directory));pc=args['parent_run']/'candidate'
                if key=='parent_false_exit':(args['parent_run']/'EXIT.json').write_text('{"exit_code":false}')
                else:
                    p=pc/'repair_cache.json';v=json.loads(p.read_text());v[key]=False if key=='face_detection_cache'else'bad';p.write_text(json.dumps(v));c=pc/'REPAIR_CACHE_COMPLETE.json';v=json.loads(c.read_text());v['repair_cache_report_sha256']=exporter.sha(p);c.write_text(json.dumps(v))
                with mock.patch.object(exporter,'verified_premesh',return_value=(None,pr)),self.assertRaisesRegex(ValueError,'actual complete all-nine parent'):exporter.export_completed_orig_cache(**args)
                self.assertFalse(args['output_directory'].exists())


if __name__ == "__main__":
    unittest.main()
