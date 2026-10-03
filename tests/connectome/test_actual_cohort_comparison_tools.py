"""Protocol and CPU-reader checks; tiny fixtures are never MRI benchmarks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools/reference"))
import benchmark_connectome_cohort_compare as compare
import compare_freesurfer_recon_outputs as anatomy

HAS_SCIENCE = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("nibabel") is not None


def manifest(root):
    return {"dataset": "fixture-only", "snapshot": "fixture", "license": "fixture", "cases": [
        {"case_id": f"sub-{n:02d}", "subject": f"{n:02d}", "t1w": str(root / f"raw/{n:02d}/T1w.nii.gz"),
         "input_files": [{"kind": "raw_t1w", "path": str(root / f"raw/{n:02d}/T1w.nii.gz"), "sha256": "0" * 64}]}
        for n in range(10)]}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = manifest(self.root)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_exactly_ten_distinct_subjects(self):
        self.assertEqual(len(compare.manifest_cases(self.manifest)), 10)
        self.manifest["cases"].pop()
        with self.assertRaises(ValueError): compare.manifest_cases(self.manifest)

    def test_duplicate_subject_refused(self):
        self.manifest["cases"][1]["subject"] = self.manifest["cases"][0]["subject"]
        with self.assertRaises(ValueError): compare.manifest_cases(self.manifest)

    def test_path_traversal_case_refused(self):
        self.manifest["cases"][0]["case_id"] = "../../elsewhere"
        with self.assertRaises(ValueError): compare.manifest_cases(self.manifest)

    def test_missing_raw_sha_refused(self):
        self.manifest["cases"][0]["input_files"][0]["sha256"] = "missing"
        with self.assertRaises(ValueError): compare.manifest_cases(self.manifest)

    def test_missing_driver_is_waiting(self):
        with self.assertRaises(compare.WaitingForActualResults):
            compare.ready_case(self.root / "not-started.json", "candidate", "sub-00")

    def validation_options(self):
        return type("Options", (), {"baseline_anatomy_root": self.root / "raw-anatomy",
            "baseline_root": self.root / "old-baseline", "baseline_driver": self.root / "old-driver/status.json"})()

    def test_same_round_revalidation_precedes_anatomy_binding(self):
        options = self.validation_options()
        revalidation = self.write(options.baseline_anatomy_root / "baseline/sub-00/recon_report.revalidated.json", {"fixture_only": True})
        self.write(options.baseline_root / "baseline/sub-00/anatomy_origin.json", {"fixture_only": True})
        self.assertEqual(compare.baseline_validation_report(options, "sub-00"), (revalidation, None))

    def test_explicit_replacement_anatomy_origin_for_original_not_dispatched(self):
        options = self.validation_options()
        new_root = self.root / "actual-declared-new-baseline"
        options._gpu_origins = {("baseline", "sub-00"): {"declaration": {"replacement": {
            "root": str(new_root), "driver_status": str(self.root / "new-driver/status.json")}}}}
        path = self.write(new_root / "baseline/sub-00/recovery_binding.json", {"fixture_only": True})
        self.assertEqual(compare.baseline_validation_report(options, "sub-00"), (path, options._gpu_origins["baseline", "sub-00"]))

    def test_undeclared_new_anatomy_namespace_is_not_discovered(self):
        options = self.validation_options()
        self.write(self.root / "undeclared-new-baseline/baseline/sub-00/anatomy_origin.json", {"fixture_only": True})
        with self.assertRaises(compare.WaitingForActualResults):
            compare.baseline_validation_report(options, "sub-00")

    def test_existing_linked_validation_is_rejected_not_hidden_by_replacement(self):
        options = self.validation_options()
        outside = self.write(self.root / "outside.json", {"fixture_only": True})
        link = options.baseline_anatomy_root / "baseline/sub-00/recon_report.revalidated.json"
        link.parent.mkdir(parents=True); link.symlink_to(outside)
        self.assertEqual(compare.baseline_validation_report(options, "sub-00"), (link, None))
        with self.assertRaises(ValueError): compare.safe_json(link)

    def recovery_anatomy_fixture(self):
        case = self.manifest["cases"][0]; subject = self.root / "raw-anatomy/baseline/sub-00/FS/00"
        original_report = {"path": str(subject.parent / "original.json"), "sha256": "1" * 64}
        failure = {"type": "RuntimeError", "message": "known int32 serialization failure fixture"}
        report = {"error": failure, "anatomy": {"scripts/recon-all.done": {"sha256": "2" * 64}}}
        original = {}
        for key in ("configuration", "driver_snapshot"):
            path = self.write(self.root / (key + ".json"), {"fixture_only": True})
            original[key] = {"path": str(path), "sha256": anatomy.sha(path)}
        new_root = self.root / "new-actual-raw-DWI"
        context = {"declaration": {"original": original, "replacement": {"root": str(new_root)}}}
        path = new_root / "baseline/sub-00/recovery_binding.json"
        proof = {"mode": "selected_same_round_anatomy_fresh_raw_dwi_monitor_recovery", "arm": "baseline", "case_id": "sub-00",
            "case": case, "FS_recomputed": False, "old_DWI_outputs_used": False, "raw_DWI_execution_policy": "complete_from_raw",
            "original_config": original["configuration"], "original_driver": original["driver_snapshot"],
            "anatomy_subject_dir": str(subject), "anatomy": report["anatomy"], "same_round_anatomy": {
                "original_recon_report": original_report, "original_reconstruction_failure": failure,
                "anatomy_geometry": {"status": "actual_images_surfaces_annotations_read"}}}
        return proof, context, path, original_report, case, subject, report

    def test_actual_selected_anatomy_proof_accepts_only_explicit_original_bindings(self):
        fields = self.recovery_anatomy_fixture(); touched = {}
        result = anatomy.validate_recovery_anatomy(*fields, touched)
        self.assertEqual(result["status"], "actual_images_surfaces_annotations_read"); self.assertEqual(len(touched), 2)

    def test_selected_anatomy_changed_case_original_source_or_FS_rejected(self):
        for field, value in (("case", {"wrong_case": True}), ("anatomy_subject_dir", "/wrong/FS"),
                             ("old_DWI_outputs_used", True), ("FS_recomputed", True), ("original_config", {"path": "/fake", "sha256": "0" * 64})):
            with self.subTest(field=field):
                fields = list(self.recovery_anatomy_fixture()); fields[0][field] = value
                with self.assertRaises(ValueError): anatomy.validate_recovery_anatomy(*fields, {})

    def test_selected_anatomy_changed_original_recon_SHA_rejected(self):
        fields = list(self.recovery_anatomy_fixture()); fields[0]["same_round_anatomy"]["original_recon_report"]["sha256"] = "0" * 64
        # Avoid shared fixture dictionary hiding the deliberately changed identity.
        fields[3] = {"path": fields[3]["path"], "sha256": "1" * 64}
        with self.assertRaises(ValueError): anatomy.validate_recovery_anatomy(*fields, {})

    def test_running_driver_does_not_mean_ready(self):
        path = self.write(self.root / "status.json", {"cases": {"candidate/sub-00": {"status": "gpu_queued"}}})
        with self.assertRaises(compare.WaitingForActualResults): compare.ready_case(path, "candidate", "sub-00")

    def test_actual_failure_is_terminal(self):
        path = self.write(self.root / "status.json", {"cases": {"candidate/sub-00": {"status": "failed_gpu_execution"}}})
        with self.assertRaises(ValueError): compare.ready_case(path, "candidate", "sub-00")

    def test_terminal_driver_with_pending_case_is_failure_not_infinite_wait(self):
        path=self.write(self.root/"status.json",{"status":"failed_or_incomplete_staged_raw_cohort","cases":{"candidate/sub-00":{"status":"bound_GPU_not_dispatched"}}})
        with self.assertRaises(ValueError):compare.ready_case(path,"candidate","sub-00")

    def test_terminal_driver_missing_case_is_failure(self):
        path=self.write(self.root/"status.json",{"status":"failed_staged_binding","cases":{}})
        with self.assertRaises(ValueError):compare.ready_case(path,"candidate","sub-00")

    def test_completed_with_timing_error_refused(self):
        path = self.write(self.root / "status.json", {"cases": {"candidate/sub-00": {"status": "completed", "timing_error": {"type": "clock"}}}})
        with self.assertRaises(ValueError): compare.ready_case(path, "candidate", "sub-00")

    def test_case_completion_independent_of_batch_failure(self):
        path = self.write(self.root / "status.json", {"status": "failed_or_incomplete_anatomy_preparation", "cases": {"candidate/sub-00": {"status": "completed"}}})
        self.assertEqual(compare.ready_case(path, "candidate", "sub-00")[0]["status"], "completed")

    def test_missing_scientific_outputs_refused(self):
        with self.assertRaises(ValueError): compare.required_outputs(self.root, ["fs-aparc"])

    def test_unsafe_atlas_refused(self):
        with self.assertRaises(ValueError): compare.required_outputs(self.root, ["../../outside"])

    def test_original_json_symbolic_link_refused(self):
        path = self.write(self.root / "real.json", {"status": "completed"})
        link = self.root / "link.json"; link.symlink_to(path)
        with self.assertRaises(ValueError): compare.safe_json(link)

    def test_input_ledger_original_hash_is_not_actual_verification(self):
        case = self.manifest["cases"][0]
        entry = dict(case["input_files"][0])
        with self.assertRaises(ValueError): compare.validate_input_ledger([entry], case, "fixture")
        entry["actual_sha256"] = entry["sha256"]
        compare.validate_input_ledger([entry], case, "fixture")
        entry["actual_sha256"] = "1" * 64
        with self.assertRaises(ValueError): compare.validate_input_ledger([entry], case, "fixture")

    def source(self):
        root = self.root / "source"
        for relative in ("src/fnit/cli.py", "src/fnit/__init__.py", "pyproject.toml", "environment.yml"):
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
        hashes = {str(path.relative_to(root)): anatomy.sha(path) for path in root.rglob("*") if path.is_file()}
        return root, {"directory": str(root), "source_sha256": hashes,
                      "source_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}

    def test_actual_source_ledger_passes(self):
        _, identity = self.source()
        self.assertEqual(compare.verify_source(identity)["file_count"], 4)

    def test_actual_source_tampering_refused(self):
        root, identity = self.source(); (root / "src/fnit/cli.py").write_text("changed")
        with self.assertRaises(ValueError): compare.verify_source(identity)

    def test_extra_source_file_refused(self):
        root, identity = self.source(); (root / "src/fnit/extra.py").write_text("new")
        with self.assertRaises(ValueError): compare.verify_source(identity)

    def test_incomplete_source_contract_refused(self):
        root, identity = self.source(); (root / "environment.yml").unlink()
        hashes = {str(path.relative_to(root)): anatomy.sha(path) for path in root.rglob("*") if path.is_file()}
        identity.update(source_sha256=hashes, source_fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
        with self.assertRaises(ValueError): compare.verify_source(identity)

    def test_source_symlink_refused_even_if_claimed_hash_matches(self):
        root, identity = self.source(); original = root / "src/fnit/cli.py"; original.unlink()
        outside = self.root / "outside.py"; outside.write_text("src/fnit/cli.py"); original.symlink_to(outside)
        with self.assertRaises(ValueError): compare.verify_source(identity)

    def test_source_fingerprint_cannot_be_metadata_only(self):
        _, identity = self.source(); identity["source_fingerprint"] = "0" * 64
        with self.assertRaises(ValueError): compare.verify_source(identity)

    def origins(self):
        declarations=[]
        cases=self.manifest["cases"]
        for index, subset in enumerate((cases[:2], cases[2:6], cases[6:])):
            root=self.root / f"prep{index}"; root.mkdir()
            self.write(root / "input_manifest.json", self.manifest)
            config={"run_root":str(root),"scope":"staged_anatomy_preparation_only","candidate_source":"unknown","gpu_started":False,"cpu_threads":8,
                    "selected_cases":[case["case_id"] for case in subset],"atlases":["fs-aparc"],
                    "official_origin":{"identity":{"version":"official-fixture","executable_sha256":"0"*64}},
                    "future_gpu_parameters":{"n_seeds":100000,"seed":0}}
            path=self.write(root / "anatomy_prep_config.json",config)
            declarations.append({"prep_config":str(path),"prep_driver_report_dir":str(self.root/f"driver{index}"),"case_ids":config["selected_cases"]})
        path=self.write(self.root / "bindings.json",{"bindings":declarations})
        return path,declarations

    def test_A2_B4_C4_exact_mapping(self):
        path,_=self.origins();origins,mapping,_=compare.load_origins(path,self.manifest["cases"])
        self.assertEqual([len(origin["case_ids"]) for origin in origins],[2,4,4])
        self.assertEqual(len(mapping),10)

    def test_duplicate_origin_mapping_refused(self):
        path,declarations=self.origins();declarations[1]["case_ids"][0]=declarations[0]["case_ids"][0]
        self.write(path,{"bindings":declarations})
        with self.assertRaises(ValueError): compare.load_origins(path,self.manifest["cases"])

    def test_incomplete_origin_mapping_refused(self):
        path,declarations=self.origins();declarations[2]["case_ids"].pop()
        self.write(path,{"bindings":declarations})
        with self.assertRaises(ValueError): compare.load_origins(path,self.manifest["cases"])

    def test_faked_preparation_source_refused(self):
        path,declarations=self.origins();cp=Path(declarations[0]["prep_config"]);config=json.loads(cp.read_text());config["sources"]={"candidate":"fake"};self.write(cp,config)
        with self.assertRaises(ValueError): compare.load_origins(path,self.manifest["cases"])

    def test_preparation_settings_difference_refused(self):
        path,declarations=self.origins();cp=Path(declarations[1]["prep_config"]);config=json.loads(cp.read_text());config["future_gpu_parameters"]["n_seeds"]=10;self.write(cp,config)
        with self.assertRaises(ValueError): compare.load_origins(path,self.manifest["cases"])

    def test_json_differences_keep_absence_explicit(self):
        diff=compare.json_differences({"timer":1,"seed":0},{"timer":2})
        self.assertEqual(len(diff),2)
        self.assertTrue(next(item for item in diff if item["path"]=="/seed")["missing_on_one_side"])

    def test_node_semantics_read_exact_order(self):
        p=self.root/'nodes.tsv';p.write_text('index\toriginal_label\themisphere\tname\n1\t1001\tL\tone\n2\t2001\tR\ttwo\n')
        self.assertEqual(compare.node_semantics(p)[1][1]["name"],"two")
        p.write_text('index\toriginal_label\themisphere\tname\n2\t1001\tL\tone\n')
        with self.assertRaises(ValueError): compare.node_semantics(p)

    def test_missing_actual_results_never_count_as_completed(self):
        cases=self.manifest["cases"]
        options=type('Options',(),{'report_dir':self.root/'reports'})();options.report_dir.mkdir()
        state={"cases":{case["case_id"]:{"case_id":case["case_id"],"status":"waiting_actual_outputs"} for case in cases}}
        origins=[{"driver_dir":str(self.root/'missing-driver')}];mapping={case["case_id"]:0 for case in cases}
        with patch.object(anatomy,"compare_fresh_case") as compute:
            compare.compare_once(options,state,cases,origins,mapping)
        compute.assert_not_called()
        self.assertEqual(state["completed_cases"],0)
        self.assertEqual(state["status"],"waiting_actual_outputs")
        self.assertEqual(len(state["cases"]),10)

    def test_report_namespace_cannot_mutate_original_source(self):
        with self.assertRaises(ValueError):compare.check_report_namespace(self.root/'source'/'reports',[self.root/'source'])

    def test_report_namespace_cannot_contain_original_driver(self):
        with self.assertRaises(ValueError):compare.check_report_namespace(self.root,[self.root/'driver'])

    def test_report_namespace_alias_into_original_source_refused(self):
        original=self.root/'source';original.mkdir();alias=self.root/'alias';alias.symlink_to(original,target_is_directory=True)
        with self.assertRaises(ValueError):compare.check_report_namespace(alias/'reports',[original])

    def test_tools_do_not_import_production_or_torch(self):
        import ast
        for path in (Path(compare.__file__),Path(anatomy.__file__)):
            tree=ast.parse(path.read_text())
            imports=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports.extend(item.name for item in node.names)
                if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
            self.assertFalse(any(name=='torch' or name.startswith('fnit') for name in imports))

    def prior_binding_fixture(self):
        original=self.root/'prior';original.mkdir()
        fields=('manifest','prep_bindings','baseline_anatomy_root','baseline_root','candidate_root','baseline_driver','candidate_driver')
        options=type('Options',(),{key:self.root/key for key in fields})()
        state={'manifest':{'sha256':'1'*64},'preparation_bindings':{'sha256':'2'*64},'case_origin_index':{'sub-00':0},
               'cases':{'sub-00':{'status':'waiting_actual_outputs'}}}
        result_path=self.write(original/'sub-00.official_anatomy.json',{'case_id':'sub-00','status':'completed','all_requested_scientific_data_equal':True})
        prior={'status':'failed_actual_comparison','end_utc':'actual-previous-end','manifest':state['manifest'],
               'preparation_bindings':state['preparation_bindings'],'case_origin_index':state['case_origin_index'],
               'configuration':{key:str(getattr(options,key)) for key in fields},'tool_sha256':'3'*64,'anatomy_tool_sha256':'4'*64,
               'cases':{'sub-00':{'status':'failed_comparison','error':{'type':'ValueError','message':'nonfinite scientific array'},
                                  'anatomy':{'path':str(result_path),'sha256':anatomy.sha(result_path),'all_requested_scientific_data_equal':True}}}}
        self.write(original/'status.json',prior)
        return original,options,state,prior

    def test_prior_binding_reuses_only_verified_anatomy_and_preserves_failure(self):
        original,options,state,prior=self.prior_binding_fixture();before=(original/'status.json').read_bytes();watched={}
        with patch.object(compare,'verify_cached_anatomy') as verify:
            compare.bind_prior_anatomy(original,options,state,watched)
        verify.assert_called_once();self.assertEqual((original/'status.json').read_bytes(),before)
        self.assertEqual(state['cases']['sub-00']['status'],'waiting_actual_outputs')
        self.assertEqual(state['prior_comparison_binding']['reused_actual_anatomy_cases'],['sub-00'])
        self.assertEqual(len(watched),2)

    def test_prior_running_controller_refused(self):
        original,options,state,prior=self.prior_binding_fixture();prior['status']='waiting_actual_outputs';self.write(original/'status.json',prior)
        with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})

    def test_prior_different_failure_refused(self):
        original,options,state,prior=self.prior_binding_fixture();prior['cases']['sub-00']['error']['message']='missing raw source';self.write(original/'status.json',prior)
        with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})

    def test_prior_raw_manifest_change_refused(self):
        original,options,state,prior=self.prior_binding_fixture();prior['manifest']['sha256']='9'*64;self.write(original/'status.json',prior)
        state['manifest']={'sha256':'1'*64}
        with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})

    def test_prior_completed_anatomy_file_change_refused(self):
        original,options,state,prior=self.prior_binding_fixture();(original/'sub-00.official_anatomy.json').write_text('{}')
        with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})

    def test_prior_anatomy_with_scientific_difference_refused(self):
        original,options,state,prior=self.prior_binding_fixture();path=original/'sub-00.official_anatomy.json';self.write(path,{'case_id':'sub-00','status':'completed','all_requested_scientific_data_equal':False})
        prior['cases']['sub-00']['anatomy']['sha256']=anatomy.sha(path);self.write(original/'status.json',prior)
        with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})

    def test_prior_cached_input_change_is_not_bypassed(self):
        original,options,state,_=self.prior_binding_fixture()
        with patch.object(compare,'verify_cached_anatomy',side_effect=ValueError('actual original input changed')):
            with self.assertRaises(ValueError):compare.bind_prior_anatomy(original,options,state,{})


@unittest.skipUnless(HAS_SCIENCE,"numpy/nibabel are provided by the FNIT conda environment")
class ActualCPUReaderFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.nib,self.np=anatomy.scientific_modules()

    def test_signed_zero_has_distinct_raw_bits(self):
        n=self.np;result=anatomy.compare_arrays(n.array([0.],dtype='float32'),n.array([-0.],dtype='float32'))
        self.assertEqual(result['numeric_neq'],0);self.assertEqual(result['raw_scalar_bits_neq'],1)
        self.assertFalse(result['exact_scientific_array_equal'])

    def test_default_scientific_array_still_refuses_nan(self):
        n=self.np;a=n.array([n.nan],dtype='float32')
        with self.assertRaises(ValueError):anatomy.compare_arrays(a,a.copy())

    def test_matching_undefined_image_values_preserve_exact_payload(self):
        n=self.np;a=n.array([1.,n.nan,n.inf,-n.inf],dtype='float32')
        result=anatomy.compare_arrays(a,a.copy(),allow_matching_nonfinite=True)
        self.assertTrue(result['exact_scientific_array_equal']);self.assertEqual(result['nonfinite']['finite_intersection_elements'],1)
        self.assertEqual(result['nonfinite']['patterns']['NaN']['baseline_count'],1)
        self.assertEqual(result['nonfinite']['payload_raw_bits_neq'],0);self.assertEqual(result['rmse'],0)

    def test_new_or_lost_nan_in_image_refused(self):
        n=self.np;a=n.array([1.,n.nan],dtype='float32');b=n.array([1.,2.],dtype='float32')
        with self.assertRaises(ValueError):anatomy.compare_arrays(a,b,allow_matching_nonfinite=True)
        with self.assertRaises(ValueError):anatomy.compare_arrays(b,a,allow_matching_nonfinite=True)

    def test_inf_sign_change_in_image_refused(self):
        n=self.np
        with self.assertRaises(ValueError):anatomy.compare_arrays(n.array([n.inf]),n.array([-n.inf]),allow_matching_nonfinite=True)

    def test_nan_payload_change_in_image_refused(self):
        n=self.np;a=n.array([0x7fc00000],dtype='uint32').view('float32');b=n.array([0x7fc00001],dtype='uint32').view('float32')
        with self.assertRaises(ValueError):anatomy.compare_arrays(a,b,allow_matching_nonfinite=True)

    def test_finite_error_denominator_is_actual_finite_intersection(self):
        n=self.np;a=n.array([1.,n.nan],dtype='float32');b=n.array([3.,n.nan],dtype='float32')
        result=anatomy.compare_arrays(a,b,allow_matching_nonfinite=True)
        self.assertEqual(result['rmse'],2);self.assertEqual(result['numeric_neq'],1)

    def test_image_keeps_all_elements_and_nan_payload(self):
        n=self.np;a=n.zeros((2,2,2,2),dtype='float32');a[1,1,1,1]=n.nan
        paths=[self.root/'a.nii.gz',self.root/'b.nii.gz']
        for path in paths:self.nib.save(self.nib.Nifti1Image(a.copy(),n.eye(4)),path)
        result=compare.image_compare(*paths);self.assertTrue(result['strict_scientific_equal'])
        self.assertEqual(result['data']['elements'],16);self.assertEqual(result['data']['finite_intersection_elements'],15)
        self.assertEqual(result['data']['nonfinite_frames'][0]['frame'],1)

    def test_matrix_nonfinite_gate_is_not_relaxed(self):
        n=self.np;a=self.matrix('a.csv',[[1,n.nan],[n.nan,2]]);b=self.matrix('b.csv',[[1,n.nan],[n.nan,2]])
        for kind in compare.MATRICES:
            with self.assertRaises(ValueError):compare.matrix_compare(a,b,2,kind)

    def test_single_float32_ulp_is_reported_not_declared_atomic(self):
        n=self.np;a=n.array([1.],dtype='float32');b=n.nextafter(a,n.array([2.],dtype='float32'))
        result=compare.scalar_ulp_diagnostic(a,b,'float32');self.assertEqual(result['max_ulp'],1)
        self.assertIn('does not prove',result['interpretation'])

    def test_double_ulp_handles_negative_values(self):
        n=self.np;a=n.array([-1.,0.,1.]);b=n.nextafter(a,n.array([0.,1.,2.]))
        self.assertEqual(compare.scalar_ulp_diagnostic(a,b,'float64')['max_ulp'],1)

    def matrix(self,name,data):
        p=self.root/name;self.np.savetxt(p,data,delimiter=',',fmt='%.9g');return p

    def test_count_self_connections_retained(self):
        a=self.matrix('a.csv',[[3,2],[2,4]]);b=self.matrix('b.csv',[[3,2],[2,4]])
        result=compare.matrix_compare(a,b,2,'count');self.assertTrue(result['strict_count_equal']);self.assertEqual(result['baseline_upper_sum'],9)

    def test_fractional_count_refused(self):
        a=self.matrix('a.csv',[[1,0.1],[0.1,2]]);b=self.matrix('b.csv',[[1,0.1],[0.1,2]])
        with self.assertRaises(ValueError):compare.matrix_compare(a,b,2,'count')

    def test_matrix_node_dimension_mismatch_refused(self):
        a=self.matrix('a.csv',[[1,2],[2,3]]);b=self.matrix('b.csv',[[1,2],[2,3]])
        with self.assertRaises(ValueError):compare.matrix_compare(a,b,3,'sift2_fbc')

    def test_FBC_error_and_ulp_are_separate(self):
        n=self.np;value=n.nextafter(n.float32(1),n.float32(2));a=self.matrix('a.csv',[[0,1],[1,0]]);b=self.matrix('b.csv',[[0,value],[value,0]])
        result=compare.matrix_compare(a,b,2,'sift2_fbc')
        self.assertGreater(result['numeric_neq'],0);self.assertEqual(result['support_neq'],0);self.assertEqual(result['ULP_diagnostics'][1]['max_ulp'],1)

    def image(self,name,data,affine=None):
        p=self.root/name;self.nib.save(self.nib.Nifti1Image(data,self.np.eye(4) if affine is None else affine),p);return p

    def test_last_4D_frame_is_fully_compared(self):
        n=self.np;data=n.zeros((2,2,2,3),dtype='float32');other=data.copy();other[1,1,1,2]=1
        a=self.image('a.nii.gz',data);b=self.image('b.nii.gz',other)
        result=compare.image_compare(a,b);self.assertEqual(result['data']['numeric_neq'],1);self.assertEqual(result['data']['elements'],24);self.assertFalse(result['strict_scientific_equal'])

    def test_image_affine_difference_not_hidden_by_equal_data(self):
        n=self.np;affine=n.eye(4);affine[0,3]=1;a=self.image('a.nii.gz',n.zeros((2,2,2),dtype='float32'));b=self.image('b.nii.gz',n.zeros((2,2,2),dtype='float32'),affine)
        result=compare.image_compare(a,b);self.assertEqual(result['data']['numeric_neq'],0);self.assertFalse(result['strict_scientific_equal'])

    def subjects(self):
        n=self.np
        coords=n.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype='float32');faces=n.array([[0,1,2],[0,2,3]],dtype='int32')
        volume_info={'head':n.array([20]),'valid':'1','filename':'','volume':n.array([2,2,2]),'voxelsize':n.ones(3),'xras':n.array([1.,0,0]),'yras':n.array([0,1.,0]),'zras':n.array([0,0,1.]),'cras':n.zeros(3)}
        roots=[]
        for index in (0,1):
            root=self.root/f'subject{index}';roots.append(root)
            for directory in ('mri','surf','label'):(root/directory).mkdir(parents=True,exist_ok=True)
            for name in ('brain','aparc+aseg','ribbon'):self.nib.save(self.nib.MGHImage(n.ones((2,2,2),dtype='float32'),n.eye(4)),root/f'mri/{name}.mgz')
            for hemi in ('lh','rh'):
                for name in ('white','pial.T1','sphere.reg'):
                    info=dict(volume_info,filename=f'private-case-{index}')
                    self.nib.freesurfer.write_geometry(root/f'surf/{hemi}.{name}',coords,faces,create_stamp=f'creation-{index}',volume_info=info)
                (root/f'surf/{hemi}.pial').symlink_to(f'{hemi}.pial.T1')
                for name in ('aparc','aparc.a2009s'):
                    self.nib.freesurfer.write_annot(root/f'label/{hemi}.{name}.annot',n.array([0,1,0,1]),n.array([[255,0,0,0],[0,255,0,0]]),[b'one',b'two'])
        return roots

    def test_internal_FS_pial_links_and_metadata_bytes_are_handled(self):
        a,b=self.subjects();result=anatomy.compare_subjects(a,b)
        self.assertEqual(len(result['files']),13);self.assertTrue(result['all_requested_scientific_data_equal'])
        self.assertIn('surf/lh.pial',result['file_byte_differences'])
        self.assertEqual(result['files']['surf/lh.pial']['baseline_file']['internal_link']['link_target'],'lh.pial.T1')
        self.assertEqual(result['files']['surf/lh.pial']['stored_payload']['coordinate_uint32_bits_neq'],0)
        self.assertEqual(result['files']['surf/lh.pial']['stored_payload']['face_uint32_bits_neq'],0)

    def test_cross_namespace_shared_inode_refused(self):
        a,b=self.subjects();target=b/'mri/brain.mgz';target.unlink();os.link(a/'mri/brain.mgz',target)
        with self.assertRaises(ValueError):anatomy.compare_subjects(a,b)

    def test_surface_link_escape_refused(self):
        a,b=self.subjects();link=b/'surf/lh.pial';link.unlink();link.symlink_to(a/'surf/lh.pial.T1')
        with self.assertRaises(ValueError):anatomy.compare_subjects(a,b)


if __name__=='__main__':unittest.main()
