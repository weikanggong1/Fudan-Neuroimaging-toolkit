import hashlib,json,pathlib
r=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001")
assert (r/"integrated_ff372d7_stage_regressions.finished").exists()
inputs={}
for sub in ["01","02"]:
 for prefix in ["synth_main_ff372d7_sub"+sub+"/run.json","synth_main_ff372d7_sub"+sub+"_cache_off/run.json",
                "aux_policy_ff372d7_sub"+sub+".json","mni_affine_ff372d7_sub"+sub+"_retry1.json"]:
  p=r/prefix;j=json.loads(p.read_text());assert j["code_commit"]=="ff372d73f106e850b999fbc95ae0b324cd315cbf"
  inputs[prefix]=hashlib.sha256(p.read_bytes()).hexdigest()
 j=json.loads((r/("synth_main_ff372d7_sub"+sub+"_cache_off/run.json")).read_text())
 assert j["comparisons"]["SynthSeg"]["different_voxels"]==0
 assert j["comparisons"]["SynthStrip"]["geometry_equal"] and j["comparisons"]["SynthStrip"]["dtype_equal"]
 for name in ["synth_main_ff372d7_sub"+sub+"_cache_off_monitor/monitor.json","mni_affine_ff372d7_sub"+sub+"_retry1_monitor/monitor.json"]:
  p=r/name;j=json.loads(p.read_text());assert j["exit_code"]==0
  assert j["peak_sampled_process_bytes"]<20000000000
  inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
resource=r/"runtime_fingerprints_ff372d7.json";j=json.loads(resource.read_text())
assert j["code_commit"]=="ff372d73f106e850b999fbc95ae0b324cd315cbf" and not j["mismatches"]
for n in ["aux_policy_ff372d7_summary.json","mni_affine_inverse_residual_diagnosis.json","runtime_fingerprints_ff372d7.json","installation_ff372d7/report.json"]:
 p=r/n;inputs[n]=hashlib.sha256(p.read_bytes()).hexdigest()
row={"code_commit":"ff372d73f106e850b999fbc95ae0b324cd315cbf","source_tree":"319ff4ccd7632c637721b55522640714ce0d37f2","status":"ready_for_raw_T1_whole_benchmark",
 "decision_scope":"root-reviewed execution readiness; not formal numerical/metric equivalence acceptance",
 "same_input_results":"SynthSeg and auxiliary labels exact vs prior; shapes/dtypes/actual GPU flags checked",
 "known_changes":["main SynthStrip boundary image 51/31 voxels; assess downstream in raw whole runs",
 "GPU MNI affine FP32 reduces field difference; strict transform diagnostic still fails",
 "inverse full-field vector maxima 0.57475/55.83623 mm outside baseline brainmask; brain maxima 0.000563/0.002872 mm, no brain voxel over diagnostic 0.1 mm",
 "cache-enabled Synth diagnostic exceeds20GB; low-memory cache-disabled monitored stages below20GB; retain policy"],
 "pending":["two final raw-T1 whole runs","complete final metrics and mesh quality","clean environment runtime isolation"],
 "overall_metric_equivalence":"not_assessed; no confirmed thresholds","input_report_sha256":inputs}
p=r/"integrated_ff372d7_stage_acceptance.json";assert not p.exists();p.write_text(json.dumps(row,indent=2)+"\n")
print("stage review saved; final raw-T1 runs may start",flush=True)
