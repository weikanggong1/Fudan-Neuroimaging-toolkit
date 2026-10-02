import hashlib,json,pathlib,nibabel as nib,numpy as np
root=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001"); reports={}
for sub in ["01","02"]:
 tag="aux_policy_29f_sub"+sub
 run=json.loads((root/(tag+".json")).read_text())
 assert run["code_commit"]=="29f07d7171e5033da2cb04672385d3ad31addfca"
 forwards=run["precision"]["entowm"]+run["precision"]["mni_aux"]["auxiliary_forwards"]
 assert len(forwards)==5
 for f in forwards:
  assert f["device"]=="cuda:0" and f["input_dtype"]=="torch.float32"
  assert f["matmul_tf32"] == (f["model"] != "affine")
  assert not f["cudnn_tf32"] and not f["autocast"]["enabled"]
 comparisons={}
 for name in ["entowm","mca-dura","vsinus"]:
  a=root/("stage1r2_sub"+sub+"_oldcpu")/"mri"/(name+".mgz")
  b=root/tag/"mri"/(name+".mgz")
  x,y=nib.load(a),nib.load(b)
  n=int(np.count_nonzero(np.asarray(x.dataobj)!=np.asarray(y.dataobj)))
  assert n==0 and np.array_equal(x.affine,y.affine) and x.get_data_dtype()==y.get_data_dtype()
  comparisons[name]={"different_voxels":n,"geometry_equal":True,"dtype_equal":True,
   "cpu_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),"gpu_sha256":hashlib.sha256(b.read_bytes()).hexdigest()}
 reports[sub]={"run":run,"comparisons":comparisons,
  "monitor":json.loads((root/(tag+"_monitor")/"monitor.json").read_text())}
out=root/"aux_policy_29f_summary.json"
assert not out.exists()
out.write_text(json.dumps({"code_commit":"29f07d7171e5033da2cb04672385d3ad31addfca","scope":"same-input-stage; not whole","cases":reports},indent=2)+"\n")
print("two real-input auxiliary stage regressions passed",flush=True)
