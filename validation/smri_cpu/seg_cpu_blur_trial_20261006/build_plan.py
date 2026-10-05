"""Build a source/geometry bound private blur trial plan, using stdlib only."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(source, cls, name=None):
    tree=ast.parse(source)
    item=next(value for value in tree.body if getattr(value,"name",None)==cls)
    if name is not None:
        item=next(value for value in item.body if getattr(value,"name",None)==name)
    return item


def dump(value):
    if isinstance(value,list):
        return [ast.dump(item,include_attributes=False) for item in value]
    return ast.dump(value,include_attributes=False)


def main():
    leaf=Path(__file__).resolve().parent
    repo=leaf.parents[2]
    src=repo/"src/fnit/synthseg_parc"
    saved_plan=leaf.parent/"seg_cpu_profile_20261006/PLAN.json"
    plan=json.loads(saved_plan.read_text())
    profile_path=leaf.parent/"seg_cpu_profile_20261006/PROFILE.public.json"
    profile=json.loads(profile_path.read_text())
    join_path=leaf.parent/"seg_memory_20261005/CPU_JOIN_STAGE.public.json"
    join=json.loads(join_path.read_text())
    old_ref="4ec078cb"
    old_sources={name:subprocess.check_output(["git","show",old_ref+":src/fnit/synthseg_parc/"+name],cwd=repo).decode()
                 for name in join["source"]["producer_files"]}
    for name,expected in join["source"]["producer_files"].items():
        assert hashlib.sha256(old_sources[name].encode()).hexdigest()==expected
    source_now={name:sha(src/name) for name in plan["source_files"]}
    assert source_now==plan["source_files"]==profile["source_files"]
    assert source_now["cpu_conv.py"]==join["source"]["producer_files"]["cpu_conv.py"]
    assert dump(node(old_sources["model.py"],"_Block"))==dump(node((src/"model.py").read_text(),"_Block"))
    proofs={"cpu_conv_complete_file_same":True,"last_block_class_AST_same":True}
    for method in ("__init__","load_h5"):
        assert dump(node(old_sources["segment.py"],"SegmentUNet",method))==dump(node((src/"segment.py").read_text(),"SegmentUNet",method))
        proofs["SegmentUNet_"+method+"_AST_same"]=True
    original_tail=node(old_sources["segment.py"],"SegmentUNet","forward").body[-3:]
    current_tail=node((src/"segment.py").read_text(),"SegmentUNet","forward").body[-3:]
    assert dump(original_tail)==dump(current_tail)
    proofs["likelihood_delete_softmax_terminal_AST_same"]=True
    prior_kernel=node(old_sources["segment.py"],"_blur").body[:4]
    current_kernel=node((src/"segment.py").read_text(),"_blur").body[:4]
    trial_kernel=[value for value in node((leaf/"channel_batch.py").read_text(),"blur").body
                  if isinstance(value,ast.Assign)][:4]
    assert dump(prior_kernel)==dump(current_kernel)==dump(trial_kernel)
    proofs["gaussian_kernel_expression_AST_same"]=True
    assert join["source"]["weight_sha256"]==plan["weights"]["synthseg_2.0.h5"]["sha256"]
    assert join["input"]["sha256"]==plan["input_sha256"]
    assert join["source"]["helper_sha256"]==source_now["cpu_join.py"]
    original_shape=(1,33,192,224,256)
    _,channels,depth,height,width=original_shape
    plane_bytes=channels*height*width*4
    slab_depth=max(1,min(32,256*1024**2//plane_bytes-2))
    spatial=slab_depth*height*width
    calls=[event for event in profile["events"] if event["name"]=="functional_conv" and event["metadata"]["groups"]==33]
    assert len(calls)==12
    assert all(event["metadata"]["shape"]==[1,33,34,224,256] and event["output"]["shape"]==[1,33,32,224,256] for event in calls)
    memory={}
    for name,block in (("original_group33",33),("all33_as_batch_inference_only",33),("declared_batch11",11)):
        memory[name]={"logical_slab_depth":slab_depth,"block_channels":block,
          "column_elements_per_call":block*27*spatial,
          "column_bytes_per_call":block*27*spatial*4,
          "contiguous_chunk_bytes_per_call":block*(slab_depth+2)*height*width*4,
          "convolution_output_bytes_per_call":block*spatial*4,
          "same_M_N_K_per_channel":[spatial,1,27]}
    output={"schema":"fnit_seg_cpu_group33_channel_batch_trial_plan/v1",
      "status":"declared_before_real_trial","production_changes":False,
      "scope":"One existing real final-decoder resume plus a single old/new ABBA blur replay, not a whole-T1 benchmark",
      "source_commit":plan["source_repository_commit"],"source_files":source_now,
      "producer_commit":old_ref,"producer_files":join["source"]["producer_files"],
      "terminal_resume_proofs":proofs,"prior_join_proof_report_sha256":sha(join_path),
      "prior_join_bit_identity_sha256":join["identity"]["values_sha256"],
      "checkpoint_files":join["input"]["checkpoint_files"],
      "weights":plan["weights"],"input_sha256":plan["input_sha256"],
      "prepared_values_sha256":join["input"]["prepared_values_sha256"],
      "profile_sha256":sha(profile_path),"profile_actual_blur_conv_calls":len(calls),
      "profile_observed_group33_conv_seconds":sum(event["inclusive_seconds"] for event in calls),
      "profile_observed_gaussian_seconds":sum(event["inclusive_seconds"] for event in profile["events"] if event["name"]=="gaussian33"),
      "cpu_affinity":plan["cpu_affinity"],"threads":8,"channel_block":11,
      "memory_geometry":memory,"full_posterior_bytes":channels*depth*height*width*4,
      "memory_limitation":"Column/input/output estimates do not include BLAS workspaces, allocator caches or complete process RSS",
      "no_new_full_CNN":True,"no_official":True,"no_GPU":True,
      "real_checkpoint_status":"Not yet resumed; existing skip/value only. One current terminal resume will create preblur posterior.",
      "optional_profiler":{"only":"up[3].conv0 in that unique terminal resume","activities":["CPU"],
        "record_shapes":False,"with_stack":False,"profile_memory":False,"module_hooks":False,
        "rusage_user_system_seconds":True,"limitations":"Native Unfold3dCopyCPU/cpublas may be inside slow_conv3d without separate ATen events; absence does not prove no unfold/GEMM."},
      "gates":{"full_FP32_posterior_bit_exact":True,"shape_stride_finite_dtype_exact":True,
        "input_unchanged_no_alias":True,"flags_unchanged":True,"GPU_original_fallback":True,
        "maximum_RSS_bytes":32000000000,"numerical_difference_action":"Stop before whole-T1 testing", "whole_map_CSV_status":"not_assessed"},
      "lock_fnit_relative_path":plan["lock_fnit_relative_path"],
      "wait_for_existing_GEMS_full":True,
      "upstream_sources":[
        "https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27",
        "https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Unfold3d.cpp#L224",
        "https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp#L551",
        "https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/ParallelOpenMP.h#L16"],
      "upstream_scope":"2.5.1 source inference; actual 2.5.1 build config, operation trace and real bit/timing gates remain to be recorded",
      "prototype_files":{name:sha(leaf/name) for name in ("channel_batch.py","check_contracts.py","build_plan.py","real_trial.py","run_trial.py")}}
    (leaf/"PLAN.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"status":output["status"],"proofs":proofs,"memory":memory}))


if __name__=="__main__":
    main()
