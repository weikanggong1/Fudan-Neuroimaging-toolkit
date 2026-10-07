"""Build a prepared single-layer plan from saved scalar provenance; no Torch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from trial_bindings import identity, require


def select(source, name, method=None):
    item = next(node for node in ast.parse(source).body if getattr(node, "name", None) == name)
    if method:
        item = next(node for node in item.body if getattr(node, "name", None) == method)
    return item


def ast_digest(item):
    nodes = item if isinstance(item, list) else [item]
    text = json.dumps([ast.dump(node, include_attributes=False) for node in nodes], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    leaf = Path(__file__).resolve().parent
    repository = leaf.parents[2]
    source = repository / "src/fnit/synthseg_parc"
    v2 = leaf.parent / "seg_columns_reuse_v2_20261006"
    v1 = leaf.parent / "seg_columns_reuse_20261006"
    v2plan = json.loads((v2 / "PLAN.json").read_text())
    readonly = json.loads((leaf / "READONLY_TARGET.json").read_text())
    prior = json.loads((leaf.parent / "seg_cpu_conv_slab_trial_20261006/PLAN.json").read_text())
    require(readonly["producer_files"] == prior["producer_files"]
            and readonly["checkpoint_declared_files"] == prior["checkpoint_files"]
            and readonly["input_sha256"] == prior["input_sha256"]
            and readonly["prepared_values_sha256"] == prior["prepared_values_sha256"], "existing real checkpoint provenance differs")
    require(not readonly["new_workspace_exists"] and not readonly["new_run_exists"], "new target absent at preparation")
    for name, digest in v2plan["source14"].items():
        require(identity(source / name)["sha256"] == digest, "current local production changed")
    old = {}
    for name, digest in prior["producer_files"].items():
        data = subprocess.check_output(["/usr/bin/git", "show", "4ec078cb:src/fnit/synthseg_parc/" + name], cwd=repository)
        require(hashlib.sha256(data).hexdigest() == digest, "actual producer Git source differs")
        old[name] = data.decode()
    now = {name: (source / name).read_text() for name in ("model.py", "segment.py")}
    pairs = {
        "last_block_class": (select(old["model.py"], "_Block"), select(now["model.py"], "_Block")),
        "SegmentUNet_init": (select(old["segment.py"], "SegmentUNet", "__init__"), select(now["segment.py"], "SegmentUNet", "__init__")),
        "SegmentUNet_load_h5": (select(old["segment.py"], "SegmentUNet", "load_h5"), select(now["segment.py"], "SegmentUNet", "load_h5")),
        "terminal_likelihood_delete_softmax": (select(old["segment.py"], "SegmentUNet", "forward").body[-3:], select(now["segment.py"], "SegmentUNet", "forward").body[-3:]),
        "Gaussian_kernel": (select(old["segment.py"], "_blur").body[:4], select(now["segment.py"], "_blur").body[:4]),
    }
    proofs = {}
    for name, (before, after) in pairs.items():
        proofs[name] = {"producer_AST": ast_digest(before), "current_AST": ast_digest(after)}
        require(proofs[name]["producer_AST"] == proofs[name]["current_AST"], "producer/current terminal AST changed")
    require(prior["producer_files"]["cpu_conv.py"] == v2plan["source14"]["cpu_conv.py"], "whole CPU convolution file changed")
    proofs["whole_cpu_conv_file"] = {"producer_sha256": prior["producer_files"]["cpu_conv.py"], "current_sha256": v2plan["source14"]["cpu_conv.py"]}
    dependencies = {}
    for name in v2plan["prototype_sources"]:
        item = identity(v2 / name)
        require(item["sha256"] == v2plan["prototype_sources"][name], "frozen v2 source changed")
        dependencies["v2_" + name] = {"fnit_relative_path": v2plan["workspace"] + "/" + name, **item}
    for name, filename in (("v2_PLAN", "PLAN.json"), ("v2_contract_report", "CONTRACTS.json"),
                           ("v2_metadata_report", "INTERFACE_LOAD.json"), ("v2_queue_receipt", "QUEUE.json")):
        prefix = v2plan["workspace"] if filename == "PLAN.json" else v2plan["runs"]
        dependencies[name] = {"fnit_relative_path": prefix + "/" + filename, **identity(v2 / filename)}
    for name, record in v2plan["previous_frozen_files"].items():
        key = {"columns_reuse.so": "v1_binary", "COMPILE.json": "v1_compile_receipt", "PLAN.json": "v1_PLAN", "columns_reuse.cpp": "v1_CPP"}[name]
        dependencies[key] = record
    dependencies["weight"] = v2plan["weight"]
    dependencies["checkpoint_manifest"] = {"fnit_relative_path": readonly["checkpoint_fnit_relative"] + "/manifest.private.json", **readonly["checkpoint_manifest"]}
    for name, record in prior["checkpoint_files"].items():
        dependencies["checkpoint_" + name] = {"fnit_relative_path": readonly["checkpoint_fnit_relative"] + "/" + name, **record}
    shape = [1, 72, 192, 224, 256]
    plane_bytes = 72 * 224 * 256 * 4
    slab = max(1, min(32, 256 * 1024**2 // plane_bytes - 2))
    require(slab == 14, "original mature slab geometry differs")
    plan = {
        "schema": "fnit_columns_saved_real_layer_prepared/v1", "status": "prepared_only_not_uploaded_not_dispatched",
        "scope": "Four fresh-process calls of the same saved-real-input up3.conv0, one ABBA; no tail/whole CNN/native/GPU",
        "canonical_main_observed": readonly["canonical_main"], "Git_HEAD_not_a_source_gate": True,
        "production_source_commit": v2plan["production_source_commit"], "source14": v2plan["source14"],
        "workspace": "workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-real-layer-v1",
        "runs": "runs/smri_cpu_20261004/remaining_20261006/seg-columns-real-layer-v1",
        "checkpoint": readonly["checkpoint_fnit_relative"], "checkpoint_files": prior["checkpoint_files"],
        "producer_commit": "4ec078cb", "producer_files": prior["producer_files"], "terminal_six_proofs": proofs,
        "T1_input_sha256": prior["input_sha256"], "prepared_values_sha256": prior["prepared_values_sha256"],
        "joined_input_value_sha256": prior["prior_join_bit_identity_sha256"],
        "prior_join_report": identity(leaf.parent / "seg_memory_20261005/CPU_JOIN_STAGE.public.json"),
        "input_dataset": "OpenNeuro ds003138 v1.0.1, CC0, case02", "input_shape": shape, "output_shape": [1, 24, 192, 224, 256],
        "layer": "SegmentUNet.up[3].conv0", "dependencies": dependencies,
        "headers": v2plan["headers"], "TorchCPU_binary": {"bytes": 314174561, "sha256": v2plan["libtorch_cpu_sha256"]},
        "provider_basename": v2plan["provider_basename"], "provider_sha256": v2plan["provider_sha256"],
        "same_ATen_wrapper_claimed": False, "new_compile_authorized": False,
        "geometry": {"slab_depth": 14, "last_slab_depth": 10, "calls": 14, "K": 1944, "N": 24,
                     "M_full": 802816, "M_last": 573440, "transpose": "NN", "alpha": 1, "beta": 1,
                     "lda_ldc": "actual M", "ldb": 1944, "K_order": "C,kD,kH,kW", "bias": "same Torch copy_ before SGEMM",
                     "columns_maximum_bytes": 6242697216, "columns_lifetime": "one layer-call only; no cross-call cache"},
        "ABBA": ["A1_baseline", "B1_candidate", "B2_candidate", "A2_baseline"],
        "A1": "unique reference generation; comparison_executed=False never counted as old/new exactness gate",
        "comparisons": "B1/B2/A2 each full preELU FP32 bits against SHA-bound A1; three actual gates",
        "baseline": "canonical mature layer(image), original default256MiB depth14; no altered backend/cap/layout",
        "candidate": "exact unchanged v2 helper and v1.so with original14-plane compact-M columns reuse",
        "timing": "one layer-call wall including candidate guard/provider checks/counters; paired clocks observations, no whole/official speed ratio",
        "actual_counts_expected": {"A1_baseline": [0, 0], "B1_candidate": [14, 14], "B2_candidate": [14, 14], "A2_baseline": [0, 0]},
        "flags_policy": "FP32/inference_mode/oneDNNFalse numerical scope restores caller booleans; no TF32/autocast setter; no Module hooks/profiler",
        "affinity": v2plan["affinity"], "threads": 8, "interop_threads": 8,
        "common_CPU_lock": v2plan["scheduling"]["common_CPU_lock"],
        "limits": {"outer_seconds": 23000, "each_worker_seconds": 180, "maximum_RSS_bytes": 32000000000,
                   "address_space_cap_bytes": 32000000000, "maximum_workers": 4,
                   "joined_input_bytes": 3170893824, "single_output_value_bytes": 1056964608,
                   "maximum_new_private_array_bytes_with_header": 1056968704,
                   "new_private_arrays": "only A1 preELU; other arms only scalar reports, no joined tensor or other outputs saved"},
        "gates": ["all source/input/weight/provider/runtime bytes before-after", "joined input value SHA identical",
                  "complete preELU bits0, finite, dtype/shape/stride/noalias", "normal/error flags unchanged", "RSS<=32GB",
                  "first arm failure stops all remaining arms; no retry/looser gate"],
        "prohibited": ["new CNN capture", "other tail layers/ELU/BN/head/softmax", "whole MRI reconstruction",
                       "official/native/GPU", "new compile/provider/cap/dtype/parameter search", "production edits"],
        "scientific_sources": {name: identity(leaf / name)["sha256"] for name in ("trial_bindings.py", "layer_trial.py", "run_abba.py", "build_plan.py")},
        "execution_at_freeze": {"uploaded": False, "INDEX_registered": False, "workers": 0, "numerical_calls": 0},
    }
    (leaf / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"PLAN": identity(leaf / "PLAN.json"), "source_count": len(plan["scientific_sources"]),
                      "dependency_count": len(dependencies), "terminal_six_proofs": True, "worker_calls": 0}))


if __name__ == "__main__":
    main()
