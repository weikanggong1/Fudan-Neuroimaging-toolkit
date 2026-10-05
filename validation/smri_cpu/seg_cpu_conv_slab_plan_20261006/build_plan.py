"""Static-only workspace derivation for one proposed CPU slab change."""
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    leaf = Path(__file__).resolve().parent
    repo = leaf.parents[2]
    previous = leaf.parent / "seg_cpu_blur_trial_20261006"
    previous_path = previous / "RESULT.public.json"
    evidence = json.loads(previous_path.read_text())
    assert sha(previous_path) == "7631b641cae7e5dc1d3a68d903ee0f4fe8001fe90dcd6111de39c95eaf0b288b"
    source = repo / "src/fnit/synthseg_parc"
    for name, digest in evidence["source_files"].items():
        assert sha(source / name) == digest, name
    tree = ast.parse((source / "segment.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SegmentUNet")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    widths = next(ast.literal_eval(node.value) for node in constructor.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "widths")
    assert widths == (24, 48, 96, 192, 384)
    batch, channels, depth, height, width = evidence["arms"]["resume"]["preblur_file"]["shape"]
    assert (batch, channels, depth, height, width) == (1, 33, 192, 224, 256)
    cin, cout, element_bytes = widths[0] + widths[1], widths[0], 4
    plane_bytes = batch * max(cin, cout) * height * width * element_bytes
    variants = {}
    for label, cap in (("existing_256MiB", 256 * 1024**2), ("proposed_64MiB_only", 64 * 1024**2)):
        slab_depth = max(1, min(32, cap // plane_bytes - 2))
        slices = [min(slab_depth, depth - start) for start in range(0, depth, slab_depth)]
        m = slab_depth * height * width
        variants[label] = {
            "maximum_slab_bytes": cap, "slab_depth": slab_depth, "calls": len(slices),
            "last_output_depth": slices[-1], "max_padded_chunk_depth": slab_depth + 2,
            "max_GEMM_M_N_K": [m, cout, cin * 27],
            "max_column_bytes": batch * cin * 27 * m * element_bytes,
            "max_contiguous_chunk_bytes": batch * cin * (slab_depth + 2) * height * width * element_bytes,
            "max_temporary_output_bytes": batch * cout * m * element_bytes,
            "total_column_elements_written": batch * cin * 27 * depth * height * width,
            "total_padded_chunk_depth_planes": sum(size + 2 for size in slices),
            "original_source_depth_planes_consumed_with_halo": sum(size + 2 for size in slices) - 2,
            "output_depths": slices
        }
    assert variants["existing_256MiB"]["slab_depth"] == 14 and variants["existing_256MiB"]["calls"] == 14
    assert variants["proposed_64MiB_only"]["slab_depth"] == 2 and variants["proposed_64MiB_only"]["calls"] == 96
    result = {
        "schema": "fnit_seg_single_CPU_slab_static_plan/v1", "status": "static_plan_only_no_computation",
        "production_modified": False, "scientific_worker_implemented": False, "contracts_executed": False,
        "real_layer_executed": False, "full_CNN_executed": False, "new_native_or_GPU_execution": False,
        "builder_sha256": sha(__file__), "evidence_sha256": sha(previous_path),
        "evidence_pending_trial_commit": "08d6f8826e56af0844786643e655fb30a0a34496",
        "evidence_report_commit": "84fbd765e436f1d0727ec5b4f2a0af015292f7f5",
        "source_commit": "46eead65807265395982e6968b0ce48c750c1672",
        "source_files": evidence["source_files"], "input_sha256": evidence["input_sha256"],
        "checkpoint_files": evidence["checkpoint_files"], "weights": evidence["weights"],
        "layer": "SegmentUNet.up[3].conv0", "input_shape": [batch, cin, depth, height, width],
        "weight_shape": [cout, cin, 3, 3, 3], "bias_shape": [cout], "dtype": "float32",
        "layout": "contiguous NCDHW before original CPU slab slicing", "groups": 1,
        "stride": [1, 1, 1], "dilation": [1, 1, 1], "padding": [1, 1, 1],
        "bytes_per_input_plane": plane_bytes,
        "retained_full_input_bytes": batch * cin * depth * height * width * element_bytes,
        "retained_full_output_bytes": batch * cout * depth * height * width * element_bytes,
        "variants": variants,
        "workspace_limit": "Derived column/input/output geometry; excludes BLAS workspace, allocator/page behavior and overlapping lifetimes; not measured RSS.",
        "existing_observation": evidence["arms"]["resume"]["up3_conv0_observed"],
        "single_candidate": "Use existing convolution_slabs with maximum_slab_bytes=64MiB only for this saved layer; same kernel, FP32, NCDHW, backend and halo algorithm.",
        "risk": "M/leading dimension changes from802816 to114688; BLAS packing/reduction may change FP32 bits even with the same1944 neighbors and bias. More calls and repeated halos can erase memory gains in speed.",
        "proposed_contracts_not_executed": {
            "in_out_channels": [72, 24], "kernel": [3, 3, 3], "bias": True,
            "shapes": [[1,72,d,13,17] for d in (1,2,7,31,33)],
            "caps": "16 versus4 logical input planes to reproduce depth14 versus2 on bounded shapes",
            "values": "Fixed FP32 seed, finite cancellation-rich mixed signs, zeros and signed zeros; no dtype change.",
            "gates": ["All pre-ELU output uint32 bits equal", "Shape/stride/dtype/finite equal",
                      "Input/weight/bias unchanged, output independent", "True-boundary padding and internal halos",
                      "Contiguous and strided CPU inputs", "Flags/no-grad/training/autocast/GPU fallback unchanged"]
        },
        "future_real_gate_needs_root_authorization": "Same saved skip/value and weight SHA, rejoin once, old/new single convolution only, common eight-core lock; compare all pre-ELU uint32 values first. Stop on any difference, do not use final labels as a substitute.",
        "upstream": "https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27"
    }
    (leaf / "PLAN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "variants": variants, "plan_sha256": sha(leaf / "PLAN.json")}))


if __name__ == "__main__":
    main()
