"""导出完整两线程目标搜索并行回归，保留源绑定与真实运行范围。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from publish_pial_norm_receipts import sanitize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    summary = dict(schema_version=1, scope="two complete frozen same-input LH stages; not raw T1 recon-all",
        production_target_backend="serial unchanged", overall_metric_equivalence="not_assessed",
        precision="TF32 enabled; no half precision", performance="one shared-load serial/parallel pair per stage; includes first JIT, output writes and synchronization",
        references="previous same-input references reused by input/program/output SHA; no new native repeat or native performance comparison",
        process_tree_GPU_memory="not measured; Torch allocator peaks are not total GPU occupancy", stages={})
    mapping = {"source_manifest": "workspaces/border_parallel_complete_20261010_v1/SOURCE_MANIFEST.json"}
    mapping.update({name: f"runs/border_parallel_complete_20261010_v1/{name}/report.json" for name in ("preaparc", "pial")})
    for name, relative in mapping.items():
        path = args.private_root/relative
        raw = json.loads(path.read_text())
        row = dict(private_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   private_relative_artifact=relative, report=sanitize(raw))
        (args.output_directory/f"{name}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n")
        if name != "source_manifest":
            serial,parallel = raw["runs"]["serial"],raw["runs"]["parallel"]
            summary["stages"][name] = dict(status=raw["status"],regression_exact=raw["regression_exact"],
                serial_seconds=serial["wall_seconds"],parallel_seconds=parallel["wall_seconds"],
                observed_wall_reduction_percent=(1-parallel["wall_seconds"]/serial["wall_seconds"])*100,
                reference_program_sha256=raw["reference_program_sha256"],border_source_sha256=raw["loaded_source_sha256"]["fnit.recon_all.place_surface_border"],
                complete_steps=parallel["result"]["steps"],pass_ends=parallel["result"]["pass_ends"],
                target_threads=parallel["result"]["target_search_threads"],
                private_receipt_sha256=row["private_report_sha256"])
    (args.output_directory/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
