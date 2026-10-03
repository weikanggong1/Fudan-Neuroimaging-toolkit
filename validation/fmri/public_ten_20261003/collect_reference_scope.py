"""只读记录本轮 fresh fMRIPrep 的请求空间与实际内部配置。

不运行 MRI，不读取 license，不把尚未完成的运行当作 benchmark 结果。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tomllib


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(reference_root: Path) -> dict:
    rows = []
    for report_path in sorted((reference_root / "cases").glob("sub-CON*/attempt-*/report.public.json")):
        report = json.loads(report_path.read_text())
        row = {
            "case_id": report_path.parent.parent.name.removeprefix("sub-"),
            "attempt": report_path.parent.name,
            "run_status_at_snapshot": report.get("status"),
            "report_sha256_at_snapshot": sha256(report_path),
            "configuration_status": "not_available",
        }
        configs = sorted((report_path.parent / "work").glob("*/config.toml"))
        if len(configs) > 1:
            raise ValueError("fresh attempt contains multiple runtime configuration files")
        if configs:
            config_path = configs[0]
            config = tomllib.loads(config_path.read_text())
            execution = config.get("execution", {})
            workflow = config.get("workflow", {})
            row.update(
                configuration_status="observed",
                runtime_configuration_sha256=sha256(config_path),
                requested_output_spaces=execution.get("output_spaces"),
                effective_internal_spaces=workflow.get("spaces"),
                run_msmsulc=workflow.get("run_msmsulc"),
                use_syn_sdc=workflow.get("use_syn_sdc"),
                cifti_output=workflow.get("cifti_output"),
            )
        rows.append(row)
    return {
        "status": "configuration_snapshot",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "collector_sha256": sha256(Path(__file__)),
        "scope": "Actual fixed-container runtime configuration only; includes failed attempts. This snapshot does not establish complete MRI outputs or whole-run success.",
        "timing_implication": "Whole reference wall time includes all default internal workflow spaces. Requested MNI152NLin6Asym output does not imply that only that template normalization was computed.",
        "attempts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="本轮独立参考根，含 cases/sub-CONxx/attempt-NN")
    parser.add_argument("--output", type=Path, required=True, help="匿名配置证据 JSON；不包含完整配置或私人路径")
    args = parser.parse_args()
    result = collect(args.reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
