"""Single-subject FNIT spherical registration commands."""

import argparse
from dataclasses import fields
import json
from pathlib import Path


def load_inputs(path, input_type):
    """Read a private L/R file manifest; preserve optional unset paths."""
    manifest = Path(path).expanduser().resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"L", "R"}:
        raise ValueError("input manifest must contain exactly L and R objects")
    names = {field.name for field in fields(input_type)}
    result = {}
    for hemisphere, values in data.items():
        if not isinstance(values, dict) or set(values) - names:
            raise ValueError(f"invalid {hemisphere} input fields")
        paths = {}
        for name, value in values.items():
            if value is None:
                paths[name] = None
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{hemisphere}.{name} must be a file path or null")
            item = Path(value).expanduser()
            paths[name] = (manifest.parent / item).resolve() if not item.is_absolute() else item.resolve()
        result[hemisphere] = input_type(**paths)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-msm", description=__doc__)
    parser.add_argument("method", choices=("msmsulc", "msmall"))
    parser.add_argument("--inputs-json", required=True, help="Private L/R input-file manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", help="Official configuration file; defaults to the canonical schedule")
    parser.add_argument("--execution", choices=("optimized", "reference"), default="optimized")
    parser.add_argument("--qc-policy", choices=("report", "repair", "error"), default="report",
                        help="MSMSulc native sphere QC; ignored for msmall")
    parser.add_argument("--no-parallel", action="store_true", help="Run L/R hemispheres sequentially")
    parser.add_argument("--cpu-threads", type=int, help="Total host thread budget, split between L/R")
    args = parser.parse_args(argv)
    if args.method == "msmsulc":
        from . import MSMSulcInputs, run_msmsulc
        input_type, register = MSMSulcInputs, run_msmsulc
    else:
        from . import MSMAllInputs, run_msmall
        input_type, register = MSMAllInputs, run_msmall
    options = dict(device=args.device, config=args.config, execution=args.execution,
                   parallel=not args.no_parallel, cpu_threads=args.cpu_threads)
    if args.method == "msmsulc":
        options["qc_policy"] = args.qc_policy
    elif args.qc_policy != "report":
        parser.error("--qc-policy applies only to msmsulc")
    result = register(load_inputs(args.inputs_json, input_type), args.output_dir, **options)
    for hemisphere in ("L", "R"):
        print(f"{hemisphere}: {result[hemisphere]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
