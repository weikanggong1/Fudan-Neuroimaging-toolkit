"""只读取已完成报告/文件，绑定真实输入、固定源码、程序和本轮资源身份。"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--itk-prefix", type=Path, required=True)
parser.add_argument("--native-program", type=Path, required=True)
parser.add_argument("--hardware-report", type=Path, required=True)
parser.add_argument("--input-raw", type=Path, required=True)
parser.add_argument("--official-program", type=Path)
parser.add_argument("--code-base", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
include = args.itk_prefix / "include/ITK-5.4"
paths = [args.input_raw, args.native_program, args.workspace / "build/fnit_n4_profile",
         args.workspace / "refinement_diagnostic/build/fnit_dump_refinement",
         args.workspace / "diagnostic/source_manifest.json", args.itk_prefix / "bin/x86_64-conda-linux-gnu-g++",
         args.itk_prefix / "bin/cmake"]
paths.extend(include / name for name in ("itkVersion.h", "itkVersionConfig.h", "itkN4BiasFieldCorrectionImageFilter.h",
    "itkN4BiasFieldCorrectionImageFilter.hxx", "itkBSplineScatteredDataPointSetToImageFilter.hxx",
    "itkBSplineControlPointImageFilter.hxx", "itkBSplineKernelFunction.h", "itkMath.h"))
if args.official_program is not None:
    paths.append(args.official_program)
files = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths if p.is_file()}
native = json.loads((args.run / "native_bundle.profile.json").read_text())
diagnostic = json.loads((args.run / "diagnostic.profile.json").read_text())
internal = json.loads((args.run / "diagnostic.profile.json.internal.json").read_text())
hardware = json.loads(args.hardware_report.read_text())
reference = args.run / "native_bundle.final.raw"
clock_output = args.run / "diagnostic.final.raw"
receipt = {"collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "receipt_collected_host": platform.node(), "benchmark_host": "gpucw1",
    "code_base": args.code_base, "uncommitted_operator_code_identity": "per-report loaded module SHA256",
    "files": files, "weights_and_assets": "N4 has no external model weights or atlases",
    "unavailable_files_at_receipt_collection": [str(p) for p in paths if not p.is_file()],
    "input_sha256_used_in_benchmark": json.loads((args.run / "full_cuda0/report.json").read_text())["input_sha256"],
    "actual_hardware_identity": {"source_path": str(args.hardware_report), "source_sha256": sha(args.hardware_report),
        "cpu": hardware.get("cpu"), "host": hardware.get("host"),
        "affinity_of_other_same_host_run": hardware.get("cpu_affinity"),
        "not_n4_process_affinity_or_completed_recon_all_evidence": True},
    "clock_diagnostic_output": {"sha256": sha(clock_output), "bytes": clock_output.stat().st_size},
    "fresh_current_native_output": {"sha256": sha(reference), "bytes": reference.stat().st_size},
    "clock_only_output_byte_identical_to_current_native": sha(clock_output)==sha(reference),
    "native_profile": native, "clock_profile": diagnostic, "clock_internal": internal,
    "native_profile_sum_seconds": sum(v for k,v in native.items() if k.endswith("seconds")),
    "diagnostic_profile_sum_seconds": sum(v for k,v in diagnostic.items() if k.endswith("seconds")),
    "bspline_fit_fraction_of_diagnostic_fit": internal["bspline_fit"]["seconds"]/diagnostic["fitting_seconds"],
    "profiling_is_inclusive_nested": True, "diagnostic_export_seconds_in_update": internal["diagnostic_export"]["seconds"],
    "official_repetition": "existing same-host frozen task02 repeat verified; no new official process launched this run",
    "implementation_changes": "complete fixed N4 Torch experimental backend, no production/default changes",
    "full_recon_all": "not measured by this N4 stage receipt", "isolation_validation": "not tested",
    "shared_load": "root end-to-end tasks and other GPU users; current single-pair N4 observations are not isolated ABBA throughput"}
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
