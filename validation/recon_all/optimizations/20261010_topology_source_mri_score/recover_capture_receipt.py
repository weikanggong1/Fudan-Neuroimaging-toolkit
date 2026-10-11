"""恢复已完成观察的事后收据；保留原驱动失败状态，不重跑或改旧输出。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(*, observer_run, observer_workspace, prior_reference_edges, output_dir):
    """具名路径→新恢复收据；仅修复旧收据把str传给Path.read_bytes的bug。"""
    observer_run, observer_workspace, output_dir = map(Path, (observer_run, observer_workspace, output_dir))
    if output_dir.exists():
        raise FileExistsError("fresh receipt directory required")
    log = (observer_run / "driver.log").read_text()
    references, _ = json.JSONDecoder().raw_decode(log)
    if len(references) != 2 or any(item["exit_code"] != 0 for item in references):
        raise ValueError("both actual reference processes must already have exited zero")
    for reference in references:
        for name, expected in reference["files_sha256"].items():
            if sha(observer_run / reference["label"] / name) != expected:
                raise ValueError("actual frozen capture changed")
    program = observer_workspace / "mris_fix_topology_active_transform_observer"
    build = observer_workspace / "build.private.json"
    if any(sha(program) != item["program_sha256"] for item in references):
        raise ValueError("actual observer program changed")
    files = references[0]["files_sha256"]
    repeated = references[1]["files_sha256"]
    result = dict(status="complete_postprocessing_recovery_only", full_ga_completed=False,
                  production_changed=False, original_driver_exit_code=int((observer_run / "driver.exitcode").read_text()),
                  original_postprocessing_error="sha(__file__) used str.read_bytes; observed programs/captures already complete",
                  observer_program_sha256=sha(program), build_receipt_sha256=sha(build),
                  original_driver_sha256=sha(observer_workspace / "driver.py"), recovery_script_sha256=sha(__file__),
                  original_driver_log_sha256=sha(observer_run / "driver.log"), references=references,
                  unchanged_before_scores=files["edges.before.bin"] == sha(prior_reference_edges),
                  repeat_capture_bytes_equal={name: digest == repeated[name] for name, digest in files.items()})
    if not result["unchanged_before_scores"] or not all(result["repeat_capture_bytes_equal"].values()):
        raise ValueError("observation changes source trajectory or reference is unstable")
    output_dir.mkdir(parents=True)
    (output_dir / "capture_recovery.private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["observer-run", "observer-workspace", "prior-reference-edges", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    recover(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
