"""Compare only saved real-prefix outputs and state gates; no inference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads((args.run / "queue.private.json").read_text())
    assert queue["status"] == "complete" and all(r["returncode"] == 0 for r in queue["jobs"])
    reports = {arm: json.loads((args.run / arm / "stage.private.json").read_text())
               for arm in ("baseline", "candidate")}
    assert all(report["status"] == "complete" for report in reports.values())
    report = {"schema": "fnit_parc_tf32_real_prefix_pairs/v1", "collector_sha256": sha(__file__),
        "queue": queue, "stage_reports": reports, "default_pairs": [], "inheritance_pairs": [],
        "scope": "Real saved 32-cubed feature-prefix/weight diagnostic; neither original T1 end-to-end nor image geometry benchmark."}
    def compare(first, second):
        return {name: {"value_bits_exact": entry["value_sha256"] == second["outputs"][name]["value_sha256"],
                       "saved_npy_files_exact": entry["saved_file_sha256"] == second["outputs"][name]["saved_file_sha256"],
                       "shape_and_dtype_exact": entry["shape"] == second["outputs"][name]["shape"] and
                           entry["dtype"] == second["outputs"][name]["dtype"]}
                for name, entry in first["outputs"].items()}
    for fast in (False, True):
        old = next(row for row in reports["baseline"]["cases"] if row["fast"] is fast)
        new = next(row for row in reports["candidate"]["cases"] if row["fast"] is fast and row["policy"] is True)
        report["default_pairs"].append({"fast": fast, "outputs": compare(old, new)})
        for policy, inherited in ((False, False), (True, True)):
            declared = next(row for row in reports["candidate"]["cases"] if row["fast"] is fast and row["policy"] is policy)
            none = next(row for row in reports["candidate"]["cases"] if row["fast"] is fast and row["policy"] is None and row["caller_cudnn"] is inherited)
            report["inheritance_pairs"].append({"fast": fast, "declared_policy": policy,
                                                "inherited_cudnn": inherited, "outputs": compare(declared, none)})
    report["passed"] = (all(all(all(row.values()) for row in pair["outputs"].values())
                            for name in ("default_pairs", "inheritance_pairs") for pair in report[name])
        and len(reports["candidate"]["cases"]) == 8 and len(reports["candidate"]["failures"]) == 5
        and reports["candidate"]["cache_rejected_before_preprocess"]
        and all(case["before"] == case["after_construction"] == case["after"]
                for case in reports["candidate"]["cases"])
        and all(row["before"] == row["after"] for row in reports["candidate"]["failures"]))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "device": queue["device"],
                      "cases": len(reports["candidate"]["cases"]),
                      "failures": len(reports["candidate"]["failures"])}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
