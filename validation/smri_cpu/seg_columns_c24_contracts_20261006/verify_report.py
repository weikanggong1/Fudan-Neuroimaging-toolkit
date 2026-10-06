"""Stdlib-only C24 report verification; no model/image/math/compiler execution."""
import hashlib,json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/"RESULTS_MANIFEST.json").read_text())
    for name,binding in manifest["files"].items():
        data=(root/name).read_bytes()
        assert len(data)==binding["bytes"] and hashlib.sha256(data).hexdigest()==binding["sha256"],name
    report=json.loads((root/"RESULTS.json").read_text())
    queue=json.loads((root/"receipts/run/QUEUE.json").read_text())
    compile_report=json.loads((root/"receipts/run/compile/COMPILE.json").read_text())
    numeric=json.loads((root/"receipts/run/CONTRACTS.json").read_text())
    post=json.loads((root/"POST_BINDINGS.json").read_text())
    assert queue["valid_queue"] and queue["sources_unchanged"] and [arm["returncode"] for arm in queue["arms"]]==[0,0]
    assert compile_report["valid_interface"] and compile_report["compile_calls"]==1
    assert numeric["valid_bounded_contracts"] and len(numeric["numeric_rows"])==6
    assert len(numeric["copy_rows"])==numeric["copy_oracle_calls"]==13
    assert numeric["candidate_copy_calls"]==numeric["candidate_SGEMM_calls"]==12
    assert len(numeric["guard_rows"])==numeric["fallback_calls"]==30
    assert all(row["different_bits"]==0 for row in numeric["numeric_rows"]+numeric["copy_rows"])
    assert queue["source_before"]==queue["source_after"]==numeric["sources_before"]==numeric["sources_after"]==post["current_source_bindings"]
    assert numeric["flags_before"]==numeric["flags_after"] and not numeric["flags_after"]["CUDA_initialized"]
    assert not numeric["postcondition_failures"]
    assert report["contracts"]["numeric_rows"]==numeric["numeric_rows"]
    assert report["compile"]["binary"]==numeric["binary"]==post["new_binary"]
    for name,entry in report["provenance"].items():
        data=(root/name).read_bytes()
        assert len(data)==entry["public"]["bytes"] and hashlib.sha256(data).hexdigest()==entry["public"]["sha256"]
    assert report["scope"]["MRI_calls"]==report["scope"]["GPU_calls"]==report["scope"]["native_calls"]==report["scope"]["production_changes"]==0
    print(json.dumps({"status":"metadata_verification_passed","files":len(manifest["files"]),"new_scientific_calls":0,"numeric_cases":6,"source_bindings":28}))


if __name__=="__main__":
    main()
