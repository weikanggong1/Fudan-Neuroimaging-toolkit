"""Independent original Surfa reader/writer verifies FNIT's saved CTAB.

Run exclusively through the original FreeSurfer fspython reference. This
script is never imported or executed by production FNIT.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import surfa as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--official-roundtrip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected = sf.load_label_lookup(args.lut)
    candidate = sf.load_volume(args.candidate)
    if candidate.labels is None:
        raise RuntimeError("Original Surfa reader found no color table")
    labels_equal = set(expected) == set(candidate.labels)
    mismatches = []
    for label in expected:
        if label not in candidate.labels:
            mismatches.append({"label": label, "reason": "missing"})
            continue
        first, second = expected[label], candidate.labels[label]
        if first.name != second.name or not np.array_equal(first.color, second.color):
            mismatches.append({"label": label, "reason": "name_or_RGBA"})
    candidate.labels = expected
    candidate.save(args.official_roundtrip)
    restored = sf.load_volume(args.official_roundtrip)
    data_equal = bool(np.array_equal(candidate.data, restored.data))
    result = {"schema": "fnit_smri_ctab_original_reader/v1",
              "reference_library": sf.__file__, "reference_version": sf.__version__,
              "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              "lut_sha256": hashlib.sha256(args.lut.read_bytes()).hexdigest(),
              "official_roundtrip_sha256": hashlib.sha256(args.official_roundtrip.read_bytes()).hexdigest(),
              "expected_entries": len(expected), "saved_entries": len(candidate.labels),
              "label_ids_equal": labels_equal, "label_mismatches": mismatches,
              "official_save_load_data_equal": data_equal,
              "scope": "Original library reads FNIT extension and writes a CTAB roundtrip; no network rerun."}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    if not labels_equal or mismatches or not data_equal:
        raise RuntimeError("Color table did not match original lookup/read/write")
    print(json.dumps({k: result[k] for k in ("expected_entries", "label_ids_equal",
                                           "official_save_load_data_equal")}), flush=True)


if __name__ == "__main__":
    main()
