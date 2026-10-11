"""只读定位自产MGZ文件差异的header、data、footer及tag31层次。"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

import nibabel as nib
import numpy as np

sha = lambda value: hashlib.sha256(value).hexdigest()


def main(args):
    report = {"scope": "readonly full MGH sections and own-transform tag31 filename; raw strict SHA remains unequal",
              "diagnostic_script_sha256": sha(Path(__file__).read_bytes()), "cases": {}}
    for case in ("sub06", "sub07"):
        comparison = json.loads((args.comparisons / (case + ".json")).read_bytes())
        subjects = [(args.runs_root / name / "subject").resolve() for name in comparison["producer_directories"]]
        rows = {}
        for name, detail in comparison["volumes"].items():
            if detail["files_bytes_equal"]:
                continue
            files = [subject / "mri" / name for subject in subjects]
            compressed = [path.read_bytes() for path in files]
            assert [sha(x) for x in compressed] == detail["files_sha256"]
            payloads = [gzip.decompress(x) for x in compressed]
            images = [nib.load(str(path)) for path in files]
            parts, tags, ends, normalized_payloads = [], [], [], []
            for subject, payload, image in zip(subjects, payloads, images):
                start = int(image.header.get_data_offset())
                end = int(start + math.prod(image.shape) * image.get_data_dtype().itemsize)
                ends.append(end)
                tag_bytes, offset, taglist, normalized_tags = payload[end+20:], 0, [], bytearray()
                while offset < len(tag_bytes):
                    code = struct.unpack(">i", tag_bytes[offset:offset+4])[0]
                    if code == 1:
                        begin = offset
                        _, version, maximum, name_length = struct.unpack(">iiii", tag_bytes[offset:offset+16])
                        assert version == -2 and name_length > 0
                        offset += 16 + name_length
                        count = struct.unpack(">i", tag_bytes[offset:offset+4])[0]
                        offset += 4
                        for _ in range(count):
                            _, length = struct.unpack(">ii", tag_bytes[offset:offset+8])
                            offset += 8 + length + 16
                        assert offset <= len(tag_bytes)
                        table = tag_bytes[begin:offset]
                        normalized_tags.extend(table)
                        taglist.append({"tag_code": 1, "format": "TAG_OLD_COLORTABLE version2 (not generic int64-length tag)",
                            "payload_sha256": sha(table), "normalized_payload_sha256": sha(table),
                            "payload_bytes": len(table), "entries": count, "maximum_entries": maximum})
                        continue
                    code, length = struct.unpack(">iq", tag_bytes[offset:offset+12])
                    value = tag_bytes[offset+12:offset+12+length]
                    assert length > 0 and len(value) == length and code in (3, 31) and value.endswith(b"\0")
                    normalized = value.replace(str(subject).encode(), b"<SUBJECT_DIR>")
                    detail = {"tag_code": code, "payload_bytes": length,
                        "payload_sha256": sha(value), "normalized_payload_sha256": sha(normalized),
                        "subject_path_occurrences": value.count(str(subject).encode())}
                    if code == 31:
                        path = Path(value[:-1].decode())
                        relative = path.relative_to(subject)
                        assert relative.as_posix() == "mri/transforms/talairach.xfm" and path.is_file()
                        detail.update(normalized_payload=normalized.rstrip(b"\0").decode(),
                            own_transform_file_sha256=sha(path.read_bytes()),
                            meaning="TAG_MGH_XFORM points to own talairach.xfm; no voxel/world matrix in tag31")
                    else:
                        assert value.startswith(b"fnit.recon_all.fill_cutting_plane_python ") and detail["subject_path_occurrences"] == 4
                        detail.update(meaning="TAG_CMDLINE retains own WM, aseg, LTA and filled output paths; shared LUT unchanged; normalized text withheld to avoid private resource pathname")
                    taglist.append(detail)
                    normalized_tags.extend(struct.pack(">iq", code, len(normalized)) + normalized)
                    offset += 12 + length
                parts.append({"header": payload[:start], "data": payload[start:end],
                              "footer_20bytes": payload[end:end+20], "tags": payload[end+20:]})
                tags.append(taglist)
                normalized_payloads.append(payload[:end+20] + bytes(normalized_tags))
            size = min(map(len, payloads))
            unequal = np.flatnonzero(np.frombuffer(payloads[0][:size], np.uint8) != np.frombuffer(payloads[1][:size], np.uint8))
            row = {"decompressed_bytes": [len(p) for p in payloads],
                "decompressed_sha256": [sha(p) for p in payloads],
                "first_decompressed_different_byte_offset": int(unequal[0]) if len(unequal) else size,
                "data_end_byte_offsets": ends,
                "regions": {key: {"bytes": [len(p[key]) for p in parts],
                    "sha256": [sha(p[key]) for p in parts], "bytes_equal": parts[0][key] == parts[1][key]}
                    for key in parts[0]},
                "gzip_metadata": [{"flags": x[3], "mtime": int.from_bytes(x[4:8], "little"),
                                  "FNAME_present": bool(x[3] & 8)} for x in compressed],
                "tags": tags,
                "subject_path_only_normalized_tag_payloads_equal": [x["normalized_payload_sha256"] for x in tags[0]] == [x["normalized_payload_sha256"] for x in tags[1]],
                "entire_MGH_subject_path_only_normalized_sha256": [sha(x) for x in normalized_payloads],
                "entire_MGH_subject_path_only_normalized_bytes_equal": normalized_payloads[0] == normalized_payloads[1],
                "strict_raw_MGZ_match": False, "strict_decompressed_MGH_match": payloads[0] == payloads[1]}
            rows[name] = row
        report["cases"][case] = rows
    report["metadata_writer_source_sha256"] = {name: sha((args.source_root / "src/fnit/recon_all" / name).read_bytes()) for name in ("conform_gpu.py", "mgh_compat.py", "fill_cutting_plane_python.py")}
    report["source_rule"] = "conform tag31 own xfm path preserved downstream; filled old CTAB uses version2 layout, followed by TAG_CMDLINE own paths; only subject path substitutions and resulting tag lengths normalized diagnostically, never production"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report_sha256": sha(args.output.read_bytes()), "cases": {
        case: {name: {"region_equal": {key: value["bytes_equal"] for key, value in row["regions"].items()},
            "tag_normalized_equal": row["subject_path_only_normalized_tag_payloads_equal"],
            "entire_MGH_normalized_equal": row["entire_MGH_subject_path_only_normalized_bytes_equal"],
            "first_offset": row["first_decompressed_different_byte_offset"],
            "data_end": row["data_end_byte_offsets"], "gzip_metadata": row["gzip_metadata"]}
            for name, row in rows.items()} for case, rows in report["cases"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runs-root", "comparisons", "source-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
