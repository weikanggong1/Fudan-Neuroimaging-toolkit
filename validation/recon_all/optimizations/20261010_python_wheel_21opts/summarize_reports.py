"""校验21项配方安装收据的公开SHA/原始绑定并生成摘要和本目录公开清单。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    rows = []
    for version in ("v1", "v2"):
        reports = root/"reports"/version
        manifest = json.loads((reports/"PUBLIC_EXPORT_MANIFEST.json").read_text())
        files = {entry["path"]:entry for entry in manifest["files"]}
        for entry in files.values():
            path = reports/entry["path"]
            assert digest(path) == entry["published_sha256"]
            assert path.stat().st_size == entry["published_bytes"] and entry["numeric_leaves_preserved"]
        binding = json.loads((reports/"SOURCE_BINDING.json").read_text())
        check = json.loads((reports/"CHECK.json").read_text())
        audit = json.loads((reports/"CLI_AUDIT.json").read_text())
        assert files["SOURCE_BINDING.json"]["original_sha256"] == check["source_binding_sha256"] == audit["source_binding_sha256"]
        assert check["status"] == audit["status"] == "complete"
        assert check["wheel"]["sha256"] == audit["wheel_sha256"]
        assert check["profile_option_count"] == len(check["snapshot_profile"]) == 21 and len(check["imports"]) == 8
        for module in check["imports"].values():
            relative = module["file"].split("$TASK_OUTPUT/installed/",1)[1]
            assert binding["source_files"]["src/"+relative]["sha256"] == module["sha256"]
        assert audit["cuda_initialized_before"] == audit["cuda_initialized_after"] == False
        script = root/("run_wheel_checks_v1.py" if version == "v1" else "run_wheel_checks.py")
        assert digest(script) == check["check_script_sha256"]
        assert digest(root/"audit_installed_cli.py") == audit["audit_script_sha256"]
        rows.append({"version":version,"wheel":check["wheel"],"git_head":binding["git_head_at_snapshot_start"],
                     "source_file_count":binding["source_file_count"],"snapshot_diff_from_git_head":binding["snapshot_diff_from_git_head"],
                     "profile_options":check["snapshot_profile"],"verified_reports":len(files),
                     "numeric_bool_null_leaves_preserved":sum(f["numeric_leaf_count"] for f in files.values()),
                     "installation_checks":"pass","scope":"wheel and isolated target imports only",
                     "overall_equivalence":"not_assessed"})
    assert rows[1]["snapshot_diff_from_git_head"] == {}
    return {"status":"complete","reports":rows,"MRI_GPU_peak_and_OS_isolation":"not_tested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only",action="store_true",help="不写文件，仅校验已发布摘要/清单")
    args = parser.parse_args();root=args.report_root
    summary = summarize(root)
    if args.verify_only:
        assert json.loads((root/"summary.json").read_text()) == summary
    else:
        (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    pattern = re.compile(rb"/(?:cpfs01|cwStorage|home/wgong)/|gwk_[0-9]{4,}|(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
    entries=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "PUBLICATION_MANIFEST.json" or "__pycache__" in path.parts:
            continue
        assert path.suffix in {".py",".json",".md"},path
        assert pattern.search(path.read_bytes()) is None,path
        entries.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"published_sha256":digest(path)})
    publication = {"scope":"actual21opts wheel installation only; notMRI/cleanConda/OSisolation",
                   "raw_report_hashes":"reports/v1 and v2 PUBLIC_EXPORT_MANIFEST original_sha256",
                   "public_payload_hashes":"published_sha256 below hashes public bytes",
                   "wheel_binary_or_MRI_published":False,"files":entries}
    if args.verify_only:
        assert json.loads((root/"PUBLICATION_MANIFEST.json").read_text()) == publication
    else:
        (root/"PUBLICATION_MANIFEST.json").write_text(json.dumps(publication,indent=2)+"\n")
    print(json.dumps({"verified_payloads":len(entries),"verified_install_reports":10,
                      "v2wheel_sha256":rows_sha(summary)}))


def rows_sha(summary):
    return summary["reports"][1]["wheel"]["sha256"]


if __name__ == "__main__":
    main()
