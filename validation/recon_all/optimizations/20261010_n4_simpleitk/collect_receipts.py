"""选择两例完整N4数组/文件收据并调用现有数值保真导出器。"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import shutil
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-directory","output-directory","replacements-file","exporter"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("existing_receipt_exporter",args.exporter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="n4-json-",dir=args.output_directory.parent) as scratch:
        staged=Path(scratch)
        baseline=args.fnit_directory/"runs/recon_n4_python_adapter_20261010_v1"
        for case in ("sub07","sub06"):
            shutil.copyfile(baseline/case/"report.json",staged/(case+"_array.json"))
            for index in (0,1):
                shutil.copyfile(baseline/case/("native-"+str(index)+"-profile.json"),
                                staged/(case+"_native_"+str(index)+"_profile.json"))
            source=args.fnit_directory/"runs/recon_n4_python_file_api_20261010_v2"/case/"report.json"
            shutil.copyfile(source,staged/(case+"_file.json"))
        shutil.copyfile(baseline/"dependency_receipt.json",staged/"dependency_receipt.json")
        print(module.export_receipts(input_directory=staged,output_directory=args.output_directory,
            replacements_file=args.replacements_file))


if __name__=="__main__":main()
