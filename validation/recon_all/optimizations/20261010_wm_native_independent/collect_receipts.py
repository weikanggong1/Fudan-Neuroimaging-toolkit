"""收集十份实际WM数值JSON并复用现有数值保真导出器，不复制MRI。"""
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
    with tempfile.TemporaryDirectory(prefix="wm-json-",dir=args.output_directory.parent) as scratch:
        staged=Path(scratch)
        for case in ("sub07","sub06"):
            for version in ("v1","v2"):
                source=args.fnit_directory/"runs"/("recon_wm_native_independent_20261010_"+version)/(case[:3]+"-"+case[3:])/"report.json"
                shutil.copyfile(source,staged/("wm_segment_"+case+"_"+version+".json"))
            source=args.fnit_directory/"runs/recon_wm_aseg_independent_20261010_v1"/case/"report.json"
            shutil.copyfile(source,staged/("wm_aseg_"+case+".json"))
            source=args.fnit_directory/"runs/recon_wm_final_source_20261010_v3"/(case[:3]+"-"+case[3:])/"report.json"
            shutil.copyfile(source,staged/("wm_final_source_"+case+".json"))
            source=args.fnit_directory/"runs/recon_wm_cpu_final_20261010_v5"/case/"report.json"
            shutil.copyfile(source,staged/("wm_cpu_final_source_"+case+".json"))
        print(module.export_receipts(input_directory=staged,output_directory=args.output_directory,
            replacements_file=args.replacements_file))


if __name__=="__main__":main()
