"""在官方 fMRIPrep 镜像中运行原 NiWorkflows 参考网格生成器。"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
from niworkflows.interfaces.nibabel import GenerateSamplingReference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixed', required=True)
parser.add_argument('--moving', required=True)
parser.add_argument('--mask', required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
interface = GenerateSamplingReference()
interface.inputs.fixed_image = args.fixed
interface.inputs.moving_image = args.moving
interface.inputs.fov_mask = args.mask
interface.inputs.keep_native = True
result = interface.run(cwd=str(args.work))
shutil.copyfile(result.outputs.out_file, args.output)
args.report.write_text(json.dumps({
    'implementation': 'original niworkflows.interfaces.nibabel.GenerateSamplingReference',
    'niworkflows_version': importlib.metadata.version('niworkflows'),
    'nilearn_version': importlib.metadata.version('nilearn'),
    'nibabel_version': importlib.metadata.version('nibabel'),
    'keep_native': True,
    'fov_mask': 'fresh original T1 SynthStrip mask',
    'output_copied_without_header_changes': True,
}, indent=2) + '\n')
