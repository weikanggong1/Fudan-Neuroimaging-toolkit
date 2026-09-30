"""比较整例与独立 MNI 重放的三个输出，复用既有严格体积比较器。

固定输入为 e036f57 的 FNIT 整例与本轮 MNI 隔离目录，不读取官方结果。
两张 warp 为向量 NIfTI、检查图为强度 NIfTI；沿用原网格、dtype、仿射及
现有 1e-6 浮点容差，不重采样。写出新的比较 JSON，保留输入及脚本哈希；
缺失文件或未通过时抛异常。该工具没有独立官方等价命令。
"""
import json
from pathlib import Path

from compare_complete_subject import _volume
from fingerprint_e036f57 import fingerprint


def main() -> None:
    """使用本轮已声明的固定路径；不接受隐含的被试或阈值选项。"""
    root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930')
    baseline = root / 'full_sub01_e036f57_uuid/mri/transforms/synthmorph.1.0mm.1.0mm'
    candidate = root / 'sub01/mni_e036f57_after_full/mri/transforms/synthmorph.1.0mm.1.0mm'
    names = ['warp.to.mni152.1.0mm.1.0mm.nii.gz',
             'warp.to.mni152.1.0mm.1.0mm.inv.nii.gz', 'test.nii.gz']
    report = {'code_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68',
              'script': fingerprint(Path(__file__)),
              'comparator': fingerprint(root / 'compare_complete_subject.py'), 'files': {}}
    for name in names:
        report['files'][name] = {'baseline': fingerprint(baseline / name),
                                 'candidate': fingerprint(candidate / name),
                                 'comparison': _volume(baseline / name, candidate / name)}
    with (root / 'sub01/mni_e036f57_after_full_comparison.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    assert all(row['comparison']['pass'] for row in report['files'].values())


if __name__ == '__main__':
    main()
