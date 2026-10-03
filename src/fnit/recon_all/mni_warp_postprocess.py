"""GPU MNI warp postprocessing callable and standalone CLI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .mni_warp_sampling import convert_mni_warp, resample_mni_check
from .mni_warp_inverse import invert_mni_warp


def run_mni_warp_postprocess(*, ras_warp, cropped_source, original, full_target,
                            crop_to_original_lta, crop_to_full_lta, output_dir,
                            device="cuda:0", chunk_slices=16):
    """从自产deform与两条LTA运行连续转换→完整求逆→最近邻检查图。

    六个输入为文件路径；output_dir为输出目录，默认CUDA、X分块16。
    返回forward/inverse/check路径与含加载/传输/保存的分步报告。
    失败抛异常，不回退、不读取参考结果；输出目录须不存在，防止覆盖。
    """
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    forward=output/'warp.to.mni152.1.0mm.1.0mm.nii.gz'
    inverse=output/'warp.to.mni152.1.0mm.1.0mm.inv.nii.gz'
    check=output/'test.nii.gz'
    conversion=convert_mni_warp(ras_warp,cropped_source,original,full_target,
        crop_to_original_lta,crop_to_full_lta,forward,device=device,chunk_slices=chunk_slices)
    inversion=invert_mni_warp(forward,inverse,device=device)
    sampling=resample_mni_check(original,forward,check,device=device,chunk_slices=chunk_slices)
    return {'forward':str(forward),'inverse':str(inverse),'check':str(check),
            'postprocess_backend':'gpu','conversion':conversion,'inverse_report':inversion,
            'check_report':sampling}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ras-warp','cropped-source','original','full-target',
                 'crop-to-original-lta','crop-to-full-lta','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--chunk-slices',type=int,default=16)
    args=vars(parser.parse_args())
    print(json.dumps(run_mni_warp_postprocess(**args),indent=2))
if __name__=='__main__':main()
