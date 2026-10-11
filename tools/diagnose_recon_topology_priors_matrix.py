"""只读定位自产拓扑RAS矩阵的数值与rawbit差别；不修改矩阵或原始报告。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def run(*,creation_capture,brain,original_report,output):
    """完整创建快照/同原MRI头/已完成报告→矩阵逐元素数值、符号位及SHA。

    全部参数为路径，无默认值；output须不存在。creation_capture为
    FNITPDF1 phase0参考，仅诊断；brain为同一自产MRI，使用nibabel头
    独立生成surfaceRAS(mm)→连续体素索引矩阵。original_report为真实
    stage JSON，不覆盖它。返回并落盘JSON，F64 uint64位模式和数值
    差别独立记录；零符号差不替换原report中的严格False，不调用原程序。
    缺损/错误phase或原MRI的SHA不匹配原报告时明确ValueError。
    """
    if output.exists():raise FileExistsError('fresh diagnostic report required')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    fields=load_volume_preparation_capture(capture_path=creation_capture)
    if int(fields['header'][7])!=0:raise ValueError('creation phase0 required')
    actual=json.loads(original_report.read_bytes())
    if sha(brain)!=actual['brain_sha256_before'] or sha(creation_capture)!=actual['reference_capture_sha256']['creation']:
        raise ValueError('same actual stage input required')
    own=source_topology_ras_to_voxel(volume_header=nib.load(brain).header).astype(np.float64)
    ref=fields['ras_to_voxel']
    positions=np.argwhere(own.view(np.uint64)!=ref.view(np.uint64))
    rows=[dict(row=int(i),column=int(j),candidate_value=float(own[i,j]),reference_value=float(ref[i,j]),
        candidate_uint64=int(own.view(np.uint64)[i,j]),reference_uint64=int(ref.view(np.uint64)[i,j]),
        numeric_equal=bool(own[i,j]==ref[i,j]),both_zero=bool(own[i,j]==0 and ref[i,j]==0)) for i,j in positions]
    result=dict(status='complete_readonly_matrix_bit_diagnosis',original_candidate_report_sha256=sha(original_report),
        capture_sha256=sha(creation_capture),brain_sha256=sha(brain),script_sha256=sha(__file__),
        candidate_matrix=own.tolist(),reference_matrix=ref.tolist(),different_raw_elements=len(rows),
        different_numeric_elements=int(np.count_nonzero(own!=ref)),all_raw_differences_are_signed_zero=all(row['both_zero'] for row in rows),
        differences=rows,original_strict_full_local_fields_passed=actual['strict_full_local_fields_passed'],
        mature_matrix_and_sampler_not_modified=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['creation-capture','brain','original-report','output']:p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
