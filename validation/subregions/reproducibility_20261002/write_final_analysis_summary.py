"""Write evidence summary from completed final saved-output metrics only."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from collections import Counter
from build_repeatability_manifest import read_json,sha256

NAMES={'brainstem':'脑干','thalamus':'丘脑细核','hippocampus_left':'左海马','hippocampus_right':'右海马','amygdala_left':'左杏仁核','amygdala_right':'右杏仁核'}
SPACES={'stage_native':'同阶段原网格','stage_hr':'同阶段高分辨率','raw_native':'原始 T1 原网格','raw_hr':'原始 T1 高分辨率'}

def number(value,digits=6):return 'NA' if value is None else f'{value:.{digits}f}'
def observed_range(values,digits=3):
    valid=[value for value in values if value is not None]
    if not valid:return 'NA'
    return number(min(valid),digits)+'–'+number(max(valid),digits)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory
    status=read_json(directory/'final_full_analysis_status.json')
    if status['state']!='completed':raise ValueError('Final completed 24-group result required')
    for name,identity in status['artifacts'].items():
        path=directory/name
        if path.stat().st_size!=identity['bytes'] or sha256(path)!=identity['sha256']:raise ValueError('Final artifact SHA differs')
    result=read_json(directory/'final_full_repeatability.json')
    timing=read_json(directory/'timing_complete/final_step_timing.json')
    if timing['final_result_sha256']!=sha256(directory/'final_full_repeatability.json'):raise ValueError('Actual timer source differs')
    groups=result['groups'];previous=result['before_groups_on_joint_grid']
    if len(groups)!=24 or sum(len(g['regions']) for g in groups)!=440:raise ValueError('Expected24groups440rows')
    count=Counter(region['own_repeat_status'] for group in groups for region in group['regions'])
    lines=['# 最终完整流程重复性与精度','',
        f"不可变源码清单 SHA `{result['snapshot_audit']['source_manifest_sha256']}`，核实 {result['snapshot_audit']['verified_files']} 个文件、{result['snapshot_audit']['verified_runtime_python_files']} 个运行时 Python 文件。优化前 `{result['before_fix_commit'][:7]}`、最终实现及三次全新官方结果分别统计。",'',
        '原始 T1 和同阶段输入分别三个独立完整进程，每次均分割四个结构。对照为同一公开 T1 的 FreeSurfer 8.2.0-1 三次全新分割，固定 norm/aseg/wmparc、4 线程和输入/图谱/源码 SHA。', '',
        '重复列为逐标签对称体积加权 Dice；跨实现列使用官方标签体积加权。不同 voxel 为家族网格上不同标签的数量。3 次重复提供观察范围，不提供统计置信区间或人群随机范围。双方均空的标签 Dice 为 NA。', '',
        '## 家族结果','']
    summary=[]
    for space,title in SPACES.items():
        lines.extend(['### '+title,'','| 家族 | 官方重复 Dice / 不同 voxel | 优化前 FNIT 重复最低 Dice / 最多不同 voxel | 最终 FNIT 重复最低 Dice / 最多不同 voxel | 跨实现 Dice 优化前 → 最终 |','|---|---:|---:|---:|---:|'])
        for group,old in zip(groups,previous):
            if group['space']!=space:continue
            def noise(g,method):
                s=g['pair_summary'][method]
                return number(s['weighted_label_dice']['min'])+' / '+str(int(s['different_voxels']['max']))
            old_cross=old['pair_summary']['cross_method']['first_reference_weighted_label_dice']['min']
            new_cross=group['pair_summary']['cross_method']['first_reference_weighted_label_dice']['min']
            lines.append('| '+NAMES[group['family']]+' | '+noise(group,'official_repeat')+' | '+noise(old,'fnit_repeat')+' | '+noise(group,'fnit_repeat')+' | '+number(old_cross)+' → '+number(new_cross)+' |')
            summary.append({'family':group['family'],'space':space,'official_repeat':group['pair_summary']['official_repeat'],
                            'before_fnit_repeat':old['pair_summary']['fnit_repeat'],'final_fnit_repeat':group['pair_summary']['fnit_repeat'],
                            'cross_before':old_cross,'cross_final':new_cross,'cross_change':None if old_cross is None or new_cross is None else new_cross-old_cross})
        lines.append('')
    lines.extend(['## 逐 ROI 重复与低 Dice','',
        '440 行逐标签记录见 [完整 TSV](final_full_repeatability.tsv)，含官方/FNIT 内部 Dice、不同 voxel、软体积 CV、跨实现 Dice、优化前指标及变化；[全部配对和来源](final_full_repeatability.json)。', '',
        '自身重复波动与跨实现偏差分别评价。稳定的实现可以有 voxel 差异；官方重复 Dice=1 不设定跨实现精度必须为1的门槛。最终自身重复分类：'+ '；'.join(f'{k}: {v}' for k,v in sorted(count.items()))+'。','',
        '| 空间 | 家族 | 最低跨实现 ROI（Dice；官方 voxel） |','|---|---|---|'])
    lowest=[]
    for group in groups:
        candidates=[r for r in group['regions'] if r['cross_method_dice']['min'] is not None]
        worst=sorted(candidates,key=lambda r:r['cross_method_dice']['min'])[:3]
        lines.append('| '+SPACES[group['space']]+' | '+NAMES[group['family']]+' | '+'；'.join(r['name']+'（'+number(r['cross_method_dice']['min'],3)+'；'+str(r['official_voxels'][0])+'）' for r in worst)+' |')
        lowest.append({'family':group['family'],'space':group['space'],'lowest':worst})
    lines.extend(['','## 实测时间','',
        '| 输入 | API 计算（秒） | API 含保存（秒） | 子进程时间（秒） | observer 回调子计时（秒） | 自身采样显存峰值（MiB） |','|---|---:|---:|---:|---:|---:|'])
    for mode,label in (('stage','同阶段'),('raw','原始 T1')):
        records=[r for r in result['run_audit'] if r['mode']==mode]
        lines.append('| '+label+' | '+observed_range([r['api_compute_seconds'] for r in records])+' | '+observed_range([r['api_total_seconds'] for r in records])+' | '+observed_range([r['process_wall_seconds'] for r in records])+' | '+observed_range([r['context_observer_seconds'] for r in records])+' | '+str(max(r['sampled_peak_own_memory_mib'] for r in records))+' |')
    lines.extend(['',
        'API/子进程时间均保留本轮 observer 实际开销。子进程时间为 Popen 前至独立 wait 确认退出，包含导入、observer 源码核对、API、评分及收尾；来源检查和 GPU 预算等待单列。observer 回调子计时不包含包装器开始/结束的源码 hash 及最终 JSON 保存。共享 GPU 其他进程占用有记录，采样显存峰值不等于瞬时峰值。','',
        '分步骤时间见 [逐步骤表](timing_complete/final_step_timing.tsv) 和 [计时出处](timing_complete/final_step_timing.json)。TH/HIP 的合成、工作图准备、多层强度准备/拟合已从对应 API/solver timer 提取；未独立计量的 affine/后处理保留缺测，recipe 剩余开销合并列出。官方显式 timer 为整数秒，合成/强度阶段包括各自准备与拟合；官方后处理没有独立 timer，保留缺测。中间阶段没有共同保存的标签时不提供虚构 Dice，最终输出精度使用上述原网格/高分辨率比较。'])
    if 'historical_raw_stage_sum' in timing:
        historical=timing['historical_raw_stage_sum'];lines.extend(['',
            f"官方历史 recon-all 耗时 {historical['historical_reconall_wall_seconds']:.0f} 秒；加上本轮三个细核子流程得到分段合计 {historical['summed_observed_wall_seconds_min']:.2f}–{historical['summed_observed_wall_seconds_max']:.2f} 秒。这是来源核实的历史预处理与本轮细核时间相加，未作为本轮一条完整原始 T1 重跑计时。"])
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    (directory/'final_doc_summary.json').write_text(json.dumps({'source_manifest_sha256':result['snapshot_audit']['source_manifest_sha256'],
        'own_repeat_status_counts':dict(count),'family_space_summary':summary,'lowest_regions':lowest,
        'result_sha256':sha256(directory/'final_full_repeatability.json'),'script_sha256':sha256(Path(__file__))},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'groups':len(groups),'rows':440,'README_sha256':sha256(directory/'README.md')}))


if __name__=='__main__':main()
