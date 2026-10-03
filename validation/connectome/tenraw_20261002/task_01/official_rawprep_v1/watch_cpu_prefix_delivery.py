#!/usr/bin/env python3
"""Collect actual first-nine CPU references incrementally; reuse prior evidence by path/SHA."""
import argparse,hashlib,json,math,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

SUBJECTS=['CON01','CON03']+[f'CON{x:02d}' for x in range(4,11)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('nonfinite actual result')
    if isinstance(x,dict):
        for v in x.values():finite(v)
    if isinstance(x,list):
        for v in x:finite(v)
def save(p,v):
    q=p.with_suffix('.partial');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prior-delivery','official-root','comparison-root','fnit-root','output-root','references-file','plot-python'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False);prior=json.loads((a.prior_delivery/'summary.json').read_text());rows={}
    for subject,row in prior['cases'].items():
        d=a.prior_delivery/subject
        if sha(d/'report.json')!=row['report_SHA256'] or sha(d/'completed_contract_verified.json')!=row['verified_contract_SHA256'] or sha(d/'comparison.json')!=row['comparison_SHA256']:raise ValueError('prior delivery evidence changed')
        rows[subject]=dict(row,evidence_directory=str(d),evidence_reused=True,evidence_files_SHA256={f.name:sha(f) for f in d.iterdir() if f.is_file()})
    previous=0
    while True:
        for subject in SUBJECTS:
            if subject in rows:continue
            root=a.official_root/f'sub-{subject}';vp=root/'completed_contract_verified.json';cp=a.comparison_root/f'{subject}_comparison.json'
            if not vp.exists() or not cp.exists():continue
            rp=root/'report.json';r=json.loads(rp.read_text());v=json.loads(vp.read_text());m=json.loads(cp.read_text())
            if not r['completed'] or sha(rp)!=v['report_sha256'] or m['official_report_sha256']!=sha(rp) or v['EDDY_solver']!='cpu' or v['GPU_UUID'] is not None:raise ValueError('not verified actual CPU result')
            for path,digest in r['input_sha256'].items():
                if sha(path)!=digest:raise ValueError('raw source changed')
            for path,digest in r['output_sha256'].items():
                if sha(root/path)!=digest:raise ValueError('verified output changed')
            finite(m);d=a.output_root/'new_cases'/subject;d.mkdir(parents=True);shutil.copy2(rp,d/'report.json');shutil.copy2(vp,d/vp.name);shutil.copy2(cp,d/'comparison.json')
            stages={x['stage']:x for x in r['commands']};qpath=a.fnit_root/f'sub-{subject}/connectome/preproc/eddy/data.eddy_qc.json';q=json.loads(qpath.read_text())
            if sha(qpath)!=m['FNIT_QC_SHA256']:raise ValueError('actual FNIT QC changed')
            shutil.copy2(qpath,d/'FNIT_EDDY_qc.json')
            for name in ('gpu_report.json','raw_bids_wall.json'):
                source=a.fnit_root/f'sub-{subject}'/name
                if source.exists():shutil.copy2(source,d/('FNIT_'+name))
            rows[subject]={'completed':True,'report_SHA256':sha(rp),'verified_contract_SHA256':sha(vp),'AP_PA_indices':[r['selection']['ap_index'],r['selection']['pa_index']],
                'CPU_reference_activation_wall_seconds':r['CPU_reference_activation_wall_seconds'],'new_CPU8_EDDY_command_wall_seconds':stages['official_EDDY_CPU']['wall_seconds'],
                'restored_original_CPU_preparation_wall_seconds':r['source_CPU_stage_lineage']['original_CPU_wall_seconds'],
                'restored_original_CPU_command_times_seconds':{k:stages[k]['wall_seconds'] for k in ('roi_AP','roi_PA','merge_pair','official_topup','official_b0_mean','official_synthstrip_CPU')},
                'actual_FNIT_packing_SHA256':r['source_CPU_stage_lineage']['actual_FNIT_packing_sha256'],'comparison_state':'actual_completed','comparison_SHA256':sha(cp),
                'packing_exact':m['selected_pair_same_voxels_and_affine'],'GP_seed_matches':m['GP_seed_matches'],'mask_Dice':m['brain_mask']['dice'],
                'TOPUP_brain_field_RMSE_Hz':m['TOPUP']['fieldmap_fout.nii.gz']['official_brain_mask']['rmse'],
                'EDDY_brain_DWI_RMSE':m['EDDY']['data.nii.gz']['official_brain_mask']['rmse'],'EDDY_brain_DWI_p99_abs':m['EDDY']['data.nii.gz']['official_brain_mask']['p99_abs'],'EDDY_brain_DWI_max_abs':m['EDDY']['data.nii.gz']['official_brain_mask']['max_abs'],
                'gradient_RMS_degrees':m['gradient_angle_degrees']['rms'],'gradient_max_degrees':m['gradient_angle_degrees']['max'],
                'FNIT_EDDY_QC_elapsed_seconds':q['elapsed_seconds'],'FNIT_EDDY_QC_SHA256':sha(qpath),'evidence_directory':str(d),'evidence_reused':False,'evidence_files_SHA256':{f.name:sha(f) for f in d.iterdir() if f.is_file()}}
        ordered={s:rows[s] for s in SUBJECTS if s in rows};finite(ordered);summary={'observed_UTC':datetime.now(timezone.utc).isoformat(),'scope':'actual first-nine official CPU references and actual old-baseline comparisons; CON11 separate explicit origin, no all-ten claim',
            'cases':ordered,'completed_CPU_reference_count':len(rows),'actual_comparison_count':len(rows),'all_nine_prefix_completed':len(rows)==9,'all_ten_CPU_references_completed':False,'pending_prefix_cases':[s for s in SUBJECTS if s not in rows],'CON11':'separate actual origin route pending; not included or guessed',
            'prior_delivery':str(a.prior_delivery),'prior_summary_SHA256':sha(a.prior_delivery/'summary.json'),'tool_SHA256':sha(Path(__file__)),'timing_policy':'new activation/CPU8 EDDY/restored original CPU stages separately recorded; no new continuous raw end-to-end wall; FNIT QC internal elapsed and native command wall have different scopes','equivalence_assessed':False,'speedup_claim':None}
        save(a.output_root/'summary.json',summary)
        if len(rows)!=previous:
            save(a.output_root/f'prefix_summary_{len(rows):02d}.json',summary);write_readme(a,summary)
            if len(rows) in (6,9):
                subprocess.run([str(a.plot_python),str(Path(__file__).with_name('plot_verified_cpu_reference.py')),'--official-root',str(a.official_root),'--delivery-summary',str(a.output_root/f'prefix_summary_{len(rows):02d}.json'),'--output-root',str(a.output_root/f'figures_{len(rows):02d}')],check=True)
            save(a.output_root/'SHA256_manifest.json',{str(f.relative_to(a.output_root)):sha(f) for f in a.output_root.rglob('*') if f.is_file() and f.name!='SHA256_manifest.json'})
            previous=len(rows)
        if len(rows)==9:break
        time.sleep(30)

def write_readme(a,s):
    n=len(s['cases']);text=f'''# 官方CPU参考当前前缀：{n}/9例\n\n## 1. 功能与流程\n\n只读收集实际verified的官方CPU rawprep和实际正式baseline比较，不执行原软件或GPU。前四例引用既有04快照的文件路径/SHA，不重复复制大型payload；新增病例独立保存。目标是前九例，CON11另用root真实origin路由，当前不是十例完成。\n\n```mermaid\nflowchart LR\n A[实际canonical raw] --> B[自产官方CPU TOPUP/SynthStrip]\n B --> C[新官方CPU8 EDDY]\n C --> D[102帧及输入输出SHA verified]\n D --> E[实际正式FNIT比较]\n E --> F[增量证据、分列计时、真实脑图]\n```\n\n## 2. Python调用与输入输出\n\n输入AP为96×96×60×102，PA为同空间2帧，bval为102值，bvec为3×102；实际共同b0从packing逐值/raw SHA确认，官方数值自产。输出summary.json、每新增病例report/verified合同/comparison/FNIT QC、不可变prefix_summary_NN.json、SHA索引和已完成前缀脑图。既有病例evidence_directory与每个evidence_files_SHA256指向原已确认快照，保持2/4历史衔接。\n\n```python\nimport subprocess\n# 以下变量均为已有实际目录或新的交付目录；不会启动科学计算。\nsubprocess.run(["python", "watch_cpu_prefix_delivery.py",\n    "--prior-delivery", prior_verified_delivery_directory,\n    "--official-root", official_cpu_reference_directory,\n    "--comparison-root", actual_comparison_directory,\n    "--fnit-root", actual_formal_baseline_directory,\n    "--output-root", new_incremental_delivery_directory,\n    "--references-file", verified_reference_readme_path,\n    "--plot-python", existing_plot_python_path], check=True)\n```\n\n七项参数均必填：prior-delivery为已SHA核对的04快照；official-root为实际CPU预算结果；comparison-root为实际成功比较；fnit-root为比较对应的真实baseline；output-root必须新目录；references-file为既有参考说明；plot-python为已有CPU绘图Python，科学环境不变。\n\n## 3. 命令行\n\n```bash\npython watch_cpu_prefix_delivery.py --prior-delivery /results/actual04 \\\n --official-root /results/official_CPU --comparison-root /results/actual_comparison \\\n --fnit-root /results/formal_baseline/baseline --output-root /results/new_prefix \\\n --references-file /docs/verified_README.md --plot-python /existing/plot/python\n```\n\n## 4. 原软件调用\n\n成功自产TOPUP、SynthStrip命令、原程序和模型SHA见每例report的原命令及source_CPU_stage_lineage；新EDDY均原生FSL6.0.7.4 eddy_cpu，每例8线程、并发2、GPU不可见。FNIT运行时不调用这些原软件。\n\n```bash\neddy_cpu --imain=raw/AP.nii.gz --mask=mask/nodif_brain_mask.nii.gz \\\n --acqp=topup/acqparams.txt --index=eddy/eddy_index.txt \\\n --bvecs=raw/AP.bvec --bvals=raw/AP.bval --topup=topup/fieldmap_out \\\n --out=eddy/data --flm=quadratic --resamp=jac --slm=linear --niter=8 \\\n --fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move --nvoxhp=1000 \\\n --repol --rms --initrand=12345 --ref_scan_no=ACTUAL_RAW_AP_INDEX\n```\n\n## 5. 实际精度与耗时\n\n|病例|AP/PA|mask Dice|脑内field RMSE Hz|脑内DWI RMSE|梯度RMS °|\n|---|---|---:|---:|---:|---:|\n'''
    for subject,r in s['cases'].items():text+=f"|{subject}|{r['AP_PA_indices'][0]}/{r['AP_PA_indices'][1]}|{r['mask_Dice']:.8f}|{r['TOPUP_brain_field_RMSE_Hz']:.6f}|{r['EDDY_brain_DWI_RMSE']:.6f}|{r['gradient_RMS_degrees']:.6f}|\n"
    text+='\n|病例|新activation wall s|新CPU8 EDDY wall s|恢复原CPU准备wall s|原TOPUP wall s|原SynthStrip wall s|FNIT QC elapsed s|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for subject,r in s['cases'].items():
        values=[r['CPU_reference_activation_wall_seconds'],r['new_CPU8_EDDY_command_wall_seconds'],r['restored_original_CPU_preparation_wall_seconds'],r['restored_original_CPU_command_times_seconds']['official_topup'],r['restored_original_CPU_command_times_seconds']['official_synthstrip_CPU'],r['FNIT_EDDY_QC_elapsed_seconds']]
        text+='|'+subject+'|'+'|'.join(f'{v:.3f}' for v in values)+'|\n'
    text+='\n上述为累积rawprep差异，各自场/mask不同，未检验等价阈值。命令wall与FNIT QC内部elapsed边界不同，不算speedup。新activation包括恢复/核验/等待/EDDY，原stage时间单列，不拼成新连续端到端。正式baseline未单记的阶段时间不补值；外层CLI含建模/追踪，不能当rawprep wall。\n\n'
    graph=9 if n==9 else 6 if n>=6 else None
    if graph:text+=f'![实际{graph}例CPU校正reference b0及自产mask](figures_{graph:02d}/verified_CPU_EDDY_brain.png)\n\n![新EDDY与原恢复CPU阶段时间分列](figures_{graph:02d}/verified_CPU_command_times.png)\n\n'
    text+='## 6. 更新和benchmark记录\n\n当前版以真实完成合同/比较增量收集，既有大型payload只引用。旧2/4快照保留；旧wrong-B0 JSON和GPU预算SIGTERM证据保留，不带回未采用OOM分支或旧错误脑图。比较器只修复已确认native outlier首行格式，不补NaN。CON11在新实际baseline出现后通过独立origin守卫，不假定旧路径。\n\n## 7. 参考文献和原代码\n\n'
    refs=a.references_file.read_text();text+=refs.split('## 7.',1)[-1].split('\n',1)[-1];(a.output_root/'README.md').write_text(text)
if __name__=='__main__':main()
