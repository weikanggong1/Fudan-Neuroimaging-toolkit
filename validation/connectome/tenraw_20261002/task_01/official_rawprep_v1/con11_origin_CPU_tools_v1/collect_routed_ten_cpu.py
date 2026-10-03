#!/usr/bin/env python3
"""Collect actual ten verified CPU cases through explicit routes, reusing the nine-case evidence."""
import argparse,json,hashlib,math,shutil,time,subprocess
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 q=Path(p).with_suffix('.partial');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def finite(v):
 if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite actual value')
 if isinstance(v,dict):
  for x in v.values():finite(x)
 if isinstance(v,list):
  for x in v:finite(x)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('routes','prefix-delivery','con11-comparison','output-root','plot-python'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False)
 prefix=a.prefix_delivery/'prefix_summary_09.json'
 while True:
  route=load(a.routes)
  ready=route['all_ten_actual_verified_ready'] and prefix.exists() and a.con11_comparison.exists()
  save(a.output_root/'status.json',{'state':'waiting_actual_ten_verified_routes_nine_prefix_and_CON11_comparison','verified_route_cases':route['verified_ready_cases'],'first9_prefix_exists':prefix.exists(),'actual_CON11_comparison_exists':a.con11_comparison.exists(),'observed_UTC':datetime.now(timezone.utc).isoformat()})
  if ready:break
  time.sleep(30)
 original=load(prefix);rows={}
 if len(original['cases'])!=9 or not original['all_nine_prefix_completed']:raise ValueError('actual nine-case prefix missing')
 for subject,row in original['cases'].items():
  item=route['cases']['sub-'+subject];d=Path(item['official_case_directory']);rp=d/'report.json';vp=d/'completed_contract_verified.json';r=load(rp);v=load(vp)
  if item['state']!='actual_verified_CPU_contract_ready' or sha(rp)!=row['report_SHA256'] or sha(vp)!=row['verified_contract_SHA256'] or sha(rp)!=v['report_sha256']:raise ValueError('actual prefix route/evidence mismatch')
  if not r.get('source_CPU_stage_lineage') or item['execution_kind']!='restored_own_CPU_stages_plus_new_CPU_EDDY':raise ValueError('restored first-nine provenance missing')
  for name,digest in row['evidence_files_SHA256'].items():
   if sha(Path(row['evidence_directory'])/name)!=digest:raise ValueError('prior evidence changed')
  for name,digest in r['output_sha256'].items():
   if sha(d/name)!=digest:raise ValueError('actual verified output changed')
  rows[subject]=dict(row,official_case_directory=str(d),actual_FNIT_case_directory=item['actual_FNIT_case_directory'],execution_kind=item['execution_kind'],evidence_reused=True)
 item=route['cases']['sub-CON11'];d=Path(item['official_case_directory']);rp=d/'report.json';vp=d/'completed_contract_verified.json';r=load(rp);v=load(vp);m=load(a.con11_comparison)
 if not r['completed'] or v['EDDY_solver']!='cpu' or v['GPU_UUID'] is not None or r.get('source_CPU_stage_lineage'):raise ValueError('fresh CON11 CPU provenance invalid')
 if item['execution_kind']!='fresh_official_CPU_rawprep' or v['execution_kind']!=item['execution_kind'] or sha(rp)!=item['report_SHA256'] or sha(vp)!=item['verified_contract_SHA256'] or sha(rp)!=v['report_sha256'] or m['official_report_sha256']!=sha(rp):raise ValueError('fresh actual CON11 route mismatch')
 if item['actual_FNIT_case_directory']!=v['actual_selection_origin']['actual_FNIT_case_directory']:raise ValueError('actual selected FNIT path mismatch')
 for path,digest in r['input_sha256'].items():
  if sha(path)!=digest:raise ValueError('raw source changed')
 for path,digest in r['output_sha256'].items():
  if sha(d/path)!=digest:raise ValueError('CON11 verified output changed')
 finite(m);dest=a.output_root/'new_cases/CON11';dest.mkdir(parents=True)
 for src,name in [(rp,'report.json'),(vp,vp.name),(a.con11_comparison,'comparison.json')]:shutil.copy2(src,dest/name)
 fnit=Path(item['actual_FNIT_case_directory']);qc=fnit/'connectome/preproc/eddy/data.eddy_qc.json';q=load(qc)
 if sha(qc)!=m['FNIT_QC_SHA256']:raise ValueError('actual CON11 QC changed')
 shutil.copy2(qc,dest/'FNIT_EDDY_qc.json')
 for name in ('gpu_report.json','raw_bids_wall.json'):
  if (fnit/name).exists():shutil.copy2(fnit/name,dest/('FNIT_'+name))
 commands={x['stage']:x for x in r['commands']}
 if any(x['returncode']!=0 for x in commands.values()):raise ValueError('actual CON11 native command failed')
 rows['CON11']={'completed':True,'execution_kind':'fresh_official_CPU_rawprep','official_case_directory':str(d),'actual_FNIT_case_directory':str(fnit),'report_SHA256':sha(rp),'verified_contract_SHA256':sha(vp),'AP_PA_indices':[r['selection']['ap_index'],r['selection']['pa_index']],'fresh_contiguous_CPU_rawprep_wall_seconds':r['observed_rawprep_total_wall_seconds'],'fresh_CPU_command_times_seconds':{k:x['wall_seconds'] for k,x in commands.items()},'new_CPU8_EDDY_command_wall_seconds':commands['official_EDDY_CPU']['wall_seconds'],'comparison_state':'actual_completed','comparison_SHA256':sha(a.con11_comparison),'packing_exact':m['selected_pair_same_voxels_and_affine'],'GP_seed_matches':m['GP_seed_matches'],'mask_Dice':m['brain_mask']['dice'],'TOPUP_brain_field_RMSE_Hz':m['TOPUP']['fieldmap_fout.nii.gz']['official_brain_mask']['rmse'],'EDDY_brain_DWI_RMSE':m['EDDY']['data.nii.gz']['official_brain_mask']['rmse'],'EDDY_brain_DWI_p99_abs':m['EDDY']['data.nii.gz']['official_brain_mask']['p99_abs'],'EDDY_brain_DWI_max_abs':m['EDDY']['data.nii.gz']['official_brain_mask']['max_abs'],'gradient_RMS_degrees':m['gradient_angle_degrees']['rms'],'gradient_max_degrees':m['gradient_angle_degrees']['max'],'FNIT_EDDY_QC_elapsed_seconds':q['elapsed_seconds'],'FNIT_EDDY_QC_SHA256':sha(qc),'evidence_directory':str(dest),'evidence_reused':False,'evidence_files_SHA256':{f.name:sha(f) for f in dest.iterdir() if f.is_file()}}
 finite(rows);result={'observed_UTC':datetime.now(timezone.utc).isoformat(),'scope':'actual ten CPU references: first-nine restored own CPU stages; CON11 fresh all-phase actual selected origin','cases':rows,'completed_CPU_reference_count':10,'actual_comparison_count':10,'all_ten_CPU_references_completed':True,'all_ten_actual_comparisons_completed':True,'routes':str(a.routes),'routes_SHA256':sha(a.routes),'prior_nine_prefix_summary':str(prefix),'prior_nine_prefix_SHA256':sha(prefix),'tool_SHA256':sha(Path(__file__)),'equivalence_assessed':False,'speedup_claim':None,'timing_policy':'First-nine restored CPU stage walls and new EDDY/activation separately; CON11 fresh uninterrupted CPU wall separately. Never concatenate stage times or mix FNIT internal QC elapsed with native process wall.'}
 save(a.output_root/'summary.json',result);shutil.copy2(a.routes,a.output_root/'explicit_actual_CPU_case_routes_v1.json')
 subprocess.run([str(a.plot_python),str(Path(__file__).with_name('plot_routed_ten_cpu.py')),'--delivery-summary',str(a.output_root/'summary.json'),'--output-root',str(a.output_root/'figures')],check=True)
 write_readme(a,result)
 save(a.output_root/'SHA256_manifest.json',{str(f.relative_to(a.output_root)):sha(f) for f in a.output_root.rglob('*') if f.is_file() and f.name!='SHA256_manifest.json'})
 save(a.output_root/'status.json',{'state':'actual_ten_verified_and_compared_CPU_delivery_collected','summary_SHA256':sha(a.output_root/'summary.json'),'scientific_source_changed':False})
def write_readme(a,result):
 text="""# 十例实际官方 CPU 参考与 FNIT rawprep 比较

## 1. 功能与流程

只读收集真实 raw 数据产生的官方 CPU 参考、完成合同与 FNIT 比较，不修改科学实现。

```mermaid
flowchart LR
 A[逐例实际路径与SHA] --> B[九例恢复自产CPU准备阶段]
 A --> C[CON11新CPU完整流程]
 B --> D[实际102帧与梯度验证]
 C --> D
 D --> E[实际FNIT逐例比较与脑图]
```

## 2. Python调用、输入与输出

输入 routes 为逐例实际路径合同；prefix-delivery 为已核验九例增量证据；con11-comparison 为新 CON11 实际比较；output-root 必须新目录；plot-python 为已有绘图 Python。原始 AP 为96×96×60×102，PA为同空间2帧，bval为102值，bvec为3×102。输出summary、仅新增CON11证据、SHA索引与实际脑图；前九例按路径和SHA引用。

```python
import subprocess
from pathlib import Path
# 变量指向已有实际证据，新交付目录必须不存在。
remote_task_directory = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_01')
existing_plot_python_path = '/home1/gongwk/anaconda3/bin/python'
subprocess.run(['python', 'collect_routed_ten_cpu.py',
 '--routes', str(remote_task_directory / 'explicit_actual_CPU_case_routes_v1.json'),
 '--prefix-delivery', str(remote_task_directory / 'CPU_reference_prefix9_delivery_v1'),
 '--con11-comparison', str(remote_task_directory / 'CON11_fresh_origin_verified_comparison_v1/CON11_comparison.json'),
 '--output-root', str(remote_task_directory / 'new_actual_ten_delivery'),
 '--plot-python', existing_plot_python_path], check=True)
```

## 3. 命令行

```bash
python collect_routed_ten_cpu.py --routes /results/explicit_routes.json \
 --prefix-delivery /results/verified_prefix9 \
 --con11-comparison /results/CON11_comparison.json \
 --output-root /results/new_actual_ten --plot-python /existing/plot/python
```

## 4. 原软件调用

原软件仅用于外部 benchmark；FNIT 运行时不调用原软件。实际 TOPUP/SynthStrip/EDDY 完整命令、输入与原程序SHA见逐例 report.commands。CPU8 EDDY 参数保持原实验设置：

```bash
eddy_cpu --imain=raw/AP.nii.gz --mask=mask/nodif_brain_mask.nii.gz \
 --acqp=topup/acqparams.txt --index=eddy/eddy_index.txt \
 --bvecs=raw/AP.bvec --bvals=raw/AP.bval --topup=topup/fieldmap_out \
 --out=eddy/data --flm=quadratic --resamp=jac --slm=linear --niter=8 \
 --fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move --nvoxhp=1000 \
 --repol --rms --initrand=12345 --ref_scan_no=ACTUAL_RAW_AP_INDEX
```

## 5. 实际精度、运行时间与脑图

|病例|AP/PA|mask Dice|脑内field RMSE Hz|脑内DWI RMSE|梯度RMS °|CPU8 EDDY wall s|FNIT QC elapsed s|
|---|---|---:|---:|---:|---:|---:|---:|
"""
 for subject,row in result['cases'].items():
  text+=f"|{subject}|{row['AP_PA_indices'][0]}/{row['AP_PA_indices'][1]}|{row['mask_Dice']:.8f}|{row['TOPUP_brain_field_RMSE_Hz']:.6f}|{row['EDDY_brain_DWI_RMSE']:.6f}|{row['gradient_RMS_degrees']:.6f}|{row['new_CPU8_EDDY_command_wall_seconds']:.3f}|{row['FNIT_EDDY_QC_elapsed_seconds']:.3f}|\n"
 text+='\n前九例的恢复准备wall、新activation及原阶段wall保存在summary各自字段；CON11为新的完整CPU运行，实际连续wall为'+str(result['cases']['CON11']['fresh_contiguous_CPU_rawprep_wall_seconds'])+'秒，逐阶段wall见fresh_CPU_command_times_seconds。未单记的FNIT阶段wall不补值；原生进程wall与FNIT内部QC elapsed边界不同，不计算speedup，未检验等价阈值。\n\n![十例真实CPU校正b0与自产mask](figures/verified_CPU_EDDY_brain.png)\n\n![各自实际命令wall](figures/verified_CPU_command_times.png)\n\n## 6. 更新与benchmark记录\n\n保留2/4/6/9例实际快照、原GPU预算SIGTERM证据及旧错误packing的JSON。当前十例按显式路径收集，CON11使用真实新origin AP0/PA0，不写旧namespace、不假造恢复lineage。outlier首行格式修复只改变读取格式，不填NaN。\n\n## 7. 参考文献与原实现\n\n'
 references=(a.prefix_delivery/'README.md').read_text().split('## 7.',1)[-1].split('\n',1)[-1]
 (a.output_root/'README.md').write_text(text+references)

if __name__=='__main__':main()
