"""Assemble actual verified final-ten metadata and readable current documentation, preserving remote originals."""
from pathlib import Path
import json,hashlib,shutil,argparse
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--download-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--frozen-tools',type=Path,required=True);a=p.parse_args()
 s=load(a.download_root/'delivery/summary.json');audit=load(a.download_root/'audit/audit.json');raw=load(a.download_root/'audit/raw_index_audit.json');meta=load(a.download_root/'metadata/index.json')
 assert len(s['cases'])==len(meta['cases'])==10 and audit['state']=='actual_ten_CPU_independently_verified_and_compared'
 expected={'CON01','CON03'}|{f'CON{x:02d}' for x in range(4,12)};assert set(s['cases'])==set(meta['cases'])==expected and len({raw['cases']['sub-'+x]['AP']['SHA256'] for x in expected})==10
 a.output_root.mkdir(parents=True,exist_ok=False)
 for src,dest in [('delivery/summary.json','summary.json'),('delivery/README.md','original_collector_README.md'),('delivery/explicit_actual_CPU_case_routes_v1.json','explicit_actual_CPU_case_routes_v1.json'),('delivery/SHA256_manifest.json','original_collector_SHA256_manifest.json'),('delivery/status.json','original_collector_status.json')]:shutil.copy2(a.download_root/src,a.output_root/dest)
 for source,dest in [('delivery/figures','figures'),('audit','independent_audit'),('metadata','cases')]:shutil.copytree(a.download_root/source,a.output_root/dest)
 tools=a.output_root/'frozen_metadata_tools';tools.mkdir()
 for name in ['collect_routed_ten_cpu.py','plot_routed_ten_cpu.py','freeze.json','actual_collector_launch_record_v1.json']:
  shutil.copy2(a.frozen_tools/name,tools/name)
 text=(a.download_root/'delivery/README.md').read_text();time_table='\n逐例时间边界如下。所有数值均为实际报告的 wall seconds；“不适用”不是补零。\n\n|病例|新activation wall|fresh完整wall|恢复原准备wall|TOPUP命令wall|SynthStrip命令wall|新CPU8 EDDY命令wall|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for subject,row in s['cases'].items():
  if subject=='CON11':
   cmd=row['fresh_CPU_command_times_seconds'];values=['不适用',f"{row['fresh_contiguous_CPU_rawprep_wall_seconds']:.3f}",'不适用']
  else:
   cmd=row['restored_original_CPU_command_times_seconds'];values=[f"{row['CPU_reference_activation_wall_seconds']:.3f}",'不适用',f"{row['restored_original_CPU_preparation_wall_seconds']:.3f}"]
  values += [f"{cmd['official_topup']:.3f}",f"{cmd['official_synthstrip_CPU']:.3f}",f"{row['new_CPU8_EDDY_command_wall_seconds']:.3f}"]
  time_table+='|'+subject+'|'+'|'.join(values)+'|\n'
 time_table+='\n恢复原准备 wall 可能包含等待实际 packing 的时间，原命令 wall 单列；不能拼成一次新的连续 coldwall。CON11 的 fresh wall 是本次真实连续运行的总 wall，不宣称冷缓存。FNIT 未独立记录的 rawprep 总 wall 与分阶段 wall 不补值。\n'
 text=text.replace('## 6. 更新与benchmark记录',time_table+'\n## 6. 更新与benchmark记录')
 text=text.replace('## 6. 更新与benchmark记录','独立完整体素审计：十例 AP/PA 共二十份原始影像负值和非有限值均为 0；校正输出的实际负值计数见 independent_audit/audit.json，不裁剪，不更改数据。\n\n## 6. 更新与benchmark记录')
 text=text.replace('## 7. 参考文献与原实现','原 collector 的 SHA 索引在更新最终 status.json 之前生成，状态文件这一项过时；原索引与状态文件均保留。independent_audit 记录原差异和实际服务器索引，本交付使用 LOCAL_delivery_SHA256.json 对本地实际内容逐项校验。精简 report_execution_metadata.json 只省略高频 memory_samples，完整不可变原报告路径、大小及 SHA 留在每例文件中；原始完整报告、失败尝试和旧 namespace 均未覆盖。\n\n## 7. 参考文献与原实现')
 (a.output_root/'README.md').write_text(text)
 (a.output_root/'delivery_provenance.json').write_text(json.dumps({'assembled_actual_UTC':datetime.now(timezone.utc).isoformat(),'case_count':10,'distinct_actual_raw_AP_SHA256':10,'actual_server_summary_SHA256':sha(a.download_root/'delivery/summary.json'),'independent_audit_SHA256':sha(a.download_root/'audit/audit.json'),'original_collector_status_index_difference_preserved':audit['original_collector_index_mismatches_preserved'],'remote_sources_or_contracts_changed':False,'science_changed':False,'speedup_claim':None,'equivalence_assessed':False,'assembler_SHA256':sha(__file__)},indent=2)+'\n')
 index={str(f.relative_to(a.output_root)):sha(f) for f in a.output_root.rglob('*') if f.is_file()};(a.output_root/'LOCAL_delivery_SHA256.json').write_text(json.dumps(index,indent=2)+'\n')
 for name,digest in index.items():assert sha(a.output_root/name)==digest
 print(json.dumps({'case_count':10,'file_count':len(index),'output':str(a.output_root)}))
if __name__=='__main__':main()
