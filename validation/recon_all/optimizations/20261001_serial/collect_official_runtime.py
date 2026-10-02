"""只读既有真实官方 recon-all 完整日志，提取整例和各命令墙钟。
--config JSON提供cases(id,subject)、programs(名字:绝对路径)、fs_time脚本路径。
--output必须不存在；输出源码/日志/程序SHA、命令原文行号、FSTIME e/S/U。
e为墙钟，S/U为累积CPU秒；并行半球和父子包装不相加，不算新配对加速。
本工具不执行官方程序，不读取许可证，不把旧记录改标为当前结果。
"""
import argparse,datetime,hashlib,json,re
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=json.loads(a.config.read_text())
    fstime=Path(c["fs_time"]);text=fstime.read_text()
    if "e %e" not in text or "/usr/bin/time" not in text:raise ValueError("unverified timing definition")
    result={"scope":"archived official same-subject full runs; not new paired benchmark",
        "script_sha256":sha(__file__),"config_sha256":sha(a.config),"fs_time_sha256":sha(fstime),
        "time_definition":{"e":"elapsed wall seconds","S":"accumulated system CPU seconds","U":"accumulated user CPU seconds",
         "restriction":"nested wrappers and concurrent hemispheres are separate; do not sum as whole wall"},
        "program_sha256":{n:sha(path) for n,path in c["programs"].items()},"subjects":{}}
    pat=re.compile(r"^@#@FSTIME\s+(\S+)\s+(\S+)\s+N\s+\d+\s+e\s+(\S+)\s+S\s+(\S+)\s+U\s+(\S+)")
    date_re=r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d+:\d+:\d+\s+\S+\s+\d{4}"
    for case in c["cases"]:
        root=Path(case["subject"]);log=root/"scripts/recon-all.log";lines=log.read_text().splitlines()
        done=root/"scripts/recon-all.done";metadata=done.read_text();times={}
        for key in ["START_TIME","END_TIME"]:
            m=re.search(key+r"\s+("+date_re+r")",metadata)
            if not m:raise ValueError("missing "+key)
            words=m[1].split();date=" ".join(words[:4]+words[5:]);times[key]=datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %Y")
        rows=[];stage=None
        for i,line in enumerate(lines):
            if line.startswith("#@#") and re.search(date_re,line):stage={"line":i+1,"raw":line}
            m=pat.match(line)
            if not m:continue
            stamp,program,e,s,u=m.groups()
            cmd=None;rx=re.compile(r"^\s*(?:\S*/)?"+re.escape(program)+r"(?:\s|$)")
            for j in range(i-1,-1,-1):
                if rx.match(lines[j]) and lines[j].strip()!=program+" done":
                    cmd={"line":j+1,"raw":lines[j]};break
            rows.append({"line":i+1,"raw":line,"timestamp":stamp,"program":program,
             "elapsed_wall_seconds":float(e),"cpu_system_seconds":float(s),"cpu_user_seconds":float(u),
             "command":cmd,"stage_marker":stage})
        result["subjects"][case["id"]]={"subject":str(root),"log_sha256":sha(log),"done_sha256":sha(done),
             "historical_metadata":metadata,"whole_wall_seconds":(times["END_TIME"]-times["START_TIME"]).total_seconds(),
             "command_wall_rows":rows,"synth_device_evidence":[{"line":i+1,"raw":line} for i,line in enumerate(lines)
                if "Configuring model on" in line or "hiding all CUDA_VISIBLE_DEVICES" in line]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({s:{"whole_wall_seconds":j["whole_wall_seconds"],"commands":len(j["command_wall_rows"])} for s,j in result["subjects"].items()}))
if __name__=="__main__":main()
