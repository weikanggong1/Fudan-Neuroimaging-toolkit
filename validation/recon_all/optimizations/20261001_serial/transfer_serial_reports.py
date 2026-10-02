"""只传输 FNIT benchmark 报告，不传 MRI、权重或许可证。
--mapping JSON为source/target条目；source须是已授权远端绝对路径。
--output为新的tar.gz；--label记录范围/代码；输出内含逐文件SHA及大小清单。
可传JSON/CSV/LOG/PNG/PY/SH/MD/TXT；IMG数据、模型和许可证路径拒绝。
该脚本在SSH授权主机执行，不建立连接、不运行脑影像算法。
"""
import argparse,hashlib,io,json,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--label",required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    mapping=json.loads(a.mapping.read_text());seen=set();entries=[]
    with tarfile.open(a.output,"w:gz") as tar:
        for row in mapping:
            src=Path(row["source"]);target=Path(row["target"])
            if not src.is_absolute() or target.is_absolute() or ".." in target.parts:
                raise ValueError("invalid report path")
            if src.suffix.lower() not in {".json",".csv",".log",".png",".py",".sh",".md",".txt"}:
                raise ValueError("MRI/weights/binary export prohibited")
            if "license" in src.name.lower() or target.as_posix() in seen:
                raise ValueError("license or duplicate target prohibited")
            seen.add(target.as_posix());data=src.read_bytes()
            info=tarfile.TarInfo(target.as_posix());info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
            entries.append({"source":str(src),"target":target.as_posix(),"size_bytes":len(data),
                            "sha256":hashlib.sha256(data).hexdigest()})
        result={"scope":a.label,"mapping_sha256":hashlib.sha256(a.mapping.read_bytes()).hexdigest(),
            "collector_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"entries":entries}
        data=(json.dumps(result,indent=2)+"\n").encode();info=tarfile.TarInfo("transfer_manifest.json")
        info.size=len(data);tar.addfile(info,io.BytesIO(data))
    print(json.dumps({"archive_sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"files":len(entries)}))
if __name__=="__main__":main()
