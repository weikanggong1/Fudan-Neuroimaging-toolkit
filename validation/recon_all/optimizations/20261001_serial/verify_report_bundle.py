"""核验本轮公开报告包，不运行影像计算或读取外部资源。
输入--root为报告根、--manifest为带相对path/size_bytes/sha256的JSON；
可选--output为新JSON。输出stdout和可选文件包含逐文件失败列表、数量及清单SHA。
禁止相对路径越界和符号链接逃逸；文件丢失、大小或SHA不符时非零退出。
不解析pstats/pickle，不加载MRI/模型。属于benchmark内部步骤，无官方独立CLI。
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def verify(root: Path, manifest: Path) -> dict:
    """核验root内文件的原始字节；返回JSON兼容结果，不修改文件或判断指标等效。"""
    root=root.resolve()
    raw=manifest.read_bytes()
    entries=json.loads(raw)["files"]
    failures=[]
    for e in entries:
        relative=Path(e["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("report path outside root")
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError("symlink escapes report root")
        if not path.is_file():
            failures.append({"path":e["path"],"reason":"missing"});continue
        contents=path.read_bytes()
        if len(contents)!=e["size_bytes"] or hashlib.sha256(contents).hexdigest()!=e["sha256"]:
            failures.append({"path":e["path"],"reason":"size_or_sha256"})
    return {"scope":"raw public report bytes; not numerical equivalence or isolated deployment",
            "manifest_sha256":hashlib.sha256(raw).hexdigest(),
            "files_checked":len(entries),"failures":failures,"status":"passed" if not failures else "failed"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.output and a.output.exists():raise FileExistsError(a.output)
    result=verify(a.root,a.manifest)
    text=json.dumps(result,indent=2)+"\n"
    if a.output:a.output.write_text(text)
    print(text,end="")
    if result["failures"]:raise SystemExit(1)

if __name__=="__main__":main()
