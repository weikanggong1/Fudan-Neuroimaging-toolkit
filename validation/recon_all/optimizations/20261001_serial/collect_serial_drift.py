"""只读两例成功整例，复用固定比较器定位体积前段及white.preaparc输入变化。

--config为整例比较JSON（cases、scripts_dir、code_commit），--output为新的报告路径。
体积处于conform网格；表面使用surface RAS mm；LTA保留原类型，不把矩阵元素当mm。
输出JSON含各体积不同数/最大/P99误差、dtype/几何、有序网格、两组LTA矩阵和
自动目标强度参数。固定比较器容差不变，缺文件或运行未完成抛异常。
属于benchmark诊断，无独立官方等价CLI，不写回生产subject。
"""
import argparse
import importlib.util
import json
from pathlib import Path


def matrix(path):
    """解析必填LTA文件path，返回其原类型4×4数值列表；矩阵缺失时抛异常。"""
    lines=path.read_text().splitlines()
    first=next(i for i,line in enumerate(lines) if line.strip()=="1 4 4")
    return [[float(x) for x in line.split()] for line in lines[first+1:first+5]]


def parameters(path):
    """读取必填autodet统计路径，返回参数名到数值的字典，单位遵循原统计定义。"""
    return {parts[0]:float(parts[1]) for line in path.read_text().splitlines()
            if len(parts:=line.split())==2}


def main():
    """读取具名CLI，绑定脚本/比较器/输入SHA并写新的只读诊断JSON。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    config=json.loads(a.config.read_text())
    source=Path(config["scripts_dir"])/"compare_complete_subject.py"
    spec=importlib.util.spec_from_file_location("fixed_subject_comparator",source)
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    names=("orig", "synthstrip", "nu", "T1", "brainmask", "norm", "brain", "wm.seg", "wm.asegedit",
           "wm", "filled", "aseg.presurf", "entowm", "mca-dura", "vsinus", "brain.finalsurfs")
    result={"scope":"current_self_generated_whole_chain; comparisons_only",
            "code_commit":config["code_commit"],"script_sha256":compare._sha256(Path(__file__)),
            "comparator_sha256":compare._sha256(source),"cases":{},
            "causal_attribution":"volume and parameter differences are observations; see four-way white diagnostic"}
    for case in config["cases"]:
        before,after=Path(case["baseline"]),Path(case["candidate"])
        for root in (before,after):
            if json.loads((root/"fnit-native-free-run.json").read_text())["status"]!="complete":
                raise ValueError("whole execution incomplete: "+str(root))
        row={"baseline":str(before),"candidate":str(after),"volumes":{},"meshes":{},"lta":{},"target_stats":{}}
        for name in names:
            first,second=[root/"mri"/(name+".mgz") for root in (before,after)]
            value=compare._volume(first,second)
            value.update(baseline_sha256=compare._sha256(first),candidate_sha256=compare._sha256(second))
            row["volumes"][name]=value
        for name in ("talairach.lta","synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta"):
            row["lta"][name]={kind:{"matrix":matrix(root/"mri/transforms"/name),
                               "sha256":compare._sha256(root/"mri/transforms"/name)}
                               for kind,root in (("baseline",before),("candidate",after))}
        for hemi in ("lh","rh"):
            for name in ("orig.nofix","orig.premesh","orig","white.preaparc"):
                row["meshes"][hemi+"."+name]=compare._surface(
                    before/"surf"/(hemi+"."+name),after/"surf"/(hemi+"."+name))
            name="autodet.gw.stats."+hemi+".dat"
            row["target_stats"][hemi]={kind:{"values":parameters(root/"surf"/name),
                                           "sha256":compare._sha256(root/"surf"/name)}
                                       for kind,root in (("baseline",before),("candidate",after))}
        result["cases"][case["id"]]=row
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
