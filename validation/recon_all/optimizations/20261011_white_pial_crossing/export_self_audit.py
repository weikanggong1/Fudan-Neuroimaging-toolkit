"""固定schema导出实际完整 selfpair 精确分类，不带私人路径或坐标。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import re

spec=importlib.util.spec_from_file_location("self_pair_qc_schema",Path(__file__).with_name("export_qc.py"))
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
exact,digest,number=schema.exact,schema.digest,schema.number
KINDS={"none","proper","edge_point_contact","coplanar_area_overlap","coplanar_edge_point_contact"}
INPUTS={"surf/lh.white","surf/autodet.gw.stats.lh.dat","label/lh.cortex.label","label/lh.cortex+hipamyg.label", "mri/brain.finalsurfs.mgz","mri/wm.mgz","mri/aseg.presurf.mgz"}


def boolean(value):
    if type(value) is not bool:raise ValueError("typed boolean required")
    return value


def state(value, name, faces):
    keys={"source_directed_pairs","float64_pairs","union_pairs","exact_classes","float64_nonproper_pairs", "source_nonproper_unordered_pairs","pairs_with_shared_vertex_ids","pairs_with_fixed_corners","rows","components", "state","enumeration_seconds","exact_audit_seconds"}
    exact(value,keys)
    if value["state"]!=name:raise ValueError("unknown state")
    result={k:number(value[k],True) for k in keys-{"exact_classes","rows","components","state","enumeration_seconds","exact_audit_seconds"}}
    result.update(state=name,enumeration_seconds=number(value["enumeration_seconds"]),exact_audit_seconds=number(value["exact_audit_seconds"]))
    exact(value["exact_classes"],KINDS);classes={k:number(v,True) for k,v in value["exact_classes"].items()}
    if type(value["rows"]) is not list or type(value["components"]) is not list:raise ValueError("complete lists required")
    rows=[];rkeys={"pair","exact_kind","source_directed_orientations","float64_positive","shared_vertex_ids", "shared_coordinate_corners","unique_corner_vertices","fixed_corner_vertices","all_corners_fixed","component"}
    for row in value["rows"]:
        exact(row,rkeys)
        if row["exact_kind"] not in KINDS or type(row["pair"]) is not list or len(row["pair"])!=2:raise ValueError("unknown relation/pair")
        pair=[number(v,True) for v in row["pair"]]
        if not pair[0]<pair[1]<faces:raise ValueError("ordered pair outside surface")
        r={k:number(row[k],True) for k in rkeys-{"pair","exact_kind","float64_positive","all_corners_fixed"}}
        r.update(pair=pair,exact_kind=row["exact_kind"],float64_positive=boolean(row["float64_positive"]),all_corners_fixed=boolean(row["all_corners_fixed"]))
        if r["source_directed_orientations"]>2 or r["shared_vertex_ids"]>3 or r["shared_coordinate_corners"]>3 or not 3<=r["unique_corner_vertices"]<=6 or r["fixed_corner_vertices"]>r["unique_corner_vertices"] or r["all_corners_fixed"]!=(r["fixed_corner_vertices"]==r["unique_corner_vertices"]):
            raise ValueError("inconsistent corner/source counts")
        if not r["source_directed_orientations"] and not r["float64_positive"]:raise ValueError("row absent from both positive sets")
        rows.append(r)
    if [tuple(r["pair"]) for r in rows]!=sorted({tuple(r["pair"]) for r in rows}):raise ValueError("rows not sorted unique")
    if classes!={k:sum(r["exact_kind"]==k for r in rows) for k in KINDS} or result["union_pairs"]!=len(rows) or result["source_directed_pairs"]!=sum(r["source_directed_orientations"] for r in rows) or result["float64_pairs"]!=sum(r["float64_positive"] for r in rows):raise ValueError("classes/pairs inconsistent")
    required_counts={"float64_nonproper_pairs":sum(r["float64_positive"] and r["exact_kind"]!="proper" for r in rows), "source_nonproper_unordered_pairs":sum(r["source_directed_orientations"]>0 and r["exact_kind"]!="proper" for r in rows), "pairs_with_shared_vertex_ids":sum(r["shared_vertex_ids"]>0 for r in rows), "pairs_with_fixed_corners":sum(r["fixed_corner_vertices"]>0 for r in rows)}
    if any(result[k]!=v for k,v in required_counts.items()):raise ValueError("classification aggregate inconsistent")
    components=[];ckeys={"component","pairs","unique_vertices","fixed_vertices","exact_classes"}
    for index,c in enumerate(value["components"]):
        exact(c,ckeys);exact(c["exact_classes"],KINDS)
        v={k:number(c[k],True) for k in ckeys-{"exact_classes"}}
        v["exact_classes"]={k:number(n,True) for k,n in c["exact_classes"].items()}
        group=[r for r in rows if r["component"]==index]
        if v["component"]!=index or v["pairs"]!=len(group) or not group or v["fixed_vertices"]>v["unique_vertices"] or v["exact_classes"]!={k:sum(r["exact_kind"]==k for r in group) for k in KINDS}:raise ValueError("component counts inconsistent")
        components.append(v)
    if any(r["component"]>=len(components) for r in rows):raise ValueError("missing component")
    return result|{"exact_classes":classes,"rows":rows,"components":components}


def export(*, report_path:Path,timer_path:Path,output:Path):
    """三个必填Path：只读完整审查和冷计时，新公共JSON须不存在。

    返回固定typed dict，保留全部类别/面对/组件和原始时间，不改变
    quality门、不判定整例。未知字段/路径字符串、非法类型/计数或
    哈希、false伪退出码、错误资源或来源明确失败且不写文件。
    """
    if output.exists():raise FileExistsError(output)
    raw=report_path.read_bytes();d=json.loads(raw)
    keys={"schema_version","execution_status","scope","script_sha256","pair_helper_sha256","exact_audit_sha256", "input_sha256","terminal_sha256","candidate_sha256","inputs_unchanged","coordinate_space","coordinate_unit", "vertices","faces","fixed_vertices","threads","cpu_affinity","states","source_version_bindings", "wall_seconds_before_report_write","peak_process_cpu_rss_bytes","gpu_computation","gpu_memory_bytes", "original_stage_program_called","overall_metric_equivalence","quality_gate_changed"}
    exact(d,keys)
    if type(d["schema_version"]) is not int or d["schema_version"]!=1 or d["execution_status"]!="complete_readonly_exact_pair_audit" or d["scope"]!="all existing self-detector positive pairs on frozen own entry and candidate; no whole or new optimizer":raise ValueError("unknown scope")
    if d["coordinate_space"]!="surface_RAS" or d["coordinate_unit"]!="mm" or d["overall_metric_equivalence"]!="not_assessed" or d["inputs_unchanged"] is not True or any(d[k] is not False for k in ("gpu_computation","original_stage_program_called","quality_gate_changed")) or d["gpu_memory_bytes"] is not None:raise ValueError("unexpected route/space/quality claims")
    if type(d["threads"]) is not int or d["threads"]!=4 or d["cpu_affinity"]!=[112,113,114,115]:raise ValueError("fixed four-core budget required")
    exact(d["input_sha256"],INPUTS)
    input_map={k:digest(v) for k,v in d["input_sha256"].items()}
    hashes={k:digest(d[k]) for k in ("script_sha256","pair_helper_sha256","exact_audit_sha256","terminal_sha256","candidate_sha256")}
    sources={}
    if type(d["source_version_bindings"]) is not dict or not d["source_version_bindings"]:raise ValueError("actual source modules required")
    for name,value in d["source_version_bindings"].items():
        if not re.fullmatch(r"fnit\.recon_all\.[A-Za-z0-9_.]+",name):raise ValueError("unknown module identifier")
        sources[name]=digest(value)
    mesh={k:number(d[k],True) for k in ("vertices","faces","fixed_vertices")}
    if not mesh["vertices"] or not mesh["faces"] or mesh["fixed_vertices"]>mesh["vertices"]:raise ValueError("invalid mesh counts")
    if type(d["states"]) is not list or len(d["states"])!=2:raise ValueError("exact two states required")
    states=[state(v,name,mesh["faces"]) for v,name in zip(d["states"],("entry","candidate"))]
    raw_timer=timer_path.read_bytes();t=json.loads(raw_timer)
    exact(t,{"schema_version","scope","command_sha256","timer_script_sha256","gpu_memory_bytes", "simultaneous_parent_child_peak_cpu_memory_bytes","execution_status","child_exit_code","cold_cli_wall_seconds","max_child_cpu_rss_bytes"})
    if type(t["schema_version"]) is not int or t["schema_version"]!=1 or t["scope"]!="single frozen Python benchmark cold CLI; not end-to-end recon-all" or t["execution_status"]!="complete" or type(t["child_exit_code"]) is not int or t["child_exit_code"]!=0:raise ValueError("completed cold CLI required")
    if t["gpu_memory_bytes"] is not None or t["simultaneous_parent_child_peak_cpu_memory_bytes"] is not None:raise ValueError("unmeasured resources must remain null")
    timer={"child_exit_code":0,"cold_cli_wall_seconds":number(t["cold_cli_wall_seconds"]), "max_child_cpu_rss_bytes":number(t["max_child_cpu_rss_bytes"],True), "command_sha256":digest(t["command_sha256"]),"timer_script_sha256":digest(t["timer_script_sha256"]), "gpu_memory_bytes":None,"simultaneous_parent_child_peak_cpu_memory_bytes":None,"raw_timer_sha256":hashlib.sha256(raw_timer).hexdigest()}
    result={"schema_version":1,"execution_status":d["execution_status"],"scope":d["scope"],**hashes,**mesh, "input_sha256":input_map,"inputs_unchanged":True,"coordinate_space":"surface_RAS","coordinate_unit":"mm", "threads":4,"cpu_affinity":[112,113,114,115],"states":states,"source_version_bindings":sources, "wall_seconds_before_report_write":number(d["wall_seconds_before_report_write"]), "peak_process_cpu_rss_bytes":number(d["peak_process_cpu_rss_bytes"],True), "gpu_computation":False,"gpu_memory_bytes":None,"original_stage_program_called":False, "overall_metric_equivalence":"not_assessed","quality_gate_changed":False,"cold_cli":timer, "raw_report_sha256":hashlib.sha256(raw).hexdigest(),"raw_timer_sha256":hashlib.sha256(timer_path.read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n");return result
