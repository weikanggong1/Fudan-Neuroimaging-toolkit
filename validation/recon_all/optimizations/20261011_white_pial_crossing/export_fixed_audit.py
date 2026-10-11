"""固定子结构审查的完整typed导出；仅数值/枚举/哈希与公开网格编号。"""
from pathlib import Path
import hashlib
import importlib.util
import json

spec=importlib.util.spec_from_file_location("fixed_qc_schema",Path(__file__).with_name("export_qc.py"))
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
exact,digest,number=schema.exact,schema.digest,schema.number
FACE={"none","proper_interior","closed_contact","coplanar_contact"}
EDGE={"none","transverse_interiors","closed_contact","collinear_overlap"}
INPUTS={"surf/lh.white","surf/autodet.gw.stats.lh.dat","label/lh.cortex.label","label/lh.cortex+hipamyg.label","mri/brain.finalsurfs.mgz","mri/wm.mgz","mri/aseg.presurf.mgz"}


def boolean(value):
    if type(value) is not bool:raise ValueError("typed boolean required")
    return value


def edge(value,vertices):
    if type(value) is not list or len(value)!=2:raise ValueError("edge pair required")
    result=[number(v,True) for v in value]
    if result[0]==result[1] or max(result)>=vertices:raise ValueError("nondegenerate bounded edge required")
    return result


def export(*,report_path:Path,timer_path:Path,output:Path):
    """三个必填Path：完整固定障碍审查、冷CLI收据、新公共JSON。

    只发布固定字段的全部面对/固定来源/精确交会/时间和SHA，保留0
    障碍或未解决结果，不判整例。未知字段/私人字符串、非法类型或
    计数、false退出码、错资源明确失败且不写文件；已有output不覆盖。
    """
    if output.exists():raise FileExistsError(output)
    raw=report_path.read_bytes();d=json.loads(raw)
    keys={"schema_version","execution_status","scope","script_sha256","exact_report_sha256","input_sha256","terminal_sha256","candidate_sha256","inputs_unchanged","coordinate_space","coordinate_unit","vertices","faces","fixed_vertices","threads","cpu_affinity","states","wall_seconds_before_report_write","peak_process_cpu_rss_bytes","gpu_computation","gpu_memory_bytes","original_stage_program_called","quality_gate_changed","overall_metric_equivalence"}
    exact(d,keys)
    if type(d["schema_version"]) is not int or d["schema_version"]!=1 or d["execution_status"]!="complete_readonly_fixed_structure_audit" or d["scope"]!="all fixed substructures of existing frozen exact selfpair lists; no joint optimizer or new quality gate":raise ValueError("unknown fixed audit scope")
    if d["inputs_unchanged"] is not True or d["coordinate_space"]!="surface_RAS" or d["coordinate_unit"]!="mm" or d["overall_metric_equivalence"]!="not_assessed" or any(d[k] is not False for k in ("gpu_computation","original_stage_program_called","quality_gate_changed")) or d["gpu_memory_bytes"] is not None:raise ValueError("unexpected space/route/quality claims")
    if type(d["threads"]) is not int or d["threads"]!=4 or d["cpu_affinity"]!=[112,113,114,115]:raise ValueError("four-core fixed budget required")
    exact(d["input_sha256"],INPUTS);inputs={k:digest(v) for k,v in d["input_sha256"].items()}
    hashes={k:digest(d[k]) for k in ("script_sha256","exact_report_sha256","terminal_sha256","candidate_sha256")}
    mesh={k:number(d[k],True) for k in ("vertices","faces","fixed_vertices")}
    if not mesh["vertices"] or not mesh["faces"] or mesh["fixed_vertices"]>mesh["vertices"]:raise ValueError("bounded positive mesh required")
    if type(d["states"]) is not list or len(d["states"])!=2:raise ValueError("two actual states required")
    states=[]
    for state,name in zip(d["states"],("entry","candidate")):
        exact(state,{"state","pairs","pairs_with_immutable_proper_obstruction","pairs_with_any_immutable_contact","face_edge_classes","edge_edge_classes","exact_checks_seconds","rows"})
        if state["state"]!=name or type(state["rows"]) is not list:raise ValueError("known state/list required")
        rows=[]
        for row in state["rows"]:
            exact(row,{"pair","fixed_corners","face_edge_checks","edge_edge_checks","immutable_proper_obstruction","immutable_contact_obstruction"})
            pair=edge(row["pair"],mesh["faces"])
            if not pair[0]<pair[1]:raise ValueError("canonical ordered pair required")
            if any(type(row[k]) is not list for k in ("fixed_corners","face_edge_checks","edge_edge_checks")):raise ValueError("complete typed lists required")
            corners=[]
            for c in row["fixed_corners"]:
                exact(c,{"vertex","fixed_source","equals_own_white"});v=number(c["vertex"],True)
                if v>=mesh["vertices"] or c["fixed_source"] not in {"rip","outside_cortex","rip_and_outside_cortex"}:raise ValueError("bounded corner/fixed provenance required")
                corners.append({"vertex":v,"fixed_source":c["fixed_source"],"equals_own_white":boolean(c["equals_own_white"])})
            if [c["vertex"] for c in corners]!=sorted({c["vertex"] for c in corners}) or len(corners)>6:raise ValueError("unique sorted bounded corners required")
            face=[];edges=[]
            for v in row["face_edge_checks"]:
                exact(v,{"fixed_face","fixed_edge","kind"});f=number(v["fixed_face"],True)
                if f not in pair or v["kind"] not in FACE:raise ValueError("known fixed face relation required")
                face.append({"fixed_face":f,"fixed_edge":edge(v["fixed_edge"],mesh["vertices"]),"kind":v["kind"]})
            for v in row["edge_edge_checks"]:
                exact(v,{"first_edge","second_edge","kind"})
                if v["kind"] not in EDGE:raise ValueError("known fixed edge relation required")
                edges.append({"first_edge":edge(v["first_edge"],mesh["vertices"]),"second_edge":edge(v["second_edge"],mesh["vertices"]),"kind":v["kind"]})
            proper=boolean(row["immutable_proper_obstruction"]);contact=boolean(row["immutable_contact_obstruction"])
            if proper!=any(v["kind"]=="proper_interior" for v in face) or contact!=any(v["kind"]!="none" for v in face+edges):raise ValueError("obstruction type contradiction")
            rows.append({"pair":pair,"fixed_corners":corners,"face_edge_checks":face,"edge_edge_checks":edges,"immutable_proper_obstruction":proper,"immutable_contact_obstruction":contact})
        if [tuple(r["pair"]) for r in rows]!=sorted({tuple(r["pair"]) for r in rows}):raise ValueError("canonical complete pair list required")
        exact(state["face_edge_classes"],FACE);exact(state["edge_edge_classes"],EDGE)
        face={k:number(v,True) for k,v in state["face_edge_classes"].items()};edges={k:number(v,True) for k,v in state["edge_edge_classes"].items()}
        if face!={k:sum(v["kind"]==k for r in rows for v in r["face_edge_checks"]) for k in FACE} or edges!={k:sum(v["kind"]==k for r in rows for v in r["edge_edge_checks"]) for k in EDGE}:raise ValueError("fixed relation totals inconsistent")
        counts={k:number(state[k],True) for k in ("pairs","pairs_with_immutable_proper_obstruction","pairs_with_any_immutable_contact")}
        if counts["pairs"]!=len(rows) or counts["pairs_with_immutable_proper_obstruction"]!=sum(r["immutable_proper_obstruction"] for r in rows) or counts["pairs_with_any_immutable_contact"]!=sum(r["immutable_contact_obstruction"] for r in rows):raise ValueError("obstruction aggregates inconsistent")
        states.append({"state":name,**counts,"face_edge_classes":face,"edge_edge_classes":edges,"exact_checks_seconds":number(state["exact_checks_seconds"]),"rows":rows})
    raw_timer=timer_path.read_bytes();t=json.loads(raw_timer)
    exact(t,{"schema_version","scope","command_sha256","timer_script_sha256","gpu_memory_bytes","simultaneous_parent_child_peak_cpu_memory_bytes","execution_status","child_exit_code","cold_cli_wall_seconds","max_child_cpu_rss_bytes"})
    if type(t["schema_version"]) is not int or t["schema_version"]!=1 or t["scope"]!="single frozen Python benchmark cold CLI; not end-to-end recon-all" or t["execution_status"]!="complete" or type(t["child_exit_code"]) is not int or t["child_exit_code"]!=0:raise ValueError("cold CLI complete required")
    if t["gpu_memory_bytes"] is not None or t["simultaneous_parent_child_peak_cpu_memory_bytes"] is not None:raise ValueError("unmeasured resources must remain null")
    timer={"child_exit_code":0,"cold_cli_wall_seconds":number(t["cold_cli_wall_seconds"]),"max_child_cpu_rss_bytes":number(t["max_child_cpu_rss_bytes"],True),"command_sha256":digest(t["command_sha256"]),"timer_script_sha256":digest(t["timer_script_sha256"]),"gpu_memory_bytes":None,"simultaneous_parent_child_peak_cpu_memory_bytes":None,"raw_timer_sha256":hashlib.sha256(raw_timer).hexdigest()}
    result={"schema_version":1,"execution_status":d["execution_status"],"scope":d["scope"],**hashes,**mesh,"input_sha256":inputs,"inputs_unchanged":True,"coordinate_space":"surface_RAS","coordinate_unit":"mm","threads":4,"cpu_affinity":[112,113,114,115],"states":states,"wall_seconds_before_report_write":number(d["wall_seconds_before_report_write"]),"peak_process_cpu_rss_bytes":number(d["peak_process_cpu_rss_bytes"],True),"gpu_computation":False,"gpu_memory_bytes":None,"original_stage_program_called":False,"quality_gate_changed":False,"overall_metric_equivalence":"not_assessed","cold_cli":timer,"raw_report_sha256":hashlib.sha256(raw).hexdigest(),"raw_timer_sha256":hashlib.sha256(raw_timer).hexdigest()}
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n");return result
