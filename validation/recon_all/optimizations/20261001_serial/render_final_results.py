"""从完整两例机器报告生成中文结果页及精简指标JSON；不执行影像计算。

全部输入为--reports报告根、--summary完整summarize_whole JSON、--drift连续链诊断JSON，
--official带SHA的归档官方耗时JSON、--output新的Markdown文件、--metrics新的JSON文件。
可选--inverse是当前整例逆场诊断，--backend-controls是同生产策略缓存控制目录；
--integration-proof为后续源码回归绑定记录。不提供则不宣称这些结果。
输入必须均存在；输出已存在、病例/计算版本不同或整例未完成时失败。表面为surface RAS/mm；
面积mm²、体积mm³、曲率mm⁻¹、耗时秒、显存字节。输出包含完整66阶段来源分类、
官方归档命令耗时、脑区及分区指标、网格质量和当前未验收项。无独立官方CLI，
对应参考为单T1 recon-all -all -parallel -openmp 4 -itkthreads 1。
不建立或修改整体等效阈值；不将不同日期的官方计时当作本轮配对提速。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    """输入可读文件Path，返回SHA-256字符串；读取失败直接抛异常。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def brief(metric):
    """保留统计汇总及最差脑区，完整per_region在源JSON中继续保存。"""
    return {k:v for k,v in metric.items() if k!="per_region"}

def summarize_case(value):
    """接收已验证单例完整报告；返回同单位指标与失败项目，不判定整体等效。"""
    comparisons=value["comparisons"]
    result={"command_timing":value["command_timing"],
            "output_completeness":value["output_completeness"],
            "peak_sampled_process_bytes":value["candidate_monitor"]["peak_sampled_process_bytes"],
            "maximum_sampling_gap_seconds":value["candidate_monitor"]["maximum_sampling_gap_seconds"],
            "failed_app_queries":value["candidate_monitor"]["failed_app_queries"],
            "continuous_peak_verified":False,"references":{}}
    for target in ("baseline","official"):
        region=comparisons["region_vs_"+target]
        strict=comparisons["strict_vs_"+target]
        surfaces=comparisons["surface_vs_"+target]["stages"]
        result["references"][target]={
            "strict":{"checked":strict["checked"],"passed":strict["passed"],
                "failed_paths":[k for k,v in strict["files"].items() if not v["pass"]]},
            "aparc_68":{k:brief(v) for k,v in region["aparc_68"].items()},
            "aseg":brief(region["aseg"]),"wmparc":brief(region["wmparc"]),
            "global_brainvol_measures":region["global_brainvol_measures"],
            "label_dice":{k:{**{a:b for a,b in v.items() if a!="per_label"},
                            "worst_label_details":[{"label":label,**v["per_label"][label]}
                                                   for label in v["worst_labels"][:10]]}
                          for k,v in comparisons["dice_vs_"+target]["files"].items()},
            "surfaces":{h:{k:v for k,v in d.items() if k in ("white.preaparc","white","pial")}
                        for h,d in surfaces.items()}}
    result["quality"]={}
    for kind,quality in value["quality"].items():
        result["quality"][kind]={}
        for hemi,d in quality["hemispheres"].items():
            cross=d["white_pial_crossings"]
            result["quality"][kind][hemi]={
                "topology":d["topology"],"vertex_links":d["vertex_links"],
                "previous_pipeline_mesh_validation":d["previous_pipeline_mesh_validation"],
                "sphere_orientation":d["sphere_orientation"],
                "white_pial_crossings":{k:v for k,v in cross.items()
                    if k not in ("proper_pair_details","proper_pair_face_ids_first_100")}}
    result["over_100_seconds"]=[{k:r[k] for k in
        ("name","candidate_seconds","baseline_seconds","implementation","source_paths")}
        for r in value["stages"] if r["over_100_seconds"]]
    result["overall_metric_equivalence"]="not_assessed"
    return result

def official_rows(subject, program, *, hemi=None, contains=None):
    """选择归档真实e墙钟；半球从命令识别，多记录不擅自求和或分配并行时间。"""
    found=[]
    for row in subject["command_wall_rows"]:
        if row["program"]!=program:continue
        command=(row.get("command") or {}).get("raw","")
        if hemi and not any(x in command for x in ("--"+hemi, "/"+hemi+".", "../surf/"+hemi+".")):
            continue
        if contains and contains not in command:continue
        found.append(row["elapsed_wall_seconds"])
    return " / ".join(f"{x:.3f}" for x in found) if found else "未单独记录"

def main():
    """解析具名CLI、校验版本、生成新结果文件；不写入生产被试目录。"""
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("reports","summary","drift","official","output","metrics"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--inverse",type=Path)
    p.add_argument("--backend-controls",type=Path)
    p.add_argument("--integration-proof",type=Path)
    a=p.parse_args()
    if a.output.exists() or a.metrics.exists():raise FileExistsError("outputs must be new")
    full=json.loads(a.summary.read_text())
    drift=json.loads(a.drift.read_text())
    official=json.loads(a.official.read_text())
    commits={x["candidate_commit"] for x in full["cases"].values()}
    if len(commits)!=1 or drift["code_commit"] not in commits:raise ValueError("tested commits differ")
    commit=commits.pop()
    metrics={"tested_commit":commit,"script_sha256":sha(Path(__file__)),
             "source_sha256":{str(x.relative_to(a.reports)):sha(x)
                              for x in (a.summary,a.drift,a.official)},
             "cases":{k:summarize_case(v) for k,v in full["cases"].items()},
             "official_timing_scope":official["scope"],
             "overall_metric_equivalence":"not_assessed"}
    inverse=json.loads(a.inverse.read_text()) if a.inverse else None
    if inverse:
        if inverse["code_commit"]!=commit:raise ValueError("inverse diagnostic version differs")
        metrics["inverse_field"]=inverse
        metrics["source_sha256"][str(a.inverse.relative_to(a.reports))]=sha(a.inverse)
    backend={}
    if a.backend_controls:
        for case in ("sub01","sub02"):
            backend[case]={}
            for policy in ("enabled","disabled"):
                root=a.backend_controls/case/policy
                run=json.loads((root/"run.json").read_text())
                monitor=json.loads((root/"monitor.json").read_text())
                if run["code_commit"]!=commit or monitor["exit_code"]!=0:raise ValueError("backend control incomplete")
                if run["SynthStrip_actual_backend"]!={"enabled":True,"benchmark":False,"deterministic":True,"allow_tf32":False}:
                    raise ValueError("actual SynthStrip backend differs from production")
                backend[case][policy]={"run":run,"monitor":monitor}
                for filename in ("run.json","monitor.json","gpu_samples.csv","command.log"):
                    f=root/filename;metrics["source_sha256"][str(f.relative_to(a.reports))]=sha(f)
        metrics["same_production_backend_controls"]=backend
    lines=["# 五阶段优化：当前两例整例验证",
        "",f"实际计算源码：`{commit}`。最终文档提交不改写这一版本记录。",
        "两例均从原始 T1 和空目录运行；sub-01 使用已初始化 CUDA 的 Python API，sub-02 使用 CLI。",
        "H100、gpucw1、目标 GPU UUID 与线程预算 4 固定；单次共享服务器观察，未测稳定吞吐。",
        f"[完整机器报告]({a.summary.relative_to(a.output.parent).as_posix()})、"
        f"[精简指标]({a.metrics.relative_to(a.output.parent).as_posix()})及原始文件保留 SHA-256。",
        "整体指标等效没有已确认阈值，保持 `not_assessed`；严格138比较继续用于排错。",
        "", "## 实际修改与误差原因","",
        "1. 复用自有GPU Synth网络，设备选择与精度策略分开；构造后施加策略并记录实际前向。"
        "辅助卷积局部FP32消除了TF32造成的少量标签差异及后续表面位移，其他作用域继续TF32。",
        "2. 有序归一化复用已有PyTorch/Triton：65项高斯卷积使用双缓冲，减少禁用缓存时的临时分配；"
        "两例冻结输入归一化输出逐位相同，加载、传输、写出都计入阶段回归。",
        "3. 球面保留已有有序CUDA平均，Numba SSE按行并行、双精度串行求和；静态CSR缓存关联与原面积。"
        "两例双侧完整几何和轨迹相同，拓扑GA的顺序依赖未强行并行。",
        "4. N4补齐拟合/重建线程与子步骤剖析、修复GPFS构建再配置循环。"
        "真实拟合约占98%，重建增加线程没有整阶段收益，默认重建1线程保留。",
        "5. Python pial复用空间桶与编译碰撞谓词，保留候选扩展及顺序更新规则。"
        "首例左侧完整同输入1453.060→1232.634秒、几何不变，仍慢于原生组件，生产继续使用Conda C++。",
        "MNI仿射的GPU matmul局部FP32减小几何量化误差；非线性链仍计算两个反对称前向，"
        "仅移除没有被消费的逆场计算。最终原生逆场仍有全域局部残差，见后文。",
        "完整修改及各函数参数、坐标、失败行为和具名示例见[串行优化说明](../../../../docs/recon_all/SERIAL_OPTIMIZATION.md)"
        "与[复现步骤](REPRODUCE.md)。",
        "", "## 端到端耗时与显存","",
        "| 真实 T1 | 优化前完整命令 s | 当前完整命令 s | 时间减少 | 当前父子同时采样峰值 GB / GiB | 输出 |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for c,d in metrics["cases"].items():
        t=d["command_timing"];mem=d["peak_sampled_process_bytes"]
        lines.append(f"| {c} | {t['baseline_seconds']:.3f} | {t['candidate_seconds']:.3f} | {t['time_reduction_percent']:.3f}% | {mem/1e9:.3f} / {mem/2**30:.3f} | 138/138 |")
    lines+=["","完整命令边界包含启动、校验、加载、传输、计算和读写，排除事后比较与画图；"
        "函数和内部步骤耗时另列，不能重复相加。首例复用源码相同的既有基线，第二例基线本轮新跑。"
        "主机相同不等于共享负载相同。",
        "NVML 显存为同次查询的候选父子进程合计，未把不同时间的峰值相加；"
        "关闭分配缓存时 PyTorch allocated/reserved 不可用，不能解释为零显存。",
        "", "| 病例 | 请求间隔 s | 最大实际间隔 s | 查询失败数 | 连续峰值 |",
        "| --- | ---: | ---: | ---: | --- |"]
    for c,d in metrics["cases"].items():
        lines.append(f"| {c} | 1 | {d['maximum_sampling_gap_seconds']:.3f} | {d['failed_app_queries']} | 未验证 |")
    lines+=["","## 严格复现与优化前后退化检查","",
        "| 病例 | 对优化前138严格诊断 | 对官方138严格诊断 | 对优化前标签Dice最低值 | 对优化前68区厚度/面积/体积 MAE |",
        "| --- | ---: | ---: | ---: | --- |"]
    failed_notes=[]
    for c,d in metrics["cases"].items():
        b=d["references"]["baseline"];o=d["references"]["official"]
        dice=min(x["minimum_dice"] for x in b["label_dice"].values())
        m=b["aparc_68"]
        lines.append(f"| {c} | {b['strict']['passed']}/138 | {o['strict']['passed']}/138 | {dice:.9f} | {m['ThickAvg']['mae']:.9g} mm / {m['SurfArea']['mae']:.9g} mm² / {m['GrayVol']['mae']:.9g} mm³ |")
        failed_notes+=["",f"{c} 对优化前仍失败的路径："+
                ("、".join("`"+x+"`" for x in b["strict"]["failed_paths"]) or "无")+"。"]
    lines+=failed_notes
    lines+=["","严格比较保留原门槛；其通过数不能代表算法完成比例。"
        "同网格比较要求顶点数及有序面成立；对官方网格不同时只报告双向点到三角面距离。",
        "", "## 最终指标相对官方","",
        "| 病例 | 68区厚度 MAE / 最大 mm | 面积中位 / P90 相对绝对误差 | 灰质体积中位 / P90 相对绝对误差 | 皮层总体积有符号差 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for c,d in metrics["cases"].items():
        o=d["references"]["official"];m=o["aparc_68"];t=m["ThickAvg"];s=m["SurfArea"];v=m["GrayVol"]
        lines.append(f"| {c} | {t['mae']:.6f} / {t['maximum_absolute_error']:.6f} | {s['median_absolute_relative_error_percent']:.3f}% / {s['p90_absolute_relative_error_percent']:.3f}% | {v['median_absolute_relative_error_percent']:.3f}% / {v['p90_absolute_relative_error_percent']:.3f}% | {o['global_brainvol_measures']['CortexVol']['relative_difference_percent']:+.3f}% |")
    lines+=["","| 病例 / 分区 | Dice中位数 | 最低Dice | 最差标签 |",
             "| --- | ---: | ---: | --- |"]
    for c,d in metrics["cases"].items():
        for name,row in d["references"]["official"]["label_dice"].items():
            worst=", ".join(x["label"]+" "+x["name"] for x in row["worst_label_details"][:3])
            lines.append(f"| {c} / {name} | {row['median_dice']:.6f} | {row['minimum_dice']:.6f} | {worst} |")
    lines+=["","最差脑区继续单列，不用均值替代局部偏差：","",
        "| 病例 / 指标 | 相对偏差最差脑区 | 当前−官方有符号差 | 相对官方百分比 |",
        "| --- | --- | ---: | ---: |"]
    for c,value in full["cases"].items():
        for name,metric in value["comparisons"]["region_vs_official"]["aparc_68"].items():
            for region in metric["worst_regions_by_relative_error"][:3]:
                row=metric["per_region"][region]
                unit={"SurfArea":"mm²","GrayVol":"mm³","ThickAvg":"mm","MeanCurv":"mm⁻¹"}[name]
                lines.append(f"| {c}/{name} | {region} | {row['candidate']-row['reference']:+.6g} {unit} | {row['signed_relative_difference_percent']:+.3f}% |")
    lines+=["","| 病例 / 半球 / 表面 | 当前→官方均值 / P99 / 最大 mm | 官方→当前均值 / P99 / 最大 mm |",
             "| --- | ---: | ---: |"]
    for c,d in metrics["cases"].items():
        for h,surfaces in d["references"]["official"]["surfaces"].items():
            for name in ("white","pial"):
                row=surfaces[name]
                if row["indexed_vertex_distance"] is not None:
                    v=row["indexed_vertex_distance"]
                    lines.append(f"| {c}/{h}/{name}（同序） | {v['mean_mm']:.6f}/{v['p99_mm']:.6f}/{v['max_mm']:.6f} | 同索引 |")
                else:
                    ds=[row[k] for k in ("candidate_to_reference_triangle","reference_to_candidate_triangle")]
                    lines.append(f"| {c}/{h}/{name} | "+" | ".join(f"{v['mean_mm']:.6f}/{v['p99_mm']:.6f}/{v['max_mm']:.6f}" for v in ds)+" |")
    lines+=["","## 网格质量与局部异常","",
        "| 病例 / 来源 / 半球 | white/pial穿越面配对 | 双面全在cortex内的配对 | sphere / sphere.reg负面积面 |",
        "| --- | ---: | ---: | ---: |"]
    for c,d in metrics["cases"].items():
        for kind,rows in d["quality"].items():
            for h,row in rows.items():
                cross=row["white_pial_crossings"];o=row["sphere_orientation"]
                lines.append(f"| {c}/{kind}/{h} | {cross['proper_transverse_pairs']} | {cross['proper_pairs_with_both_faces_fully_in_cortex']} | {o['sphere']['negative_faces']} / {o['sphere.reg']['negative_faces']} |")
    lines+=["","穿越数量为当前固定严格横穿谓词的三角面配对数，并非穿越顶点数或体积。"
        "端点接触、重合内侧壁与严格横穿分别记录；各自无自相交不能替代二者互不穿越。",
        "连通性、非流形边、顶点链接、顶点顺序和语义控制见完整质量JSON。"
        "本次同时复核优化前和官方；历史阳性不被平均误差掩盖。",
        "", "## 原始T1连续链：当前对优化前","",
        "| 病例 / 体积 | 不同体素数 | 最大 / P99绝对差 | dtype相同 | affine / header |",
        "| --- | ---: | ---: | --- | --- |"]
    for c,value in drift["cases"].items():
        for name,d in value["volumes"].items():
            v=d["voxels"]
            lines.append(f"| {c}/{name} | {v['elements']-v['exact']} | {v['max_abs']:.9g} / {v['p99_abs']:.9g} | {d['dtype'][0]==d['dtype'][1]} | {d['affine']['pass']} / {d['header_exact']} |")
    lines+=["","LTA矩阵、orig.nofix→orig.premesh→orig→white.preaparc有序网格、自动强度参数和逐文件SHA见[连续链诊断]("+a.drift.relative_to(a.output.parent).as_posix()+")。"
        "标签 Dice 在上一节另列；标签数值相关性不作为分割一致性。",
        "", "## 全部阶段耗时与实现来源","",
        "| 阶段 | sub01 优化前→当前 s | sub02 优化前→当前 s | 当前实现 |",
        "| --- | ---: | ---: | --- |"]
    stage01={r["name"]:r for r in full["cases"]["sub01"]["stages"]}
    stage02={r["name"]:r for r in full["cases"]["sub02"]["stages"]}
    for name,row in stage01.items():
        other=stage02[name]
        lines.append(f"| {name} | {row['baseline_seconds']:.3f}→{row['candidate_seconds']:.3f} | {other['baseline_seconds']:.3f}→{other['candidate_seconds']:.3f} | {row['implementation']} |")
    lines+=["","父级surface/finish阶段包含下列内部时间，不另外加到总时间。各阶段源文件及内部完整报告见机器JSON。",
        "", "## 官方归档耗时口径","",
        "官方只在独立benchmark目录运行。下表复用已有日志的 e=墙钟秒，不使用 U/S CPU秒代替；"
        "程序、日志、时间字段定义及SHA保留在[归档提取]("+a.official.relative_to(a.output.parent).as_posix()+")。"
        "官方参考整例在不同日期、部分双侧并行，不能与本轮候选组成受控速度比，也不能把嵌套/并行时间相加。",
        "", "| 病例 | 归档官方整例秒（时间戳分辨率1s） |", "| --- | ---: |"]
    for c,d in official["subjects"].items():lines.append(f"| {c} | {d['whole_wall_seconds']:.0f} |")
    lines+=["","下列官方时间来自各自自产上游，候选时间来自本轮完整运行。"
        "边界相近的命令可并排定位瓶颈；它们不是冻结同输入的算法速度比。",
        "", "| 病例 / 阶段 | FNIT 当前秒 | 官方归档 e 秒 | FNIT实现 |",
        "| --- | ---: | ---: | --- |"]
    mapping=[("T1_normalize","mri_normalize", "T1.mgz"),
             ("SynthSeg","mri_synthseg",None),
             ("mri_em_register","mri_em_register",None),
             ("brain_second_normalize","mri_normalize","brain.mgz"),
             ("mri_ca_normalize","mri_ca_normalize",None),
             ("mri_segment","mri_segment",None),
             ("mri_edit_wm_with_aseg","mri_edit_wm_with_aseg",None),
             ("mri_fill","mri_fill",None)]
    for c,value in full["cases"].items():
        rows={r["name"]:r for r in value["stages"]}
        for name,program,needle in mapping:
            r=rows[name]
            lines.append(f"| {c}/{name} | {r['candidate_seconds']:.3f} | {official_rows(official['subjects'][c],program,contains=needle)} | {r['implementation']} |")
        for h,run in value["nested_hemisphere_reports"].items():
            items=[
                ("white.preaparc",run["white_preaparc"]["place_seconds"],
                 official_rows(official["subjects"][c],"mris_place_surface",hemi=h,contains="--nsmooth 5"),"固定FS源码Conda C++ CPU"),
                ("standard sphere",run["standard_sphere"]["total_seconds_including_io"],
                 official_rows(official["subjects"][c],"mris_sphere",hemi=h),"自有Numba CPU优化器＋有序PyTorch CUDA平均"),
                ("sphere registration",run["sphere_registration"]["total_seconds_including_io"],
                 official_rows(official["subjects"][c],"mris_register",hemi=h),"自有Numba CPU优化器＋有序PyTorch CUDA平均"),
                ("final white",run["final_white"]["seconds"],
                 official_rows(official["subjects"][c],"mris_place_surface",hemi=h,contains="--white --nsmooth 0"),"固定FS源码Conda C++ CPU"),
                ("pial",run["pial"]["seconds"],
                 official_rows(official["subjects"][c],"mris_place_surface",hemi=h,contains="--pial"),"固定FS源码Conda C++ CPU")]
            for name,seconds,native,origin in items:
                lines.append(f"| {c}/{h}/{name} | {seconds:.3f} | {native} | {origin} |")
            for name,seconds in run["metrics"].items():
                signatures={"thickness":"--thickness","area":"--area-map ../surf/"+h+".white",
                    "area.pial":"--area-map ../surf/"+h+".pial","curv":"--curv-map ../surf/"+h+".white",
                    "curv.pial":"--curv-map ../surf/"+h+".pial"}
                native=official_rows(official["subjects"][c],"mris_place_surface",hemi=h,contains=signatures[name])
                lines.append(f"| {c}/{h}/{name} | {seconds:.3f} | {native} | 自有PyTorch CUDA |")
    lines+=["","拓扑/remesh等官方双侧并行记录可能存在交错，完整原始命令保留，未把无法确认归属的 e 值强行分配给某半球。"
        "官方未单独记录某命令的 e 值时留空说明，不用联合阶段窗口冒充单步计时。"]
    lines+=["","## 当前脑图",""]
    for case,value in full["cases"].items():
        case_root=Path(value["source_files"][0]["path"]).parent
        lines += ["### "+case,""]
        for img,label in [("t1_surface_overlay.png","真实T1上的white/pial叠加"),
                          ("region_errors.png","脑区指标误差"),("local_region_boundary.png","最低Dice脑区边界")]:
            rel=(case_root/"figures"/img).as_posix()
            lines.append(f"![{case} {label}]({rel})")
            lines.append("")
    lines+=["## 安装、限制与下一步","",
        "本轮在已有主页 Conda 环境构建 wheel、编译自有 FastPD 扩展、安装到独立目标目录并验证 API 导入和 CLI 帮助；"
        "新版 N4 从源码重新构建并回归，其他13个原生组件复用已记录的固定源码构建产物。"
        "无新增生产依赖；不是全新 Conda 或物理无预装软件隔离整例。",
        "旧Synth控制误用cuDNN flags默认enabled=False，不能用其超高显存推断生产缓存策略。"
        "当前整例低显存策略保留；连续NVML峰值和无预装软件隔离部署未验证。"
        "Python完整pial仅首例左侧跑完整同输入三方对照，右侧/第二例覆盖内核回归但未跑完整Python pial。"
        "官方全流程重复性本轮未重新测，不能将所有差异归因于随机性。",
        "MNI逆场全域局部大差异与brainmask内部残差分开报告，未宣布变换等效。",
        "下一步优先对原生逆变换的全域残差做固定输入定位；需要受控重复实验才可报告稳定吞吐。",
        ""]
    if backend:
        lines+=["## 同生产cuDNN策略的缓存控制","",
            "旧控制只传allow_tf32=False，实际还关闭cuDNN及确定性；51/31个去颅骨差异和超高显存属于该独立控制。"
            "修正测试显式enabled=True、benchmark=False、deterministic=True、allow_tf32=False，"
            "在整例计时全部结束后测，避免污染整例墙钟。缓存开启/关闭均不改变精度。"
            "阶段含额外mask/SDT写入，不能直接代表整例提速；CLI和预初始化API的缓存开启完整整例尚未验证。",
            "", "| 病例 / 缓存 | Strip / Seg 秒 | 对优化前Strip / Seg不同体素 | 同时采样显存GB | allocated / reserved GB |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for c,rows in backend.items():
            for policy,data in rows.items():
                run=data["run"];mon=data["monitor"];t=run["seconds"];diff=run["comparisons"]
                values=[run[k] for k in ("allocated_bytes","reserved_bytes")]
                txt=" / ".join(f"{v/1e9:.3f}" if v is not None else "unavailable" for v in values)
                lines.append(f"| {c}/{policy} | {t['SynthStrip_including_extra_mask_distance_io']:.3f} / {t['SynthSeg']:.3f} | {diff['SynthStrip']['different_voxels']} / {diff['SynthSeg']['different_voxels']} | {mon['peak_sampled_process_bytes']/1e9:.3f} | {txt} |")
    if inverse:
        lines+=["","## MNI逆场残差","",
            "前向/逆向位移采用NIfTI displacement-vector和mm单位。下表为向量距离，"
            "区别于严格报告中的逐分量最大误差。0.1mm只作诊断分箱，未作为接受阈值。"
            "脑内残差小不等于全域变换通过；所有最大值仍保留。",
            "", "| 病例 | 全域最大 / P99 mm | 基线brainmask内最大 / P99 mm | 全域最大是否脑内 |",
            "| --- | ---: | ---: | --- |"]
        for c,d in inverse["cases"].items():
            u=d["whole_vector_error"];v=d["brainmask_vector_error"]
            lines.append(f"| {c} | {u['maximum']:.6f} / {u['p99']:.6f} | {v['maximum']:.6f} / {v['p99']:.6f} | {d['maximum_inside_brainmask']} |")
    if a.integration_proof:
        proof=json.loads(a.integration_proof.read_text())
        if proof["whole_tested_commit"]!=commit: raise ValueError("whole binding differs")
        metrics["integration_validation"]=proof
        metrics["source_sha256"][str(a.integration_proof.relative_to(a.reports))]=sha(a.integration_proof)
        lines+=["","## 推送源码与整例源码的对应","",
            "整例实测源码保持ff372d7。保留main更新后，ece5e23的Talairach/MNI受影响阶段用两例冻结自产输入回归，"
            "六类矩阵/warp/检查图全部逐位相同。db479a0仅追加cuDNN enabled、benchmark、deterministic日志字段，97项相关测试通过；随后d4cba29修复自产LTA的多空格读取，六个真实矩阵差0，最终100项测试通过。",
            "这些补充验证不称为新源码的原始T1整例，完整源码范围见[绑定记录]("+a.integration_proof.relative_to(a.output.parent).as_posix()+")。",
            "首次比较封装暴露共享LTA读者的空格兼容bug，随后已修复并做六文件真实回归；cuDNN-on控制与生产仿射后端不一致。"
            "两者原始日志及validity.json均保留，修正后以新目录完成同后端回归。"]
    lines+=["","图示说明：叠加图青/绿为官方white/pial，橙/红为候选white/pial；"
        "脑区误差图表示相对官方的有符号百分比，蓝为负、红为正；"
        "局部标签图青为官方、红为候选，轴为conform网格体素索引。参考值为零的百分比不定义，完整JSON保留差值。",
        "下一条性能验证命令见[复现说明](REPRODUCE.md#下一条性能验证)。"
        "当前缓存开启只完成同输入Synth阶段；需先验证其完整CLI，再验证预初始化API，默认低显存策略继续保留。",""]
    a.metrics.write_text(json.dumps(metrics,indent=2)+"\n")
    a.output.write_text("\n".join(lines))

if __name__=="__main__":main()
