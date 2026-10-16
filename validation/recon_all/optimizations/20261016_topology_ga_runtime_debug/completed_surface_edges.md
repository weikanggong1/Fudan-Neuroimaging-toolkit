# 完成整例后的表面长边复查

[只读脚本](audit_completed_surface_edges.py)用于追踪同输入 C++ premesh 在 sub-07 右半球约 [29.023, 45.086, 2.920] mm（surface RAS）附近的局部尖刺。它只接收**已完成**的 subject；两份 FNIT 输入必须具有 fnit-native-free-run.json 中的 complete、138/138 输出检查通过和生产网格检查通过，且各自双侧四种表面均存在。可选的独立官方参考只用于相同的独立网格统计，官方运行完成性由调用者预先核对。整例运行期间不要调用。

Python 接口 audit 的全部输入为 baseline_subject（纯 Python GA 完整 subject 目录）、candidate_subject（混合 C++ 拓扑完整 subject 目录）、可选 reference_subject（已完成的官方 subject 目录）、focus_ras_mm（surface RAS 三元组，单位 mm，默认上述位置）、focus_radius_mm（以边中点到焦点的距离选取局部边，默认 20 mm，必须为正）、output_report（不存在的新 JSON 路径）。输出是同一个 JSON 字典；每份表面记录文件 SHA-256、顶点数、面数、唯一边长最大/P99.9/>5 mm 数量、面面积最大值、零面积面数、焦点附近的边长摘要及局部最长边端点。无法读取文件、FNIT 报告未完成、网格索引错误、非有限坐标、输出已存在时抛异常。脚本只写新报告，不改输入文件。

命令行调用使用全部具名参数：

~~~bash
# 固定本次仓库版本中的诊断脚本
SCRIPT=/path/to/FNIT/validation/recon_all/optimizations/20261016_topology_ga_runtime_debug/audit_completed_surface_edges.py
# 已完成的纯 Python GA subject
BASELINE=/path/to/completed/python/subject
# 已完成的混合 C++ 拓扑 subject
CANDIDATE=/path/to/completed/hybrid/subject
# 尚不存在的新私有 JSON 报告
OUT=/path/to/new/diagnostics/surface_edges.json
python "$SCRIPT" \
  --baseline-subject "$BASELINE" \
  --candidate-subject "$CANDIDATE" \
  --focus-ras-mm 29.023 45.086 2.920 \
  --focus-radius-mm 20 \
  --output-report "$OUT"
~~~

有已完成的官方参考时，另加 --reference-subject /path/to/completed/official/subject。脚本同时读取 lh 和 rh 的 orig.premesh、orig、white、pial，共每份 subject 八个表面。所有长度是 mm，面积是 mm²；>5 mm 是本次异常的追踪字段，**不是新验收阈值**。不同网格独立统计，输出不含同索引顶点误差、双向表面距离或整体指标等效判断。完整质量仍需调用现有网格结构、严格自相交、white/pial 穿越和脑区指标检查。该诊断本身没有原软件对应的独立 CLI；相关官方步骤为 mris_fix_topology、mris_remesh 和 mris_place_surface，源代码见 [固定 FreeSurfer 版本](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)，参考 Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781。

本脚本尚未在新的两例混合整例上运行，结果应保持 not_assessed。已完成的同输入 premesh 局部观察和原始 SHA 见[异常记录](premesh_local_abnormality.md)；不能把阶段观察当作最终 orig、white 或 pial 的结论。
