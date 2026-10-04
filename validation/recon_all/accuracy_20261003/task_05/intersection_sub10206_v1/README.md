# 第五例 LH 表面相交：只读 CPU 定位

## 结论与边界

本报告对应公开真实 T1 `ds000030_sub-10206` 的冻结基线 `816e5610417a4c587caf321049438a9554139016`，不是任务5候选整例。输入 SHA 为 `1d1f71a21c9415aa8a91c55b8505eab16c0b091ad71263f0da976839a6bc9ab4`。失败回执保留 exit 1；138项文件齐全不能代替几何质量通过。整体等效仍为 `not_assessed`，没有修改门槛或生产算法。

轻量诊断在已授权 gpucw1 运行，隐藏 CUDA，CPU线程预算2；只读已有表面，不重新放置、不修补被试、不启动整例。影像和表面原文件留在服务器，报告仅保存哈希、面/顶点编号、面积和数值见证。该主机安装了官方 benchmark 软件，本次不是物理隔离部署验证。

| 同一冻结Python检测器，各自原网格 | 相交面 / 面对 | 零面积面 | 最小面积 mm² | 精确有理数严格穿越面对 |
|---|---:|---:|---:|---:|
| 候选 LH white.preaparc | 16 / 10 | 0 | 8.522835e-10 | 1 |
| 候选 LH white | 16 / 10 | 0 | 8.522835e-10 | 1 |
| 候选 LH pial | 16 / 10 | 0 | 7.332581e-11 | 3 |
| 官方 LH white.preaparc | 0 / 0 | 0 | 4.964126e-4 | — |
| 官方 LH white | 0 / 0 | 0 | 8.574761e-4 | — |
| 官方 LH pial | 0 / 0 | 0 | 1.414138e-4 | — |

候选与官方有序faces不同，不能进行同索引位移比较；这里是在每个保存网格上应用同版本检测器。每一命中面对均无共享顶点、共享边或完全相同坐标的角点。候选white命中27个顶点、pial命中28个，全部在各自cortex label内。极小面积没有被改标为零面积或放宽阈值排除。

独立的精确有理数Cramer求解使用文件中浮点坐标的精确分数表示，要求线段参数严格在 `(0,1)`、目标三角形三个重心坐标严格大于0，不使用原生 `1e-6` 容差。white面对 `[9366,31624]` 与pial三对 `[1211,221475]`、`[3725,221475]`、`[3860,17720]` 均有严格穿越见证，详见 `hit_refinement.json`。因此不能将全部失败归为检测误报。其余容差命中未由本项严格测试判真或判假，未删除。

## 已定位的阶段与尚未定位的原因

保存的LH prewhite日志先报告14相交面，连续16次清理仍为14；final white清理从14增加到20，最终恢复最少14的状态。这些日志与脚本/产物哈希见 `failure_binding.json`。最终报告为white/pial各16、RH两者0。本次未重复RH检测，RH0依据冻结完整pipeline报告，不能冒充新实测。

候选orig与后续表面有序faces相同，涉事orig三角面面积为white命中面 `0.096049–0.246202 mm²`、pial命中面 `0.039354–0.246202 mm²`，远大于放置后极小面。现存异常已在white.preaparc保存阶段出现，继续传播到final white；每轮坐标未保存，不能进一步断言是哪一轮或哪个算子造成坍缩。

Python检测器和生产placement谓词在全部30个报告面对上均判断相交。另将原环境保存的 `utils/tritri.cpp` 编译为一次性CPU谓词共享库，30/30亦判断相交；源码和临时库SHA写入报告，临时目录自动删除。它验证保存源码的三角面谓词，**没有调用安装的官方二进制，也没有证明该临时库就是既有生产程序实际链接的库**。

原生MHT排除共享顶点面，并通过面散列候选集调用tritri；Python使用包围球与AABB候选集。原生日志14与重新读取文件的16差异尚未隔离到MHT候选集、运行时rip状态或坐标状态，不能据此断定Pythonbug。源码契约：`mrisurf_metricProperties.cpp:mrisMarkIntersections`、`mrishash.cpp:MHTexpandToTouchingFaces/MHTdoesTriangleIntersect`、`mrisurf_deform.cpp:MRISremoveIntersections`。原生清理会在无进展超过15次后结束，即使命中仍非零；其成功返回本身不能保证最终质量门槛通过。

当前未证明可直接修复的通用算法bug，因此生产代码保持原实现，失败产物保留。下一步应定位完整放置/清理阶段中的小面生成与运行时检测差异，再决定一般修复；不能针对该被试强制平滑或复制官方坐标。

## 工具、输入输出与复现

`probe_intersections.py` 是只读诊断，无新增生产API。输入为两个标准subject目录的 `surf/<半球>.<表面>`，surface RAS、mm；输出stdout JSON含文件SHA、有序faces SHA、面积mm²、面/顶点ID、共享面排除数及浮点线段穿越见证。CPU诊断耗时只标记观察，不是生产性能比较。

全部参数：`--source-root` 必填，固定FNIT源码根（其下有src）；`--candidate`/`--reference` 必填，候选/隔离官方subject；`--hemispheres` 默认lh，可选lh/rh列表；`--surfaces` 默认white.preaparc、white、pial列表。缺失表面输出missing和null，不计为0；读文件/源码错误直接失败。脚本关闭Numba JIT，防止写入冻结源码缓存；不执行GPU。

`probe_hit_pairs.py` 复核既有命中，无需再次扫描完整表面。参数 `--subject` 为保存候选subject；`--native-source` 为已保存上游源码根，其下包含utils/tritri.cpp和include；`--pairs-json` 为 `{表面名: [[面编号A,面编号B],...]}` JSON字符串。三项必填。输出stdout JSON含精确穿越分数、原生谓词、cortex归属、orig同faces检查和源码/文件SHA。需要现有g++，不是FNIT新增运行依赖。首次临时库载入因无关MatrixFree符号失败；后续用隐藏可见性和函数节回收仅保留三角面谓词，已成功；没有修改上游源码。

```bash
# 在本报告目录执行，复用已认证SSH；各路径先按FNIT统一索引复验。
fnit_ssh=/tmp/fnit-recon-five-20261002/coordination/ssh_gpucw1.sh
fnit_legacy_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
fnit_accuracy_root="$fnit_legacy_root/accuracy_20261003"  # 保留现有冻结实体
fnit_python="$fnit_legacy_root/fnit_main_env/bin/python" # 原实际prefix
# 只读保存网格，标准输出写新的报告路径，不覆盖已有失败记录。
"$fnit_ssh" "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 '$fnit_python' - --source-root '$fnit_accuracy_root/baseline_runtime_816e5610' \
 --candidate '$fnit_accuracy_root/baseline/subjects/ds000030_sub-10206' \
 --reference '$fnit_accuracy_root/official/subjects/ds000030_sub-10206'" \
 < probe_intersections.py > new_cpu_geometry.json
# run_refinement.sh固定本次已发现的面对，使用同一只读方式进行精确复核。
# 复用时将其stdout路径改为新文件，避免覆盖已保存报告。
```

内部检测器没有独立官方CLI；本次原软件没有补跑命令。对应原生完整放置命令仍为此前pipeline日志中的 `mris_place_surface` white/prewhite/pial调用，不用简化命令替代已记录的完整参数。`diagnosis_summary.json`绑定原失败报告、输入、脚本与结果哈希；`mesh_summary.csv`提供可直接读取的逐表面对比。

## 参考与记录

- [FreeSurfer tritri.cpp](https://github.com/freesurfer/freesurfer/blob/v8.2.0/utils/tritri.cpp)、[mrishash.cpp](https://github.com/freesurfer/freesurfer/blob/v8.2.0/utils/mrishash.cpp)、[mrisurf_deform.cpp](https://github.com/freesurfer/freesurfer/blob/v8.2.0/utils/mrisurf_deform.cpp)。本次事实依据服务器保存源码和真实产物，不是未核对网页内容。
- Möller T. A Fast Triangle-Triangle Intersection Test. Journal of Graphics Tools 2(2), 1997.
- 2026-10-03 22:25 UTC：协调者采集第五例终点失败，保存完整日志和138项。
- 2026-10-04：本任务执行上述六表面CPU只读回放、独立精确见证及临时原生谓词核对；未提交算法修改。
