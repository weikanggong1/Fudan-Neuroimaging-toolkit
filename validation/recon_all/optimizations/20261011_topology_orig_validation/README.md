# 最终orig完成检查修复

## 1. 功能与原因

修复独立最终orig验证工具的两个完成检查漏洞：无体积几何头的闭合表面
此前仍可返回passed，两个空footer也会被当作保留；父回执此前只核对
表面，没有逐一核对声明的末态/映射产物及必要输入/模块哈希。
本轮不改变GA、remesh、相交算法、原轮数、输出坐标或精度。

## 2. Python、输入和输出

完整工具的具名接口、全部参数与surface RAS/mm语义见
[功能页](../../../../docs/recon_all/TOPOLOGY_GENETIC_ORIG.md)。
父目录必须包含完整GA报告、COMPLETE、H.orig.premesh、末态NPZ及
映射NPZ；全部声明输出逐文件SHA匹配，三个输入与实际模块哈希完整。
最终头必须包含valid=1、正体积尺寸/体素大小、有限且可逆的RAS几何，
原footer字节比较继续保留。源valid字段的尾部注释及微小方向分量允许。

## 3. CLI和复现

```bash
# 原完整验证工具；只能消费实际完整自产premesh，不手工填充回执。
PYTHONPATH=src:tools python tools/benchmark_recon_topology_genetic_orig.py \
  --premesh-directory /data/own/complete-genetic-premesh \
  --hemisphere lh --output /data/runs/new-genetic-orig --threads 4 \
  --remesh-scalar-storage numpy

# 只读两个既有自产表面头，不执行网格算法，不等于上述完整阶段。
PYTHONPATH=src:tools python validation/recon_all/optimizations/20261011_topology_orig_validation/reproduce_geometry.py \
  --input-surfaces /data/own/sub06/lh.orig.nofix /data/own/sub07/lh.orig.nofix \
  --case-names sub06 sub07 --output /data/runs/new-geometry-report.json
```

复现脚本的input_surfaces是同意使用的三角表面路径列表，case_names为
等长公开名称，output为不存在的JSON。返回输入SHA、顶点/面数、几何
门与工具版本；缺输入或无效头报错。脚本属于本次整理版本；实际服务器
执行的是从工具逐字符提取同一函数的短观测脚本，其SHA另记，
不将本脚本SHA回标实测。

## 4. 原软件

本轮修改的是FNIT的完成验证，没有独立官方CLI。所验证算法仍对应
`mris_fix_topology -ga`之后的`mris_remesh --remesh --iters 3`与
`mris_remove_intersection`，原命令只用于隔离参考。

## 5. 实测范围

本地6项测试及19个子测试通过7.41秒；JUnit套件记录7.396秒，两者均
不是MRI benchmark。覆盖缺失/被改动/未声明产物、非局部输出名称、
输入/模块哈希缺失、几何头缺失/无效以及合法源注释头。

现场复用79a41cdd自产两例左侧nofix，输入SHA详见
[真实头读取](reports/real_geometry.json)：sub06为126952顶点/253976面，
sub07为110534顶点/221116面。两者均256³、1mm，valid实际为
`1  # volume info valid`，方向可逆且校验通过。完整冷观测CLI2.051321秒；
内部含导入/读取/哈希/检查1.895394秒。此处没有运行GA、remesh或清理，
不报告整例提速、最终表面距离或指标等效。GPU隐藏，显存记null。

原先排队的75项冻结orig工具SHA仍为ad008be…，未被热修改。
它的正常产物须再经本次增强验证，独立tmux任务已经排队，
[启动回执](reports/strengthened_launch.json)绑定新工具及两级父清单，
只有原先orig任务退出0、COMPLETE与报告SHA有效后才读取实际产物；
不重复网格算法，也不更改旧冻结目录。旧排队收据不代表本次版本已完成
完整orig。全九缺陷退出、最终orig和原始T1整例仍未评估；无新的脑图，
既有真实脑图见[首缺陷](../20261010_topology_genetic_search/README.md)。

## 6. 版本和失败记录

- 旧ad008be版本可由小四面体控制复现错误接受空头/缺父产物；不是
  已确认的真实被试失败原因。
- 本次工具SHA bd60fb4e…，只修完成验证；函数提取SHA及实际观测脚本
  SHA绑定实测报告，JUnit保留原始SHA并只移除公开版hostname属性。
- 不新增依赖，不修改已运行的冻结目录、图像或他人的工作。

## 7. 来源

- [自有最终工具与七节说明](../../../../docs/recon_all/TOPOLOGY_GENETIC_ORIG.md)。
- [Nibabel格式读写](https://nipy.org/nibabel/reference/nibabel.freesurfer.html)。
- [固定源拓扑流程](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
