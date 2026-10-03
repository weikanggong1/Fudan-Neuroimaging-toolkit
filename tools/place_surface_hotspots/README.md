# 表面放置原生热点：未消费面哈希构建

固定 FreeSurfer d932c45 的 `MRISpositionSurface` 在非零 l_repulse 时每轮构建 mht_f_current，只作为 mrisComputeRepulsiveTerm 的 mht_faces 参数传入；该参数在固定定义中未读。此候选删除这一副本构建，动态碰撞 MHT、顶点 repulse MHT、完整四轮、拒绝恢复、固定顶点与相交清理均沿用成熟 Conda 源码控制流。没有新精度策略或额外 GPU/CPU 并行。

## 构建调用与参数

```bash
python tools/place_surface_hotspots/build_native.py \
  --build "$EXISTING_CONDA_FS_BUILD" \
  --source "$FIXED_CONDA_FS_SOURCE" \
  --output "$NEW_PRIVATE_BUILD_OUTPUT" \
  --ninja "$CONDA_PREFIX/bin/ninja"
```

- `--build`：已完成的 Conda Ninja 完整 FS 构建目录，须有 utils/libutils.a 及 mris_place_surface 链接命令。
- `--source`：与其匹配的固定源码根，六个入口、计算与哈希支持源文件须严格匹配 builder 内声明 SHA-256。不是任意兼容源码或系统二进制目录。
- `--output`：新的独立目录；必须不存在，不覆盖源目录/共享安装/其他任务产物。
- `--ninja`：该 Conda 环境中的 Ninja 可执行文件。编译器、参数、库来自已有构建命令，不更换编译器/数值选项。

没有参数默认值。源码偏移、未读参数条件变化、缺编译对象、原构建不匹配、输出已存在、编译/链接失败均立即报错。

输出 `control/` 与 `candidate/` 各含 mris_place_surface_fnit_hotspot、私有patched源、source.patch、对象与libutils.a及build.json。control 保留未改代码；candidate 删除构建，并增加只读能力查询。build.json 绑定六份输入源码、原archive、编译器/命令、builder、最终程序hash。patched完整上游源和二进制只留授权构建目录，不提交进FNIT仓库。

## Python调用与产物

现有白面与pial入口的binary参数均可选择candidate路径，其余参数/默认/失败行为保留：

```python
from fnit.recon_all.final_white_conda import run_final_white

final_white_result = run_final_white(
    subject_dir="/data/subject_fnit",              # 已自产MRI、white.preaparc、标签、aparc及阈值
    hemi="lh",                                    # lh/rh
    binary="/private/build/candidate/mris_place_surface_fnit_hotspot",  # 仅在实际验收后选择
    assets_dir="/data/fnit_fixed_assets",           # 校验过大小和SHA-256的固定资源
    threads=4,                                    # CPU总线程预算，不新增并发
)
```

输入/输出空间与原stage保持：surface RAS/mm、有序三角面/顶点、原MGZ网格。white.preaparc写表面、autodet阈值及mrisps.wpa；final white写white与mrisps.white；当前生产pial写pial.T1和日志，没有显式outvol参数。后续曲率阶段保持原调用顺序。完整契约见专项INTEGRATION.md。

## 官方命令、验证与版本

对应完整官方 mris_place_surface --white/--pial（参数见既有三个wrapper与专项文档）；本热点无独立官方CLI。真实同输入测试脚本为 validation/recon_all/optimizations/20261002_parallel/task_02/benchmark_native.py。比较当前Conda、独立未改重建control、候选及可选官方参考，完整阶段包含加载、计算、保存和wrapper检查；隔离复制另计。严格检查同索引表面、stats、诊断MGZ数组/affine/dtype和辅助曲率文件。官方偏差单列；本实验不得自动改变默认生产或整体等效结论。

2026-10-02候选：固定哈希的未消费面哈希构建删除；实测状态见专项报告。没有以旧native_cuda_pilot或20261001时间作为本轮收益，也未补齐完整Python white。

来源：[mrisurf_mri.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_mri.cpp)、[mrisurf_compute_dxyz.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_compute_dxyz.cpp)。保留仓库 licenses/FreeSurfer.txt 与 THIRD_PARTY_NOTICES.md 的适用条款。没有复制或发布无关上游代码、许可证或真实影像。

能力查询 `mris_place_surface_fnit_hotspot --fnit-placement-capabilities` 在常规参数解析前返回 JSON，包含 schema_version、program、features、upstream_commit、uses_cuda；退出码为 0。仅对这一精确单参数调用触发，正常 white/pial 调用不改变。安装应检查特性 skip-unconsumed-repulse-face-table 与固定上游提交，不根据文件名判断能力。

## 本轮真实完整阶段记录

2026-10-02，在同一真实被试左半球的冻结自产前置上，四种程序独立复制输入并完成三个步骤。benchmark源码c8c1db9；构建使用同一既有Conda编译器/参数。所有control与candidate对当前Conda的有序表面坐标误差为零，stats及诊断MGZ数组、affine、dtype精确一致。没有复制真实MRI/表面进仓库。

| 完整步骤 | 当前Conda/秒 | 未改重建control/秒 | 候选/秒 | 官方8.2.0-1/秒 |
|---|---:|---:|---:|---:|
| white.preaparc | 215.676 | 212.808 | 187.169 | 143.846 |
| final white | 193.708 | 200.878 | 171.721 | 135.492 |
| pial | 181.396 | 231.256 | 188.863 | 129.463 |

时间包含wrapper中的加载、完整计算、保存和检查；隔离复制另计，见native_three_stage_v1/report.json每行字段。白面候选相对当前单次耗时少13.217%和11.350%；pial多4.117%，其control与当前时间也有明显差别，pial收益未确认，仍保留当前生产选择。此次顺序固定、每程序每阶段单次；不能推广为整例加速。默认调度/安装尚未切换，协调者须完成集成的两例空目录整例和138严格诊断。

官方与当前Conda差异单列：white.preaparc的最大/P99为1.148342/0.014154mm，final white为0.220274/0.005112mm，pial为2.284583/0.134769mm。候选没有改变当前Conda几何；本轮不据此宣布与官方总体等效，不扩大等效阈值。程序、输入、四个固定LUT资源的大小/SHA及全部产物清单见公开统计报告。两个白面步骤都有诊断MGZ，生产pial入口没有outvol/curv参数；后续曲率生产阶段仍由原流程执行。
