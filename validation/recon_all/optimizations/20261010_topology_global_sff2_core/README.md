# 第二次全局曲率拟合：完整同输入核心回归

## 1. 功能和范围

复用已提交的 `global_second_fundamental_form()`，核对公开 ds000114 sub-07 LH、`79a41cdd` 自产上游上的 SFF2 完整 K1/K2 核。110,534 顶点、221,116 面、2,166,570 项实际有序 vtotal。未运行新的源观察器，没有后态曲率输入、原版程序生产调用或默认切换。

## 2. Python 输入输出

`tools/benchmark_recon_topology_global_sff2_core.py:run()` 的具名路径参数为 `statistics`、`statistics_repeat`（FNITFIT1 两重复）、`calibration`、`calibration_repeat`（对应原域 FNITHST1 phase0 两重复）、`source_manifest` 和新 `output`；`threads` 为正整数总预算。源码相位审计和完整面、原坐标、rip、一圈前缀确保 actual vtotal 来自同一未变原网格，而非候选 corrected 网格。

producer 仅消费捕获的当前 XYZ、法向、rip、实际有序邻域及独立置 0 的初始 K；参考 K1/K2 在候选写出后用于比较。全部顶点进入 `fit_kind=1`，没有 rip/degree0 保留分支消费未知初始值。输出完整 candidate NPZ 留私有目录，公开报告仅聚合指标、哈希和版本；输入/输出单位见[七节功能页](../../../../docs/recon_all/TOPOLOGY_GLOBAL_SFF.md)。未捕获 E1/E2、H/K 与表面摘要，明确不作为已验收字段。

## 3. CLI

```bash
python tools/benchmark_recon_topology_global_sff2_core.py \
  --statistics statistics.bin \
  --statistics-repeat statistics_repeat.bin \
  --calibration calib.before.bin \
  --calibration-repeat calib_repeat.before.bin \
  --source-manifest SOURCE_MANIFEST.json \
  --output new_sff2_core \
  --threads 4
```

完整参数中文解释及变量注释示例见功能页第 3 节。错误相位/索引域/一圈前缀抛异常；已有输出拒绝覆盖。K 核严格门失败返回 1 并保留报告，未覆盖的保留状态不能通过输出参考填补。

## 4. 原软件

固定 `d932c45b7941662ea380a05efef580568b98d41a`，内部 `mrisComputeSurfaceStatistics()` 调用 `MRIScomputeSecondFundamentalForm()`；没有独立 CLI。已有隔离 Conda 观察器运行 `mris_fix_topology -ga -seed 1234 -threads 4`，仅供参考。其两个完整捕获的程序哈希、源码相位和重复性见[来源审计](reports/source_phase.json)。

## 5. 实测

| 项目 | 结果 |
| --- | --- |
| K1/K2 全顶点不同值、不同字节 | 0 / 0；最大/P99 = 0 mm⁻¹ |
| 自有 11 数组两次重复 | 全部逐位相同 |
| 两次参考、两个完整捕获 | 文件 SHA 完全相同 |
| 首次 API，含 JIT/校验 | 9.035304 秒 |
| 同进程热 API | 0.348773 秒 |
| 脚本含导入/加载/候选写出，截止报告写入前 | 17.748947 秒 |
| 冷解释器启动与最终报告写入 | 未单独计时 |

CPU 四核/总线程 4，未执行 CUDA、不修改 TF32/半精度，无新增依赖。参考观察器包含上游及其他评分，不给相对速度比。完整全局状态、SFF1/PDF、dot、GA、网格质量和整例均 `not_assessed`；这次不是两个被试整例结果。完整机器报告：[report_v1.json](reports/report_v1.json)。

![公开 T1 衍生 SFF2 曲率及误差](figures/sff2_curvature_error.png)

图每四点显示一处，所有误差计算仍覆盖全部 110,534 顶点。原始/检查点数组和源程序不发布。

## 6. 版本

- 新阶段冻结 tracked HEAD `373281933f456782cb71462c85cab1c6ee350324` 加专属验证脚本；global 模块 SHA `03427708…` 与提交 `29b7434d` 相同。
- 原全局 prepared scope 的 43 项契约及共享 256-bin 回归保留在[旧独立版本目录](../20261010_topology_global_priors/README.md)。本次新增 11 项域/相位/保留分支与参考隔离契约；JUnit 10.717 秒、pytest 10.74 秒。
- 旧 SFF2 K 对 SFF1 PDF 的错误相位比较不改标通过。本次只把相同 SFF2 核闭合，没有放宽阈值或修改算法。

## 7. 来源

- [固定拓扑 caller](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。
- [全局 SFF](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)。
- Fischl B et al. Automated manifold surgery. *NeuroImage* (2001). [DOI](https://doi.org/10.1006/nimg.2001.0781)。
