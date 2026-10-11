# 2026-10-10 完整 MRI 拓扑似然子项回归

公开 ds000114 sub-07 ses-test LH，输入来自 `79a41cdd` 的原始 T1
连续链。范围是完整初始化 surface-statistics 同输入算子测试，
没有重连候选、GA、整例提速或整体指标等效声明。

## 实际结果

110,534 顶点、221,116 面；选择 108,538 顶点、216,722 面。
全部联合 PDF 65,536 项、MRI 变换以及四个参考标量严格不同数均为 0。
自有重复调用完全一致，固定源码两次完整检查点 SHA 相同。

| 自有字段 | 实测值 |
| --- | ---: |
| face_ll | −9.599752426147461 |
| vertex_ll | −6.82485818862915 |
| 面积加权面似然 | +6.5949201583862305 |
| 面积加权顶点似然 | −6.870598316192627 |

API 首次 PDF/JIT 0.758368 秒，完整 MRI 子项首次 JIT 0.949482 秒，
热调用 0.117425 秒；含检查点/MRI 读写的脚本 3.743876 秒，含
解释器启动/导入的进程 7.048504 秒。参考入口包含初始准备与所有统计，
两次至检查点为 7.614243/5.089190 秒，不与单子项构造速度比。
CPU 四线程/四核，GPU 未使用。完整硬件、CPU affinity、线程环境
和程序/源码/输入哈希见 JSON 收据。

## 失败边界

真实 v3 输入包含 86,280 个有向负面面积。旧参数校验错误要求所有
面积为正，因此在候选计算前失败；v4 保留源有向符号，只拒绝总面积
为零，四个子项与固定源码精确一致。v3 原失败和源码绑定未删。
最终网格/球面翻折/相交质量门保持独立，不以本字段符号放行网格。

## 源码与收据绑定

- v4 实际候选 `topology_mri_likelihood.py` SHA-256：
  `9f001c8df921eeb31c3140d916de548e2d59374391278946004b2d94c4fa6302`。
- 参考固定源码：`d932c45b7941662ea380a05efef580568b98d41a`。
- 独立 Conda 统计观察器 SHA-256：
  `1c41ff4d8c58931de1a81ae5b68ef5c42b0427170d029d0374be5fbf01c960b1`。
- 两次统计 capture SHA-256：
  `6259f860c7e6f6a4f157ca0705d611df4083072e381933ee7d476867751a873f`。
- [公开导出 manifest](reports/PUBLIC_EXPORT_MANIFEST.json)保留原始收据哈希，
  服务器侧去除地址、账户及私有路径，全部数值/布尔/null 叶保持不变。
  MRI、原始几何/检查点、程序及许可证留在授权私有服务器目录，不进 Git。
- [实际 v4 报告](reports/v4/candidate.json)、[进程墙钟](reports/v4/stage_process.json)、
  [v3 失败](reports/v3/failure.json)、[参考重复性](reports/statistics/reference.json)。

## 复现

具名参数及完整字段说明见[中文功能页](../../../../docs/recon_all/TOPOLOGY_MRI_LIKELIHOOD.md)。
`tools/benchmark_recon_topology_mri_likelihood.py` 使用冻结同输入完整状态；
`build_statistics_observer.py` 在新 benchmark 构建目录只插入观察逻辑，
保留固定源完整编译/链接参数。7 项本地合同用于接口和边界，不能代替真实数据。

## v5：有向顶点与源顺序分母

当前源码 SHA `c3932110db5cff03b1f9b8539b0a7d24768531450e04a5a6a42ac22dabebd585`。
零面积检查改用实际内核的原顺序 double 分母，不用 NumPy 另一棵归约树。
9/9 本地合同通过 1.139 秒；同完整真实数据 PDF、矩阵、四标量及自身重复
仍严格差 0。PDF 首次 JIT 0.483241 秒、完整 MRI 子项首次 JIT 0.987338 秒，
热调用 0.117249 秒、含读写脚本 3.635384 秒；含子解释器进程 9.564424 秒。
v4 原收据与源码绑定保留，当前结果不覆盖旧版本。

[新v5完整报告](reports_v5/v5/candidate.json)、[新v5源清单](reports_v5/v5/source_manifest.json)、
[9合同](reports_v5/unit_receipt.json)及[公开导出清单](reports_v5/PUBLIC_EXPORT_MANIFEST.json)。
