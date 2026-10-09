# 实际生产内部完整球面链：v9 同输入配对

## 范围与版本

公开 ds000114 sub-07 的双侧 `smoothwm` 来自已完成的 803aec50 原始 T1 FNIT 候选整例。本次输入固定，直接调用实际生产内部 `_run_accurate_sphere_pair`，完整执行 `smoothwm → inflated/sulc → standard sphere`。没有重新执行 conform/N4/WM/GA/remesh，也没有执行 sphere.reg；不能称为新的原始 T1 整例或官方整例比较。

候选只替换标准 inflation 为 FNIT 完整 Torch 实现，并在 fresh surface worker 局部开启分配缓存。球面法向仍为 Numba，平均复用既有 GPU 实现。对照的 inflation 为固定 FreeSurfer 源码独立 Conda 构建产物；两方 sphere 都是 FNIT 成熟实现。参考仅用于候选输出后的比较，候选不读参考修补。

实际基底 `c886a003` 加明确五文件接线 overlay；[冻结清单](tested_root_frozen_source.json)与[模块/硬件收据](source_hardware_receipt.json)绑定实际文件，不重标后续提交：

| 身份 | SHA-256 |
|---|---|
| 实际 `native_free.py` | `be1e044e2db43a4d59c9b6752f997643a4d7e5f9749ba6b30b464b37b557002b` |
| 本次只读 observer | `98ec3a86ad87294d50241da927de0562263db945efd9c0cfbb3a765c14c6fa1b` |
| hemisphere child-cache 接口 | `c618fc805bcb779a2ace1cbc55def2a7412a535270815e31cb76504160485e2f` |
| 完整 inflation runner | `1186db4d73b43edc7bea7580a57b73f490b5ef3669241e3c99837f630718d3bb` |
| 完整 inflation Torch 积分 | `a782b5ec493d4c390131e8882b0b07ebf3d5b5e9ad39da59328b964500550aab` |
| 独立构建 `mris_inflate` | `8c3e5f688a635a4bf7e1c49efec9737bef2310a2fcc5b221332fffc54667be17` |

## 耗时和精度

A100-SXM4-80GB、Xeon Platinum 8369B，同 CPU64–67、总4线程、每侧2线程，Torch 2.5.1/CUDA11.8；TF32 开启，未用半精度或 autocast。父 CUDA 已初始化并持有活张量，父分配缓存关闭。两种策略分别使用新空 NumBa/Triton JIT 缓存，完整组包含私有复制、fresh exec、导入、初始化、JIT、全部计算、传输、读写与发布。双方只读 trace 记录每轮状态，计时含该诊断开销。

| 完整实际阶段 | native＋inherit，s | Torch inflation＋子缓存 enabled，s |
|---|---:|---:|
| 完整双侧组 | 265.534329 | 248.454963 |
| LH inflation | 24.014664 | 5.500657 |
| LH standard sphere | 233.062893 | 233.446349 |
| RH inflation | 19.089170 | 5.487209 |
| RH standard sphere | 156.661795 | 144.324836 |

完整双侧组本次观察缩短 **6.432075%**。这是共享节点上的单次冷配对，不是 ABBA 稳定吞吐。inflation-only 双侧组约50%的先前阶段收益不能解释为完整球面链或 recon-all 整例提速；600s/subj 目标未由此测试证明。解释器导入不在 `main_seconds` 中，但组计时包含子解释器启动及导入。

- LH 114247 顶点/228490 面，RH 114951 顶点/229898 面。
- inflated 和 sulc 的全部 FP32 元素差异数、最大/P99/RMSE 均0，有序面及九项体积几何头全部相同。
- 两侧 sphere 坐标、有序面、几何头相同，各半球整个 sphere 文件 SHA 也相同。
- LH 168轮、RH 189轮的坐标 SHA、梯度 SHA、全步长搜索候选及 SSE、尺度接受/更新状态相同；末段清理计数轨迹相同。
- 两组父活张量、父缓存环境、子实际策略、线程预算和 TF32/no-autocast 共10项合同通过。

执行完成、严格配对复现通过；本次未观察到优化退化。整体脑区指标等效尚未判定。

## 网格质量与显存

最终文件重读后的 FP64 径向检验，LH 两组均有 **47 个负面，面积合计0.05596049539646856 mm²**，负面索引 SHA 相同；RH 两组均0。零面积面均0，全部坐标有限。LH 内部末段清理最后计数48与最终文件 FP64 检查使用不同检查路径，分别保留。本次没有新增翻折，但既有翻折未修复。三维自相交、white/pial 穿越和后续 sphere.reg 不在此测试范围。

| 指定目标 GPU 采样 | native，字节 | Torch，字节 |
|---|---:|---:|
| 整卡观测峰值，含其他进程与驱动开销 | 1,354,760,192 | 2,290,089,984 |
| 全部计算进程同期合计上界 | 1,337,982,976 | 2,269,118,464 |
| 精确父子进程树同期峰值 | null | null |
| 最大实际采样间隔，秒 | 3.985930 | 3.280537 |

名义采样间隔0.5秒，无查询失败。容器/驱动 PID 归属未解决，不能把未知树占用记为0。整卡和进程列表来自连续查询，时间并非完全同时；上述上界不等于精确父子树峰值。这个脚本没有收集 allocated/reserved 峰值，不补记0，也不借用其他试验峰值。采样结果不保证连续峰值，不能宣布整例20,000,000,000字节验收。物理无预装脑影像软件的干净环境隔离未验证。

## 失败诊断与复现

[v8 失败报告](v8_failed_observer/summary.json)和 worker 日志保留：只读 Python trace 对多行 `updates.append` 收到两次行事件，记录数合同明确失败，组错误处理取消本组另一侧。没有更改或修补生产算法，也没有中断其他任务。v9 按真实迭代 index 只记录一次，重新完整运行同一冻结生产源码，最终 [summary](v9/summary.json) 为 `complete_actual_sphere_chain_pair`，严格配对 `passed`。退出0表示计算完成，质量与严格比较字段分别判断。

复现入口是本目录上两级的 `benchmark_sphere_prepare_group.py`，全部参数和具名调用见[七节功能页](../../../../../../docs/recon_all/INFLATE_TORCH_20261009.md)。必须使用新输出目录和匹配 SHA 的冻结源码、公开输入包及声明 native/资产目录；默认生产后端保持 native，配准 child-cache 仍 inherit。

40份机器收据仅替换私有路径和主机名，283,241个数字/布尔/null字段逐项不变。原始包私有保留，SHA `34b4b9673d5e6e6bda6ae4c726318a3f8e2588cbd6dcaa227574d4556e15a9ac`，2,284,814字节。[映射清单](public_export_manifest.json)记录原始与公开字节 SHA，[发布收据](publication_receipt.json)记录验证范围；不发布影像、网格、权重、许可证或凭据。
