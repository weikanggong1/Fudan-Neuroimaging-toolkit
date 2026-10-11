# 完整表面放置的分配缓存验证

本目录绑定未改动的 v4 自产输入和源码；不调用参考来修改候选、不改变生产默认。使用公开 ds000114 sub07 LH，同逻辑 GPU、CPU 四核和四线程，每项新进程拥有独立冷 Numba 缓存。CLI 和父 CUDA 已初始化 API 共五项完整 white 的实际子进程退出码均为 0。

## 已完成的完整 white

| 对照 | API 秒 | fresh exec 秒 |
| --- | ---: | ---: |
| enabled 1 | 114.845595 | 122.367409 |
| disabled 1 | 1374.054040 | 1383.346664 |
| disabled 2 | 1571.527018 | 1583.506027 |
| enabled 2 | 108.461225 | 119.828927 |
| 父 CUDA 已初始化、子 enabled | 107.863718 | 117.953264 |

ABBA API 中位数为 1472.790529→111.653410 s，观察比 13.190735 倍。每次包含完整四轮、22 个接受步、最终相交清理和输出读写；GPU 首尾同步。五项坐标、面顺序、全部接受/拒绝判定、white 与预处理 MRI 字节一致，对原同输入 v4 自产输出也为零差异。整体指标等效和原始 T1 整例提速未在本实验判断。

首对的 24 个试步候选构建累计为 1112.554836→16.438807 s，有序 CPU 接受为 29.874123→29.542597 s。两者候选计数、motion bounds、动态分块和不截断标记相同；没有收集 CUDA malloc/free API 轨迹，不断言唯一驱动根因。共享背景占用在第二次实验中变化，全部负载、CPU 时间和 cgroup 快照保留。

缓存开启的 Torch peak allocated/reserved 为 690,560,000 / 1,069,547,520 字节；关闭时不可用字段为 null。API 父 live tensor 为 33,554,432 字节，值、指针和环境前后不变。树 PID 归属不明，精确父子同期峰值为 null；整卡/全进程占用上界包含其他任务，不代替精确峰值或连续峰值保证。

![计时与设备总占用](reports_v1/white_abba_v1/timing_and_device_load.png)

## 收据、源码与已知控制脚本问题

- [全部阶段及同期采样](reports_v1/white_abba_v1/summary.private.json)。
- [逐项机器摘要](reports_v1/white_abba_v1/derived-final/metrics.json)、[CSV](reports_v1/white_abba_v1/metrics.csv)。
- [对原自产输出的只读核验](reports_v1/white_abba_v1/evaluation-final.private.json)。
- [实际源码清单](reports_v1/white_abba_v1/frozen_source_manifest.private.json)、[输入与全部 Python SHA](reports_v1/white_abba_v1/manifest.private.json)。
- [完整来源检查和退出标记问题](reports_v1/white_abba_v1/artifact_verification.private.json)。
- [逐数值叶保真导出清单](reports_v1/PUBLIC_EXPORT_MANIFEST.json)：58,380 个数字/布尔/null 叶；其余服务器原件保留。

最初两个 shell launcher 的 `printf` 格式转义错误，导致外层退出标记为空，原文件未改写。外层退出码标为未捕获；五个完整阶段的真实 `Popen.returncode` 都是 0，summary 和只读评价也确实完成。后续 launcher 以 SSH stdin 传完整脚本并正确引用格式；不能回填旧退出码或重标旧墙钟。

v1 完整 white 驱动 SHA `1afcb5df8fe7d47e14c0d422580c2c399c2d6fd1627dd364daf4394949bd1732`；[冻结复现副本](provenance/benchmark_final_white_v1.py)。评价 SHA `c9c99386b1a94b7aa86ff59b6401d6cacf2a220b9ebbcbde57bf99205259c35e`；[冻结副本](provenance/evaluate_final_white_v1.py)。这些是当前收据所需的只读参考，不是第二套生产实现。

## pial 失败与已完成的隔离参考

[pial v2 失败收据](reports_v1/pial_failed_v2/summary.private.json)来自完整四轮 47 步后的残余相交拒绝。没有写 pial.T1，子进程墙钟 254.506 s 不能用于成功提速；父初始化 API 按失败保护没有运行。47 步 SHA 与所有判定保存在[轨迹](reports_v1/pial_failed_v2/01-cli-enabled/accepted-trace.private.json)。失败时该 v2 工具没有保留 allocated/reserved，仍为未知；不能用后来工具补测数值冒充原值。

[冻结 v2 驱动](provenance/benchmark_pial_v2.py) SHA `e15454ff1951a2f49ba8cfc398e61fcb0efedff4ea1fa017808f18ddbb64766a`。v3 实际驱动 SHA `c45bbe5769e8e6ec0ddfeeebf5eebd3cb9a64058a13f9ce565aac77e50671d65`，[冻结副本](provenance/benchmark_pial_v3.py)保留。v3 47 步的所有坐标 SHA 和试步判定与 v2 相同；[只读清理记录](reports_v3/pial_cleanup_v3/01-cli-enabled/cleanup.private.json)显示 415→32 面停滞，源门继续拒绝写出。失败 API 经过 245.518 s，不是成功耗时；本次 Torch peak allocated/reserved 为 689,376,256 / 2,650,800,128 字节，不能回填 v2。

[两次隔离固定源码参考](reports_v3/pial_reference_v3/summary.private.json)均退出 0，完整优化器 164.174/162.559 s，fresh exec 164.252/162.666 s；均为 95,045 顶点、190,086 面。参考重复坐标完全相同，但原生写出也保留 32 个相交面、35 个标记顶点。[失败终态只读对照](reports_v3/pial_cleanup_v3/evaluation-cleanup.private.json)确认 Python 清理返回坐标与两次参考逐元素一致，有序面相同，mean/P99/max 距离均为 0，列出残余 32 面索引。原生文件差异仅为创建时间注释，注释之后全部字节相同。这不能用来宣布网格质量、pial 或整体指标等效通过，也不能归因缓存。

`reports_v3/FIRST_EXPORT_MANIFEST.json` 绑定原始 JSON→路径/主机脱敏；第二层 `PUBLIC_EXPORT_MANIFEST.json` 再去除创建注释中的私有账号，两层分别校验数值、布尔、null 的值与类型，原始文件 SHA 由链式清单追溯。私有终态 NPY、原始影像和许可证未下载。

生产缓存接线由协调者另行验证；本目录不改父进程或默认策略。旧 v4 cacheoff pial 未有最终收据，不能称其成功或将失败诊断用于整例提速。

## 汇总复现接口

`summarize_and_plot.summarize(summary_json=Path(...), evaluation_json=Path(...), output_dir=Path(...))` 只接受完整 producer 的 summary；可选评价 JSON 提供旧自产数值比较。输出新的 metrics JSON/CSV 和资源 PNG。秒为墙钟，显存为字节；父端查询时间与子算法并发，不能从 API 时间直接减去。不存在的输出目录、完整 producer 和实际子进程成功是必要条件。

```bash
python validation/recon_all/optimizations/20261010_surface_allocator/summarize_and_plot.py \
  --summary-json /runs/white-cache/summary.private.json \
  --evaluation-json /runs/white-cache/evaluation-final.private.json \
  --output-dir /runs/white-cache/derived-new
```

三个参数依次是完整原始报告、只读对照报告和新派生目录。`matplotlib`、nibabel、NumPy、Numba、Torch 均已在主页 Conda 声明，无新依赖。完整七节功能说明、逐项输入/输出及具名调用见[表面缓存专页](../../../../docs/recon_all/SURFACE_CUDA_ALLOCATOR.md)。
