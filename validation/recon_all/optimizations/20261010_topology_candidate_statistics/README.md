# 真实拓扑候选统计回归

## 范围

2026-10-10，公开 ds000114 sub07 左半球，三次实际候选的完整
`updateVertexStatistics` 前后状态。候选统计是拓扑 GA 内部的反馈累积，
不是脑区指标、完整候选生成、GA 完成或原始 T1 整例。

FNIT 使用独立 NumPy，不执行或链接原版程序；固定源码参考仅用于
benchmark，透明捕获原函数前后字段。算法只使用 before、当前 curvbak、
边状态和原缺陷映射；after 仅在输出生成后用于比较。

## 版本与复现

- 实测快照：`8de83f0f17d51b91a52d5b97a4ad6ed2810bef7a` 加并发未提交文件，
  实际上传的 593 个文件均绑定大小和 SHA，不能只用提交号代替。
- 模块 SHA：`149214d1afa777ee23abd560ceb5e27702cb8b5a7f53d85ebb0bc67bd6c1d12f`。
- benchmark SHA：`2ae18fe944fe6ed39cc51d4a5eb1c437aa6df39eda046b8b12e3b17f3a764fbd`。
- 原参考程序 SHA：`9e48942dee5aa3369e047f659454245f9064bbdbc190b43ecdd6077456d1c574`。
- 原源码固定 d932；源 observer builder SHA：
  `e60cc0bcccf562fd58a3b4960ae25095049d8f6d830e6fc285ac68824485c739`。
- source manifest SHA：`76eca1326fefa420b1d202e22e4a2803c03e32bd870fc39a96f3e77528aef867`。
- 公开指标及图导出 tar：194,560 字节，SHA
  `ceac447c968d1a84ea5b13c475e10f0ebd49ca7d90191abc52502764ef5abe65`。

完整原状态、日志、observer 收据及工作区源码清单位于已授权私有
benchmark 目录。公开目录保留 7 个白名单指标报告及 PNG，另附导出
manifest；完整精度、重复性、计时和资源字段保留。白名单排除主机名、
路径、日志和 observer 全文；已选数值、bool 和 null 叶全部保持。
manifest 逐项绑定原与公开 SHA，逐文件代码版本绑定仍保留在私有目录。

对私有已授权捕获复现时，使用[中文功能页](../../../../docs/recon_all/TOPOLOGY_CANDIDATE_STATISTICS.md)
的具名 Python/CLI 参数；`FNITRST1` schema 与严格解析在
`tools/benchmark_recon_topology_candidate_statistics.py`。原参考 run v1
是启动路径命名错误的失败目录，未进行计算；正式捕获 v2 不覆盖它。

## 结果

完整状态：110,534 个 marks、67 个原缺陷条目、29 个 DISCARD、4,723 条候选边。
三次真实调用分别验证历史使用次数 0、1、2 的完整更新，每次更新 38 个
条目。marks、次数、运行均值与原参考全数组逐位一致，最大/P99 误差 0。
原参考两次运行的所有捕获字节重现，FNIT 两次计算所有返回字段重现。

| 原调用 | 最大次数前→后 | 首次完整 API / s | 重复完整 API / s | 加载、NPZ、PNG / s |
| --- | --- | ---: | ---: | ---: |
| 0 | 0→1 | 0.000683157 | 0.000597072 | 0.417467600 |
| 1 | 1→2 | 0.000480369 | 0.000409016 | 0.134537202 |
| 2 | 2→3 | 0.000488716 | 0.000423742 | 0.132061023 |

最后一列计至各自 JSON 报告写出前，不含 JSON 写出。driver 导入后
计时 0.727513172 秒，包含 8 个契约测试、三次回归及结果读写；工具顶层
导入 0.146408407 秒另记，不能把导入后的 driver 时间称冷进程墙钟。
8 项契约测试全部通过，服务器实测 0.029540309 秒。

## 资源、安装与未完成项

Xeon 8369B、CPU affinity 120–123、固定 4 线程；Python 3.11.16。
CUDA 上下文前后均未初始化，TF32 调用方设置不变（matmul false、cuDNN
true），CPU 标量统计不使用 Tensor Core。未采样进程 GPU 显存，字段为
`null`；没有原统计函数独立配对计时，不发布速度比。

只用主页 Conda 环境已有 NumPy，无新依赖，未进行干净环境整例安装
验收。严格复现：本同输入统计阶段通过；优化是否引入退化：本阶段
未观察到新差异；整体指标等效：`not_assessed`。未替换默认拓扑调度，
也不声称已闭合完整 GA、10 分钟整例或 20 GB 全流程显存门槛。

![三次累积后运行均值与逐条误差](reports/call2/statistics_error.png)

图中横轴是原缺陷条目序号，不是脑区或表面顶点编号。
