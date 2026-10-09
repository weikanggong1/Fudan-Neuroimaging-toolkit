# WM/aseg PyTorch 子阶段与完整混合编辑验证

## 范围

本目录是2026-10-09的公开ds000114 sub-07/sub-06、冻结同输入WM编辑验证。
不是原始T1整例，不包含N4、WM初始`mri_segment`或最终脑区指标。
候选只读取FNIT自产WM/brain/aseg/EntoWM；参考WM只用于输出比较。
源码基线为`937263e04eb1be1a5053952ff2b673213cdf966f`，实际执行代码是
该基线及报告中逐模块SHA绑定的overlay。不能将这里的阶段提速写成整例提速。

## 实现

- GPU：静态fill候选、固定filled的26邻域传播、海马下方WM、MTL上缘
  spackle、SCM-HA、EntoWM及复用既有ACJ内核。
- CPU：反馈依赖的有序Numba扫描、尚只覆盖可证明no-op的remove_paths。
- fill在原生首遍WM分支消费，不预填整卷、不重复后置fill。
- 既有Numba核心漏掉的cerebellar exterior 6/45、temporal WM 186/187已修复。
- 无半精度。二值max-pool以float32表示0/1，整数标签和uint8编辑值保持不变。
- 不新增依赖；PyTorch、NumPy、nibabel、Numba、SciPy沿用主页环境。

完整混合文件API只接受aseg**原存储**int32或float32精确整数。
固定原生源码MRIread后直接MRIclone，byte-access范围由原itemsize决定；
int16/uint8等不能先转int32再套用证明。几何no-op证明不成立仍抛
NotImplementedError，MTL原CPU未定义边界候选仍拒绝；没有按被试SHA例外。
旧固定被试公共入口及原生生产默认保留。late-only API可接受更多整数存储。

## 当前结果

同H100 PCIe、显式cuda:1、4线程、相同文件、ABBA，包含加载、转换、
传输、计算、压缩和写出；JIT与页缓存已热。共享节点秒数是本轮观察。

| 输入 | Conda原生中位数 | 混合中位数 | 阶段速度比 | 不同体素 | 最大/P99 | 原生重复体素差 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| sub-07 | 34.4076 s | 8.1870 s | 4.2027× | 0 | 0/0 | 0 |
| sub-06 | 34.5813 s | 7.9455 s | 4.3523× | 0 | 0/0 | 0 |

两例geometry/dtype精确一致，每个整数值Dice1。WM保存强度和编辑值，
这里的逐值Dice不是完整aseg/aparc脑区等效验收。完整WMAPI使用
`-keep-in -fix-ento-wm ... 3 255 255 -fix-acj ... 255 255 -fill-seg-wm -fix-scm-ha 1`。
参考是固定源码Conda构建程序，程序SHA、源文件SHA、cma.h SHA均见JSON。

进程采样最大1,377,828,864字节（1.378GB，1.283GiB），PyTorch
allocated/reserved最大738,197,504/826,277,888字节。采样按UUID/PID筛选，
不把他人进程加入本任务。请求间隔0.25秒，实际最大间隔4.481/3.138秒，
查询延迟可能漏过短峰值；只报告采样最大，不能保证连续峰值。
原生子进程只运行CPU，本API没有GPU子进程。

## 文件与版本

- `sub07-stage-v8/report.json`、`sub06-stage-v3/report.json`：原始未改写的
  ABBA报告；包括全部输入/模块SHA、affine、dtype、逐值Dice、原生重复性。
- 各自`process_memory.json`：完整采样；raw report遗留的
  `cuda_process_sampling.status=not_measured`旧占位字段由真正的
  `process_memory`字段覆盖，本次采样不能解释为0。复现脚本已修正兼容字段。
- `summary.json`、`summary.csv`：相同结果的简表，不改变验收标准。
- `final_guard_report.json`、`final_guard_tests.log`：最后只增加原存储guard
  后的源码SHA、12项CPU直接函数回归、两例真实aseg的存储/整数校验。
  两例原存储均`>f4`且每个体素是精确整数。
- `reproduce.sh`：按显式变量运行独立benchmark；不含许可证或私有认证信息。

ABBA执行Torch模块SHA为`3cf6368da9ff...`；最后模块SHA为
`920dc12aee18...`，仅多出原存储guard及中文说明；这两例接受的算子未变。
原始报告保持其实际执行SHA，不将它改标为最后版本。完整原始T1候选
由协调者使用最后模块重新验证。12项测试通过直接调用test函数及raises
兼容器运行，服务器无pytest包；不宣称完整pytest运行。

## 尚未完成

完整Torch N4、`mri_segment`、通用byte-path搜索和有序扫描纯GPU化未完成。
本目录不包含整例10分钟、整体脑区等效或无预装软件隔离部署的结论。
曾有首次CUDA context/1–12字节上传OOM而H100仍有充足空闲，失败日志留在
独立运行目录；不加入生产自动重试，也不把它归因于本算法显存超限。

后续可先优化remove_paths几何证明的整卷5次SciPy膨胀：只需比较最小MTL
种子x减5是否触及byte-addressable区间，严格保持接受/拒绝集合；本次尚未
实施，避免改变已冻结的两例算子。第一轮有序扫描仍是本混合阶段CPU热点。
