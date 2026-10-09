# 2026-10-09：完整 filled 的边界与有序传播优化

本目录是冻结同输入阶段回归，基线 `80e3a661a6234d547417f9771b7ab87266581721`
加各报告列出的实际执行模块 SHA；最终提交由协调者整合。两例来自既有
ds000114 CC0 公开 cohort，原始 SHA 及最小冻结输入见
`public_wm_fill_inputs_v1/manifest.json`。参考为旧 FNIT 冻结 filled。
该阶段当前已经使用 FNIT Python，不是原生 C++。

## 已完成

- `boundary_cpu_abba_v1_*`：只换 Torch CPU 静态边界，Python 有序传播保留。
  完整文件 API 两例各 ABBA，0 差异体素、Dice 1，约 1.31×。
- `numba_cpu_abba_v3_*`：Torch CPU 边界＋修复的有序 Numba 传播，八个
  完整输出全部 0 差异体素、max/P99 为 0、shape/dtype/affine 相同。
- `numba_distance_regression_v3.json`：两例两侧四个完整 float32 距离场
  全部元素相同，CC 查询及左右选择也相同。
- `numba_cpu_tests_v3.log`：7 个 CPU 回归通过、1 个 GPU 条件测试未执行。

同一 Xeon Gold 6418H、线程 4，完整文件 API 包括读取、计算和写出。
sub-07 配对中位 36.8056→7.7177 秒，sub-06 38.7433→8.5171 秒。
Numba 堆为 CPU 单线程，不能称纯 GPU 或原始 T1 整例提速。
首次新 signature JIT 完整 sub-07 为 10.5965 秒，单列冷 API。
共享负载及旧调用时间波动原样保留。与不同硬件历史计时不求速度比。

## 失败版与原因

`numba_cpu_abba_v1_*`、`numba_distance_regression_v1.json` 及
`numba_cpu_typeguard_v2` 均为排错记录，曾改变 sub-07/sub-06 的
30/16 个 filled 体素，未用于生产。首差定位到初始 trial 队列；
Numba 嵌套 `initial=True/False` 调用未保留初始仅更新 far 的语义。
分开初始与传播循环后，完整同输入场和最终输出恢复一致。
float32→Python float 的 double 运算也通过显式 np.float64 保留。

`stage_records.csv` 是全部完整 API 原始记录的索引，不删除失败行。
`profile_v1` 为第一次剖析，`profile_phase_v1` 增加初始化/传播分段，
二者用于定位热点，不当作本轮替换收益的配对基线。

## 尚未验证

新同主机原生参考重复性、A100 完整 GPU 文件 API、整例资源/提速和
干净隔离部署尚未完成。整体指标等效保持 `not_assessed`。
GPU0/1的既有其他阶段、H100历史结果不能代替本目录 CPU 回归。

## 复现

`reproduce.sh` 使用用户声明的冻结 input、LUT 及 code commit，逐次
写入新目录。生产路径不读取比较参考文件。完整函数参数、坐标、标签、
异常、原软件命令与出处见 [功能页](../../../../docs/recon_all/FILL_BOUNDARY_TORCH.md)。
