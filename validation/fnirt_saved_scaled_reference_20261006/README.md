# FNIRT 保存参考图与 ScaledRef：同点逐位核对

## 1. 功能与结果

复用[冷初始化缓存捕获](../fnirt_native_cache_prefix_result_20261006/README.md)和已有 Python checkpoint，在同一固定参数点核对参考图、强度参数及缩放后的参考图。两道输入身份门通过后，16128个派生 FP32 ScaledRef 值与原生保存图全部逐位相同。生产实现、GPU和原阈值保持。

| 核对项 | 结果 |
| --- | --- |
| checkpoint fixed 改为原生 X连续字节序 | SHA 与原生借用 Ref 相同 |
| checkpoint scale 与原 FP64 参数末8字节 | 逐位相同 |
| fixed 转F64 × scale，再转F32 | 16128/16128逐位同，maxabs0、relative L2 0 |

## 2. 输入与输出

只恢复 checkpoint 的 `fixed`（F32，XYZ=24×28×24，C-order）和 `scale`（F64标量）；只读取原1177元素F64参数文件末8字节及原捕获的 `native_sref.f32`。显式将XYZ C-order重排为X连续存储，不翻转图像。

输出仅为私密原始 JSON 的身份门、bit计数、幅度统计、源码/输入前后SHA、内存和诊断时间；公开摘要见[manifest](manifest.public.json)。不发布图像、参数值或私密路径。

## 3. Python与命令行范围

此处没有新增生产 API/CLI。诊断沿当前 `registration.py` 的原表达式执行一次：`(scale * fixed.to(scale.dtype)).to(fixed.dtype)`。私密控制使用相同的IEEE64乘法和IEEE32舍入，不重估scale。仅当参考图字节及scale末8字节都相同才进入乘法；任一身份不匹配即停。

## 4. 对应原实现

原 `ScaledRef` 是 FNIRT 的内部步骤，没有单独官方CLI；固定GLOBAL_LINEAR mapper以Double系数乘Float参考图，再存Float。本次没有重跑官方程序，使用上轮已经保存的原生操作数。源身份与原数学位置已单独核对，见manifest中的7个source roles。

## 5. 实测范围、时间和资源

一次保存状态控制完成，worker0.699538秒、supervisor0.839380秒；worker最大RSS19,849,216B。时间包含身份校验、NPZ读取、16128次乘法及统计，既不是完整FNIRT速度，也不与native命令比较。全部source/input字节前后相同；实际8核亲和设置、有限期限、进程监督及共同锁门通过。原始8份文本和当前owner退出由根任务复核。

新native、scale reduction、residual、SSD、gradient、H/PCG、完整配准与GPU调用均为0。没有新增脑图或重跑T1。

## 6. 解释与下一步

前轮checkpoint没有直接保存scaled_fixed，本次结果是**新增同点乘法后派生**，不能改写为原NPZ直接成员比较。它排除了本次固定状态下参考图、scale参数和ScaledRef乘法的差异；原生重采样图仍有2425个不同FP32 word。下一步核对重采样缓存与导数路径，再看梯度、优化轨迹和完整非线性输出。历史solve3 warm cache及完整配准等价仍未证明。

## 7. 来源

原FSL 6.0.7.4的fnirt_costfunctions与intensity_mappers接口，来源和固定SHA见[上轮来源](../fnirt_native_cache_prefix_result_20261006/SOURCE_BINDINGS.public.json)。当前FNIT数学文件SHA与`7ff215ee`基线相同；没有复制原SDK、二进制或MRI数据。
