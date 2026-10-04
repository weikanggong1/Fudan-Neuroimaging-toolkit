# v6 固定 RM 状态只读审计

服务器产物：`/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/cuda_ioctl_context_v6`。执行时间 2026-10-04 09:10:54–09:11:34 UTC。本审计只读既有结果，没有启动 GPU 或修改生产程序。

## 实际结果与覆盖

original_worker 4/4、direct_priority 4/4、context_driver 8/8 成功，本轮没有复现 CUDA OOM。

16 个子进程每个记录 401–402 条固定 RM ioctl；合计 6,420 条，与每个子进程 strace 中 type F、read/write、CONTROL size32 / ALLOC size32或48 的调用计数逐一一致。固定头复制失败 0，全部 syscall ret=0。其中 RM_CONTROL32 为 5,044 条，RM_ALLOC32 为48条，RM_ALLOC48为1,328条。strace 所有 ioctl 调用合计31,936条，因此这里的完整覆盖只适用于白名单，不能称为所有驱动内部状态解码；UVM及其他结构没有解码。

## status 86 的官方含义

535.216.03 官方 `nvstatuscodes.h:115` 明确定义 **86 / 0x56 = NV_ERR_NOT_SUPPORTED**。本轮 6,356 条 status=0，64条 status=86。每个成功子进程都是同两个 cmd 各重复两次：

| cmd 十进制 | 十六进制 | 官方可确认的操作 | 本轮观察 |
| --- | --- | --- | --- |
| 545259915 | 0x2080018b | NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS，查询活跃 GPU partition ID；该接口文档列出 NV_ERR_NOT_SUPPORTED 为可能返回值 | 每子进程2次 status86 |
| 8389259 | 0x0080028b | 535.216.03 公共 ctrl0080gpu.h 没有定义此编号；不能猜私有接口的操作 | 每子进程2次 status86 |

64 条非零状态均伴随最终 context/scalar 成功，没有观察到内存不足 RM 状态。不能把成功路径的“不支持”能力查询归因为先前 CUDA OOM，也不能据此宣布初始化修复成功或排除间歇故障。public header 对分区接口的含义不证明本次硬件/驱动为何返回“不支持”。

全部 syscall ret=0；日志 errno=17 有3,040条，errno=2有1,140条，errno=0有2,240条。成功 ioctl 后 errno 没有定义为此次错误，工具按要求保留原值，**errno17 不能解释为 ioctl 失败**。

本轮启动锁内 GPU used=69,392,662,528 bytes，free=15,549,333,504 bytes，是这次观察。它不能解释此前 annotation 或 v4 context 失败的历史原因。v6为 scalar 独立诊断，minimum_free_bytes=2,000,000,000，不能与整例调度预先声明的20,000,000,000 bytes准入要求混同。

## 结论

probe 已在本轮成功子进程实际产生完整白名单记录。v6没有失败臂，因此仍需要未来复现失败时对比固定 RM status、CUDA code 与 syscall 证据；本轮没有新增可实测 OOM 根因。详细计数、原路径、summary SHA、源码依据 SHA 见同目录 `cuda_ioctl_v6_status_audit_20261004.json`。

## 官方固定版本依据

- [nvstatuscodes.h：NV_ERR_NOT_SUPPORTED](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/common/sdk/nvidia/inc/nvstatuscodes.h#L115)
- [ctrl2080gpu.h：活跃 partition ID 查询及可能状态](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h#L3370)
- [ctrl0080gpu.h：本次核查的公开设备 GPU 接口](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gpu.h)
