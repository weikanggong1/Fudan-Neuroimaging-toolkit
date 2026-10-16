# A100完整filled文件API配对

公开ds000114 CC0的两例冻结WM、aseg、LTA和LUT。旧Python→GPU边界+Numba→GPU+Numba→旧Python，同机四线程/taskset4-7，明确cuda:1。每次独立进程，完整API含读取、变换、H2D/D2H、堆传播及压缩写出；解释器/顶层导入/CUDA上下文另计。

| 例 | 旧API中位秒 | GPU初始化+Numba秒 | 速度比 | 四次完整输出不同体素 |
| --- | ---: | ---: | ---: | ---: |
| sub-07 | 51.8721 | 14.9040 | 3.480× | 全部0 |
| sub-06 | 59.4067 | 10.1074 | 5.878× | 全部0 |

最大/P99为0，0/127/255标签Dice1，dtype/affine相同。GPU只生成完整确定性边界及firstvisit；堆/eikonal为保留顺序的CPU Numba，不能称纯GPU。候选不读取参考。参考是既有FNIT冻结filled，迁移环境没有新原生mri_fill程序，不冒称本次官方复现。

`profile.json`绑定输入、LUT、代码基线+执行模块SHA、设备UUID、实际精度、allocated/reserved和目标卡采样。父进程NVML匹配失败为None，非0；整卡上界包含共享进程，不代表自身占用或连续峰值。完整suite退出0、cgroup未增加failcnt见WM planar同目录共享suite记录。

[完整功能与参数](../../../../../docs/recon_all/FILL_BOUNDARY_TORCH.md)。本目录是冻结完整阶段，原始T1整例时间、20GB整例预算、整体指标等效和隔离安装均待验证。公开报告去除私有主机/存储路径；原始/副本SHA在sanitization_manifest.json。复现命令见reproduce.sh；输入和参考必须已获得相应公开许可，原始影像不提交Git。
