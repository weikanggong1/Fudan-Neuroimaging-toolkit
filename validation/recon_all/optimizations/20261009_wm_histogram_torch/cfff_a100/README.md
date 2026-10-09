# A100完整WM三方配对

sub-07 V1完整ABBA，sub-06 V1仅3/6行，不完整；独立V2完整结果单列。保留未完成日志，不把它作为完整性能对照。

- 只测完整WM文件API/原生命令，包含读取、搬运和压缩写出；不含解释器、导入和CUDA初始化，不是原始T1整例。
- “cpu”只是旧NumPy histogram；两种Python完整路径都选择同一GPU，其他既有阶段相同。原生命令不使用diag-write。
- 两例旧/新Python完整输出均没有新增差异，原生已有64/235体素差异单列；不得称随机尾差。
- V1进程显存PID采样没有匹配行，峰值None不等于0。V2 target-card total-free是共享整卡上界，不是FNIT自身占用。
- 冻结包基线完整commit为a756fffbcff0ed46e8849aef0c25123768c67d60。V1原code_commit短值保留，执行模块/程序/输入/脚本均另有SHA。V2直接使用完整commit。
- 原始报告存于授权环境，公开副本去除私有CFFF主机和存储路径，原始/副本SHA见sanitization_manifest.json。

详见[中文功能页](../../../../../docs/recon_all/WM_HISTOGRAM_TORCH.md)。整体等效、原始T1整例提速和整例显存保持未评估。
