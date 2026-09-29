# TorchBEDPOSTX 当前验证

[功能、参数、输入输出和官方命令](../../docs/bedpostx/README.md) · [真实数据汇总报告](report.public.json) · [方向轴绘图脚本](plot_real_axes.py)

一例真实 UK Biobank DWI 分别取 14 个弱纤维体素和 64 个交叉纤维富集体素，使用三纤维、model 2、ARD=1、burn-in 1000、后续 1250 跳、每 25 跳保存一次、种子 8665904。FSL 6.0.7.22 的 `xfibres` CPU 输出用于精度对照，14 体素另运行 `xfibres_gpu` 核心测速。FNIT 在 gpucw1 的 H100 GPU 1 上运行；该卡已有其他任务，测试顺序执行，并以 `torch.cuda.set_per_process_memory_fraction(0.2)` 限制 PyTorch 分配。[报告](report.public.json)给出源码及图片 SHA-256、两种裁剪的时间、峰值已分配显存和逐项精度。

14 体素优化前后 19 个 NIfTI 输出逐元素相同。64 体素的未编译结果与旧版逐元素相同，编译后 MCMC 链发生分叉；报告采用编译后输出重新计算对 FSL 的指标。该裁剪按已有 FSL 后验富集次要纤维，适合检查弱纤维和方向轴，不代表全脑无偏精度。FSL GPU 的时间仅为 `xfibres_gpu` 核心，不包含拆分、合并或调度。

![真实 DWI 诊断 ROI 的次要纤维方向轴](../../docs/bedpostx/figures/bedpostx_real_ukb_direction_axes.png)

图片来自当前编译版与 FSL 的真实 64 体素输出。只绘制双方平均纤维分数均 ≥0.1 的体素：f2 为 61 个，f3 为 26 个。原始 DWI 和完整后验留在授权服务器；仓库仅保存汇总指标和去标识图。
