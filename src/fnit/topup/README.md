# TorchTOPUP 源码目录

`TorchTOPUP` 用 PyTorch 复现 FSL 6.0.7.4 `b02b0.cnf` 的 AP/PA 两帧 b0 路径。它写出 FSL 角色对应的 coefficient、movement、Hz field、校正 b0 和 Jacobian 文件；运行时不调用 FSL。

```python
from fnit import TorchTOPUP

model = TorchTOPUP(
    device="cuda:0",  # 计算设备；可改为 "cpu"
)
result = model.run(
    imain="B0_AP_PA.nii.gz",  # 输入：两帧相反相位编码 b0
    datain="acqparams.txt",  # 输入：每帧的相位编码方向和总读出时间
    out="fieldmap_out",  # 输出根名：coefficient 与 movement 文件
    fout="fieldmap_fout",  # 输出根名：Hz 场图
    iout="fieldmap_iout",  # 输出根名：两帧校正图
    jacout="fieldmap_jacout",  # 输出根名：两帧 Jacobian
    overwrite=False,  # 是否覆盖已有输出
)
```

对应的原软件命令：

```bash
topup --imain=B0_AP_PA.nii.gz --datain=acqparams.txt \
  --config=b02b0.cnf --out=fieldmap_out \
  --fout=fieldmap_fout --iout=fieldmap_iout --jacout=fieldmap_jacout
```

`imain`、`datain` 与六类输出的 shape、坐标和文件结构见[完整说明](../../../docs/topup/README.md)。2026-10-02 补齐默认 regrid，并修复周期平滑、共同几何 mask、cubic 采样精度及层间场传递，前五层改为联合 GN/LM、后四层 SCG；复用 FNIRT 的矩阵无关 PCG 和样条算子。图像与输出 float32，系数和求解器 float64。旧 `optimizer_steps_per_iteration` 已移除，`maximum_iterations` 按官方 `miter` 语义解释。

一例真实 `104×104×72×2` AP/PA 输入的独立完整估计：场图信号区 `r=0.9999998778`、RMSE `0.010799 Hz`；固定原软件脑区的两帧 iout `r=0.9999965618`、RMSE `12.849295`、最大误差 `1037.257813`。最大平移/旋转差分别为 `5.340487×10⁻⁵ mm` / `6.659318×10⁻⁶ rad`。固定官方保存参数的独立渲染检查则得到脑内 iout RMSE `0.011135`，支持区 Dice `1`，分清了估计和渲染的差异。新版仍非逐元素一致；[详细结果与方法](../../../validation/topup/README.md)分别记录三项真实检查。历史 FNIT 0.14.0/0.16.0 L-BFGS 对照保留为历史基线。

H100 三次独立进程的 API 读取、计算和写盘为 `17.864061/6.704100/7.094846 s`，中位数 `7.094846 s`；内部同步计算中位数 `6.368049 s`，含启动和统计的完整进程中位数 `10.82 s`，峰值 CUDA 已分配显存 `389,119,488 bytes`。主要精度报告使用第 2 次运行，其 API 时间为 `6.704100 s`。三次精度统计相同，第一轮时间差的原因未隔离。

同输入 FSL 两次独立运行的外层同步进程墙钟为 `216.987430/205.233591 s`，中位数 `211.110510 s`，相对 FNIT 含启动和统计的完整进程中位数 `10.82 s`，观测时间比 `19.51×`。FSL CPU 16–23、FNIT CPU 8–15，均 8 线程；系统 cache 未清空，节点未独占。FSL GNU time 两次原生进程墙钟另为 `216.98/205.22 s`。完整参考链 CPU 0–7 的单次 `215.4733 s` 没有混入两次中位数。API 与完整进程的计时范围不同，输出尚非逐元素相同，该时间比不表示数值等价或整链加速。

## 目录结构

- `core.py`：输入检查、官方九层 schedule、周期预滤和平滑、场/运动目标函数、解析梯度、联合矩阵无关 GN 与 LM/SCG、结果 QC。
- `_sampling_cuda.py`：固定图像系数的周期 cubic 强度与三个空间导数融合计算；本路径使用 `official_precision=True` 的 double 权重/累加。
- `io.py`：intent 2016 coefficient、intent 2018 Hz 场及 Analyze-style Jacobian header。
- `ukb.py`：独立的 UKB b0 选择准备步骤，本次 TOPUP 核心修复未修改该文件。
- `cli.py`：`fnit topup`/`fnit-topup` 参数入口，仅选择 `b02b0.cnf`。

无新增运行依赖或外置权重。上游实现、许可、输入输出结构、完整变量注释示例及历史 benchmark 见[功能说明](../../../docs/topup/README.md)。数值正确性的 CPU 数学测试不替代真实数据 benchmark；新版仍不声明 FSL 数值等价。

2026-10-02 最终源码删除未调用的 core helper 和旧 float CUDA 分支。`core.py` SHA-256 为 `d6b9838ca62ffeaa32b608a860520fc3feb5e66582064303a6de47f199e2b8e8`，`_sampling_cuda.py` 为 `ee19a764849bda80137312ed3ab8f1bf0aaef5e13f852f49f435e511b40d2427`；上述数值及固定参数、初始梯度、sampler 检查均已在该源码重跑完成。真实 sampler 的全部强度、三个导数和 valid 精确匹配独立 double tensor 数学参照，暖采样时间比分别为 `89.002×/91.043×`，不代表 FSL 或整组件速度。[104 项组合回归](../../../validation/dmri_pipeline/regression_synthstrip_topup_20261002.public.json)通过；完整 dMRI 链的精度和计时单独记录。
