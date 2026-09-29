# TorchBEDPOSTX 源码目录

`TorchBEDPOSTX` 是单被试 Python API，读取 FSL 风格的 DWI、脑掩膜、b-value 和 b-vector，写出概率纤维追踪所需的方向与纤维分数后验样本。CUDA 路径以 float32 和默认 TF32 运行，并编译重复的似然计算；运行时不调用 FSL。

```python
from fnit.bedpostx import TorchBEDPOSTX

result = TorchBEDPOSTX(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU
    nfibres=3,  # 模型：每个体素最多三条纤维
    model=2,  # 模型：多壳层 Gamma 扩散率
    burnin=1000,  # MCMC：burn-in 跳数
    njumps=1250,  # MCMC：正式采样跳数
    sample_every=25,  # MCMC：保存间隔
    ard_weight=1.0,  # 先验：次要纤维 ARD 权重
    chunk_size=4096,  # 资源：每批体素数
    seed=8665904,  # 随机性：固定随机种子
    threads=1,  # CPU 线程数
)(
    subject_dir="/absolute/path/subject",  # 输入：单被试目录
    output_dir="/absolute/path/subject.bedpostX",  # 输出：后验目录
    overwrite=False,  # 写盘策略：不覆盖已有文件
)
```

命令行为 `fnit bedpostx --subject-dir SUBJECT --output-dir SUBJECT.bedpostX --device cuda:0`，独立入口 `fnit-bedpostx` 参数相同。输入文件、全部参数、输出结构、FSL 对应命令和真实数据验证见 [TorchBEDPOSTX 完整说明](../../../docs/bedpostx/README.md)。当前真实数据覆盖 14 体素弱纤维裁剪和 64 体素 FSL 富集裁剪，并已公开[去标识方向轴图](../../../docs/bedpostx/figures/bedpostx_real_ukb_direction_axes.png)；这些小型诊断裁剪仍不能替代无偏全脑验证。
