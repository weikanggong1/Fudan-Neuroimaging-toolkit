# Talairach LTA 坐标转换

`run_input_talairach_chain` 在 SynthMorph 仿射注册后，需要将 `aff.lta` 的 RAS 到 RAS 矩阵写成体素到体素的 `talairach.xfm.lta`。这一步现由 `write_voxel_lta_from_ras` 在仓库内用 NumPy 完成，不导入 Surfa。Talairach affine 注册也已采用[NiBabel/PyTorch 实现](TALAIRACH_AFFINE_NIBABEL.md)；其他通用 SynthMorph 路径与部分重建阶段的 Surfa 依赖尚未全部替换。

## 输入、输出与调用

```python
from fnit.recon_all.input_talairach_chain import write_voxel_lta_from_ras

write_voxel_lta_from_ras(
    source_lta="subject/mri/transforms/synthmorph.mni305/aff.lta",  # 输入：单矩阵、type 1 RAS_TO_RAS LTA，含两端体积几何
    output_lta="subject/mri/transforms/talairach.xfm.lta",  # 输出：type 0 VOX_TO_VOX LTA，保留原始几何文本
)
```

`source_lta` 的 `1 4 4` 后有 4×4 仿射矩阵，`src volume info` 和 `dst volume info` 各记录体积尺寸、体素大小、x/y/z RAS 轴及中心 RAS 坐标。函数用这些字段还原两个 voxel-to-RAS 矩阵，计算 `inv(dst) @ ras @ src`；`output_lta` 是可供后续 FreeSurfer LTA 读取器使用的文本文件，包含 type、4×4 矩阵和未改变的两端几何区块。函数不返回值。

官方重建中相应的完整注册步骤为：

```bash
fs-synthmorph-reg --i synthstrip.mgz --t mni305.cor.stripped.mgz \
  --affine-only --o transforms/synthmorph.mni305 --threads 4
mri_synthmorph -m affine -t transforms/synthmorph.mni305/aff.lta \
  synthstrip.mgz mni305.cor.stripped.mgz -j 4
```

第一条命令估计 `aff.lta`，第二条命令在官方流程内部生成 Talairach 变换；官方没有单独暴露这里的 LTA 坐标转换命令。输入图像、模板和权重决定 `aff.lta`；本函数只转换已有矩阵，不重新估计仿射。

## 真实 T1 验证

输入来自去标识 OpenNeuro ds000114 sub-01，固定 `aff.lta` SHA-256 为 `49d6b8e2aaf2e2111fd86fe6ecb7c0e0ed99a9264f0b22a4f95e0e7545792d5c`。[输入](../../tests/recon_all/data/talairach_ras_real.lta)和[旧 Surfa 转换输出](../../tests/recon_all/data/talairach_voxel_oracle_real.lta)各约 1.3 KB，可在仓库内复核。NumPy 输出 1,353 字节与旧输出**逐字节相同**，SHA-256 均为 `46d0228ff5f6cc6ad213964d791e13f7e78003dd04932e82418fd6083d5b3f48`。这同时核对 type、全部矩阵元素和体积几何文本。

归档官方 `talairach.xfm.lta` 与上述两份结果的最大矩阵元素差为 `7.63e-6`，来自完整官方变换生成路径；本次替换只证明旧 FNIT 子函数的输出不变，不声称完整 Talairach 输出已逐字节匹配官方。

在 headcw 新 Conda 环境中，同一输入和相同文件写入范围各预热一次后交错运行 20 次：NumPy 实现中位数 **0.363 ms**，旧 Surfa 调用中位数 **1.187 ms**。此阶段远小于网络注册时间，不能把这项差值解释为整例提速。正式整例仍需重新核对 LTA、eTIV 和下游体素。
