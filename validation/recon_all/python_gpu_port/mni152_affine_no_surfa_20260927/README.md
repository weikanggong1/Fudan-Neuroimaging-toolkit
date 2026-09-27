# 可选 MNI152 affine 注册去除 Surfa：真实 T1 配对验证

本次只把 `register_mni152_affine` 中的 `SynthMorph(model="affine")` 调用改为 `affine_transform`。裁切、full-MNI 到原图体素矩阵组合和辅助分割代码未改；通用 `joint`、`deform`、`rigid`、`SynthMorph.__call__` 与 `apply_transform` 仍依赖 Surfa。没有重跑 MCA/dura、静脉窦、`brain.finalsurfs` 或整例 recon-all。

## 输入与输出

被试 `mri/orig.mgz` 来自去标识 OpenNeuro ds000114 sub-01 真实 T1，SHA-256 为 `7ac9286da6923e127e746249ca40137f5cff3d5914bfddf11dc1a92aab4353c9`。外置 cropped/full MNI152 模板的 SHA-256 分别是 `ef89b7aa615dcb4a68a1cad725a9b99314227f9777ed86f42aae61e4e89da19a` 和 `e4e1a25b66fef6b2cde8e4916d4a03d4369de6f24b0d3d4736baffc3d4a8c575`；affine 权重为 `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6`。程序输出 `invol.crop.nii.gz`、type 1 `aff.lta` 和 type 0 `reg.targ_to_invol.lta`，最后一项将完整 MNI152 体素映射到被试 `orig.mgz` 体素；函数返回最后一项路径。[函数参数和官方命令](../../../../docs/recon_all/MNI_AUX_CHAIN.md)逐项说明输入与输出。

在 gpucw1 的同一 H100 GPU0、同一 Conda 环境和四个 CPU 线程下，旧 Surfa 与新 NiBabel 路径成对运行。实际 cropped T1 及 MNI152 模板的图像数据、voxel-to-world 矩阵、方向、中心和 256³ 网络坐标矩阵与 Surfa 相同，见 [`mni_geometry_probe.json`](mni_geometry_probe.json)。首次生成的 `aff.lta` 仅在 NIfTI 第三轴体素大小的最后一位小数不同；将 NIfTI voxel size 直接读取 header zoom 后，三份输出均逐字节相同：

| 输出 | 新旧 SHA-256 | 新旧矩阵最大元素差 |
| --- | --- | ---: |
| `invol.crop.nii.gz` | `4a38fed518e638466ac7092793a759401bc856f4ef62b8b215c69b6199a427c0` | 图像文件逐字节相同 |
| `aff.lta` | `41dfb20bfdaed2ff29b3ffc204be885aa15f97b86abb030b22ee56ec6a1f7335` | 0 |
| `reg.targ_to_invol.lta` | `dace59d73b2d9d05a7cbd6dd746d3a01f56f186ad4a1d017029174b2eb921bce` | 0 |

从全新 Python 进程设置 `sys.modules['surfa'] = None` 后，`run_mni_aux_chain`、`mri_mcadura_seg`、`mri_vsinus_seg` 均可导入；单独运行 `register_mni152_affine` 也生成上述相同的三个哈希。两个辅助分割函数本次只检查导入与直接调用链源码，没有重新执行推理。证据见 [`mni_affine_pair.json`](mni_affine_pair.json) 与 [`fresh_no_surfa.json`](fresh_no_surfa.json)。

## 时间与官方参照边界

同机交错两次稳态运行包含裁切、权重加载、仿射网络、两份 LTA 写盘：旧 Surfa 路径 **1.347、1.686 秒**，新 NiBabel 路径 **1.039、1.110 秒**；PyTorch 峰值显存分配分别约 **4448、4427 MiB**。首次调用受进程和 CUDA 初始化影响，记录在 JSON，不用于比值。共享 GPU 上每种实现只有两次稳态计时，不能外推整例速度。

已有[同一真实 T1 的官方对照](../mni_aux_connected_20260927/README.md)报告：旧 FNIT GPU affine 组合的最终 LTA 与官方最多相差 **0.068515 体素**；另一次 CPU 链的最终 LTA 最多相差 **5.06e-5 体素**，辅助分割体素一致。本次替换与旧 FNIT 的两个 LTA 完全相同，因此继承旧路径精度，尚未消除 GPU 与官方的差距。该归档官方 `mri_synthmorph -m affine` CPU 日志约 **107.57 秒**，与本次 GPU 成对计时来自不同运行条件，不构成等价加速比。
