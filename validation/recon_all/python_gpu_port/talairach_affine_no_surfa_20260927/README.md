# Talairach 仿射注册：去除 Surfa 的同输入验收

本次只替换 recon-all 中 `register_talairach` 调用的 `SynthMorph(model="affine").affine_transform(...)` 路径。NiBabel 读取强度与 MGH 几何，PyTorch 继续运行原来的 affine 网络和空间算子，NumPy 组合矩阵并写 type 1 LTA。旧版会为了取得矩阵额外执行两次 Surfa 体积重采样；此路径没有使用这两张体积图。`joint`、`deform`、`rigid`、通用 `SynthMorph.__call__` 与 `apply_transform` 仍保留 Surfa，未纳入本次等价性结论。没有重跑整例 recon-all。

## 实际输入与比较方法

输入是 OpenNeuro ds000114 sub-01 的真实 T1 经 SynthStrip 生成的三张脑图：归档 `fs_sub01/mri/synthstrip.mgz`、重新运行的 FreeSurfer SynthStrip `official_image.mgz`、FNIT NiBabel SynthStrip `new_image.mgz`。三张图的文件 SHA-256 分别为 `326fb4e6013e059a2c7f75f3f83bd8e7c3c8233df388bf40394782e575dc0faa`、`a643b6e1d5c15aa5ebb611029f05a3795e61b1c74eaaa3984840ec7ba6db7aeb`、`0841dc73514a82d37ba8662e647a8fdacb55960a379a3e4cd71fcb6548a0ddb7`。后两张图在 SynthStrip 单独验收中体素与 affine 完全相同，MGH 文件字节因元数据不同。

固定模板 `mni305.cor.stripped.mgz` 的 SHA-256 为 `fff93f13255a8d393c0e787fbfcfaf5eb379e88e955bda7e04e31552568878a4`；affine 权重 `synthmorph.affine.2.h5` 的 SHA-256 为 `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6`。在 gpucw1 H100 上，以同一 Conda 环境、四个 CPU 线程、相同 GPU 和关闭 matmul/cuDNN TF32 的条件成对运行旧 Surfa 与新 NiBabel 路径。每组首次运行和交错的两次稳态运行均重新加载权重；先旧后新及新旧旧新的顺序见 JSON。两组用 `cuda:1`，新 SynthStrip 图像用 `cuda:0`，各组内部 GPU 一致。

| 脑图 | 4×4 元素相等 | LTA / XFM 字节相同 | eTIV 差（mm³） | 旧路径稳态中位数（秒） | 新路径稳态中位数（秒） |
| --- | ---: | --- | ---: | ---: | ---: |
| 归档 SynthStrip | 16/16 | 是 / 是 | 0 | 1.734 | 0.832 |
| 重跑 FreeSurfer SynthStrip | 16/16 | 是 / 是 | 0 | 1.446 | 0.738 |
| 新版 FNIT SynthStrip | 16/16 | 是 / 是 | 0 | 1.657 | 0.791 |

三组的 XFM SHA-256 均为 `8fe0c4b815c743eac9d47ccb7f12c5563e8e4cae466c27867ae750362cf64972`，eTIV 均为 **1,310,265.837338 mm³**。归档和重跑的 FreeSurfer SynthStrip 图像生成的 LTA SHA-256 为 `718808364d8c915e0d5bfda41645ee3b2a78e2d7eef7509ce87ec9a8ea611e2a`；新版 FNIT SynthStrip 图像的 LTA SHA-256 为 `0212a247361487569befb804b0e3af9f00a5d03274d467819f2871d95bb5b95d`，差别在源 MGH 几何元数据；同一输入的新旧 LTA 始终逐字节相同。PyTorch 峰值分配为旧 4497 MiB、新 4433 MiB。共享 GPU 上每组只有两次稳态计时，时间是本机本阶段观测值，不代表整例加速比。

在全新 Python 进程中，将 `sys.modules['surfa'] = None` 后从隔离的新源码直接导入并在新版 SynthStrip 图像上完成注册：LTA、XFM、eTIV 与旧路径相同。独立 `python -m fnit.recon_all.talairach_synthmorph --help` 也成功。此检查仅证明 Talairach affine 调用链不需 Surfa。

## 与归档 FreeSurfer 8.2 结果

同一被试的归档 FreeSurfer 流程由 `fs-synthmorph-reg` 和 `lta_convert` 产生参考文件，神经网络命令被既有 PyTorch neural hook 接管。因此它是官方包装脚本加 PyTorch 网络的参考，不能当作未修改的 FreeSurfer TensorFlow 网络独立对照。新输出与归档参考的 LTA 4×4 最大元素差 **1.2812e-5**，XFM 3×4 最大元素差 **2.288e-5**，eTIV 差 **−0.420220 mm³**。两种 XFM 的注释和写盘方式也不同，文件 SHA 不同。归档包装脚本旧日志曾记录 10.65 秒；设备、缓存和脚本范围与本次隔离计时不成对，不据此计算加速比。

数据见 [`talairach_affine_pair_archived.json`](talairach_affine_pair_archived.json)、[`talairach_affine_pair_officialstrip.json`](talairach_affine_pair_officialstrip.json)、[`talairach_affine_pair_newstrip.json`](talairach_affine_pair_newstrip.json)、[`fresh_no_surfa.json`](fresh_no_surfa.json) 与 [`official_comparison.json`](official_comparison.json)。报告的矩阵、哈希、时间和 eTIV 来自实测输出；尚未验证其他被试或下游整例指标。
