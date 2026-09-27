# SynthSR：移除 Surfa 导入后的真实 FLAIR 核对

[返回功能说明](../../docs/synthsr/README.md)

本次只改 `SynthSR` 的内存体输入识别：通过 `.data` 和 `.geom.vox2world.matrix` 读取已有体对象，不导入 Surfa。路径读取、1 mm 重采样、PyTorch 网络、锐化和 NiBabel 存盘逻辑均未改动。单例 CLI 在解析 SynthSR 参数后直接运行该模块。

## 输入、指令和输出

输入是 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) `sub-04`、`ses-1` 的原始 3D FLAIR，SHA-256 `49de51a875ade818485c239d37632dce4aafc042f7bf241188460ff9d989c622`。通用 v2 官方权重 `synthsr_v20_230130.h5` 的 SHA-256 为 `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b`。所有运行均在 headcw 的 CPU 上使用 4 线程，`ct=False`、`disable_flipping=False`、`disable_sharpening=False`。

官方 FreeSurfer 8.2.0-1 命令：

```bash
mri_synthsr --i sub-04_FLAIR.nii.gz --o sub-04_synthsr.nii.gz \
  --threads 4 --cpu
```

本包新版命令：

```bash
fnit synthsr --i sub-04_FLAIR.nii.gz --o sub-04_synthsr.nii.gz \
  --weights synthsr_v20_230130.h5 --device cpu --threads 4
```

`--i` 是一幅 3D FLAIR，`--o` 是合成后的 1 mm T1w。Python 对应 `SynthSR(weights=权重文件, device="cpu", lowfield=False, v1=False, threads=4)(image=输入文件, ct=False, disable_flipping=False, disable_sharpening=False)`，返回 `SynthSRResult.image`。`.data` 是 `uint8` 量化图，`.affine` 是 4×4 RAS 几何；`.save(path)` 保存 `.nii`、`.nii.gz`、`.mgz`，或将锐化后、量化前的 `.float_data` 写成 `vol_data` 字段的 `.npz`。本例输出大小为 `217×218×126`，与输入网格不同。

## 配对结果

旧版参考是本仓库提交 `74205a4` 中仍导入 Surfa 的 SynthSR；新版是本次改动。两版用同一 Conda 环境、同一 FLAIR 和权重顺序运行，并分别保存 NIfTI、MGZ、NPZ。

| 检查项 | 新版对旧版 PyTorch | 新版 CLI 对当场运行的官方 CPU |
|---|---:|---:|
| 量化 `uint8` 图不同体素 | 0 / 5,960,556 | 287 / 5,960,556 |
| 完全一致体素比例 | 100% | 99.995185% |
| 最大灰度差 | 0 | 1 |
| 未量化的 `float32` 图不同体素 | 0 / 5,960,556 | 官方无对应 `.npz`，未比较 |
| 输出仿射最大绝对差 | 0 mm | 0 mm |
| 已保存 NIfTI | SHA-256 相同 | 官方与 PyTorch 存盘内容不同 |
| 已保存 MGZ、NPZ | 各自 SHA-256 相同 | 本次未与官方比较 |

新旧 API 的内存数组和仿射也逐元素相同。新 CLI 与旧 API 的 NIfTI 文件 SHA-256 相同。官方 CPU 的 287 个差异均为 1 灰度级；平均绝对差为 `4.815×10⁻⁵` 灰度级。官方结果与先前保存的同例官方 CPU 输出逐体素、逐仿射一致。新 CLI 在拦截 `surfa` 导入的进程中完成，结束时没有 Surfa 模块进入 `sys.modules`。另以这份真实 FLAIR 建立 `surfa.Volume`，确认新版 `_load_image` 能在不检查其具体类的情况下读取数据和仿射；未对该内存输入重复完整网络推理。

| 当次运行 | 墙钟秒数 | 计时范围 |
|---|---:|---|
| 旧版 PyTorch API | 33.26 | 模型加载、推理、保存 NIfTI/MGZ/NPZ |
| 新版 PyTorch API | 21.10 | 同上 |
| 新版 PyTorch CLI | 30.26 | 完整单例，保存 NIfTI |
| FreeSurfer 官方 CPU CLI | 240.03 | 完整单例，保存 NIfTI |

旧版/新版 API 的模型加载、推理和三种格式保存分别为 `1.41/30.43/1.42` 秒与 `0.78/18.86/1.46` 秒。两版的计算路径相同，只各运行一次；上述差值可受缓存和共享节点负载影响，不能归因于去掉一次 Surfa 类型判断。官方与 PyTorch 使用不同的 Python/框架，240.03 秒是这次完整命令的观测值，不代表稳定加速倍数。

已测模式：真实 `.nii.gz` 路径输入、通用 v2 权重、默认图像处理选项、CPU、NIfTI/MGZ/NPZ 输出，以及 Surfa 内存输入的读取分支。尚未在本次迁移中重测完整内存输入推理、`.nii`/`.mgz`/`.npz` 输入、CT、低场或 v1 权重、禁用翻转/锐化、CUDA、批量接口和其余病例。之前 12 例 T1w 的[完整基准](../synthsr/README.md)属于旧版导入路径，不能当作这次修改的多例验收。

[机器可读结果](report.json)不包含影像、权重或服务器账户路径。
