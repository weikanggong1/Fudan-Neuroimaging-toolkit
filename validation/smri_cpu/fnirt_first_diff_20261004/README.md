# FNIRT 真实 GM 首差定位（2026-10-04）

## 1. 范围与结论

固定已保存的官方 GM、官方 FLIRT、`template_GM_v1` 和官方 mask，保留完整 GM schedule，仅设 `miter=1,0,0,0`。本轮比较首级初始状态、一个 accepted update、以及共享真实 FP64 系数的第一次 `4→2` 网格转换。没有重跑原始 T1 pipeline，没有改生产 FNIRT 或 CUDA。

**首次数值差异在 moving GM 平滑。** GM affine determinant 为 `+0.4839547278`。NEWIMAGE 内部先转换到 radiological X 方向；FNIT 在 nibabel 数组原方向累计 FP32 卷积。两者归一化数组逐值相同，但原方向平滑有 1,296,806 个值不同，最大 `3.8147e-5`、RMSE `1.4002e-6`（归一化强度）。仅在诊断中先翻转 X、调用同一成熟平滑、再翻回，完整 `224×288×288` moving 图与官方 **逐值相同**。[完整对照](comparison.public.json)、[翻转消融](flip_smoothing.public.json)。

首轮差异很小，不能解释为完整估计偏差已解决。后续 CPU 平滑修复另行验证与提交；本报告不包含该修复结果。

## 2. 固定输入与执行

源码基点 `b4a55d8ff2738d9cfd9744f9759f62dead9bc4fc`；实际调用冻结源为服务器 `gems/fnirt-jacobian-v1/source/src`。完整文件 SHA-256、原生程序及动态库、失败记录在 [binding](binding.public.json)。输入 SHA、header/affine、全部 solver QC 在 [comparison](comparison.public.json) 中。

- 真实公共 T1 来源为既有 ds003138 v1.0.1（CC0）记录；本轮只读取已保存官方 FAST GM，没有再次 FAST 或 FLIRT。
- GM：`FNIT/runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt/T1_brain_pve_1.nii.gz`。
- FLIRT：同目录 `gm_affine.mat`；转换使用实际 NIfTI header zoom，未用 affine 列范数替代。
- 模板：`FNIT/legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz`。
- mask：FSL 6.0.7.4 `MNI152_T1_2mm_brain_mask_dil.nii.gz`；完整 recipe 中前三阶段不启用它。
- nodecw7，同组 CPU `32,36,40,44,48,52,56,60`，8 Torch/OMP/BLAS threads，`nodecw7.gems.cpu8.lock`。
- 独立官方程序为已安装 FSL 6.0.7.4 `fnirt`（2203.0）。额外 C++ oracle 只用于 benchmark，编译、链接和原软件调用均不进入 FNIT 生产。
- 首级 grid `24×28×24`、voxel `8 mm`、knot `5 voxel`、control `7×8×7`。不能截短 schedule：`FullResKsp` 依赖该 process 最后一个 subsampling。

## 3. 实际首轮结果

| 状态 | 官方与 FNIT 差异 |
| --- | --- |
| 归一化 moving / fixed | 完整数组逐值相同 |
| fixed 平滑及首级采样 | 完整数组逐值相同 |
| 初始有效 mask | 逐值相同，14,906 体素 |
| 初始 warped，归一化单位 | max `1.0300e-4`，RMSE `7.3343e-6` |
| 初始 SSD | 官方 `419.1605287981`；FNIT `419.1605194904` |
| 有效 mask 内 image derivative | max `3.9101e-5`，RMSE `3.0022e-6` |
| direct gradient 相对 L2 | `1.596e-8` |
| Hessian diagonal 相对 L2 | `8.921e-12` |
| 固定向量 Hessian 乘积相对 L2 | `1.393e-9` |
| 首 accepted update（stock 官方） | 双方 mask 14,820；SSD `73.0927213271 / 73.0927191860` |
| 首 accepted coefficient（stock FP32 与 FNIT FP32） | max `1.2040e-5 mm` |
| 首 accepted warped（stock） | max `2.1362e-4`，RMSE `1.2109e-5`，归一化单位 |

自编译 oracle 的初始 fixed/warped/mask 与 stock 原生程序逐值相同，首 accepted warped 也逐值相同；26 个 FP32 系数有末位差，最大 `2.3842e-7 mm`。不能把该 oracle 首 accepted coefficients 写成 stock 逐 bit 相同。stock ASCII gradient 的 10 位输出精度差单列；oracle 使用 17 位保存 FP64 vector。

同一 FNIT 首轮 `reference / optimized` 的平滑、mask、SSD、gradient、diagonal、warped 全部逐值相同。Hessian probe 最大差 `1.1102e-16`，accepted coefficients 最大差 `7.5495e-15 mm`，PCG 均 5 次。因此此例首轮没有 Gram 量级偏差证据。

同一官方 initial 数组按 FSL Jte 的顺序（先 float derivative×residual，再 double adjoint 和除 mask count）重算 gradient，相对 L2 `2.9394e-10`；原 FNIT 数组按该顺序为 `1.4772e-8`。累计顺序存在差异，但本轮不把它作为新的生产修复。

## 4. 一次共享系数转换

使用 oracle 保存的真实首 accepted **相同 FP64 参数**，双方只做 `24×28×24, 8 mm → 46×55×46, 4 mm` 的 `ZoomField`，knot 保持 `5 voxel`。FNIT/官方 control 为 `12×14×12`；最大系数差 `1.3464e-12 mm`，RMSE `1.4034e-13 mm`，scale 相同。[转换指标](zoom.public.json)。本 checkpoint 未发现网格转换的量级错误。

既有 v4 完整估计报告在第三级触发 topology projection，范围由 `0.1116–2.4243` 调整到 `0.2008–2.4259`，第四级再次调整。旧官方完整运行未保存内部逐轮状态。因此，这些历史 QC 只能指定后续检查位置，不能直接证明 topology 是首差来源；本次没有重复完整估计。

## 5. 复现

Python：[fnit_probe.py](fnit_probe.py) 保存首轮状态；[compare_states.py](compare_states.py) 比较 native/stock/FNIT 及归约顺序；[compare_zoom.py](compare_zoom.py) 比较一次共享系数转换。输入和参考数组只在服务器保存，不随报告发布。

```bash
# 以下环境变量均指向已校验文件；使用新输出目录，保留冻结旧结果。
export PYTHONPATH="$FNIT_FROZEN_SOURCE/src"
python fnit_probe.py --moving "$OFFICIAL_GM" --fixed "$GM_TEMPLATE" \
  --mask "$REFERENCE_MASK" --affine "$OFFICIAL_FLIRT" \
  --execution reference --output "$NEW_RUN/fnit-reference"
python fnit_probe.py --moving "$OFFICIAL_GM" --fixed "$GM_TEMPLATE" \
  --mask "$REFERENCE_MASK" --affine "$OFFICIAL_FLIRT" \
  --execution optimized --output "$NEW_RUN/fnit-optimized"
```

对应独立官方命令：

```bash
# --debug 仅供官方隔离诊断；完整 schedule 保持原配置。
fnirt --in="$OFFICIAL_GM" --ref="$GM_TEMPLATE" --aff="$OFFICIAL_FLIRT" \
  --refmask="$REFERENCE_MASK" --config=GM_2_MNI152GM_2mm.cnf \
  --miter=1,0,0,0 --debug=2 --cout="$NEW_RUN/coefficients" \
  --logout="$NEW_RUN/official.log" --verbose
```

[official_probe.cpp](official_probe.cpp) 调用已安装的 FNIRT cost、image derivative、Hessian 和 optimizer；[official_zoom.cpp](official_zoom.cpp) 仅调用官方 `splinefield::ZoomField`。headcw 使用 `g++ -std=c++17 -O0`，installed include/source/libs，显式链接 `lapack/blas/pthread`，两个最终程序退出 0。最大迭代停止是有意设置，不代表优化收敛。此次不报告速度收益。

## 6. 失败与后续

错误的单级 v1 schedule（knot 改为 1）未用于有效结论。首次编译缺 lapack、node7 无 g++、第二次链接缺 pthread、一次 zoom 比较漏设 PYTHONPATH 均保留在 binding 中。没有隐藏失败或改旧结果。

最小下一步是 CPU-only 按实际 input/template affine 方向执行同一平滑，保持 CUDA 原分支，然后用固定同输入完整 FNIRT stage 核验。完整 nonlinear estimation、导数末位累计和 later topology 的精度仍需独立验证；本报告不宣布官方等价。

## 7. 原实现

[FNIRT 官方源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)、[NEWIMAGE](https://git.fmrib.ox.ac.uk/fsl/newimage)、[basisfield](https://git.fmrib.ox.ac.uk/fsl/basisfield)，许可与实际 source SHA 见项目 `_vendor_fsl` 和 binding。参考：Andersson、Jenkinson 与 Smith，*Non-linear registration, aka spatial normalisation*，FMRIB TR07JA2（2007）。
