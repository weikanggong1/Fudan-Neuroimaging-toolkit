# EDDY 匹配输入验证

[功能说明](../../docs/eddy/README.md) · [机器报告](report.matched_20260928.json) · [比较脚本](compare_matched.py)

2026 年 9 月 28 日，在 gpucw1 用一例真实 UKB 格式 AP/PA dMRI 比较当前 TorchEDDY 与 FSL 6.0.7.4 `eddy_cuda10.2`。两次运行使用完全相同的 AP 图像、bval、bvec、脑掩膜、FNIT TOPUP 系数、acqparams、index 和参考帧。报告记录这些输入和两份 4D 输出的 SHA-256。数据含 105 个 volume，其中 b0 为 5 个，b≈1000 和 b≈2000 各 50 个；掩膜内有 242,316 个体素。

| 范围 | 全部体素与 volume 的 r | 每体素跨 volume 的 r 中位数 | MAE |
|---|---:|---:|---:|
| 全部 | 0.981975 | 0.986378 | 240.98 |
| b0 | 0.986419 | 0.818843 | 429.00 |
| b≈1000 | 0.941577 | 0.780127 | 261.41 |
| b≈2000 | 0.918361 | 0.754824 | 201.74 |

旋转后 b-vector 的无符号夹角均值为 0.663°；平移参数 MAE 为 0.694 mm，旋转参数 MAE 为 0.00773 rad。TorchEDDY 与 FSL 分别标出 15 和 16 个离群切片，只有 1 个重合。当前实现**未达到逐体素数值等价**。

FSL 命令从进程启动至输出写完耗时 1,056.49 s；TorchEDDY 的 54.49 s 是完整 pipeline 中 EDDY 阶段计时。两者计时范围不同，节点当时也有其他 GPU 任务，不能据此计算可靠加速比。此报告只验证一个真实被试，不能代表群体误差分布。

比较脚本读取两份输出、相同输入掩膜及 bval，检查形状和 affine，然后计算掩膜内相关、误差、b-vector 夹角、参数误差及离群切片重合数：

```bash
python validation/eddy/compare_matched.py \
  --torch-root <FNIT_EDDY输出前缀> \
  --fsl-root <FSL_EDDY输出前缀> \
  --mask <共同脑掩膜.nii.gz> \
  --bvals <共同输入.bval> \
  --output <比较报告.json>
```

`--torch-root` 和 `--fsl-root` 都是不带 `.nii.gz` 的输出前缀；脚本还读取同前缀的 `.eddy_rotated_bvecs`、`.eddy_parameters` 和 `.eddy_outlier_map`。`--output` 是写入的 JSON 路径。公开报告附加了运行日期、源码版本和全部关键输入哈希；仓库不发布被试影像。

官方 EDDY 2111.0 源码使用球面 Gaussian-process predictor、交叉验证超参数估计、b0 与 DWI 分阶段配准、LM 参数更新和基于带符号残差的离群检测。当前 TorchEDDY 使用固定 q-space 邻居、Adam 和 median/MAD 离群统计。这些差异均可影响最终图像，尚不能把本例差距归因于单一组件。严格复刻工作未完成；这里保存的是改写前基线。
