# 当前 main 与本轮官方完整流程的配对对照

当前 raw 实测源码 `f436de588647a0de80735e4a98d53df5d88e502d`；固定同一公开 T1 的十例再次独立执行 `structures=all, optimization=fast`。逐例采用本轮同病例 fresh 官方 `recon-all -all` 加三项细分割的实际整例 wall，并先计算官方/FNIT 比值再对病例汇总。

stage 列仍为原冻结 ac692bb 的实际运行，从该病例本轮 fresh norm/aseg/wmparc 开始；对照为三项官方细分割 command wall 之和，不包括 recon-all，不称为 f436de5 新跑。这项条件测试的官方 norm 输入不会进入 raw-native 生产拟合。

## 精度及脑图的继承依据

十例 50 张标签图的体素、shape、affine 和 dtype 完全相同；1,100 条硬/软体积字典和 160 条上下文记录相同，拟合配置与模型选择相同。两版 24 个 GEMS 文件逐字节一致。因此原 F8 raw Dice 和由这些相同标签图生成的脑图可复用，精度未另作近似或重新评分。

[本轮精度与真实脑图](../benchmark_results.md)、[完整 110 ROI 逐例表](../analysis/cohort_roi.tsv)、[880 行 ROI 汇总](../analysis/cohort_roi_summary.tsv)、[脑图清单](../brain_figures/plot_manifest.json)。

## 实际耗时与同病例速度

单位：秒（显存为 MiB）；均值 ± 样本 SD（ddof=1）、中位数、最小–最大值、有效/计划及 NA。SD 描述病例间差异。raw 的 process wall 包含观察器和进程开销，输入及 GPU 预算等待计时另列；API compute/save/total 是各自实际计时，不相互代替。

### 全部 10 例

| 指标 | 实测统计 |
|---|---|
| 当前 main raw process wall | 259.366 ± 6.447; 中位 260.716; [251.255, 267.758]; 10/10，NA 0 |
| 当前 main raw API compute | 253.660 ± 6.308; 中位 254.652; [245.684, 261.887]; 10/10，NA 0 |
| 当前 main raw API total | 254.260 ± 6.340; 中位 255.249; [246.253, 262.551]; 10/10，NA 0 |
| 当前 main raw save | 0.550 ± 0.046; 中位 0.545; [0.481, 0.644]; 10/10，NA 0 |
| 当前 main raw 自身 PID 采样峰值显存 MiB | 16856.200 ± 1405.989; 中位 16488.000; [14748.000, 18428.000]; 10/10，NA 0 |
| 官方 fresh 完整流程 wall | 7526.972 ± 736.548; 中位 7745.333; [6128.053, 8334.101]; 10/10，NA 0 |
| 官方完整流程 / 当前 main raw（同病例） | 29.010 ± 2.643; 中位 29.702; [24.106, 32.966]; 10/10，NA 0 |
| 官方三项细分割 / 原冻结 stage（同病例） | 5.212 ± 0.457; 中位 5.282; [4.476, 5.786]; 10/10，NA 0 |
| 当前 main / 原冻结 raw process wall（同病例） | 0.895 ± 0.091; 中位 0.941; [0.725, 0.989]; 10/10，NA 0 |

| 实际步骤 | 耗时统计 |
|---|---|
| 官方 reconall command wall | 5981.719 ± 664.155; 中位 6126.216; [4701.855, 6904.683]; 10/10，NA 0 |
| 官方 brainstem command wall | 326.396 ± 39.559; 中位 330.462; [240.555, 371.252]; 10/10，NA 0 |
| 官方 thalamus command wall | 455.386 ± 45.265; 中位 447.600; [400.943, 543.188]; 10/10，NA 0 |
| 官方 hippo-amygdala command wall | 763.296 ± 74.775; 中位 739.344; [677.185, 898.262]; 10/10，NA 0 |
| 当前 main raw brainstem recipe 总计 | 17.779 ± 0.709; 中位 17.640; [16.884, 19.236]; 10/10，NA 0 |
| 当前 main raw thalamus recipe 总计 | 74.884 ± 4.759; 中位 75.842; [67.856, 81.642]; 10/10，NA 0 |
| 当前 main raw hippo-amygdala-left recipe 总计 | 64.916 ± 4.179; 中位 64.769; [57.597, 71.195]; 10/10，NA 0 |
| 当前 main raw hippo-amygdala-right recipe 总计 | 64.931 ± 4.032; 中位 66.777; [55.730, 69.270]; 10/10，NA 0 |

### 新 9 例（排除开发用 sub-01）

| 指标 | 实测统计 |
|---|---|
| 当前 main raw process wall | 260.213 ± 6.221; 中位 263.232; [251.255, 267.758]; 9/9，NA 0 |
| 当前 main raw API compute | 254.459 ± 6.130; 中位 256.959; [245.684, 261.887]; 9/9，NA 0 |
| 当前 main raw API total | 255.060 ± 6.166; 中位 257.572; [246.253, 262.551]; 9/9，NA 0 |
| 当前 main raw save | 0.552 ± 0.049; 中位 0.550; [0.481, 0.644]; 9/9，NA 0 |
| 当前 main raw 自身 PID 采样峰值显存 MiB | 16897.111 ± 1484.950; 中位 16488.000; [14748.000, 18428.000]; 9/9，NA 0 |
| 官方 fresh 完整流程 wall | 7607.378 ± 733.200; 中位 7984.084; [6128.053, 8334.101]; 9/9，NA 0 |
| 官方完整流程 / 当前 main raw（同病例） | 29.231 ± 2.704; 中位 30.331; [24.106, 32.966]; 9/9，NA 0 |
| 官方三项细分割 / 原冻结 stage（同病例） | 5.293 ± 0.399; 中位 5.306; [4.757, 5.786]; 9/9，NA 0 |
| 当前 main / 原冻结 raw process wall（同病例） | 0.901 ± 0.094; 中位 0.949; [0.725, 0.989]; 9/9，NA 0 |

| 实际步骤 | 耗时统计 |
|---|---|
| 官方 reconall command wall | 6038.284 ± 678.413; 中位 6202.540; [4701.855, 6904.683]; 9/9，NA 0 |
| 官方 brainstem command wall | 335.934 ± 27.150; 中位 331.961; [291.651, 371.252]; 9/9，NA 0 |
| 官方 thalamus command wall | 461.265 ± 43.774; 中位 460.913; [400.943, 543.188]; 9/9，NA 0 |
| 官方 hippo-amygdala command wall | 771.720 ± 74.107; 中位 745.272; [677.185, 898.262]; 9/9，NA 0 |
| 当前 main raw brainstem recipe 总计 | 17.749 ± 0.745; 中位 17.528; [16.884, 19.236]; 9/9，NA 0 |
| 当前 main raw thalamus recipe 总计 | 75.665 ± 4.315; 中位 77.275; [68.436, 81.642]; 9/9，NA 0 |
| 当前 main raw hippo-amygdala-left recipe 总计 | 65.515 ± 3.951; 中位 65.217; [57.597, 71.195]; 9/9，NA 0 |
| 当前 main raw hippo-amygdala-right recipe 总计 | 64.449 ± 3.960; 中位 66.558; [55.730, 67.600]; 9/9，NA 0 |

recipe 总计和内部子步骤存在包含关系，不重复相加。官方海马/杏仁核双侧只计一个 command wall。[原冻结流程及官方实际步骤](../analysis/steps/cohort_steps_summary.tsv)；当前 main raw 的全部实际 recipe timer 路径与值保存在本页 [机器可读对照](official_comparison.json)。

main 与原冻结 raw 的速度变化是在共享 GPU 上分时测得；源码、负载和时间点同时变化，这项观测不能将变化量归因于某个 buffer 优化。自身 PID 显存采样和整卡其他进程占用分开记录。

[20 行逐例配对表](official_paired_runtime.tsv)中，十行 raw 为当前 main，十行 stage 明确保留 ac692bb 的版本和实际运行来源。本次 companion 不改 F8 的文件、数字或 SHA；[原 F8 发布证据](../benchmark_evidence.json)与[main 审计](audit/summary.json)由新 JSON 的输入清单绑定。

## 文件身份

| 输入 | SHA-256 |
|---|---|
| F8 benchmark_evidence | `34725f21fc0c22688b493c994f64342e0cd0595394d021ee8bc21c327989fad2` |
| F8 paired_runtime | `fe57821c7a4f119eff792c1b933979dcfa60f4eb99ef6469c196c1fa25b55230` |
| F8 cohort_summary | `62808ffb2f74fa6a80a316cedfefc9d0a298ae13e0003354e5130c23bd0bb306` |
| F8 steps_summary | `b9dafce071cc7147c7fcc29bc2345bf8842dfc9d99de1afdd802dbd88f694d5b` |
| F8 steps_per_case | `cec8a8c05e8ce3df82271c9634163ec5ec640ebfebaedd20bb4226f7621d2a5a` |
| main regression audit | `cd497c2dd130c4238c1314a5b5cf7cb9f5f59a59b7910646459f03e1baafebf4` |
| main volume comparisons | `1e3ea8073aebcee281c6ec8305ba91b1675d2ec48626aeed52bf9a0e95dc2125` |
| main cohort_manifest | `3a81f1fe8792a32d022e8fb4e2c9dd82978b3dbf241e5995547b0ce9a518ee35` |
| main copy receipt | `5ff13be044e0154b88d50b5e7e6ac1cd198d15f8e9a03513c800ef9cecbd2bc4` |
| original source_manifest | `d8c8f7aecdae521751f30fdd09bea6acbffd2c0a50c9fb70e1dc97ad74717b07` |
| main source_manifest | `4fb86d811b92c9db6be002f2c58a46b8f07f624b42671d034bc996294ce0b602` |
