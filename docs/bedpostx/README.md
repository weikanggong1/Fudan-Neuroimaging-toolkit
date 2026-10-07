# TorchBEDPOSTX：纤维方向后验估计

| 项目 | 内容 |
|---|---|
| 输入 | 单被试DWI/mask/bvals/bvecs目录 |
| 输出 | 每纤维方向、分数后验及均值图 |
| 对应原软件 | FSL BEDPOSTX/xfibres |
| Python / CLI | TorchBEDPOSTX / fnit-bedpostx |
| CPU / GPU | CPU/CUDA；float32、CUDA允许TF32 |

## 1. 功能简介

`TorchBEDPOSTX` 为每个diffusion体素估计最多三条交叉纤维的方向轴、体积分数及后验不确定性，输出可被概率追踪读取的BEDPOSTX目录。默认model2使用Gamma分布描述多shell扩散率，model1使用单扩散率。

算法采用ball-and-stick信号、Gaussian残差、次要纤维ARD先验和Metropolis采样。FNIT用PyTorch拟合、nibabel读写，运行时不调用FSL；CUDA似然使用torch.compile融合。CPU/CUDA均为float32，允许TF32且不使用FP16/BF16。

## 2. Python 调用

```python
from fnit.bedpostx import TorchBEDPOSTX

posterior_model = TorchBEDPOSTX(
    device="cuda:0",  # 指定GPU设备
    threads=1,  # 辅助CPU计算线程数
    chunk_size=4096,  # 每批脑体素数；小于默认以减少工作区
    seed=8665904,  # 固定MCMC随机种子
)
posterior_result = posterior_model(
    subject_dir="/data/dwi/subject",  # 四个必需输入文件的目录
    output_dir="/data/results/subject.bedpostX",  # 方向后验输出目录
    overwrite=False,  # 已有非空目录时报错
)
posterior_sample_count = posterior_result.nsamples  # 每体素后验样本数
fitted_voxel_count = posterior_result.nvoxels  # 参与拟合体素数
```
### 输入数据格式

- DWI：单被试 NIfTI，shape 为 `[X,Y,Z,N]`，至少含 b0 和扩散方向；信号强度沿用输入单位。读入转 float32，不在此入口做运动、涡流或磁敏感校正。
- mask：NIfTI `[X,Y,Z]`，须与 DWI 的 affine、orientation、体素尺寸和原生 diffusion 空间一致。不自动配准或重排梯度；mask 外结果为零。
- bval：纯文本 N 个有限 b-value，单位 s/mm²，顺序与第四维一致。
- bvec：纯文本 `3×N` 或 `N×3`，每个非 b0方向对应同一 volume；已完成 EDDY 时用旋转后的 bvec。
- 路径接受字符串或 PathLike。示例使用用户自己的真实文件；多被试调度由调用方组织。
本入口从固定文件名读取：

```text
subject/
├── data.nii.gz                 # 校正DWI [X,Y,Z,N]
├── nodif_brain_mask.nii.gz      # 同shape/affine的3D脑mask
├── bvals                       # N个b-value
└── bvecs                       # 3×N或N×3方向
```

mask值>0才参与拟合；mask非空、masked DWI与梯度必须有限。至少一个b≤50的b0和六个b>50方向，程序单位化bvec。默认1250跳、每25跳保存，得到50个后验样本。

保存全部后验会占用主机内存和磁盘；`chunk_size`只限制拟合批工作区，不限制所有输出数组驻留。小ROI首次运行可能主要花在CUDA初始化和编译。

**`TorchBEDPOSTX.__init__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | `str/torch.device/None` | `'cpu'` | PyTorch 设备；None 自动选择可用 CUDA，否则 CPU。 |
| `threads` | 否 | `int或None` | `None` | 正整数 CPU 线程数；None 保留 PyTorch 设置。 |
| `nfibres` | 否 | `int` | `3` | 每体素最大纤维数，支持1–3。 |
| `model` | 否 | `int` | `2` | 1为单扩散率；2为 Gamma 多壳层扩散率。 |
| `burnin` | 否 | `int` | `1000` | 丢弃的 MCMC 预热跳数。 |
| `njumps` | 否 | `int` | `1250` | 预热后 MCMC 总跳数。 |
| `sample_every` | 否 | `int` | `25` | 保存后验的跳数间隔；样本数为 njumps//sample_every。 |
| `ard_weight` | 否 | `float` | `1.0` | 次要纤维的 ARD 稀疏先验权重。 |
| `chunk_size` | 否 | `int` | `16384` | 每批并行拟合的脑内体素数，必须为正整数。 |
| `seed` | 否 | `int` | `8665904` | 随机种子；固定并不保证不同软件随机轨迹相同。 |

**`TorchBEDPOSTX.__call__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `subject_dir` | 是 | `路径` | `—` | 一名被试的输入目录，文件结构见下文。 |
| `output_dir` | 否 | `路径` | `None` | 本次结果目录；路径按当前工作目录解析。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

### 输出

```text
subject.bedpostX/
├── merged_th1samples.nii.gz
├── merged_ph1samples.nii.gz
├── merged_f1samples.nii.gz
├── mean_f1samples.nii.gz
├── dyads1.nii.gz
├── ...                         # 按nfibres重复上述文件
├── mean_dsamples.nii.gz
├── mean_d_stdsamples.nii.gz     # 仅model2
├── mean_S0samples.nii.gz
├── nodif_brain_mask.nii.gz
└── run.json
```

| 文件 | 格式、单位与含义 |
|---|---|
| merged_th/ph | `[X,Y,Z,Nsample]` float32；球坐标角，弧度。 |
| merged_f | `[X,Y,Z,Nsample]` float32；纤维分数，无量纲。 |
| mean_f | 3D float32；后验平均分数。 |
| dyads | `[X,Y,Z,3]` float32；后验平均方向轴，正负号等价。 |
| mean_d / mean_d_std | 3D float32；平均扩散率/标准差，mm²/s。 |
| mean_S0 | 3D float32；平均基线信号，输入强度单位。 |
| run.json | 参数、体素/样本计数和完整调用时间。 |

全部NIfTI保留输入DWI原生网格、affine和orientation；mask外为零。纤维按每体素后验平均分数降序重排。`BedpostXResult`返回目录、nvoxels、nsamples和elapsed_seconds；时间包含输入读取、采样、后处理和写出。

文件命名与FSL追踪后验接口相同，但随机数、初始化及归约顺序不同，不声明全部后验逐值相等。

## 3. 命令行调用

```bash
fnit-bedpostx --subject-dir /data/dwi/subject    --output-dir /data/results/subject.bedpostX --device cuda:0    --threads 1 --nfibres 3 --model 2 --burnin 1000 --njumps 1250    --sample-every 25 --ard-weight 1 --chunk-size 4096 --seed 8665904
```

统一入口 `fnit bedpostx`接受同一组参数。

| CLI | Python参数 | 含义 |
|---|---|---|
| `--subject-dir`、`--output-dir` | 调用参数同名下划线形式 | 输入及结果目录 |
| `--device`、`--threads` | 构造参数同名 | 设备和线程 |
| `--nfibres`、`--model` | 同名 | 最大纤维数和模型 |
| `--burnin`、`--njumps`、`--sample-every` | `burnin/njumps/sample_every` | MCMC设置 |
| `--ard-weight` | `ard_weight` | ARD权重 |
| `--chunk-size`、`--seed` | `chunk_size/seed` | 批大小与随机种子 |
| `--overwrite` | `overwrite=True` | 覆盖非空目录 |

CLI和Python的device均默认cpu；使用CUDA必须明确指定。批大小影响吞吐/显存，也可能通过随机样本组织影响MCMC结果。

## 4. 原软件调用

原软件在独立参考环境读取同一subject目录：

```bash
bedpostx /data/dwi/subject --nf=3 --model=2 --fudge=1    --bi=1000 --nj=1250 --se=25
bedpostx_gpu /data/dwi/subject -NJOBS 1 -n 3 -model 2 -w 1    -b 1000 -j 1250 -s 25
```

| FNIT | 原BEDPOSTX |
|---|---|
| nfibres / model / ard_weight | --nf / --model / --fudge；GPU为-n/-model/-w |
| burnin / njumps / sample_every | --bi/--nj/--se；GPU为-b/-j/-s |
| subject_dir | wrapper位置参数 |
| device、chunk_size、threads、output_dir | FNIT运行选项，wrapper调度语义不同 |

当前不覆盖FSL的全部模型和采样扩展，支持model1/2、Gaussian残差、最多三纤维。FSL的拆分、调度与合并成本必须与核心xfibres计时分开。

## 5. 最新精度和运行时间

最新独立真实对照为[2026-09-29报告](../../validation/bedpostx/report.public.json)，以core.py SHA `a1b2f8fd…`绑定。真实DWI同一例的14弱纤维体素和64交叉纤维富集体素，后者按FSL后验筛选，不能代表无偏全脑。参照FSL6.0.7.22，H100共享GPU，CPU14体素控制为1线程；本轮未重跑该benchmark。

| 14体素完整进程 | 时间 |
|---|---:|
| FSL CPU | 10.99 s |
| FNIT CPU | 22.54 s |
| FNIT GPU优化前→后 | 39.32→18.89 s |
| FSL GPU xfibres核心 | 2.20 s，仅核心，非完整wrapper |

FNIT进程内14/64体素优化后调用为15.98/21.65 s，峰值allocated为0.27 GiB。FNIT完整进程含载入、CUDA/编译、采样和写盘；原GPU核心不含调度/合并，不能直接计算端到端比。float32、CUDA允许TF32，测试前后GPU非空闲。

| 真实精度 | 对FSL CPU结果 |
|---|---|
| 14体素f1 MAE / r / 主轴角中位数 | 0.01088 / 0.99918 / 0.66° |
| 64体素共同f≥0.1的f2/f3 r | 0.727 / 0.714；61/26体素 |
| 64体素f2/f3轴角中位数 | 5.39° / 7.97° |

14体素优化前后19张图逐值相同；64体素融合引起MCMC链分叉，已重新对FSL统计。没有独立全脑精度、全脑端到端时间或完整逐阶段配对报告。

![64真实诊断体素的FSL/FNIT第二和第三纤维方向轴](figures/bedpostx_real_ukb_direction_axes.png)

图只显示共同分数支持的体素，报告保留筛选定义。合成示例仅用于功能回归，不计作科学benchmark。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-09-29 | 6f379535 | 融合重复似然计算 | 14/64真实诊断体素精度和时间 |
| 2026-09-28 | 2ad53c5b | 刷新真实数据与来源审计 | 独立FSLCPU/GPU核心边界 |
| 2026-09-26 | 12555da | 加入BEDPOSTX与后验文件合同 | 最早发布验证；不是当前全脑验收 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/bedpostx/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="输入"></a>
<a id="python-单被试调用"></a>
<a id="命令行与原软件对应"></a>
<a id="输出与结构"></a>
<a id="与-fsl-的真实数据对照"></a>
<a id="合成回归示例"></a>
<a id="来源与限制"></a>
<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 官方：[FSL BEDPOSTX](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/bedpostx.html)、[FDT代码](https://git.fmrib.ox.ac.uk/fsl/fdt)中的`xfibres`、`CUDA/xfibres_gpu`与wrapper。
- Behrens等，2007，[Probabilistic diffusion tractography with multiple fibre orientations](https://doi.org/10.1016/j.neuroimage.2006.09.018)；Jbabdi等，2012，[Model-based analysis of multishell diffusion MR data](https://doi.org/10.1002/mrm.24204)。
- FNIT：[core.py](../../src/fnit/bedpostx/core.py)、[验证索引](../../validation/bedpostx/README.md)。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
