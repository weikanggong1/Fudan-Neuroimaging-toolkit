# BWAS：全脑体素连接的群体关联

[返回主页](../../README.md) · [验证代码](../../validation/bwas/benchmark_abide.py)

`run_bwas` 接收已经完成预处理、位于 MNI152NLin6Asym 2 mm 空间的多被试 BOLD。它逐人计算体素对的时间相关并作 Fisher z 变换，再对每条连接拟合“表型 + 协变量 + 截距”的线性模型。高于连接定义阈值（CDT）的连接按两端体素的空间邻接关系聚簇，使用原版 BWAS 的六维高斯随机场公式计算簇水平 FWER p 值。运行时只使用 PyTorch、NumPy、SciPy 和 Nibabel；不调用原版 BWAS 或 FSL。

## 输入与输出

输入目录是 BIDS Derivatives；每个 `participant_id` 必须恰好对应一份 `sub-*/[ses-*]/func/*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz`。所有 BOLD 与 3D 分析掩膜必须有相同的 shape、affine 和 2 mm 体素。`participants.tsv` 的 `participant_id` 与 BIDS 目录名精确匹配，不依赖文件排序；表型和协变量必须是数值列，设计矩阵需满秩。不同被试的时间帧数可以不同。若有多个 site，把 site 编码为哑变量并省略一个参考 site。

```text
derivatives/fnit-volume/
├── dataset_description.json
├── sub-001/func/sub-001_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz
└── sub-002/func/sub-002_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz

participants.tsv: participant_id  case  age  sex  site_01  site_02  ...
graymatter_mask.nii.gz: 与 BOLD 同网格的 2 mm 灰质二值掩膜
```

输出采用 BIDS Derivatives 的数据集说明与 `group/func/` 布局；连接和簇表使用 BWAS 自定义的组水平文件名。`_desc-BWASedges_relmat.tsv.gz` 每行保存一条越阈值连接的两端体素索引、z 值和簇编号；`_desc-BWASclusters_stat.tsv` 保存簇的连接数、校正/未校正 p 值及两端区域的体素数；`_desc-BWASMA_statmap.nii.gz` 是每个体素参与显著连接簇的连接数，附 JSON sidecar。根目录有 `dataset_description.json`。没有显著簇时 MA 图全零，越阈值连接和簇表仍保留。

## Python 调用

```python
import csv
from pathlib import Path
from fnit import run_bwas

bids_root = Path("/data/derivatives/fnit-volume")  # 输入：含各被试 2 mm clean BOLD 的 BIDS Derivatives 根目录
participants_tsv = Path("/data/participants.tsv")  # 输入：participant_id、case、age、sex 与 site 哑变量
graymatter_mask = Path("/data/MNI152_2mm_graymatter_mask.nii.gz")  # 输入：与 BOLD 同网格的二值灰质掩膜
output_root = Path("/data/derivatives/fnit-bwas")  # 输出：不存在或为空的 BIDS Derivatives 目录

with participants_tsv.open(newline="") as stream:
    site_columns = tuple(name for name in csv.DictReader(stream, delimiter="\t").fieldnames
                         if name.startswith("site_"))  # 数值哑变量；已省略一个参考 site

result = run_bwas(
    bids_root=bids_root,
    participants_tsv=participants_tsv,
    mask_file=graymatter_mask,
    output_root=output_root,
    phenotype="case",  # 表型列；0/1 病例对照或连续数值
    covariates=("age", "sex", *site_columns),  # 数值协变量列，自动加截距
    cdt=5.0,  # 双侧 |z| 连接定义阈值；原版建议正式分析不低于 5
    block_size=5120,  # 每维体素分块宽度；此示例已在 H100 上验证
    subject_block_size=16,  # 每次送入 GPU 的被试数，限制显存
    num_workers=8,  # 并发读取、平滑度估计与时间序列标准化的被试数；占用更多主机内存
    cache_root=Path("/data/local-scratch"),  # 可选：本地临时盘；需容纳全部被试的标准化 BOLD
    device="cuda:0",  # PyTorch 设备；无 CUDA 时可用 "cpu"
    fwhm=None,  # 默认从所有输入 BOLD 估计平均空间平滑度；已知时可填体素单位的正数
    validate_direct_ols=False,  # 研究验证时可逐连接比较直接回归；会增加主机内存与耗时
)
print(result.edges, result.clusters, result.ma_map)
```

| 参数 | 含义 |
|---|---|
| `bids_root` | 输入 BIDS Derivatives 根目录，含 `dataset_description.json`。 |
| `participants_tsv` | 按 `participant_id` 列关联影像的 TSV；行顺序决定回归行顺序。 |
| `mask_file` | 2 mm MNI 分析掩膜，只检验非零体素间连接。 |
| `output_root` | 新建结果目录；不覆盖已有文件。 |
| `phenotype` | 欲检验的单个数值列名；系数对应模型第一列。 |
| `covariates` | 数值协变量列名元组；函数自动追加截距。多类别 site 需先转换为哑变量。 |
| `cdt` | 对双侧 z 值取绝对值的连接阈值，默认 `5.0`；原版建议正式全脑分析不低于 `5`。 |
| `block_size` | 每个体素轴的块宽，默认 `128` 仅适合小掩膜验证；全脑需显式设置较大块宽，示例 `5120` 适用于已验证的 H100 配置，其他 GPU 按可用显存调整。 |
| `subject_block_size` | 每个 GPU 批次的被试数，默认 `16`。 |
| `num_workers` | 读取、平滑度估计和标准化 BOLD 的并发 worker 数，默认 `1`；增大可缩短准备时间，但会增加主机内存和磁盘负载。 |
| `cache_root` | 可选临时缓存目录，默认使用 `output_root`；本地高速盘可加快反复读取体素块，需有约 `4 × 总帧数 × 掩膜体素数` 字节可用空间，运行结束自动清理。 |
| `device` | 默认 `cuda:0`。 |
| `fwhm` | 三轴共用的空间平滑度，单位为体素；默认从所有 BOLD 估计并至少取 `2`。 |
| `validate_direct_ols` | 默认关闭。开启后，对每个无序体素对比较被试分块与一次性直接回归的 z 值，并记录平均/最大误差及 CDT 判定分歧；额外主机内存上限约为 `8 × 被试数 × block_size²` 字节。 |

体素块和被试块共同限制显存。每块仅存当前被试组的 Fisher 连接和回归充分统计量 `QᵀY`、`YᵀY`；其中 `Q` 来自对表型和协变量的 QR 正交化，避免站点哑变量造成的病态矩阵逆。标准化时间序列按“体素×时间”连续存放在临时缓存目录，运行结束自动清理。连接和群体回归均用普通 float32；本功能关闭 TF32，BOLD 缓存与 MA NIfTI 也为 float32，不使用 float16。CUDA 输入批次使用页锁定内存传输。
超过 512 人时，Linux 版本会把当前进程的文件描述符软上限提高到 `被试数+128`；若系统硬上限仍不足，会在计算前报错并说明所需数量。

单站点的命令行调用：

```bash
fnit-bwas --bids-root /data/derivatives/fnit-volume \
  --participants /data/participants.tsv \
  --mask /data/MNI152_2mm_graymatter_mask.nii.gz \
  --output-root /data/derivatives/fnit-bwas \
  --phenotype case --covariate age --covariate sex \
  --cdt 5 --block-size 5120 --subject-block-size 16 --num-workers 8 \
  --cache-root /data/local-scratch --device cuda:0
```

多站点时为每个额外 site 列重复一次 `--covariate site_XX`。原版对应调用为：

```bash
python BWAS_main.py -toolbox_dir /path/to/BWAS \
  -image_dir /path/to/ordered_2mm_bold -output_dir /path/to/reference_output \
  -mask_file /path/to/graymatter_mask.nii.gz \
  -target_file /path/to/case.npy -cov_file /path/to/age_sex_site_dummies.npy \
  -CDT 5 -memory_limits 16 -ncore 8
```

原版按影像文件名排序读取 `target`/`covariates` 行；使用它作对照时必须先按相同排序生成 `.npy`。FNIT 的 TSV 通过 `participant_id` 显式关联。为了复现原版统计输出，FNIT 同样先以 `n-p` 估计 t 值，再以原版代码中的 `n-p-1` 自由度转成 z 值；这是与通常单一残差自由度写法不同的原版数值约定。

## 真实数据验证

使用 ABIDE II 同一采集站点 32 人（16 病例、16 对照）的真实预处理 BOLD，插值到 2 mm 后在中心 512 个体素上验证。此实验的 `CDT=3` 用于产生足够多的越阈值连接以核对聚类规则，不作疾病发现推断。参考程序从其原始 `BWAS_cpu.py` 直接加载相关、GLM、平滑度、邻接和六维校正函数；未把原版代码纳入 FNIT 运行时。[复现脚本](../../validation/bwas/benchmark_abide.py) 和[汇总结果](../../validation/bwas/abide32_summary.json)记录全部 130,816 条无序体素对的 z 值比较：与原版相比，MAE `2.28×10⁻⁶`、最大误差 `6.27×10⁻⁵`、CDT 判定分歧 `0`。777/777 条越阈值连接一致，40 个连接簇的大小一致。同一批真实连接用一次性回归与被试分块回归相比，z 值最大差 `1.08×10⁻¹²`。原版核心计算用 `0.75 s`；FNIT 含 BIDS 读写、平滑度、聚类与直接回归核验的全流程用 `1.82 s`，计时范围不同，不构成加速比。

全脑实验从 ABIDE I 与 II 的 1778 份真实预处理 BOLD 及官方表型表建立 BIDS 2 mm 输入。以本地 FSL 2 mm 灰质概率图 `>0.5` 定义初始 128,190 体素掩膜；该模板不随 FNIT 分发。源数据有一份全零影像，因此预先固定每人至少 120,000 个有效灰质体素的质量标准：保留 1748 人（792 病例、956 对照），共有 112,215 个有效灰质体素。模型包含年龄、性别、35 个站点哑变量和截距，设计矩阵满秩；正式连接阈值为 `CDT=5`。[BIDS 准备脚本](../../validation/bwas/prepare_abide.py)、[全量运行脚本](../../validation/bwas/run_abide_full.py)和[原版整脑 seed→voxel 对照脚本](../../validation/bwas/benchmark_full_seed.py)可复核输入、覆盖筛选和统计量；实验不上传个体影像、表型行或连接矩阵。

原版 CPU 的逐值对照使用全体合格被试的两枚灰质 seed，各自连接到整张共同灰质掩膜；全量体素对由 FNIT 运行并检查输出文件。种子对照不代表原版 CPU 逐条计算了所有体素对。两种计时均分别注明是否包含 BIDS 读取、缓存准备和结果写出。

## 来源

- [原版 weikanggong/BWAS](https://github.com/weikanggong/BWAS) 的 [BWAS_cpu.py](https://github.com/weikanggong/BWAS/blob/master/BWAS_cpu.py) 与 [BWAS_main.py](https://github.com/weikanggong/BWAS/blob/master/BWAS_main.py)（Apache 2.0；参考源码 SHA-256 `1b78a98efb04ae5c1b764b101ec434ed8c8277f815481ae0ca940ebd910323c2`）。
- Gong W, et al. [Statistical testing and power analysis for brain-wide association study](https://doi.org/10.1016/j.media.2018.03.014). *Medical Image Analysis* 47 (2018): 15–30.
- [ABIDE II 表型变量释义](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf)。
- [ABIDE I 官方表型表](https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv)和[ABIDE II 官方表型表](https://fcon_1000.projects.nitrc.org/indi/abide2/release/phenotypic_data/ABIDEII_Composite_Phenotypic.csv)。
