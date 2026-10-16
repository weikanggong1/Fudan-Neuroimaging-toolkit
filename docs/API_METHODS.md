# 统一接口中的算法与输入选项

FNIT 的主要公开调用以 `FNIT_` 开头。重命名直接复用已有实现；本页说明需要选择算法或输入形态的三个入口。其余主要功能的构造参数、运行方法、结果类型和数值计算均沿用对应功能页。

## 1. `FNIT_newmsm`：一种入口选择两种球面配准

MSMSulc 用脑沟特征配准，MSMAll 用多模态特征配准。`method` 只选择算法，输入的顶点顺序、特征列、配置和权重原样保留。输入准备与配准是两个明确步骤：准备函数产生下表的输入对象；统一函数执行双侧配准。

```python
from fnit import FNIT_newmsm
from fnit.msm import MSMSulcConfig, MSMSulcInputs

# 每侧对象记录既有球面、脑沟图和初始仿射的文件路径。
hemisphere_inputs = {
    "L": MSMSulcInputs(
        native_sphere="inputs/L.sphere.surf.gii",
        rotated_sphere="inputs/L.sphere.rotated.surf.gii",
        native_sulc="inputs/L.sulc.shape.gii",
        reference_sphere="templates/L.sphere.surf.gii",
        reference_sulc="templates/L.sulc.shape.gii",
        affine="inputs/L.rotation.txt",
    ),
    "R": MSMSulcInputs(
        native_sphere="inputs/R.sphere.surf.gii",
        rotated_sphere="inputs/R.sphere.rotated.surf.gii",
        native_sulc="inputs/R.sulc.shape.gii",
        reference_sphere="templates/R.sphere.surf.gii",
        reference_sulc="templates/R.sulc.shape.gii",
        affine="inputs/R.rotation.txt",
    ),
}
registered_spheres = FNIT_newmsm(
    hemisphere_inputs, "outputs/msmsulc",
    method="msmsulc",                # 也可选择 msmall，并提供 MSMAllInputs
    config=MSMSulcConfig(),           # 原有官方计划配置；也可传配置文件路径
    device="cuda:0",
    execution="optimized",          # 保留 reference 选项
    parallel=True,                   # 双侧并行
    cpu_threads=8,                   # 双侧合计线程预算
    qc_policy="report",             # 原样记录原生球面翻折；repair/error 为显式选项
)
```

| 参数 | 格式、默认值与含义 |
|---|---|
| `inputs` | 只含 `L`、`R` 的字典。`msmsulc` 对应 `MSMSulcInputs`；`msmall` 对应 `MSMAllInputs`。 |
| `output_dir` | 输出目录路径，保存双侧 `.surf.gii` 和 `registration_report.json`。 |
| `method` | `msmsulc`（默认）或 `msmall`。 |
| `device` | `cuda:0`（默认）或 `cpu`。 |
| `config` | 对应方法的配置对象、配置文件路径或 `None`（该方法的原有默认计划）。 |
| `execution` | `optimized`（默认）或 `reference`。 |
| `parallel` | 默认 `True`；是否并行左右半球。 |
| `cpu_threads` | 正整数或 `None`，双侧总线程预算；`None` 使用既有预算解析。 |
| `qc_policy` | `report`（默认）、`repair` 或 `error`，只适用于 MSMSulc。MSMAll 非默认值会报错。 |

两种输入的文件格式、完整配置与官方命令见 [MSMSulc](msm/README.md) 和 [MSMAll](msm/msmall.md)。返回值仍是 `{"L": Path, "R": Path}`，每个球面 GIFTI 含顶点坐标和有序三角形索引。

## 2. `FNIT_noddi`：模型对象复用与算法选择

`method="amico"` 使用现有离散字典拟合；`method="classic"` 使用现有 AMICO 初始化和连续 Watson 非线性拟合。两种方法估计相同含义的参数，但彼此数值不同；统一名称不改变各方法的计算。

```python
from fnit import FNIT_noddi
from fnit.amico_noddi import AMICONODDIConfig

noddi_configuration = AMICONODDIConfig(lut_batch_size=400)
noddi_model = FNIT_noddi(
    device="cuda:0",                  # 同一模型可连续处理多个输入
    method="classic",                # amico 或 classic；省略时仍默认 amico
    config=noddi_configuration,        # 原样复用完整配置对象
)
noddi_result = noddi_model.run(
    data="inputs/dwi_corrected.nii.gz", # 四维 DWI [X,Y,Z,N]
    mask="inputs/dwi_mask.nii.gz",     # 与 DWI 空间网格一致的三维 mask
    bvecs="inputs/dwi_rotated.bvec",   # eddy 旋转后的 FSL 方向文件
    bvals="inputs/dwi.bval",           # N 个 b 值，单位 s/mm²
    output_dir="outputs/noddi",
    naming="ukb",                     # ukb 或 amico 输出文件命名
    overwrite=False,
)
```

| 构造参数 | 格式、默认值与含义 |
|---|---|
| `device` | 设备或 `None`；`None` 选择可用 CUDA，否则 CPU。 |
| `method` | `amico`、`classic` 或 `None`；省略时使用旧 `fit_method`，两者都省略则是 `amico`。 |
| `config` | 完整 `AMICONODDIConfig` 或 `None`。保留扩散率、字典、正则项、求解容差及 LUT 批大小。 |
| `fit_method` | 旧参数兼容项，值仍为 `amico` 或 `classic`；同时设置 `method` 时两者必须一致。 |

`noddi_model(data, mask, bvecs, bvals)` 只返回原有 `AMICONODDIResult`；`.run(...)` 额外保存结果，参数完整继承。返回的 `ndi`、`odi`、`fwf`、`directions`、`rmse` 是保持原网格与空间头的 NIfTI；`qc` 保留原有方法、求解器、计时和精度说明。完整配置、输出文件名、AMICO 和 NODDI Toolbox 参考命令见 [NODDI 功能页](amico_noddi/README.md)。

## 3. `FNIT_mshbm`：明确输入是 profile、时序还是体积

MS-HBM 推断继续使用已验证的 CPU 实现；体积投影与标签映射复用既有 PyTorch 实现。三种输入都返回 `(labels, history)`：`labels` 是 64984 个顶点的网络编号，皮层网络为 1–17，内侧壁和无效顶点为 0；`history` 是收敛记录。

```python
import numpy as np
from fnit import FNIT_mshbm
from fnit.mshbm import load_assets

# 跨session输入保持现有fsLR32k顶点和landmark顺序；禁止pickle对象数组。
session_profiles = [
    np.load("inputs/session_01_profiles.npy", allow_pickle=False),
    np.load("inputs/session_02_profiles.npy", allow_pickle=False),
]
cortical_timeseries = np.load("inputs/cortical_timeseries.npy", allow_pickle=False)
frame_keep_vector = np.loadtxt("inputs/frame_keep.txt", dtype=np.int32)  # 每帧0/1

mshbm_assets = load_assets()              # 固定 HCP_40 fsLR32k 17 网络先验，可跨调用复用
surface_labels, inference_history = FNIT_mshbm(
    session_profiles,                    # 至少两个 [59412,1483] session profile
    mshbm_assets,
    input_kind="profiles",              # 已有 profile 输入，不额外变换
    w=200.0, c=50.0,
    max_outer=50, max_em=101, max_m=300, max_lambda=101,
)

# 一个 [T,59412] 数组可直接交给既有拆半和 profile 构造流程。
surface_labels, inference_history = FNIT_mshbm(
    cortical_timeseries,
    mshbm_assets,
    input_kind="timeseries",
    censor=frame_keep_vector,             # 每帧一个 0/1，至少保留 8 帧
)

# 体积路径复用现有 volume 入口，并额外保存体积/表面结果。
surface_labels, inference_history = FNIT_mshbm(
    "inputs/bold_mni.nii.gz",            # 四维 MNI BOLD
    mshbm_assets,
    input_kind="volume",
    left_surface="inputs/L.midthickness.surf.gii",
    right_surface="inputs/R.midthickness.surf.gii",
    cortical_mask="inputs/cortical_mask.nii.gz",
    output_dir="outputs/mshbm",
    censor=None,
    w=200.0, c=50.0,
    device="cuda:0", frame_chunk=8, max_distance_mm=3.0,
)
```

| 参数 | 格式、默认值与适用输入 |
|---|---|
| `data` | 必填。profile 列表、二维时序数组或四维 NIfTI 路径，由 `input_kind` 指定。 |
| `assets` | 已加载的先验字典或 `None`（加载包内固定先验）；传入对象原样复用。 |
| `input_kind` | `profiles`（默认）、`timeseries` 或 `volume`。不会猜测输入空间。 |
| `w`、`c` | 所有分支的先验与 MRF 权重，默认 200.0、50.0。 |
| `max_outer/max_em/max_m/max_lambda` | profile/时序推断上限，默认 50/101/300/101。volume 入口沿用这些固定默认，非默认值报错。 |
| `censor` | 时序/体积每帧一个 0/1 数组，默认 `None`；profile 入口不接受。 |
| `left_surface/right_surface` | volume 所需左右 fsLR32k 中厚度 GIFTI，坐标须在体积 scanner-RAS 空间。 |
| `cortical_mask` | volume 所需三维皮层 mask，与 BOLD 空间网格一致。 |
| `output_dir` | volume 必需，其他分支不接受。保存标签、CIFTI、网络时序/相关矩阵及 `labels_mni.nii.gz`。 |
| `device` | 仅控制 volume 的投影和标签映射，默认 `cuda:0`。 |
| `frame_chunk` | volume 每批帧数，默认 8。 |
| `max_distance_mm` | volume 到中厚度顶点映射的距离上限，默认 3.0 mm。 |

所有只适用于另一种输入的非默认参数会报错，避免静默忽略。profile/时序分支保留原 `parcellate` 的返回方式，不额外写文件；原有输出辅助函数仍可使用。完整格式、命令与官方 CBIG 对照见 [MS-HBM 功能页](mshbm/README.md)。

## 4. 命令行调用

```bash
# 同时支持 --method 和旧位置参数；省略方法时默认为 msmsulc。
FNIT_newmsm --method msmsulc --inputs-json inputs/msmsulc.json \
    --output-dir outputs/msmsulc --device cuda:0 --cpu-threads 8
FNIT_newmsm msmall --inputs-json inputs/msmall.json \
    --output-dir outputs/msmall --device cuda:0 --cpu-threads 8

# 新选项 --method 与旧 --fit-method 都可使用，冲突时明确报错。
FNIT_noddi --method classic --data inputs/dwi_corrected.nii.gz \
    --mask inputs/dwi_mask.nii.gz --bvecs inputs/dwi_rotated.bvec \
    --bvals inputs/dwi.bval --output-dir outputs/noddi --device cuda:0

# MS-HBM CLI 沿用既有时序/体积文件参数；Python 另支持预先计算的 profile。
FNIT_mshbm --timeseries inputs/cortex.dtseries.nii --output-dir outputs/mshbm
```

完整输入 JSON 结构、GIFTI/CIFTI 约定及原软件命令，分别见上述三个功能页。`FNIT_newmsm --help`、`FNIT_noddi --help` 和 `FNIT_mshbm --help` 列出命令行全部参数。

## 5. 接口验证与 benchmark 范围

本次是名称与分派整合，不改数值内核、默认配置或算法精度。针对性测试检查：直接别名与原实现是同一对象；MSM 配置、输入和执行参数原样透传；NODDI 复用原 `__call__`/`run`，兼容旧方法参数并拒绝冲突；MS-HBM 复用既有输入准备、CPU 推断和体积入口。

各功能页中已有的真实数据 benchmark 继续描述相应数值实现。本次接口测试不是新的速度或科学精度 benchmark；不得将方法之间的差异解释为改名导致的精度改善。

## 6. 版本与参考实现

- 2026-10-16：统一公开 `FNIT_` 名称；MSM/NODDI 以选项保留两种既有算法；MS-HBM 明确三种输入。
- newMSM 原项目：[rbesenczi/newMSM](https://github.com/rbesenczi/newMSM)。
- AMICO 原项目：[daducci/AMICO](https://github.com/daducci/AMICO)；经典 NODDI 原项目：[NODDI Toolbox](https://www.nitrc.org/projects/noddi_toolbox/)。
- MS-HBM 原实现：[CBIG Kong2019 MS-HBM](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Kong2019_MSHBM)。
- 原软件版本、资源许可与文献详见各功能页；本次不新增外部权重、模板或二进制依赖。
