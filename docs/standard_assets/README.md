# 标准模板资源安装

| 项目 | 内容 |
|---|---|
| 输入 | 绝对输出目录，以及 `dmri`、`registration` 或 `all` 资源组合 |
| 输出 | 保留官方文件名的 NIfTI 模板，以及 `standard-assets.json` 安装记录 |
| 对应原软件 | FSL `data/standard` 中的 FA、tensor、T1 和脑掩膜资源 |
| Python / CLI | `prepare_standard_assets` / `fnit-setup-standard-assets` |
| CPU / GPU | CPU 流式下载和 SHA-256 校验；不使用 GPU |

## 1. 功能简介

本功能为 FNIT 的 dMRI 和配准模块准备标准模板。它从 FNIT 固定 GitHub
Release 的已发布资源目录选择文件，逐个检查精确大小和 SHA-256，保存到
用户指定的目录。已验证的本地文件会直接复用；Release 下载失败时，仅使用
同一目录条目记录的原作者备用 URL，并检查相同字节。

模板下载和校验不调用 FSL 可执行程序，也不使用 Nipype、fMRIPrep 或 DIPY。
下载完成后，将返回的模板路径传给相应 FNIT API；现有 pipeline 的输入参数
和默认值不会因安装模板而改变。

| profile | 文件数 | 用途 |
|---|---:|---|
| `dmri` | 5 | FA 配准、TBSS 骨架和 MMORF 标量／tensor 参考 |
| `registration` | 7 | 1 mm、2 mm T1 配准与相应脑掩膜 |
| `all` | 11 | 上述两个组合的去重集合；Python 和 CLI 的默认值 |

profile 的必要文件如果还没有以 `published` 状态进入 FNIT 目录，安装器会
在创建目录和开始下载前列出缺项。它不会根据文件名猜测下载地址，也不会
绕过发布目录去寻找其他版本。

## 2. Python 调用

```python
from pathlib import Path
from fnit.standard_assets import prepare_standard_assets

# 模板保存目录必须是绝对路径，可以使用已经存在的目录。
template_directory = Path("/data/templates")
# 选择 dMRI 所需的五个模板，返回文件名到绝对路径的映射。
template_paths = prepare_standard_assets(template_directory, profile="dmri")
# FA 模板可用于相应 pipeline 的 fa_template 输入。
fa_template_path = template_paths["FMRIB58_FA_1mm.nii.gz"]
# 骨架模板可用于 TBSS 路径的 fa_skeleton 输入。
fa_skeleton_path = template_paths["FMRIB58_FA-skeleton_1mm.nii.gz"]
print(fa_template_path, fa_skeleton_path)
```

### 输入数据格式

此入口的输入是目录和资源组合，没有待处理的 MRI 被试影像。

- `output_dir` 为字符串或 `pathlib.Path`，展开 `~` 后必须是绝对路径。
- 目录可以尚不存在；正常安装会创建目录，`verify_only=True` 不创建目录。
- 本地发布目录是包内 JSON，仅使用 `group="standard"`、`status="published"`
  的资源条目。用户不需要提供或编辑该文件。
- 条目必须包含安全的文件 basename、大小、SHA-256 和原作者 HTTPS URL。
- 文件名不得含目录分隔符或指向输出目录之外的位置。
- 安装器不读取被试影像的 shape、affine、orientation、dtype 或单位。
- 不需要用户提供脑掩膜，也不要求用户事先运行 preprocessing。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `output_dir` | 是 | `str` / `Path` | 无 | 模板保存目录，必须是绝对路径 |
| `profile` | 否 | `str` | `"all"` | `dmri`、`registration` 或 `all` |
| `verify_only` | 否 | `bool` | `False` | 只检查本地文件，不下载、不创建目录、不改安装记录 |

返回值为 `dict[str, Path]`。字典键是标准文件名，值是保存文件的绝对路径。
不同组合包含不同的键；访问其他组合中的文件前，应先安装对应 profile。

### 输出

`profile="all"` 的目录结构如下：

```text
/data/templates/
├── FMRIB58_FA_1mm.nii.gz
├── FMRIB58_FA-skeleton_1mm.nii.gz
├── FSL_HCP1065_FA_1mm.nii.gz
├── FSL_HCP1065_tensor_1mm.nii.gz
├── MNI152_T1_1mm.nii.gz
├── MNI152_T1_1mm_brain.nii.gz
├── MNI152_T1_1mm_brain_mask.nii.gz
├── MNI152_T1_2mm.nii.gz
├── MNI152_T1_2mm_brain.nii.gz
├── MNI152_T1_2mm_brain_mask.nii.gz
├── MNI152_T1_2mm_brain_mask_dil.nii.gz
└── standard-assets.json
```

| 文件或文件组 | 格式及用途 |
|---|---|
| `FMRIB58_FA_1mm` | NIfTI FA 标准参考，用于对应 FA 配准路径 |
| `FMRIB58_FA-skeleton_1mm` | NIfTI 白质 FA 骨架，与相应 TBSS 模板配套 |
| `FSL_HCP1065_FA_1mm` | NIfTI HCP FA 参考，用于相应 MMORF 标量输入 |
| `FSL_HCP1065_tensor_1mm` | NIfTI tensor 参考，用于相应 MMORF tensor 输入 |
| `MNI152_T1_1mm` / `2mm` | 对应分辨率的完整 T1 标准参考 |
| `MNI152_T1_*_brain` | 对应分辨率的去颅骨 T1 标准参考 |
| `MNI152_T1_*_brain_mask` | 对应网格的脑掩膜 |
| `MNI152_T1_2mm_brain_mask_dil` | 2 mm 扩张脑掩膜，可供相应 FNIRT 输入使用 |
| `standard-assets.json` | 本次 profile 的逐文件来源、大小、SHA-256 和许可元数据 |

NIfTI 文件保持官方原始字节：shape、affine、orientation、dtype、单位和
NIfTI intent 均不转换，安装器不重采样、不重编码 label，也不创建变换。
文件名中的 `1mm` / `2mm` 用于区分模板版本；实际网格应以影像 header 为准。

本次发布文件的实际 header 如下；全部为 LAS 方向，空间单位为 mm。
完整 affine 与单位记录见[实测 header 清单](../../validation/assets_release_20261006/standard_headers.public.json)。

| 文件组 | shape | 原始存储 dtype | 数值含义 |
|---|---|---|---|
| `FMRIB58_FA_1mm` | `182 × 218 × 182` | `int16` | 原包 FA 参考强度编码 |
| `FMRIB58_FA-skeleton_1mm`、`FSL_HCP1065_FA_1mm` | `182 × 218 × 182` | `float32` | FA 骨架强度或 FA 值 |
| `FSL_HCP1065_tensor_1mm` | `182 × 218 × 182 × 6` | `float32` | 六通道 tensor；通道约定见 MMORF 手册 |
| `MNI152_T1_1mm`、`MNI152_T1_1mm_brain` | `182 × 218 × 182` | `int16` | T1 参考强度，无绝对物理强度单位 |
| `MNI152_T1_2mm`、`MNI152_T1_2mm_brain` | `91 × 109 × 91` | `int16` | T1 参考强度 |
| 1 mm 脑掩膜 | `182 × 218 × 182` | `uint8` | `0` 为背景，`1` 为脑内 |
| 2 mm 脑掩膜及扩张脑掩膜 | `91 × 109 × 91` | `uint8` | `0` 为背景，`1` 为掩膜内 |

`FMRIB58_FA_1mm` 的原始值为 0–8676，header 不带幅度缩放。
安装器不会将其转换为 0–1；所选 pipeline 负责自己的强度处理。

使用 tensor 时，仍须符合 [TorchMMORF](../mmorf/README.md) 的六通道顺序和
单位要求；使用多个模板时，仍须通过相应 pipeline 的空间和网格检查。
安装完成不代表任意模板组合都适合任意注册任务。

安装记录保存所选 Release URL、原作者备用 URL，以及实际使用的下载 URL。
复用已验证本地文件时，`retrieval_url` 为 `null`，表示本次未进行网络下载。
后续安装其他 profile 会更新本次安装记录，但保留目录内已有的其他模板。

## 3. 命令行调用

```bash
fnit-setup-standard-assets --output-dir /data/templates
```

只安装 dMRI 组合：

```bash
fnit-setup-standard-assets --output-dir /data/templates --profile dmri
```

离线校验已经安装的注册组合：

```bash
fnit-setup-standard-assets --output-dir /data/templates --profile registration --verify-only
```

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `--output-dir` | `output_dir` | 必需的绝对输出目录 |
| `--profile` | `profile` | 选择 `dmri`、`registration` 或 `all`；默认 `all` |
| `--verify-only` | `verify_only=True` | 仅校验本地文件 |

Python 与 CLI 的默认 profile 相同。命令成功后逐个打印验证文件的路径。
参数、发布目录或文件校验失败时，CLI 返回非零退出状态并显示原因。
若使用尚未安装为 console script 的源码检验入口，可运行：

```bash
python -m fnit.standard_assets --output-dir /data/templates --profile dmri
```

## 4. 原软件调用

FSL 的对应内容是标准资源目录，没有完成同一资源准备任务的独立影像算法
命令。已有独立 FSL 参考环境时，可从其安装目录复制所需模板，例如：

```bash
mkdir -p /data/reference_templates
cp "$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz" /data/reference_templates/
cp "$FSLDIR/data/standard/FMRIB58_FA-skeleton_1mm.nii.gz" /data/reference_templates/
cp "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" /data/reference_templates/
cp "$FSLDIR/data/standard/FSL_HCP1065_FA_1mm.nii.gz" /data/reference_templates/
cp "$FSLDIR/data/standard/FSL_HCP1065_tensor_1mm.nii.gz" /data/reference_templates/
```

| FNIT 参数 | 原软件资源准备方式 |
|---|---|
| `output_dir` | 复制模板的目标目录 |
| `profile="dmri"` | 选择上述五个标准文件 |
| `profile="registration"` | 选择 T1 full、brain、mask 及 2 mm 扩张 mask |
| `verify_only=True` | 对复制后的文件独立计算大小和 SHA-256 |

此入口实现固定资源选择、完整性校验和安装记录，不执行 FSL preprocessing、
TBSS、FLIRT、FNIRT 或 MMORF。不同 FSL 版本中的同名文件可能拥有不同字节；
对照本次资源时，应使用 FNIT 发布目录记录的版本和 SHA-256。

## 5. 最新精度和运行时间

这是文件安装功能，没有 MRI 算法精度或 GPU 运行时间 benchmark。
文件完整性依据实际字节数与 SHA-256；不将其解释为配准或分割结果等价。

| 验证范围 | 当前证据 |
|---|---|
| 下载器行为 | [定向测试](../../tests/test_standard_assets.py)：profile、校验、缓存、备用来源和安全文件名 |
| 真实资源来源 | [发布目录](../../src/fnit/_release_asset_catalog.json) 中 `group="standard"` 的已发布条目 |
| 端到端下载耗时 | 未测；受网络和本地缓存影响 |
| 分步骤耗时、峰值 GPU 显存 | 不适用；安装器不建立 CUDA 计算任务 |

测试中的小字节 fixture 仅验证下载器，不代替真实 MRI benchmark。
本功能不生成脑图；相关真实数据结果见 [dMRI pipeline](../dmri_pipeline/README.md)
和 [TorchFNIRT](../fnirt/README.md) 的验证章节。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-06 | 随本次功能提交绑定 | 新增标准模板 profile、Release 优先下载及离线校验 | 下载器定向测试；无 MRI 算法计时 |

## 7. 参考文献、原软件和资源

- 实现源码：[standard_assets.py](../../src/fnit/standard_assets.py)。
- 固定下载入口：[FNIT assets-v1](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)。
- 官方模板来源：[FSL data_standard](https://git.fmrib.ox.ac.uk/fsl/data_standard)。
- 原软件说明：[FSL 官方文档](https://fsl.fmrib.ox.ac.uk/fsl/docs)。
- 原软件许可：[FSL 6.0 许可文本](../../licenses/FSL-6.0.txt)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| dMRI 五项标准模板 | FA／骨架／tensor 参考 | FSL `data_standard` | 发布目录逐文件记录 | 发布目录逐文件记录 | 以条目许可记录为准，保留 FSL 非商业使用条件 |
| registration 七项模板 | T1 full／brain／mask 参考 | FSL `data_standard` | 发布目录逐文件记录 | 发布目录逐文件记录 | 以条目许可记录为准，保留 FSL 非商业使用条件 |

每个资源的准确大小、完整 SHA-256、固定原始 URL 和再分发依据均集中在
[发布目录](../../src/fnit/_release_asset_catalog.json)，安装后也写入本地记录。
GitHub 下载地址不改变资源的原始许可；FSL 资源保留其非商业使用限制，
商业用途应向原作者取得适用授权。尚未许可确认和发布的资源不会进入安装器。
