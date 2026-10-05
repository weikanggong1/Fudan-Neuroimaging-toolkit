# 权重、图谱和模板

FNIT仓库及wheel不包含模型权重。使用主页Conda环境中的安装器获取所需资源；
模型组只负责文件配置，不安装或运行原软件。

## 选择需要的模型

```bash
fnit-setup-weights --model synthstrip
fnit-setup-weights --model fmri --dest /data/fnit_weights
fnit-setup-weights --all --dest /data/fnit_weights
```

| 模型组 | 文件数 | 总大小（B） | 适用功能 |
|---|---:|---:|---|
| `synthstrip` | 1 | 30851709 | 对应同名模型/流程；独立调用见功能页。 |
| `fast-vbm` | 2 | 3539482133 | 对应同名模型/流程；独立调用见功能页。 |
| `fmri` | 2 | 3539482133 | 对应同名模型/流程；独立调用见功能页。 |
| `synthstrip-nocsf` | 1 | 30851709 | 对应同名模型/流程；独立调用见功能页。 |
| `synthmorph-rigid` | 1 | 51656152 | 对应同名模型/流程；独立调用见功能页。 |
| `synthmorph-affine` | 1 | 51455312 | 对应同名模型/流程；独立调用见功能页。 |
| `synthmorph-deform` | 1 | 3508630424 | 对应同名模型/流程；独立调用见功能页。 |
| `synthmorph-joint` | 2 | 3560085736 | 对应同名模型/流程；独立调用见功能页。 |
| `wmh-synthseg` | 1 | 790531383 | 对应同名模型/流程；独立调用见功能页。 |
| `synthseg` | 4 | 53087016 | 对应同名模型/流程；独立调用见功能页。 |
| `synthseg-plus` | 5 | 106177856 | 对应同名模型/流程；独立调用见功能页。 |
| `recon-all` | 11 | 3653913443 | 对应同名模型/流程；独立调用见功能页。 |
| `synthsr` | 1 | 106163752 | 对应同名模型/流程；独立调用见功能页。 |
| `synthsr-lowfield` | 1 | 106163752 | 对应同名模型/流程；独立调用见功能页。 |
| `synthsr-v1` | 1 | 53075984 | 对应同名模型/流程；独立调用见功能页。 |

`recon-all` 当前为11个文件，包含SynthMorph deform、分割及重建辅助模型，
不是旧说明中的6或10文件轻量组合。
Release列出20个不同文件，共4,845,447,631 B；当前模型组联集仅18个文件，
`--all`/默认实际安装18个，共4,845,447,015 B。
`mca-dura.ctab`和`sclimbic.volstats.csv`不被当前模型组引用。
不同模型组共享文件，不应把分组总量相加。
TorchFAST、TorchFLIRT、TorchFNIRT、DTIFIT等数值模块不读取模型checkpoint。

## 保存位置和离线校验

```bash
fnit-setup-weights --model recon-all --dest /data/recon_weights
fnit-setup-weights --model recon-all --dest /data/recon_weights --verify-only
export FNIT_WEIGHTS=/data/recon_weights
```

安装器检查实际大小和SHA-256，然后保存默认目录配置。
`--verify-only`只验证，不更改配置；目录不匹配、文件缺失或SHA错误都会报错。
可在联网机器准备目录后复制到计算节点，再运行离线校验。
`--model`可重复指定；未指定组时默认处理全部组。

## 固定 Release 与原站

[assets-v1](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)
含20个模型文件与30个HCP项（含许可证）。
安装器优先使用可校验的Release附件，失败时回退作者官方来源。
大文件 `synthmorph.deform.3.h5` 分为两个附件，本地合并后验证完整大小与SHA。

公开Release文件的原始大小、SHA、官方地址与许可见 [逐文件清单](RESOURCE_MANIFEST.md)。
本轮公开manifest下载为28,717 B、SHA-256为
`24a292cc79b0e9530559b7edb3fc09c7b157d0023b06335798ca4a902d5c1a3f`。
20权重大小/SHA和30HCP SHA与当前源码一致；SynthSeg parc原站annex URL有一处已修正的路径差异，
其内容大小/SHA相同，记录见 [核对报告](../validation/documentation/readme_manual_20261005/resources.audit.json)。
本轮没有重新下载全部大权重，清单核对与文件二进制核验分开记录。

## fMRI 表面和标准模板

<a id="fmri-templateflow-原站模板"></a>

```bash
fnit-setup-fmri-surface-assets --output-dir /data/hcp_assets --fmriprep
fnit-setup-fmri-surface-assets --output-dir /data/hcp_assets --fmriprep --msmall
```

基础HCP资源含球面、脑沟、ROI、查找表和MSM配置，优先固定Release；
TemplateFlow MNI6-2mm T1w、brain mask及HCP dseg仅从原站下载，逐文件检查大小和SHA。
MSMAll的d7–d21低维参考也从固定HCP上游下载，检查大小和SHA。
基础HCP文件当前安装器检查SHA，公开清单补充大小；不将这一记录称为全部已有显式大小验证。

| TemplateFlow文件 | 大小（B） | 用途 |
|---|---:|---|
| tpl-MNI152NLin6Asym_res-02_T1w.nii.gz | 1,412,252 | volume目标空间。 |
| tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz | 28,557 | 同目标网格脑mask。 |
| tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz | 25,762 | CIFTI皮层下BrainModel轴。 |

完整SHA和原站地址见 [逐文件清单](RESOURCE_MANIFEST.md#原站资源)。
MNI标签的空间身份不能只由文件名或相同shape推断。

## 重建和亚区图谱

```bash
fnit-setup-recon-all-assets --dest /data/recon_assets
fnit-setup-recon-all-assets --dest /data/recon_assets --verify-only
fnit-setup-subregion-atlases --output-root /data/subregion_atlases --device cpu
```

recon-all模型与默认98项核心图谱/模板是两组资源，安装、大小/SHA和native编译见 [重建手册](recon_all/README.md)。
默认核心98项共264,773,919 B；`--all`完整111项共374,437,464 B，包含独立阶段验证额外资源。
该图谱组含第三方来源，未逐项确认FNIT再分发权，因此保持FreeSurfer原站获取，不新增Release镜像。
亚区图谱由专用安装器配置，BrainstemSS、ThalamicNuclei及HippoSF来源和逐文件SHA见 [亚区手册](subregions/README.md)。
空间转换与MS-HBM的CBIG资源分别按 [空间转换](space_conversion/README.md) 和 [MS-HBM](mshbm/README.md) 配置；
未获明确再分发许可的外部atlas不上传FNIT。

## 许可和归属

<a id="权重许可与归属"></a>

SynthStrip和SynthMorph的固定Release权重选择CC BY4.0，保留原作者及论文归属：
[SynthStrip官网](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)、[SynthMorph官网](https://synthmorph.io/#code)。
其余FreeSurfer模型与查找表按固定清单记录的
[FreeSurfer Software License](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)保存许可与归属。
不能从模型代码采用Apache/MIT推断所有权重也采用相同许可。

HCP文件遵循固定上游的 [LICENSE.md](https://github.com/Washington-University/HCPpipelines/blob/f8cac6892f88bdf889d644711ff038198eb81533/LICENSE.md)，
许可随Release及下载结果保存。
TemplateFlow、第三方图谱与用户atlas不因代码许可而自动获得再分发授权。
更多代码归属见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
