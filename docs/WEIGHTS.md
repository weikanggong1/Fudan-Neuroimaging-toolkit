# 预训练权重：下载、校验与公开发布

Git 仓库和 wheel 均不包含权重。SynthStrip、SynthMorph、33 类 SynthSeg、
WMH-SynthSeg、SynthSR、Python recon-all 和可选辅助分割阶段使用 FreeSurfer 官方发布的文件；
配置脚本下载文件、核对大小与 SHA-256，并
保存权重目录。此后 Python API 和 `fnit` 命令会自动查找它，下载过程无需安装
FreeSurfer。TorchFAST、TorchFLIRT 和 TorchFNIRT 是数值算法，
不使用模型权重。FastVBM 和 fMRI 体积流程的默认 SynthMorph 分支使用
`synthstrip.1.pt` 与 `synthmorph.deform.3.h5`；fMRI 的 PyTorch FNIRT 分支仍需 SynthStrip 做脑提取。

## 一次配置，后续自动使用

在仓库根目录运行。默认下载下表 20 个文件到 `~/.cache/fnit/`。优先使用 [FNIT 固定版本 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)；Release 暂时不可用时回退到表中的官方地址。`synthmorph.deform.3.h5` 在 Release 中分为两卷，安装器合并后核对完整文件的 SHA-256。其他文件也在写入 `.part` 后通过大小和 SHA-256 校验，再更名为正式文件。

```bash
python tools/setup_weights.py --all
```

也可只下载所需模型。`joint` 需要 affine 和 deform 两份权重；下例再加默认 SynthStrip 权重，共三个文件：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
```

只运行 WMH-SynthSeg 时下载其单个 checkpoint：

```bash
python tools/setup_weights.py --model wmh-synthseg
```

Python recon-all 使用另一份 **33 类 SynthSeg** 模型，不能用 WMH-SynthSeg 的 39 类 checkpoint 替代。整例入口下载 6 个实际使用的权重/标签文件；独立 SynthSeg 只需其中四个：

```bash
python tools/setup_weights.py --model synthseg
python tools/setup_weights.py --model synthseg-plus
python tools/setup_weights.py --model recon-all
```

SynthSR 默认、低场和 v1 是三份不同权重。只需通用 v2 时下载一份；需要全部变体时把三个模型名同时传给脚本：

```bash
python tools/setup_weights.py --model synthsr
python tools/setup_weights.py --model synthsr --model synthsr-lowfield --model synthsr-v1
```

有独立模型目录时，用 `--dest` 指定一次即可。脚本成功后把绝对路径保存在用户缓存目录的 `weights.json`，之后 API 和 CLI 可以省略 `weights=` / `--weights`：

```bash
python tools/setup_weights.py --all --dest /path/to/models
python tools/setup_weights.py --all --verify-only
```

`--verify-only` 只检查当前权重目录，不下载或修改配置。已从联网机器复制了权重时，运行 `python tools/setup_weights.py --all --dest /path/to/copied/models`：现有文件校验成功后直接保存目录，无需重新下载。安装 wheel 后也可使用相同选项的 `fnit-setup-weights` 命令。

可选模型名：`synthstrip`、`synthstrip-nocsf`、`synthmorph-rigid`、`synthmorph-affine`、`synthmorph-deform`、`synthmorph-joint`、`synthseg`、`synthseg-plus`、`wmh-synthseg`、`recon-all`、`synthsr`、`synthsr-lowfield`、`synthsr-v1`、`fast-vbm` 和 `fmri`。`recon-all` 包含 SynthStrip、SynthMorph affine 和 33 类 SynthSeg 的六个文件，重复选择时只下载一次。`fast-vbm` 和 `fmri` 都是 `synthstrip.1.pt` 与 `synthmorph.deform.3.h5` 的依赖别名。端到端 fMRI 选择 `registration_backend="fnirt"` 时只需 `--model synthstrip`；仅单独调用 `register_t1_to_mni`、并已备妥去颅骨 T1 与 MNI 模板时不需要 checkpoint。`--model` 可重复；不写 `--model` 时等同 `--all`。显式 API/CLI 权重路径优先，其次是 `FNIT_WEIGHTS` 环境变量，再次是脚本保存的目录，然后是默认缓存。`XDG_CACHE_HOME` 可改变缓存根目录。单独的模型推理不联网；统一 `segment_subregions` 首次运行会下载并校验缺失的 SynthSeg/SynthSeg+ 权重。离线运行前应使用配置脚本备妥权重及图谱。

官方文件于 **2026-09-23 至 2026-09-29** 从 FreeSurfer 官方源码、git-annex 或已安装的官方发行版核对大小和 SHA-256；Release 保留原始字节，不改变权重格式。SynthStrip/SynthMorph 的 SHA-256 来自本包已完成数值验证的权重，并与 FreeSurfer 官方仓库的 git-annex 指针一致；WMH-SynthSeg 和 SynthSR v1 的 SHA-256 来自官方文件的完整下载校验。SynthSR v2 两份文件的大小和 SHA-256 与 FreeSurfer git-annex 对象名一致；配置脚本下载后还会逐字节校验。此处的版本号固定，不会自动跟随上游替换为新模型。

`synthseg-plus` 在 `synthseg` 四个文件之外增加 53,090,840 字节的皮层分区网络；该文件已与 FreeSurfer 8.2.0-1 安装文件核对 SHA-256。`--all` 的总量因此增加 53,090,840 字节。

## 官方文件

| 功能 | 文件与官方下载链接 | 字节数 | 使用场景 |
|---|---|---:|---|
| SynthStrip | [synthstrip.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt) | 30,851,709 | 默认脑提取 |
| SynthStrip | [synthstrip.nocsf.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt) | 30,851,709 | `no_csf=True` / `--no-csf` |
| SynthMorph | [synthmorph.affine.2.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.2.h5) | 51,455,312 | affine；joint 的仿射阶段 |
| SynthMorph | [synthmorph.deform.3.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.deform.3.h5) | 3,508,630,424 | deform；joint 的非线性阶段 |
| SynthMorph | [synthmorph.rigid.1.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.rigid.1.h5) | 51,656,152 | rigid |
| WMH-SynthSeg | [WMH-SynthSeg_v10_231110.pth](https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth) | 790,531,383 | `wmh-synthseg`；解剖结构与 WMH 的联合分割 |
| SynthSR | [synthsr_v20_230130.h5](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/f08/bc9/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5) | 106,163,752 | `synthsr`；默认通用 v2 |
| SynthSR | [synthsr_lowfield_v20_230130.h5](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/de0/799/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5) | 106,163,752 | `synthsr-lowfield`；低场单输入 v2 |
| SynthSR | [synthsr_v10_210712.h5](https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthsr/synthsr_v10_210712.h5) | 53,075,984 | `synthsr-v1`；2021 年通用模型 |
| 33 类 SynthSeg | [synthseg_2.0.h5](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/bee/241/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5) | 53,079,152 | `synthseg`、`recon-all`；33 类分割模型 |
| SynthSeg+ | `synthseg_parc_2.0.h5` | 53,090,840 | `synthseg-plus`；SHA-256 `83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684` |
| 33 类 SynthSeg | [synthseg_segmentation_labels_2.0.npy](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_labels_2.0.npy) | 348 | 标签编号 |
| 33 类 SynthSeg | [synthseg_segmentation_names_2.0.npy](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_names_2.0.npy) | 7,168 | 标签名称 |
| 33 类 SynthSeg | [synthseg_topological_classes_2.0.npy](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_topological_classes_2.0.npy) | 348 | 拓扑类别 |
| 可选辅助分割阶段 | [entowm.fsm31.t1.nstd00-30.nstd21-108.h5](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/entowm.fsm31.t1.nstd00-30.nstd21-108.h5) | 3,296,904 | EntoWM 模型 |
| 可选辅助分割阶段 | [entowm.ctab](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/entowm.ctab) | 318 | EntoWM 查找表 |
| 可选辅助分割阶段 | [mca-dura.both-lh.nstd21.fhs.h5](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/mca-dura.both-lh.nstd21.fhs.h5) | 3,294,856 | MCA/dura 模型 |
| 可选辅助分割阶段 | [mca-dura.ctab](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/mca-dura.ctab) | 116 | MCA/dura 查找表 |
| 可选辅助分割阶段 | [vsinus.no-sp.m.all.nstd10-070.h5](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/vsinus.no-sp.m.all.nstd10-070.h5) | 3,296,904 | 静脉窦模型 |
| 可选辅助分割阶段 | [sclimbic.volstats.csv](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_sclimbic_seg/sclimbic.volstats.csv) | 500 | 上游体积统计字段表 |

合计 **4,845,447,631 字节**，约 4.85 GB（4.51 GiB）。其中 `synthseg` 单独安装需四个文件、53,087,016 字节；`synthseg-plus` 需五个文件、106,177,856 字节；`recon-all` 组需 10 个文件、145,283,019 字节，包含尚待接入整例的 MCA/dura 和静脉窦模型；不下载 deform、额外查找表和统计字段表。只使用默认 SynthStrip 时需要第一个文件；默认 joint 配准需要 affine 和 deform 两个文件；WMH-SynthSeg 只需其单独的 `.pth`；默认 SynthSR 只需通用 v2 的 `.h5`。[33 类 SynthSeg 官方目录](https://github.com/freesurfer/freesurfer/tree/v8.2.0/mri_synthseg) · [辅助分割官方目录](https://github.com/freesurfer/freesurfer/tree/v8.2.0/mri_sclimbic_seg) · [WMH 官方目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg) · [SynthSR 官方目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthsr)

SHA-256：

```text
37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33  synthstrip.1.pt
62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28  synthstrip.nocsf.1.pt
1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6  synthmorph.affine.2.h5
95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536  synthmorph.deform.3.h5
284c145fce47e98ecf3fdeda2163f646ac3ebb0240e87dd50d71d879f4d5b3af  synthmorph.rigid.1.h5
0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988  WMH-SynthSeg_v10_231110.pth
a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b  synthsr_v20_230130.h5
a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84  synthsr_lowfield_v20_230130.h5
2fd59e96196388360eba95254fb6dfc9eb9eb8638018b590575e47e0a387f255  synthsr_v10_210712.h5
f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e  synthseg_2.0.h5
5ef25ec33fe917ac99f30b8f2185b2d77121136ee411b9c4970c0b59be615ed8  synthseg_segmentation_labels_2.0.npy
234eb6d514e10d6ebd748a8b30a1d12d9426fd874c607e37852406fae8f290fc  synthseg_segmentation_names_2.0.npy
650b4b96834485c1e6d7421de4af74da80d861e6b2a39ef1164389bde3a5e14a  synthseg_topological_classes_2.0.npy
9be55798498331f655acd75d4f0cd5036463e0f497bbb239be0167d6a9129a07  entowm.fsm31.t1.nstd00-30.nstd21-108.h5
fa46a74e7c5385b6e474553acbb34f536dac640c52586ec4193a0ea9739948f1  entowm.ctab
da6a7b994e3e804cc3dc0e98e965c28a802ddcd38fd9b5c680d75cef285657b0  mca-dura.both-lh.nstd21.fhs.h5
77faedc95badab7b01ab8ef71889724c5eda33a0862fefa845b5ba33b8bc3e13  mca-dura.ctab
3d78948741306a31337468c86be55821913edb73855116fcb063b61135b90f12  vsinus.no-sp.m.all.nstd10-070.h5
691b8e1a1d74668b65a0571e2854a4a83c484438107693d71eef5a081d17380b  sclimbic.volstats.csv
```

也可在 [provenance.json](provenance.json) 查看 SynthStrip/SynthMorph 权重与参考实现的来源记录。SynthMorph、SynthSR v2 和 33 类 SynthSeg 的 `.h5` 由 FreeSurfer 的 git-annex 管理；直接下载 GitHub 同名 `raw` 路径可能只得到链接文本。本表链接指向实际 annex 对象；新增的十个 FreeSurfer 模型及查找表均已完整下载校验。[官方 SynthMorph 目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthmorph) · [FreeSurfer git-annex 构建说明](https://surfer.nmr.mgh.harvard.edu/fswiki/BuildGuide)

## 手动下载示例

下面下载默认 SynthStrip 权重并检查 SHA-256。其他模型替换为上表的完整 URL、文件名和对应 SHA-256 即可。

```bash
mkdir -p weights
curl --fail --location --retry 3 \
  'https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt' \
  --output weights/synthstrip.1.pt.part
printf '%s  %s\n' \
  '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33' \
  'weights/synthstrip.1.pt.part' | sha256sum --check - && \
  mv weights/synthstrip.1.pt.part weights/synthstrip.1.pt
export FNIT_WEIGHTS="$PWD/weights"
```

`pip install`、导入模块和推理不下载权重。离线计算节点可从联网机器复制已校验的权重目录。

## TorchFAST 不需要权重

`TorchFAST` 和 `fnit fast` 直接运行 HMRF-EM、
bias field 和 PVE 数值计算，不读取 checkpoint，也不需要执行
`tools/setup_weights.py`。只有从原始、未去颅骨 T1 开始并先调用 SynthStrip 时，
才需要配置 `synthstrip.1.pt`。`setup_weights.py --all` 的 20 个文件属于上表
学习模型及其查找表，不含 TorchFAST 文件。

`FastVBM` / `fnit fast-vbm` 从原始 T1w 开始，默认调用 SynthStrip，因此需要
`synthstrip.1.pt`。`registration_backend="synthmorph"` 还读取
`synthmorph.deform.3.h5`；该分支传入外部线性初始化并设置 `mid_space=False`，因此
不需要 `synthmorph.affine.2.h5`。`registration_backend="fnirt"` 使用本包 PyTorch
cubic B-spline 优化器，不读取 SynthMorph 权重。独立 TorchFLIRT、TorchFAST、
TorchFNIRT、TorchApplyWarp、Jacobian 和 modulation 都不读取 checkpoint。

`python tools/setup_weights.py --model fast-vbm` 配置 SynthStrip 和 deform 两个文件，
是两个后端的权重超集；只运行 TorchFNIRT 分支可改为 `--model synthstrip`。GM
template 是独立输入，不是模型权重，也不由本仓库或配置脚本下载。fMRI 同理：`fnit-setup-weights --model fmri` 安装默认链的两份权重；MNI152 T1 2 mm 模板和可选脑掩膜由用户提供绝对路径。

## 权重许可与归属

**20 个文件中，SynthStrip 与 SynthMorph 的五个权重可选择 MIT 或 CC BY 4.0 许可。** 两个功能的官网 “Code and Weights” 均明确提供这一选择。[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)，[SynthMorph](https://synthmorph.io/#code)

Release 中这五个权重选用 CC BY 4.0，保留原作者、原始模型名称、官方来源和许可链接；文件未经转换或修改。权重归原作者所有，本项目提供独立的 PyTorch 实现及验证，不将这些模型声称为本项目训练所得。Release 说明链接原论文，清单记录逐文件 SHA-256。[MIT 条款](https://choosealicense.com/licenses/mit/) · [CC BY 4.0 条款](https://creativecommons.org/licenses/by/4.0/)

**33 类 SynthSeg、WMH-SynthSeg、FreeSurfer 辅助模型/查找表和 SynthSR 权重遵循 [FreeSurfer Software License](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)。** 官方没有为这些文件宣布上述 MIT 或 CC BY 4.0 双许可。该许可对下载、使用和再分发要求保留条款与归属信息；原文说明软件为研究用途设计，临床应用未获审查或批准。[SynthSeg 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) · [WMH-SynthSeg 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg) · [SynthSR 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR)

上述许可针对权重。改编代码及依赖继续遵守 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 中的 FreeSurfer、Apache 等条款。

## 固定版本 Release 与原站下载范围

[FNIT `assets-v1` Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)保存本页 20 个权重文件。每个附件的原始 URL、字节数、SHA-256 和许可记录在 Release 附带的 `asset-manifest.json`。`synthmorph.deform.3.h5` 拆成两个小于 2 GiB 的附件；安装器在本地合并，并核对上表中的完整 SHA-256。下载器不会调用 FreeSurfer 程序。

Release 同时保存 [HCPpipelines v4.7.0 固定提交](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533)中的 29 个公开 fMRI 表面模板及配置文件，另附原仓库的 `LICENSE.md`。`fnit-setup-fmri-surface-assets` 优先下载这些附件，失败后回退 HCP 原站；`--fmriprep` 指定的 TemplateFlow HCP dseg 继续从 [TemplateFlow 原站](https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/)下载。

FreeSurfer recon-all 的外置图谱和模板仍由 `fnit-setup-recon-all-assets` 从 FreeSurfer 原站获取。该组中含 MNI 和其他第三方来源的数据，尚未逐一确认其再分发权利，因此不进入 Release。fMRI 体积流程的 FSL MNI152 T1 与脑掩膜仍按[功能说明](fmri/README.md)从 FSL 官方数据包获取；Tian atlas 仍由用户按其来源条款提供。

权重与 HCP 模板均保留原作者归属。SynthStrip 和 SynthMorph 的五个模型在 Release 中选用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；其余 FreeSurfer 权重依 [FreeSurfer Software License](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)再分发，Release 附完整许可文本。HCP 模板依[HCPpipelines 原仓库许可](https://github.com/Washington-University/HCPpipelines/blob/f8cac6892f88bdf889d644711ff038198eb81533/LICENSE.md)再分发，许可文本也作为附件提供。具体模型论文和原实现链接见各功能说明。
