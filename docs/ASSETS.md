# 运行资源安装

模型权重、图谱、标准模板和固定native源码优先从FNIT的固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)取得。安装器只对许可已确认、已发布且大小/SHA-256匹配的文件使用Release；未收录或镜像下载失败时保留固定作者来源回退。

Conda/PyPI依赖仍按[主页环境](../README.md#安装)安装，不制作离线依赖包。个人FreeSurfer运行许可证由用户提供，不进入Git仓库、安装包或Release。

本轮新增165项获授权运行资源，另发布完整运行清单和许可汇编2项附件；Release目前共220项附件。20项再分发许可尚待确认的运行资源继续使用原作者来源，个人FreeSurfer运行许可证另行准备。

## 按功能准备资源

激活主页的Conda环境，在FNIT仓库根目录执行需要的命令。下面的`/data/fnit_resources`可以改成自己的绝对路径；未使用的功能不用安装。

| 功能 | 安装入口 | 安装内容 |
|---|---|---|
| SynthStrip、SynthMorph及分割模型 | `fnit-setup-weights` | 按模型组安装外置权重与标签数组。 |
| recon-all | `fnit-setup-recon-all-assets` | 固定图谱、网格、查找表和模板。 |
| fMRI volume/surface、MSMAll | `fnit-setup-fmri-surface-assets` | HCP/fsLR模板、MSM配置、WRN低维参考；可选MNI6-2mm资源。 |
| MNI/fsaverage/fsLR转换 | `fnit-setup-space-assets` | HCP2017球面/面积和CBIG RF-ANTs映射。 |
| 标准结构连接组atlas | `fnit-setup-connectome-atlases` | 按所选atlas安装Tian及Schaefer文件。 |
| dMRI和配准标准模板 | `fnit-setup-standard-assets` | FSL标准数据文件；不安装或执行FSL。 |
| MS-HBM volume投影 | `python -m fnit.mshbm.assets_setup` | 群体MNI中层表面和皮层估计掩膜。 |

```bash
# 权重可按实际需要选择模型组；全部组使用 --all。
fnit-setup-weights --model fmri --model recon-all --dest /data/fnit_resources/weights

# 结构重建资源；该命令不生成被试重建结果。
fnit-setup-recon-all-assets --dest /data/fnit_resources/recon-all

# HCP、MNI6和完整MSMAll低维参考。
fnit-setup-fmri-surface-assets --output-dir /data/fnit_resources/hcp --fmriprep --msmall

# 标量/标签脑图的空间转换资源。
fnit-setup-space-assets --output-dir /data/fnit_resources/space

# 按所选标准atlas安装；已有用户native模板可跳过。
fnit-setup-connectome-atlases --atlas schaefer200+tian-s1 \
  --output-dir /data/fnit_resources/connectome

# 标准数据：只需要dMRI时使用 --profile dmri。
fnit-setup-standard-assets --output-dir /data/fnit_resources/standard --profile all
```

MS-HBM体积投影的皮层掩膜要匹配自己的**已预处理MNI BOLD**网格，因此另行执行：

```bash
python -m fnit.mshbm.assets_setup \
  --reference /data/bold/clean_mni_bold.nii.gz \
  --output-dir /data/fnit_resources/mshbm-mni
```

安装器下载的源文件检查固定大小和SHA；按reference重采样生成的`cortical_mask.nii.gz`保留reference网格，不能套用原始掩膜的SHA。

GEMS亚区atlas的本地配置见[亚区手册](subregions/README.md)。固定native程序通过以下命令在激活的Conda环境内编译，源归档按发布目录优先获取：

```bash
bash tools/setup_recon_all_native_conda.sh
```

编译工具链仍通过Conda安装；此步骤不复制系统安装的FreeSurfer二进制。程序范围与运行条件见[recon-all手册](recon_all/README.md)和[构建说明](recon_all/CONDA_CPP_BUILD.md)。

## 校验与离线使用

下载结果与已有缓存都需要通过安装器校验，未通过大小/SHA检查的文件不能用于运行；离线校验会报错。下载临时文件通过校验后才发布到最终路径，联网安装保留重新获取或固定来源回退。

```bash
fnit-setup-weights --model fmri --model recon-all \
  --dest /data/fnit_resources/weights --verify-only
fnit-setup-recon-all-assets --dest /data/fnit_resources/recon-all --verify-only
fnit-setup-standard-assets --output-dir /data/fnit_resources/standard \
  --profile all --verify-only
export FNIT_WEIGHTS=/data/fnit_resources/weights
```

HCP、空间转换、连接组和MS-HBM安装器会先校验已有文件；有效缓存不重复下载。可以在联网机器准备相同目录并复制到计算节点，在节点上再次校验后使用。

资源目录作为相应功能的公开参数传入；安装资源不会自动完成MRI预处理，也不会改变已有处理参数或精度设置。权重目录的配置规则见[权重手册](WEIGHTS.md)。

## 仍需原作者来源的资源

| 资源 | 当前处理 | 原因 / 获取方式 |
|---|---|---|
| 个人FreeSurfer运行许可证 | 用户自行准备`FS_LICENSE` | 由[FreeSurfer官方](https://surfer.nmr.mgh.harvard.edu/)申请；不读取或公布许可证内容。 |
| VPNL来源的17项重建资源 | 使用安装器记录的作者来源 | 独立再分发许可尚待确认；具体路径见[重建资源定义](../src/fnit/recon_all/assets.py)。 |
| MS-HBM的两项Caret派生MNI中层表面 | 从固定CBIG作者来源下载 | 再分发许可尚待确认；来源、大小和SHA见[MS-HBM资源表](mshbm/README.md#外部资源)。 |
| Oxford `template_GM.nii.gz` | 用户从官方公开包准备 | 文件级再分发许可尚待确认；获取步骤见[VBM手册](ukb_vbm/README.md)。 |

用户自己的影像、被试重建结果和自定义atlas不属于Release资源。HCP S1200 d25/d50等未纳入安装器的外部数据也不因本次资源发布获得额外访问或再分发许可。

## 文件目录、来源和许可

- [固定Release附件](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)：下载文件。
- [完整运行资源清单](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/runtime-assets-manifest-20261006.json)：本次新增文件、许可例外和来源记录。
- [公开许可汇编](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/runtime-assets-licenses-20261006.tar.gz)：完整公开条款与native来源notice，不包含个人许可证。
- [安装器使用的发布目录](../src/fnit/_release_asset_catalog.json)：实际附件名称、大小、SHA-256和发布状态。
- [资源来源清单](RESOURCE_MANIFEST.md)：原作者来源和逐资源许可记录。
- [代码及资源归属](../THIRD_PARTY_NOTICES.md)：保留原软件、数据许可和科学引用。

下载位置改变不会改变原作者许可：引用、署名、非商业等条件仍按具体资源条款执行，FSL来源的标准模板保留原非商业条件。软件代码的许可不能代替模型、模板或第三方数据的许可。
