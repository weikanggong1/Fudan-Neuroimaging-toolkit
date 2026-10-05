# FNIT 用户手册迁移与核对

本轮基于实际checkout `140c3739ac6c7a6826bf9421202ef59bec7ffe67`，先读取当前源码、公开API、CLI、原始真实数据报告及资源清单，再整理文档。没有重新执行MRI，没有把历史结果改标为当前代码重测。

主页提供安装、最小Python/CLI例子及每项一句话功能索引；37份 `docs/*/README.md` 与独立fMRI surface手册按 [统一模板](../../../docs/README_TEMPLATE.md)迁移。普通模块约200–500行，复杂pipeline约300–700行；脑图索引没有处理API，保持简短导航。各页第5节不超过25%，逐被试表格移至已有或归档的详细验证记录。

## 核对入口

| 范围 | 记录 |
|---|---|
| 12项配准/影像模块 | [registration.audit.json](registration.audit.json) |
| 11项结构像/重建模块 | [anatomy.audit.json](anatomy.audit.json) |
| 11项扩散/统计模块 | [diffusion_statistics.audit.json](diffusion_statistics.audit.json) |
| fMRI volume/surface、connectome与主页 | [root.audit.json](root.audit.json) |
| 20权重、30HCP固定资源 | [resources.audit.json](resources.audit.json) |
| 统一结构、长度、链接、fence与图片 | [summary.public.json](summary.public.json) |
| 最终文档大小与SHA | [manual_files.sha256.public.json](manual_files.sha256.public.json) |
| 既有验证文档的反向链接兼容 | [backlinks.audit.json](backlinks.audit.json) |

API核对包含真实模块导入、`inspect.signature`、公开参数/default和示例语法；CLI核对实际parser的 `--help`，不通过删除导入或伪造模块来运行parser。依赖和资源要求与主页Conda环境相连。

这些检查证明示例调用形状与当前接口相符；它们不替代真实MRI运行。影像shape、dtype、affine、空间与变换方向依据源码和对应原报告的真实输出检查。精度与耗时按报告的冻结source SHA、输入、程序版本和计时范围保留；缺少同范围参考、分阶段计时或硬件元数据的地方明示缺项。

## 本轮纠正的事实

- SynthSegPlus Python的 `keep_geometry=True` 与CLI默认False分开说明。
- recon-all权重组现为11个文件、3,653,913,443 B；默认模型组安装18个不同文件，Release清单20个。
- recon-all默认核心资产98项，完整含阶段验证资产111项，二者范围分开说明。
- BOLD→T1w是BBR；T1w→MNI的FNIRT路径是FLIRT＋TorchFNIRT，默认后端仍为SynthMorph。
- fMRI采样的FNIT Workbench2.1.0与参考2.0.1不同，不能称同二进制；固定输入零差不推广为整链等价。
- GB/GiB、TF32与FP32例外、CPU/native/Workbench步骤按真实实现说明，未声称所有模块满足20GB或全GPU。
- NODDI方向文件经实际目录与Git跟踪核对齐全，未把ignore搜索遗漏误记为资源缺失。
- 原软件命令按实际入口核对；fMRIPrep的 `--use-syn-sdc` 不接受False字符串，MRtrix尝试预算显式使用 `-seeds` 与 `-select 0`。

现有显式官方FreeSurfer兼容backend仍在源码中；本轮未破坏公开接口。生产示例采用FNIT或只读provided来源，官方分支明确限于独立参考环境。必要独立native源码节点和Workbench需求保留在功能页，不能据文档迁移声明实现已经全部纯PyTorch。

## 资源和许可

公开 [assets-v1清单](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/asset-manifest.json)本轮下载大小28,717 B，SHA-256 `24a292cc79b0e9530559b7edb3fc09c7b157d0023b06335798ca4a902d5c1a3f`。20个权重大小/SHA与源码一致，30个HCP SHA一致；SynthSeg parc官方annex路径有一处差异，内容大小/SHA一致，采用当前源码地址。

清单核对不等于本轮重新下载所有大文件；基础HCP安装器有SHA检查，显式大小检查仅覆盖源码定义的额外文件。完整记录见 [资源表](../../../docs/RESOURCE_MANIFEST.md)。未经逐项确认再分发的图谱和TemplateFlow资源保持作者原站，不新增资源镜像。模型权重许可与代码许可分别记录。

## 历史证据和隐私

迁移前说明保存为各模块 `validation/<module>/readme_archive_20261005.md`；fMRI surface另有 [独立历史档](../../fmri/surface_manual_archive_20261005.md)。旧页顶部标为历史，当前调用说明以docs中的手册为准。仍支持结论的原始benchmark JSON/CSV、源码守卫与脑图保持原文件。

历史SuperBigFLICA说明中的受限表型名/字段标识改为匿名trait标签，不再嵌入带受限文字的历史图；不复制原始数据、具体位点/gene表、个人许可内容或服务器地址。归档文本对受限字段和运算节点代号脱敏，不改变原benchmark数值文件。
本轮文档不发布服务器地址；CPU/GPU型号、线程与计时范围仍以原正式报告为准。

复核命令：

```bash
python validation/documentation/readme_manual_20261005/check_readmes.py
```

该工具只读文档及本地链接，输出结构核对JSON；MRI和原软件对照仍按各独立验证文档复现。
