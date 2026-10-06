# FNIT 运行资源再分发许可审查（2026-10-06）

本审查对照当前 `origin/main=db61cebc9d3683720e6c0a288db3cdc7d6f30a7c` 中实际运行资源、文件大小和 SHA-256，使用原作者官方条款判断是否可镜像。Conda/PyPI 安装包和个人 FreeSurfer 运行 license 不在范围内。逐文件记录、冻结哈希和证据见 [license_audit.public.json](license_audit.public.json)。允许镜像仍须随资源发布完整许可、版权、引用及相应限制；FNIT 代码许可不会替换资源许可。

| 资源组 | 当前判断 | 随资源保留的条款 |
| --- | --- | --- |
| FreeSurfer recon-all 本体资源（81 个，含 ex-vivo） | 可按条件再分发 | FreeSurfer Software License 1.0 全文、MGH 指定前言、第三方 notice、修改说明 |
| GEMS 子分区 atlas（9 个） | 可按条件再分发 | 同上，并保留脑干、丘脑、海马/杏仁核 atlas 原论文引用 |
| MNI ICBM09c recon 配准资源（4 个） | 可按条件再分发 | MNI/McGill 原许可、FreeSurfer 派生处理归属 |
| VPNL/Juelich 皮层标签（17 个） | 尚未闭环 | 官方 atlas 来自第三方 Stanford VPNL，未找到对这些二进制标签的明确再分发授权 |
| TemplateFlow MNI6 T1 和脑 mask（2 个） | 可按非商业条款镜像 | FSL 完整许可、MNI/McGill/Andrew Janke 归属及 TemplateFlow 标准化说明 |
| TemplateFlow HCP dseg（1 个） | 可按条件再分发 | HCP 完整许可、FreeSurfer 归属、TemplateFlow 标准化说明；[数组/网格一致性证据](templateflow_hcp_lineage.public.json) |
| CBIG RF-ANTs4 和共享 cortex mask（4 个） | MIT 条款允许 | CBIG MIT 全文、Wu 等人的原始 data-release 说明和引用 |
| MSHBM 的 Caret 中层网格（2 个） | 尚未闭环 | CBIG 自述下载自 Caret；未找到网格数据级授权，不套用 CBIG 代码 MIT |
| FSL FA/skeleton/T1/HCP1065 tensor 标准资源 | 可按非商业条款镜像 | FSL 完整许可、各模板作者和 HCP 数据致谢 |
| Oxford DATA_public/template_GM | 尚未闭环 | 原 archive 内 18 个 metadata 均未授权 template_GM；公共下载或 pipeline Apache-2.0 不能替代数据许可 |
| 固定 FreeSurfer native source 原归档 | 可按各组件条件镜像 | exact 原归档源码及 65 个完整 notice；附加非商业条款和 GPL/LGPL 保留 |

## 官方证据

- [FreeSurfer 8.2.0 许可](https://github.com/freesurfer/freesurfer/blob/v8.2.0/LICENSE.txt)：许可开头明确包括 software and/or data；Part B 授予 MGH 所持权利的再分发许可，同时要求保留条款、MGH 前言及第三方权利。许可不自动覆盖无法确认的外部第三方材料。
- [MGH 子分区 atlas 作者和引用](https://surfer.nmr.mgh.harvard.edu/fswiki/SubfieldAtlasesICBMspace)：核实 GEMS atlas 来源，保留脑干、丘脑、海马与杏仁核论文。只处理运行 atlas，不打包页面所列测试扫描。
- [MNI ICBM09c 官方模板与许可](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009)：允许使用、复制、修改和再分发，须保留版权及许可。
- [VPNL 官方 atlas 页面](https://vpnl.stanford.edu/vcAtlas/) 和 [FreeSurfer 标签来源注释](https://github.com/freesurfer/freesurfer/blob/v8.2.0/distribution/average/colortable_vpnl.txt)：明确第三方来源；文章开放获取本身不能证明外置标签二进制许可。
- [FSL 官方 datasets](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html) 将相应标准资源指向 [FSL 主许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)：无 financial return 的再分发获准，接收方须同样受条款约束；因此 Release 应标注非商业资源，不能标为 MIT/CC0。
- [TemplateFlow 数据 metadata](https://github.com/templateflow/tpl-MNI152NLin6Asym/blob/master/template_description.json) 引用 `See LICENSE file`，当前根目录却缺该文件；不能用 TemplateFlow 客户端的 MIT 许可给模板授权。 [sanitize.py](https://github.com/templateflow/tpl-MNI152NLin6Asym/blob/master/scripts/sanitize.py) 说明 orientation、dtype、affine 和单位标准化。
- [HCP 固定版本许可](https://github.com/Washington-University/HCPpipelines/blob/f8cac6892f88bdf889d644711ff038198eb81533/LICENSE.md) 与 [atlas 说明](https://github.com/Washington-University/HCPpipelines/blob/f8cac6892f88bdf889d644711ff038198eb81533/global/templates/standard_mesh_atlases/00_README.txt)：BSD 风格条件允许再分发，须保留全文和 FreeSurfer 数据归属。
- [CBIG 固定版本 MIT](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/LICENSE.md) 和 [Wu RegistrationFusion 数据发布说明](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/stable_projects/registration/Wu2017_RegistrationFusion/README.md)：作者明确发布由其生成的 warp 与 cortex mask，未发现该文件夹单独的许可例外。
- [CBIG fs_LR_32k 来源 Readme](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/data/templates/surface/fs_LR_32k/Readme.txt)：中层网格来自 Caret 网站；实际 GIFTI metadata 没有许可条款。需要原作者明确许可或许可已明确、科学上可验证的替代资源。
- [Oxford FBP 官方 ancillary data 下载页](https://www.fmrib.ox.ac.uk/ukbiobank/fbp/) 未给 DATA_public/template_GM 明确许可；[另一组 4000 人 group-average 资源页面](https://www.fmrib.ox.ac.uk/ukbiobank/index.html) 的 CC BY 4.0 不能未经来源核验套给这个文件。

TF HCP dseg 与固定 HCP `Atlas_ROIs.2.nii.gz` 经 canonical orientation 后，shape、物理 affine 和每个标签都相同，不同 voxel 为 0；gzip bytes 差异来自朝向、dtype 和 header 标准化。该实测结果支持 TF 资源是 HCP atlas 派生的来源推断，并未把数值相同称为新的许可授权。

## 发布边界

“尚未闭环”表示当前证据不足，不表示作者禁止再分发。相关资源应继续从原作者官方来源获取，或在获得确切授权/确认许可附件后补充 Release。不能通过复制软件代码许可或引用论文来解除这个条件。

FSL 的 HCP1065 tensor/FA 是 FSL datasets 说明的 DTI 模板，不能混同 DSI Studio 的 HCP1065 tractography atlas 并套用后者的 CC BY-SA。Oxford 大归档只用于确认需要的运行资源及其许可，不整体上传包含其他数据的归档。

本审查未读取个人 license、密码或凭据；未上传资源、改动计算源码或运行新的 MRI 流程。

## 许可全文与 native 原源码归档

- 已保存 MNI 官方完整许可为 `licenses/MNI-ICBM152-2009.txt`、`licenses/MNI-ICBM152-NLin6.txt`。现有 `licenses/FSL-6.0.txt` 全文与当前官方网页内容规范化后完全匹配；网页证据和文件 SHA 见 [retained_license_texts.public.json](retained_license_texts.public.json)。
- 固定 native 原源码归档大小 408890859 B，SHA-256 `2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a` 本轮重新核验通过；扫描 65 个完整 license/notice 已保存在 `licenses/freesurfer-native-source-notices/`。原归档同时保留源码和组件条款，见 [notice 清单](freesurfer_native_notices.public.json)。
- 此原归档具有多个组件许可，包括 FreeSurfer/MGH、CorTechs、FSL、TetGen、BSD/MIT、Apache、GPL/LGPL。CorTechs 有非商业内部研究/教育及非营利机构条件，FSL/TetGen 有非商业条件；因此不能把完整归档标为 MIT 或单一 FreeSurfer 许可。保留原包不代表 FNIT 会运行所有原软件，也不为新的混合链接二进制授权。
- 许可目录只保存公开法律文本和 notice，不包含个人 FreeSurfer 运行 license。
