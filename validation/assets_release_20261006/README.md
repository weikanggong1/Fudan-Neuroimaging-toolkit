# 运行资源 Release 验收（2026-10-06）

**当前完整结果见[20项补充发布](supplement_20/README.md)：已补齐此前20项文件，Release共242附件。以下为首次165项发布和当时许可审查的历史记录。**

本轮核对当前 `origin/main` 的资源入口、官方数据和再分发条款，将许可明确的运行资源补充到固定 `assets-v1`。Conda/PyPI 依赖仍按主页安装；个人 FreeSurfer 运行许可证不收集、不发布。原 MRI 输入、运行环境和 benchmark 成品未修改。

| 项目 | 结果 |
|---|---:|
| 新增获许可运行资源 | 165 个文件，980,667,228 B |
| 新增许可汇编、扩展清单 | 2 个附件 |
| Release 附件总数 | 220 |
| 原有附件 ID、大小、SHA 保持不变 | 53/53 |
| 新资源完整匿名 GET / 大小 / SHA 校验 | 165/165 |
| 许可汇编和扩展清单匿名 GET | 2/2 |
| 仍从原作者获取 | VPNL 17、Caret 2、Oxford GM 1 |

[正式发布报告](publication.public.json)记录实际附件和API核验；[完整匿名下载证据](public_get_verification.public.json)保存逐文件结果，不含凭据或签名重定向URL。[许可审查](license_audit.public.md)说明确切条款与未闭环资源；[原始法律文本核验](retained_license_texts.public.json)和[native组件notice清单](freesurfer_native_notices.public.json)保留依据。

## 实际安装验证

使用匿名下载、空目标目录、隐藏 CUDA 和单线程 BLAS。对目录中已发布资源禁止原作者URL回退；一次失败会明确记录，不能被上游成功替代。

| 安装入口 | 实际验证 |
|---|---|
| 标准模板 | 11项安装及离线复查通过 |
| recon-all资源 | 94条路径通过；91条路径由Release下载，3个LUT由包内提供 |
| GEMS | 真实atlas生成四个资源pack；缓存内准备，无新增下载 |
| fMRI完整选项 | 48条资源路径由Release获取 |
| Space | HCP34、CBIG RF4，由Release获取 |
| Connectome | 10项由Release获取 |
| MS-HBM | cortex mask由Release获取；2项待授权Caret网格使用已校验的官方文件预置 |

完整结果见[标准/recon/GEMS报告](recon_standard_install.public.json)和[其余安装器报告](other_install.public.json)。[安装包验收](packaging.public.json)确认最终wheel中的源码与本次修改一致，安装后的新CLI及11项真实模板离线校验均通过。这些记录中的耗时是下载与资源准备时间，受网络和本地I/O影响，不能作为MRI处理时间或软件加速倍数。

安装不改变模板字节、affine、dtype或网格。真实文件header见[标准模板清单](standard_headers.public.json)。TemplateFlow HCP dseg与原HCP atlas的canonical标签及affine一致性见[来源核验](templateflow_hcp_lineage.public.json)，这不表示三方MRI pipeline结果数值等价。

## 成熟模块修复

真实MS-HBM资源准备发现掩膜writer丢失reference空间单位。修复仅继承xyz单位；公开API及掩膜数值、affine、shape、dtype保持原行为。缓存回放证据见[mshbm_units_fix.public.json](mshbm_units_fix.public.json)，原97次下载和修复前unknown单位观测保留在安装报告中。

## 核验方法

`verify_public_downloads.py`对新资源逐项完整读取并核对大小/SHA。其余安装验证脚本及source SHA位于本目录；安装器单元测试使用小字节fixture验证下载错误处理，正式下载验证使用实际作者资源。此轮没有新增MRI实验，也未重新标记已有MRI benchmark的代码版本。
