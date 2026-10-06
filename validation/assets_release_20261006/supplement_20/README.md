# 20项运行资源补充发布（2026-10-06）

按用户明确确认“这20个文件都可以上传”，在原220项附件之后补充17项VPNL/Juelich标签、2项Caret派生中层网格和1项Oxford GM模板，合计3,418,915 B。新增上传确认和完整资源清单后，固定`assets-v1`共242项附件。原220项附件的ID、大小、SHA和状态保持不变；个人FreeSurfer运行license不收集、不发布，Conda/PyPI依赖按主页安装。

| 项目 | 验证范围 |
|---|---|
| VPNL 17项 | recon-all安装器根据原size/SHA自动使用Release |
| Caret 2项 | MS-HBM安装器使用Release，保留原网格和来源 |
| Oxford GM 1项 | Release单文件下载及校验，仍由既有FastVBM `template`输入 |
| 完整资源清单 | 新附件`runtime-assets-manifest-20261006-complete.json`；旧清单保留为历史证据 |

[发布与API核验](publication.public.json)记录真实附件和旧附件守卫；[完整匿名GET和实际安装](install.public.json)验证20项新文件、17项重建路径、MS-HBM三项源资源及GM手动下载。没有原作者来源回退，不用模拟字节代替正式下载。[上传确认](confirmation.public.json)与独立作者许可证据分开记录。

## 来源与许可

Oxford GM与原作者官方GitLab冻结commit `9458b42e23c3476cc9bd4b6a0ae60e1df3807e55`中的文件大小及SHA完全相同，项目根许可为Apache-2.0；[官方同字节与许可证据](oxford_license.public.json)记录完整GET和原条款。保留[Oxford版权声明](../../../licenses/Oxford-UK-biobank-pipeline-LICENSE.txt)及[Apache-2.0全文](../../../licenses/Apache-2.0.txt)，不将该条款扩展到原包其它文件。

VPNL和Caret公开来源未发现针对这些文件的明确禁止条款，但独立数据再分发条款仍未确认。本次按用户明确确认上传，保留原作者来源、归属与引用；不将用户确认写成原作者书面授权，不给第三方文件赋予MIT/CC等未经证实的许可。此前[许可审查](../license_audit.public.md)是补充发布之前的证据快照，不能视为现在仍未镜像。

## 安装与科学边界

本次仅补充下载目录元数据和资源说明。VPNL及Caret的下载函数、公开API和计算参数未改变；Oxford不进入FSL standard的11文件profile，不改变FastVBM的必填输入和默认值。继承此前MS-HBM空间单位修复；实际安装检查mask的shape、affine、uint8和空间单位。原MRI输入、参考与benchmark保持不变，这里报告的耗时仅为资源下载和安装验证。

资源测试110项通过；最终源码SHA和测试记录见[source_binding.public.json](source_binding.public.json)。文档检查见[documentation_check.public.json](documentation_check.public.json)。[核验脚本](verify_installers.py)只使用公开资源和匿名初始URL，不保存签名重定向、凭据或服务器地址。
