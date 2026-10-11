# 全半球自有GA：控制合同与实际运行版本

## 1. 当前范围

独立实验从公开T1的自产nofix/qsphere/brain MRI开始，运行全部9缺陷、完整默认GA、五轮SOAP、源序压缩、CURRENT/TMP有序CSR清理，再恢复ORIGINAL写premesh。后续标准remesh、ORIGINAL清理与最终网格门仍为必要步骤。本页记录固定运行版本，执行完成和严格精度分别报告。

## 2. 实际源码绑定

运行的Git基点为`fc36d6c1c761302ed5b7425b730b17ab4951090b`，68项冻结文件逐个校验后启动。完整[模块清单](reports/runtime_source.json)、[初始启动回执](reports/launch_receipt.json)保留manifest SHA `9ddb21a2ff8209a23820a2be0067d5ed058ddb809b391f1d768fb41f9a722bee`、上传包SHA `2c0b1b6c6248512f9b5c7bb59043125373300072fd437426845f8b2cbb638fb4`和actual runtime SHA。

控制器SHA `ae4a9b3ab256ac6a58a0cc3c64674425adc68c8ec0f949964790179688132325`，完整工具SHA `477fe5f699e19b59846009f8491c9192d531cdd6415dd5de48b04f8a35cf19c3`。既有retessellation继承已验证冻结版，不覆盖他人的dirty文件；正在配对优化的邻接/状态模块没有混入这个基线。

## 3. 控制合同

9项测试和14个子测试全部通过，pytest输出10.63秒，JUnit完整秒数/测试名/原文件SHA见[local_contracts.json](reports/local_contracts.json)。它们验证共享准备一次、动态邻接域、跨缺陷状态/RNG、最终坐标槽、CURRENT两种独立残余门、非默认patch提前拒绝、nibabel实际头与网格往返及失败回执。

这些小网格/fixture是控制与I/O合同，不能替代真实MRI精度、完整运行时间或最终质量。

## 4. 实际运行状态

最初启动已完成导入预检，四线程/四核亲和性、CUDA隐藏、源默认10/10/−1、initial/剔除True、overlapFalse、seed1234、SOAP5。新目录从自身输入重新准备，参考和原程序都未传入。

北京时间2026-10-11，第一个缺陷已完成29轮（轨迹索引0–28；继承字段`search_generations=28`保留）/495次评分；最佳与最终评分均−93.04168701171875，删除计数2、阈值0.03125。完整末态检查点SHA与独立首缺陷运行完全相同，代数、全部计数、评分及连续删除状态七组检查均通过，见[本次首缺陷连续性](reports/first_defect_continuity.json)。这是与先前自产结果对照，不是原版完整赢家一致性。

首缺陷API2867.967724秒，搜索2853.689547秒、最终写回10.335202秒；本次共享主机墙钟并非ABBA，不能与另一时段的单缺陷秒数直接解释为优化回归。后续缺陷继续运行；尚无全半球成功退出、完整`report`/`COMPLETE`与premesh门回执。

成功后将自动运行[标准remesh、最终ORIGINAL清理与orig质量工具](../20261010_topology_genetic_orig/README.md)，已完成实际依赖预检，但尚无该真实阶段结果。启动/PID/日志不视为通过；GPU显存不记0，完整阶段时间和RSS待实际退出。最终orig、原始T1整例、138项、整体指标等效和600秒目标尚未验收。

## 5. 先决真实证据

[首缺陷完整默认搜索](../20261010_topology_genetic_search/README.md)已实际完成29轮（轨迹索引0–28；继承字段`search_generations=28`保留）/495次搜索评分，9组首候选与固定原源码逐位一致，最佳/最终评分相同。这是启动全半球回归的阶段依据，不能代替本页全半球结果。

## 6. 复现

完整参数、输入/输出、surface RAS/mm、具名中文调用与命令见[功能说明](../../../../docs/recon_all/TOPOLOGY_GENETIC_REPAIR.md)。运行输入来自同一连续上游；标准完整原始T1验收最终仍须从原始影像与空目录起步。诊断检查点仅保留自身阶段排错范围。

不新增依赖，沿现有主页NumPy/Numba/SciPy/nibabel/PyTorch安装路径；干净无预装脑影像软件环境的实际隔离整例未验证。

## 7. 文件与来源

本目录保存控制测试、固定源码清单、明确未完成的初始启动回执及已完成首缺陷的真实连续性报告。运行原影像、网格NPZ、许可证和服务器路径不提交；`PUBLICATION_MANIFEST.json`绑定公开字节。

- [源拓扑算法](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。
- [Fischl等，2001](https://doi.org/10.1109/42.906426)。
