# 完整自产 GA 网格比较：冻结与等待回执

## 1．功能与流程

自有全九缺陷 GA → 标准 remesh/最终清理 → 增强完成检查 → 独立只读原生对照。原生结果不进入候选算法。

## 2．Python、输入与输出

完整具名参数、结构、坐标和失败行为见[功能页](../../../../docs/recon_all/GENETIC_MESH_COMPARISON.md)。三项实际算法输入的 SHA 与历史参考相同；原生程序、报告及两张参考输出的身份记录在 [reference_preflight.json](reference_preflight.json)。当前只保存检查和等待回执，尚无真实全网格距离结果。

## 3．命令行复现

[比较工具](../../../../tools/compare_recon_genetic_mesh.py)只接收完整自产回执；按功能页固定四线程和亲和性运行。CPU 预算同时约束 BLAS、OpenMP 与 Numba；不初始化 CUDA。生产阶段与额外诊断墙钟分开。

## 4．原软件对应

只读比较固定来源的 GA `orig.premesh` 和标准 `orig`，本工具不执行原生程序。程序 SHA 和来源提交保持历史绑定，不把今日代码标作原参考版本。

## 5．实测范围

| 检查 | 当前结果 |
|---|---|
| 本地格式、完成及身份合同 | 6 项通过，pytest 7.73 秒；不是 MRI benchmark |
| accelerator-b 冻结文件校验 | 7 项 SHA/大小相同；环境和正在运行源码保持原实体路径 |
| 远端导入 | 退出 0，7.762893843 秒 |
| 真正比较 | 等待增强 orig 成功；未完成 |
| 双向距离、严格网格复现、脑图 | 未测，不能从输入相同推断通过 |
| 整例耗时、整体指标等效、整例显存和干净安装 | 未评估 |

`gpu_memory_bytes=null` 表示没有本项测量，不表示整例零显存。逐索引误差只在全部有序面和顶点数对应时报告；否则完整双向点到三角面距离覆盖全部顶点。没有添加新的等效门槛。

## 6．版本与记录

- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：冻结文件 SHA、作者基线提交及继承包身份；额外驱动脚本私有保存，只公开其 SHA。
- [contracts.json](contracts.json)、[JUnit](contracts.junit.xml)：当前工具的合同结果与脚本/测试 SHA。
- [queue_receipt.json](queue_receipt.json)：实际导入和等待状态；它不是比较完成回执。
- [PUBLIC_MANIFEST.json](PUBLIC_MANIFEST.json)：本轮公开代码、说明及回执大小和 SHA；不包含原始影像、网格、许可证、服务器入口或凭据。

## 7．源码与参考

[功能页的源码及参考文献](../../../../docs/recon_all/GENETIC_MESH_COMPARISON.md#7参考文献与源码)。沿用仓库已有完整距离实现，无新增依赖。此前第一缺陷的阶段回归、完整九缺陷控制器与原始 T1 整例分别记录。
