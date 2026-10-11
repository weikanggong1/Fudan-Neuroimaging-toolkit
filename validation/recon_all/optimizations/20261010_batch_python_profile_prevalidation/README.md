# batch Python 配方预检：当前调用链与入口修复

## 1. 功能简介

当前源码审查及 batch 成熟入口 bug 修复。`backend="python"` 先解析同一21键配方再校验，消除隔离 WM、Torch inflation 被混合缺省值错误拒绝的问题；native 检查不变。完整说明和流程图见[功能页](../../../../docs/recon_all/BATCH_PYTHON_PROFILE_PREVALIDATION.md)。

## 2. Python 调用和输入输出

输入是当前仓库源码、明确旧 Git baseline 和合法参数组合。`collect_prevalidation.py` 只生成三份 JSON：源码/本地包版本、实际阶段路由和修复前后预检；不读 MRI/权重/许可证。阶段审查以当前函数与实际消费者为依据，公开 JSON 仅保留仓库相对路径。

## 3. 命令行

```bash
# --repository指定仓库；--output是新的报告目录。
python collect_prevalidation.py --repository /data/FNIT --output /data/audit-new
# 单元与CLI fixture；耗时执行体已替换，不执行MRI计算。
python -m pytest -q tests/recon_all/test_batch_python_profile.py \
  tests/recon_all/test_batch_python.py tests/recon_all/test_python_profile.py \
  tests/recon_all/test_python_profile_n4_selection.py
```

## 4. 原软件调用

本次没有执行原软件。默认 CUDA 混合配方仍会运行声明的固定源码 N4、WM 分割/编辑、GA、inflation、placement、atlas paint、曲率统计和缺陷投射程序；CUDA 默认 GCA/顶点指标已有自有实现。Python 配方不消费这些程序，使用完整 SimpleITK/Torch N4、自有 WM/Numba 编辑和数字拓扑；数字拓扑不等于原 GA。CPU MNI 参考分支另有三程序。逐项条件和行号见 `reports/current_inventory.json`。

## 5. 本版验证与性能范围

- focused 21项测试、103项 subtests：全部通过，12.14秒。
- 两组旧入口配置真实被预检拒绝；修复后空任务预检接受，完整真实 CLI 参数解析/公开 wrapper fixture 验证配方。
- 显式 Torch/isolated N4 保留；非法 N4 组合仍拒绝；native 原检查保持；fixture 没有初始化 CUDA。
- 未做真实 MRI、batch 整例、原软件计算、OS 隔离或新性能比较。影像误差、显存、整体指标等效和整例提速均未测；不能把 fixture 的 complete 当成真实完成。

`reports/test_receipt.json` 绑定实际测试模块和当前 source SHA；`source.json` 的包版本是本地合同解释器，不冒称计算服务器当前版本。旧真实 N4/WM/整例报告保持原版本，调用链表仅指出其验收缺口。

## 6. 版本和记录

baseline 为 `fc36d6c1c761302ed5b7425b730b17ab4951090b`，修复后为已记录源码 SHA 的 overlay。报告数字、布尔值和 null 保持原类型；PUBLICATION_MANIFEST 绑定公开 payload 的实际字节，包含源实现及测试文件。当前 API/CLI 默认仍 native；Python 配方为独立候选，整体等效 `not_assessed`。

## 7. 参考文献与源码

- [FNIT 函数页](../../../../docs/recon_all/BATCH_PYTHON_PROFILE_PREVALIDATION.md)
- [当前源码](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)
- [固定 FreeSurfer d932 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)
- Dale AM et al. NeuroImage. 1999;9:179–194。Tustison NJ et al. IEEE TMI. 2010;29:1310–1320。
