# 最终实现与发布范围审阅

此次审阅不改生产源码、不启动 GPU、不迁移或删除服务器证据。

## 生产实现

六处 GEMS 改动仍与全流程六次验证使用的冻结文件 SHA 完全相同。没有新增依赖或 FSL、FreeSurfer、SPM、MRtrix3、AFNI、Nipype、DIPY 的运行时调用。`GEMSAtlas.from_freesurfer` 仅解析 atlas 文件。PyTorch、NumPy、SciPy、nibabel 与既有 Triton 后端均已在环境声明中。

完整回归为 **439 passed in 54.27 s**；后续 origin/main 快进及公共 API/CLI 验证记录另列，不能改写此前冻结身份。

## 内部 helper 的适用域

`ordered_row_gather` 对不能以共享存储展平的非连续二维 `row_ids`，反向会在布局身份检查处抛出 `ValueError`。CPU 复现索引为 `torch.tensor([[0, 1, 2], [3, 4, 0]]).T`，形状 `(3, 2)`、步幅 `(1, 3)`。

当前生产 `_compact_lookup` 通过 `torch.cat` 得到连续的一维 `selected_ids`，两个 gather 共享该索引，实际脑干/丘脑/海马路径不受此限制。该审阅没有扩大 helper 的输入域或修改已验证源码。通用非连续索引支持应另做归一化和数值梯度验证。

## 发布文件与本机归档

Git 待发布候选扩展为 JSON、Python、日志、PNG、TSV、Markdown、patch 与 diff；没有影像体积、权重、atlas、许可证内容、符号链接或强凭据模式。PNG 为用户要求的衍生脑图。

本机 `source_final` 的 424 份重复运行时源码与原始 tar 虽被 Git 忽略，当前 `MANIFEST.in` 的递归规则仍会收集其中 **426 个 Python/JSON 文件**。根代理已选择将它们迁到仓库外 `../fnit-subregions-20261002-local-archive`；不要在 `validation` 内另建 ignored archive。

最小发布清单保留：六处生产源码及五份必要测试、最新说明；最终 24 组/440 行 compact 前后指标、分步骤时间及来源审计；六次运行的 context/GPU/进程日志；八幅最终脑图及 metadata；冻结/传输身份、官方源码与计时审计、关键定位摘要和重现脚本。详见 [机器清单](publication_scope_review.json)。

已撤回 mixed-scope 的展开 JSON/manifest、临时 first-run 展开结果、旧 probe 全量 payload 及重复源码可迁到本机仓库外归档。必须先核验最终 compact 的服务器 path、大小、SHA 引用；保留 mixed_scope_diagnostic 的撤回原因。服务器原始数据及展开证据保留。

迁移前将本目录旧 README 的临时阶段说明标为历史记录，链接改向保留的前后 compact；最终协议的旧 timer 链接改为 V2。
