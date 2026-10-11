# NVML诊断当前帧故障修复

只修复独立诊断工具：forced采样前后对比sample/error长度，查询失败或无新增sample时输出null/unavailable。直接NVML查询与采样器查询分别记录跨度和进程行，不混作同一次快照。

CPU故障合同：24项通过，pytest阶段约0.12秒，含生产采样器成功→失败→恢复、无新增样本、真实pynvml异常类、proc权限/退出/身份变化、非自有PID和非3.10内核。见contracts.public.json。CUDA未初始化，TF32前后相同。111/333等注入值仅用于契约断言，不是显存测量或MRI benchmark。

补测曾因PYTHONPATH漏掉既有NVML依赖目录而失败，见contracts_v2_dependency_failure.public.json；修正独立测试任务导入路径后通过，没有新增依赖或修改共享环境。

prior_reports_unchanged.public.json证明12个历史报告、分析和CSV的Git blob与7381dfa63基线相同。JSON字节完全相同；原CSV工作/交付文件原有CRLF换行，Git内容为LF，另核对原交付字节相同。保留历史unknown/failed和实际执行源码SHA；新源码没有重新运行GPU/MRI。未修改生产监测、MNI、缓存和统计实现，未验证完整重建20GB上限。

输入、输出、参数、Python/命令行示例、版本和参考见docs/recon_all/NVML_NAMESPACE_DIAGNOSTIC.md。
