# 原始T1精度工作本地集成准备（未提交）

## 版本与目的

本工作分支 `recon-accuracy/20261004-main-integration` 从精确candidate `1be058a24b9ce702c644c13dddd7bd0f56d7c57f` 创建，HEAD仍停留在该commit。仅整合root `8eba2ace9b205d2611854ca71d80a85baf74659d` 已受跟踪的验证工具、结果及其文档依赖，便于根任务审阅。没有提交、推送、远程上传、编译或新影像计算。

实测production是 `3a0c9aba6321b4981fd8174b4b191515459aa38b`；当前src和candidate、3a的Git树完全一致，树对象 `52c48cf6e955810c2583f16a3cce0fa45c1771b9`。3a真实结果保持其原版本标识，不称为本集成分支实测。root较旧生产树 `6a17597d30c80b5738b5bef2ed62763e10cf9e3e` 未迁入。native、安装、environment及项目主页不变。

## 可审查迁移清单

`migration_manifest.json`逐路径记录root新增1156文件的SHA：1150个accuracy工具/说明/报告及其引用证据，加5篇主说明和一张脑图。Git archive只取固定commit受跟踪字节，不含root两个untracked文件。运行证据里的bound_source/tool_source仅作为原执行来源，原SHA/原失败保持；不是在新工作分支执行的代码。

全部59工具的实际导入/配置驱动依赖已存在、AST/SHA通过；11个candidate已有依赖原字节保留。242条原root链接加合并导航后实际252条本地相对链接全部可解析。各条验证和59路径列于manifest。没有用全树覆盖新版main。

13份逐文件冲突审阅记录列于manifest，其中4篇主文档保留candidate字节不变，9份实际修改：CA_NORMALIZATION、CUDA_STARTUP_RESOURCE_WAIT、NATIVE_PIAL_PLACEMENT、README、SURFACE_STATS_CACHE、SYNTHSTRIP_MGH_DTYPE及task_02/03 README和task_05 compare_placement。

|冲突|合并决定|
|---|---|
|CONDA_CPP_BUILD|保留candidate16项安装产物、mris_expand及旧记录限定，root15项旧定义不覆盖|
|PROFILING|保留半球启动前父缓存释放链接；root移除链接不应用|
|SPHERE_REGISTRATION_PERFORMANCE|保留candidate CPU有序平均优化、当前清理语义及质量诊断，不恢复root旧源码一致/整例未运行描述|
|THREAD_BUDGET|保留candidate线程设置或阶段失败仍写出完整failed/恢复墙钟的说明|
|主README|新增3a实际2例完成与精度报告入口，同时保留candidate完整CPU官方比较、父缓存、分割统计等新版内容|
|CUDA_STARTUP_RESOURCE_WAIT|完善具名参数与已完成两例annotation；区分annotation未触发重试和3a整例finish真实OOM恢复，不将局部范围推广|
|task_02/03|合并实际完成历史报告，保留其旧816输入/候选准备时点限定，不重标为3a新整例|
|task_05 compare_placement|逐diff核对，仅增加只读脚本SHA、局部CSV与簇统计；不改变生产算法或等效门槛|

全部candidate独有2444文件经Git blob批量核验字节不变，包括HEMISPHERE_GPU_MEMORY.md、SEGMENTATION_STATS.md、TEN_CASE_CANDIDATE_PREPARATION_20261004.md、VOLMASK.md及candidate1be独有真实annotation证据。完整保留路径列于manifest。

## 指定CPU回归与失败范围

在本integration的路径内运行已有meaningful测试：CPU队列12项与Popen内部取消竞争1项，13/13通过，unittest时间2.149秒。测试是标准库mock接口/控制流，不是影像benchmark，不触发GPU或算法；日志及其SHA与命令在manifest。未跑广泛测试。

本次迁移和CPU测试无失败。原metrics v1接口TypeError在算法前发生，原失败数据完整保留；其修正后的v2独立成功并不覆盖v1。未将prepared/admitted、文件复制成功或CPU测试通过记为原始T1算法完成。

后续由root独立review后决定提交；提交前仍应审阅主文档表述与文件清单。不因本次集成宣称整体数值等效或新版本耗时，十例状态以各原实际config/launch/checkpoint/数字报告为准。

源迁移脚本prepare_migration_readonly_sources.py记录只读固定Git来源和校验流程；它会创建新metadata并写集成文件，仅供审阅，不应在既有集成目录直接重新执行覆盖当前review。
