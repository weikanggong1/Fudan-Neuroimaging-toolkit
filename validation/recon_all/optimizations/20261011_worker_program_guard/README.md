# 自有worker程序边界修复交付

起点 `9dce8be979863ab98a9389066e882537b5a22763`；实际测试代码 `2f16c17e343104ac03ffb45de15874fc42ad6193`，受测源码无未提交改动。未跟踪的旧回执未进入测试。源码范围仅三个worker、新增守卫适配器、专属测试/探针与中文说明；主任务拥有的配方、其边界测试、安装验证器、根文档和共享索引均未修改。

旧EM审查结论已按起点源码纠正：实际生产两条GCA入口均进入稳定/全样本完整align_em，首次方向不足的历史结论不适用于此版。两轮接线及既有完整优化器合同已执行，不表示本次整例实测。

三个worker按继承禁令在CLI及阶段入口安装现有original_stage_guard。真实exec传播、禁止尝试、参考放行、早期/阶段内部故障、嵌套及正常/异常退出合同均通过；41项检查通过，无跳过。三个既有worker的数值函数体经过AST对比，剔除新增守卫导入/装饰器与说明后，与起点一致。

未运行MRI、权重加载、CUDA、整例、原版程序或性能benchmark；真实数据精度/耗时和整例访问证明仍由协调者验收。守卫范围是直接Python exec事件，可选隔离Torch N4不属于这次三worker修复。不能把这份合同视作动态库或外部shell子孙沙箱证明。

复现命令、运行库版本、测试分组、失败记录及源码SHA均在 `audit_and_contracts.public.json`；冻结JUnit在 `contracts_frozen.xml`，源码清单在 `source_manifest.public.json`。中文参数/调用说明见 [WORKER_PROGRAM_GUARD](../../../../../docs/recon_all/WORKER_PROGRAM_GUARD.md)。
