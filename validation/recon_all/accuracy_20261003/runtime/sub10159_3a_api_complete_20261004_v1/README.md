# 10159 候选3a 已初始化CUDA API整例完成

实际完成 2026-10-04T16:05:49.456124+00:00，source3a/03cc，原输入0cb8，138/138齐全、双侧mesh passed。真实 run-api receipt证明入场CUDA已初始化，保留4B标量。8个worker首次startup均complete且operation_entered=true；未观察新OOM。

已初始化CUDA API入口命令 2769.71966387704s（不是CLI，也不包括monitor额外包装）；内部API总墙钟 2766.121300884988s（completion.pipeline_total_seconds是其别名）；真实pipeline 2760.63337273011s。60个顺序父stage总和 2751.661368488334s；nested worker/internal指标不重复相加。

自身父子同期采样峰 10947133440B，1321个CSV样本；请求2s、最大间隔 3.5076937531121075s、应用查询失败0。不是连续峰，也不是整卡峰。

实际precision与caller策略完整保留：预初始化allocator effective=preserved_preinitialized_unknown；env=1不等于可证明统计有效。官方评估checkpoint verify_binding failed：ValueError(precision candidate actual allocator differs)。这是评估绑定校验失败，不能改称算法失败或数值差异；整体equivalence not_assessed。

输入不重读MRI，以原config/launch/admission库存SHA一致核对；原报告不修改，无GPU或CPU4重跑，原运行中快照保持。
