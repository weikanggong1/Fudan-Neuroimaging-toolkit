# task4 真实交点过滤候选交付

- 冻结父提交：`aeabc5aba9341f0358772eb00b1f07eb2cbaf643`。
- 13项新旧功能测试通过；三个真实冻结输入、16次完整检测的顶点bool数组和相交面数一致。
- 自有F64交点判据、完整清理函数、默认后端、质量门保持。无新生产依赖，无原软件运行，无影像/坐标输出进入源码。
- 候选Numba/GPU显式可用；真实高面对数量pial单轮观察有收益，其他网格未显示一致完整检测收益，因此不切换默认。
- `metrics.public.json`与`measurements.csv`保留全部完成测量、输入/源码哈希、计时边界和资源范围；`LOCAL_CONTRACTS.json`保存原判据及完整清理AST未变的核对结果。
- 旧错误记录保留：缺少pytest发生于测试启动前；缺少time发生于13项测试通过之后、检测启动之前，均不计成功benchmark。
- `TOPOLOGY_BOUNDARY_AUDIT.json`为已有冻结GA证据的只读审计，不是本轮重复GA。原版全9缺陷形状差异未解决，整例和整体等效未评估。

协调者可只合入本任务源码、测试和功能页。F64检测保持默认python；高面对数量检查显式传`separation_backend="numba"`，GPU显式传`separation_backend="torch", device="cuda:N"`。不改共享native_free/batch/安装入口，不启用受保护的GA原型，不推main。全部参数和真实复现见[功能说明](../../../../docs/recon_all/TRIANGLE_PLANE_FILTER.md)。

后续拓扑排错应绑定同一nofix、qsphere、MRI、源码、seed和RNG流，逐代比较选择、交叉/突变边序、获胜者恢复、下一缺陷输入、压缩映射及最终original恢复；不要再将局部几何差异解释为整体等效或只检查平均误差。
