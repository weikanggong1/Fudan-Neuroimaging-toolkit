# 真实 CUDA 启动恢复证据补充

real_recovery_exercised=true；algorithm_failures_retried=false。finish_surface 双侧首次 fresh-exec 的 CUDA first_allocation 边界失败，operation_entered=false；第二次 fresh-exec READY/GO 后成功，双侧算法各进入一次。四组共8次算法进入，保留10份startup响应及相应全部请求/日志。

失败响应仅有monotonic时钟：UTC按原响应文件mtime与finished_monotonic锚定，为近似对齐；mtime是响应写入时刻，不是驱动调用精确时刻。当前wall/monotonic锚交叉检查差值见JSON，不能排除历史校时。外层CSV的time_utc是原实际记录。

|半球/事件|近似UTC|CSV UTC|own父子同期B|整卡used B|整卡free推导B|
|---|---|---|---:|---:|---:|

free = 启动准入同物理GPU记录total 85520809984 B − CSV整卡used。不是13,570,670,592 B自身峰，也不是两采样器相加。CSV没有total/free字段，推导使用同UUID固定总容量声明。

外层采样请求2s、最大间隔4.622375769s；±3s采样不能排除更短瞬态容量变化，不据此宣称驱动内部原因已彻底查明。更密的group采样另存，仅用于自身进程树，不含实际整卡容量查询。

v1包已保留不覆盖；本补充形成新v2包。数值比较与整体官方等效仍未评估。
