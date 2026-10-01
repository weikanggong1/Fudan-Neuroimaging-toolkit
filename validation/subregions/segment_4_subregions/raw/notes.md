# segment_4_subregions raw 完整运行记录

输入、官方参考及 C6 对照路径均由原始 report/status 和 CPU comparison 记录。计算 425.862 s，API 含保存 426.354 s，监控进程总墙钟 452.074 s；后者还包括启动、读入和对照记录，不能代替 API 计算时间。

几何/类型/标签表/原始输入/来源/完成状态共 29 项通过；全部110 soft volume 有限且非负，4个最终最小 Jacobian 为正。实际8项输出路径逐项记录大小和 SHA，影像与权重保留服务器。

相对 frozen C6，六家族 Dice≥.97 检查为 False，六家族 hard/soft 体积变化均≤5% 检查为 False。这些与基础几何检查分开报告。

丘脑 45 个可评估细核的官方 Dice：均值 0.612455→0.634180，参考体素加权 0.764980→0.776716。全部 105 个可评估标签：加权 0.835749→0.833728。完整细核变化保存在 nucleus_differences.json。

PyTorch tensor 峰值 15.472 GiB，本进程约5秒采样峰值 18906 MiB，上限 19073 MiB。共享GPU的前后负载完整保留 gpu_load.jsonl；速度比仅为实测观察。

冻结清单457文件（runtime407）独立检查全部大小/SHA。C6 runtime389，二者共同且SHA相同的runtime文件 339 个，改变 48 个；差异路径见 source_facts.json。报告内只列30个选定新runtime文件，另有冻结清单审计覆盖全部文件。新旧math等价范围以实际文件和输出指标为依据。

对照图各用6个轴位切片：vs_official 为固定官方输出，vs_c6 为旧C6。当前独立运行未要求逐体素相同。历史C6及官方原始报告未改写。
