# 新 main 精度候选 annotation 阶段验证

候选 `3a0c9aba6321b4981fd8174b4b191515459aa38b` 对已验证 8f：4 arms 完成、exit=0；两例共12套 annotation exact regression passed。全部标签差异0、所有存在标签Dice=1；有序面/坐标/color table/names相同，annot及NPZ文件字节也相同。

| arm | group墙钟秒 | 外层monitor命令秒 | 同刻父子采样峰 bytes | 组内独立峰 bytes |
|---|---:|---:|---:|---:|
| ds000114_sub-06_baseline | 136.727012 | 146.759675 | 4475322368 | 4475322368 |
| ds000114_sub-06_candidate | 139.556746 | 147.189874 | 4475322368 | 4475322368 |
| ds000114_sub-07_candidate | 118.888445 | 128.745292 | 3984588800 | 3986685952 |
| ds000114_sub-07_baseline | 125.419761 | 132.836339 | 3984588800 | 3984588800 |

显存为独立有限采样，各自同刻父子合计；两个峰不相加，不称连续峰。实际UUID为GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba；父worker实际TF32开启，autocast未启用，线程总4。sub06先建标量context且cache关闭，sub07未预初始化；不覆盖任意长期已初始化CUDA/caching状态。

各半球startup attempt均1，operation_entered均True；候选保留实际parent_idle_cuda_cache报告，详细参数、精度、源文件及工具SHA在measured_summary.json和原stage报告。

阶段墙钟包含指定scope，锁等待单列；另有共享CPU4预算，本阶段4预算，共享环境观测不能声明完整配对提速。原始T1整例/整体官方等效未在本阶段评估。
