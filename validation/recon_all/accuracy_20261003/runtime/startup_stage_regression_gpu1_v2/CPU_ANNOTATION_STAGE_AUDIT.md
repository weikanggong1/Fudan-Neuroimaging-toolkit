# GPU1 startup annotation v2：独立 CPU 报告审阅

压缩包 SHA 校验通过：`eb2784227f79bb6914147f85702c1d0669d29ed1cfd4510c7e3adfbf8d4f02e4`。原报告解包后保持原样；新审阅仅写本目录 CPU 文件，没有 GPU 计算、算法重跑、源码修改或提交。

## 标签、文件与网格

独立读取24个语义NPZ（标准库ZIP/NPY头解析），核对每个文件SHA/size及labels/table数组回执SHA，并重新检查有序vertex索引、两臂所有数组原始字节及逐标签计数/Dice。两病例×六套配对均 exact_regression_pass=true；实际出现的全部标签Dice=1，min/p05/median均1，原始packedID差异和table-index差异均0。色表与names数组完全相同。packed RGB ID与色表行索引分别验证，不当作解剖类别编号。

| 病例 | 半球 | aparc 顶点/出现标签 | a2009s 顶点/出现标签 | DKT 顶点/出现标签 | 全部逐标签Dice | 标签差异 |
|---|---|---:|---:|---:|---:|---:|
| ds000114_sub-06 | lh | 130,346/35 | 130,346/75 | 130,346/32 | 1 | 0 |
| ds000114_sub-06 | rh | 132,837/35 | 132,837/75 | 132,837/32 | 1 | 0 |
| ds000114_sub-07 | lh | 114,342/35 | 114,342/75 | 114,342/32 | 1 | 0 |
| ds000114_sub-07 | rh | 114,824/35 | 114,824/75 | 114,824/32 | 1 | 0 |

每套原annot文件字节SHA及NPZ字节SHA在报告中均相同；四个smoothwm/sphere.reg的坐标、ordered faces及mesh文件字节均报告相同。**包中只有语义NPZ，没有原annot/mesh**：NPZ可独立核对，原annot字节与几何一致性只能审阅服务器已绑定回执，不能在本地重新读取mesh验证。所有逐标签ID、名称、顶点数、交集和Dice已完整保留在新JSON。

## 设备、精度与完成状态

四臂父/worker声明及实际worker UUID一致，实际为GPU1 `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，H100 PCIe；不是此前故障GPU0。总线程4、worker2、每worker2，实际Torch/Numba effective=2，Torch interop=1。父TF32 matmul/cudnn均true、autocast=false，worker报告TF32均true。sub06两臂初始化父CUDA标量；sub07父未预初始化，实际设备由两个worker证明。

四臂stage_complete、monitor/controller exit0、controller全部owned descendants退出。没有completion.json：stage/controller结果支持本阶段完成，不是原始T1整例完成。

## 耗时与显存（分开口径）

| 病例/臂 | group调用墙钟s | stage验证至summary s | monitor命令墙钟s | 自身父子同刻采样峰bytes | group采样峰bytes | 整卡采样峰bytes | monitor最大间隔s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ds000114_sub-06/baseline | 143.754 | 154.583 | 155.261 | 4,414,504,960 | 4,414,504,960 | 6,042,943,488 | 0.809 |
| ds000114_sub-06/candidate | 148.030 | 155.818 | 156.512 | 4,469,030,912 | 4,469,030,912 | 6,097,469,440 | 1.312 |
| ds000114_sub-07/baseline | 140.853 | 149.063 | 149.794 | 3,978,297,344 | 3,978,297,344 | 5,604,638,720 | 0.695 |
| ds000114_sub-07/candidate | 125.749 | 137.005 | 137.693 | 3,948,937,216 | 3,982,491,648 | 5,556,404,224 | 0.766 |

monitor采样请求0.5秒，四臂app/GPU查询失败均0。自身峰为同次query中父与全部子进程之和；整卡为另一次GPU query，不代表自身/整卡峰发生同一时刻。group与整stage是两个独立采样器，sub07 candidate group峰3,982,491,648高于monitor峰3,948,937,216，属于不同采样时刻/间隔，不能把monitor峰宣称为连续上限。全部continuous_peak_verified=false，缓存禁用下tensor allocator峰不可用。

group包含私有拷贝、启动、计算、发布与清理；candidate另报startup wait实际sub06=3.141秒、sub07=2.827秒，不能算成纯annotation计算。stage额外包含输入SHA、拷贝、导入及语义输出验证；monitor另含进程启动退出；controller outer再含准入、库存校验及清理。所有层的原始时间保留在JSON，不相互替代。

单次pair每例，且GPU0原始T1队列同时运行、组合声明CPU8：sub06候选group更慢、sub07更快只是本轮观测，不支持稳定加速结论。两个GPU1成功pair也不证明历史GPU0间歇OOM已修复。

## 缺口

- Archive has no completion.json; controller complete + stage_complete + monitor exit0 provide stage completion, not whole-case completion.
- Original .annot and smoothwm/sphere.reg files are not included: bytes/geometry claims audited from bound reports, cannot independently reread original mesh here.
- No tensor allocator continuous peak (cache disabled); sampled NVML peaks are lower bounds and differ across stage/group sampler cadences.
- Single annotation pair per case on GPU1; timing changes cannot establish speedup or historical GPU0 OOM mitigation.
- No official accuracy or original-T1 whole-case chain in this stage scope.

详细机器证据：`cpu_annotation_stage_audit.json`。其中24个NPZ均独立SHA校验，所有逐标签Dice已重新检查。
