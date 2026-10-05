# 不改变原CPU归约尺寸的下一步（仅静态分析）

## 已确认与未知

当前合同实际使用同一`convolution_slabs`，仅改变cap。depth1/2时两个路径的实际矩阵尺寸相同并逐位一致；depth7时旧路径一次计算，新路径分成2+2+2+1，首次出现微小FP32差异。权重、输入、输出空间邻域与dtype不变。本轮没有ATen/BLAS调用trace，不能断言具体MKL微核、packing或线程分区已经定位，也未与官方中间层比较。

PyTorch2.5.1的[Slow3d源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L245)把列矩阵交给GEMM：`M=D_out×H_out×W_out`，`N=C_out`，`K=C_in×kD×kH×kW`，`lda=ldc=M`；bias先复制到输出，再用`beta=1`。本合同K=1944、N=24，旧depth7的M=1547，新路径前三次M=442、末次M=221。真实层原计划M=802816→114688，但未执行。

[CPUBlas源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp#L552)在batch/groups=1时进入单GEMM，FP32构建可走SGEMM。**推断：** M与leading dimension变化可改变BLAS的blocking/向量累加器和bias合入时的舍入位置；数学上相同K项不保证浮点归约逐位相同。目标的MKL2022.1配置和实测小差异支持这一风险判断，尚未证明内部路径。不能称这次改动比旧CPU更接近或更偏离官方。

## 优先候选：原几何下复用工作区

| 必须固定的量 | 目的 |
| --- | --- |
| 原slab起止/halo、14次调用与末次depth10 | 保持每次原空间矩阵尺寸 |
| NCDHW、K排列、weight/bias FP32位值 | 保持列与权重条目顺序 |
| M/N/K、lda/ldb/ldc、NN转置参数、BLAS入口 | 降低不同GEMM调度造成的舍入风险 |
| bias预填与beta=1 | 保持原bias加法路径 |
| 同8核、原后端、无autocast、CUDA路径 | 隔离CPU存储优化 |

`compute_columns3d`在每次调用内部用`at::empty`创建K×M列矩阵；公开`slow_conv3d_forward.out`只复用输出，内部列矩阵仍重新创建。[官方实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L517) 因此，仅换成`.out`不能声称已经消除im2col分配。

最有希望的顺序是：先确认目标CPU allocator是否重复归还这些大页，再评估仅该层复用原大小列缓冲和连续输入/输出缓冲。列缓冲复用需自有CPU操作实现或适配同ATen/BLAS入口，当前Python接口没有可传入的columns工作区。**这里没有编写、编译或验证该操作。** 末次较短矩阵须保留原连续stride，不能直接用最大缓冲切片而悄悄改变leading dimension。

保持原6.243GB最大列缓冲不会降低理论峰值或累计展开量；潜在收益来自减少分配/归还、页处理和重复缓冲初始化。实际allocator行为尚未量化，不能承诺提速。已保存单次up3.conv0 profiler中Slow3d占16.054/16.279秒；`contiguous`12事件合计0.285秒，不能将嵌套copy耗时再相加或将整个conv的CPU核占用归因于GEMM。仅输入copy复用的可见空间有限，列/页面路径更值得先定位。

## 最小后续门（尚未获准或执行）

1. 只读目标allocator/ATen接口，证明同矩阵布局、bias路径与原列值排列；不改全局allocator或其他函数。
2. 一个私有原几何复用原型，先实际权重短合同全FP32 bit一致、alias/stride/flags/异常门。
3. 只有协调者另行授权后，再复用已有join输入，单层旧新有限回放；记录完整pre-ELU bit、RSS及页fault和操作clock。任一非exact即停。
4. 单层通过仍不能替代完整T1图/CSV和CUDA保护门；进入生产须另行验收。

改变K排列或channels_last_3d属于另一条数值策略，需真实官方中间状态/标签门；当前不自动启动，也不重复已拒绝oneDNN或低精度路线。
