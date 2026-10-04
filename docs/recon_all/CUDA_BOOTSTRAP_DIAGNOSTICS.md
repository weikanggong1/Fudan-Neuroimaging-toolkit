# 半球worker的CUDA失败诊断

本次补充记录失败位置和启动参数，尚未证明或修复CUDA OOM机制。算法、精度、CPU线程预算、CUDA首次分配顺序及半球文件发布规则不变。

原Python/CLI入口及参数不变。输入仍为自产subject及明确device、threads、workers；没有新增影像、权重或官方参考输入。正常输出仍由双侧成功屏障发布。

`subject/scripts/<operation>.hemisphere-group.json`记录各worker回执。失败worker新增：

- `failure_phase`：imports、policy、first_allocation、sync、device_properties、import_callable或operation。它是Python执行边界；仅first_allocation不能区分底层context、stream和cudaMalloc，须结合C++栈。
- `logical_device`与`precision`：原请求的逻辑设备及TF32配置，不冒称失败进程已确认物理UUID。
- `cuda_initialized_after_failure`：只读取PyTorch的Python初始化标志，不调用CUDA、不查询显存；导入失败等无法读取时为null。
- `traceback`：原异常栈。父进程仅在未设置时给fresh exec加入`TORCH_SHOW_CPP_STACKTRACES=1`，原生错误栈采集和符号化发生于抛出异常时；没有增加正常CUDA计算或同步。未提供性能benchmark。

每个已启动半球的请求保存在`subject/scripts/<operation>.<hemi>.request.json`，成功与失败都保留；group的`requests`提供其路径。请求中的private subject路径为历史参数，通常已清理，不能拿它当作现存复跑目录。记录只含原具名算法请求，不保存完整进程环境。worker已退出但没有回执时明确记录`report_unavailable`，不能推断成功。

CPU调度回归覆盖真实fresh exec请求留存、模拟首CUDA分配失败的诊断边界与错误后不调用CUDA、缺失worker回执时禁止发布。模拟失败仅验证报告契约，不代替真实OOM定位或影像benchmark。

参考：[PyTorch 2.5.1 C++栈配置](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/util/Exception.cpp)、[CUDA stream初始化](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDAStream.cpp)。该内部调度诊断没有独立原软件命令或新增生产API。

## 当前验证与调用示例

2026-10-04在gpucw1的独立源码副本做17项CPU调度回归，全部通过；unittest记录22.920秒，完整解释器起止25.419秒。CUDA_VISIBLE_DEVICES为空，原生线程预算4。输入是小型调度夹具，验证失败记录和文件发布，不能代替真实影像benchmark。[机器回执](../../validation/recon_all/accuracy_20261003/runtime/oom_report_patch_cpu_v1_completion.json)绑定两个模块及测试文件SHA；源码来自冻816，只覆盖本次诊断模块。

```python
from fnit.recon_all.hemisphere_parallel import run_hemisphere_group

# 单个自产被试，surface/register等先决结果必须已经存在。
subject_directory = "/path/to/fnit_subject"
# 显式GPU、总线程预算和并发数；参数和坐标语义保持原调度约定。
annotation_report = run_hemisphere_group(
    subject=subject_directory,  # 自产被试目录；表面使用surface RAS毫米。
    operation="annotation",  # 三套图谱；顶点顺序来自已有smoothwm/sphere.reg。
    device="cuda:0",  # 当前可见GPU的逻辑索引。
    threads=4,  # 总原生线程预算；每半球2线程。
    workers=2,  # 双侧独立exec，使用私有副本和成功发布屏障。
    profile_stages=True,  # 阶段同步计时；生产可按既有规则关闭。
    kwargs={"assets": "/path/to/fnit_assets"},  # 声明、带哈希的图谱和ico资产。
)
# 返回值含values、workers、requests、published、耗时和同期采样；失败抛
# HemisphereGroupError并在scripts保留日志、历史请求和失败group回执。
```

此次修改未改变数值算法、首次CUDA调用或分配缓存，没有OOM修复和整例加速结论。后续原生边界定位见[独立CUDA诊断](../../validation/recon_all/accuracy_20261003/CUDA_BOOTSTRAP_DIAGNOSIS.md)。
