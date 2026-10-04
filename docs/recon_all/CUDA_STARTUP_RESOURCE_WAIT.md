# CUDA 半球启动资源等待

## 功能与范围

`run_hemisphere_group` 在调用算法前让实际 worker 完成 CUDA 标量分配、同步和设备检查，原子写 READY 并保持上下文。父进程确认当批全部 READY 后发送 GO；每侧函数最多进入一次。CPU 也使用屏障，但不重启资源失败。

这是已观测上下文建立期间间歇性 OOM 的启动资源处理；没有证明具体底层资源类型，也不是驱动根因修复。没有低精度、CPU 或其他算法 fallback。

## Python 参数与输入输出

新增具名参数 `startup_wait_seconds=30`，单位秒，必须有限且非负。每批固定启动期限包括导入、bootstrap 和每次最多 2 秒的退避；不包括算法执行。`workers=2` 一批，`workers=1` 两批逐侧启动、执行、退出，各最多一个启动预算，不驻留额外半球上下文。0 禁止重启，单次正常启动仍有 90 秒硬期限。

```python
from fnit.recon_all.hemisphere_parallel import run_hemisphere_group

# subject_directory 是 FNIT 自产被试目录，包含 mri/surf/label/stats/scripts。
subject_directory = '/path/to/subject'
report = run_hemisphere_group(
    subject_directory, 'annotation', device='cuda:0', threads=4,
    workers=2, startup_wait_seconds=30,
)
```

只有可信 worker 报告明确 `operation_entered=False`、CUDA OOM，且失败阶段为 `first_allocation`、`sync` 或 `device_properties`，才允许启动重启。先回收失败进程及其自有后代。imports、policy、缺失报告、callable 导入、函数内失败及函数后的 sync 都直接失败。成功 READY 半球保持上下文等同批另一侧，不重复执行算法。worker 等 GO 时也检查同一启动期限及实际父进程身份；父进程消失或超时即退出，不给算法增加执行期限。CUDA OOM 必须带明确消息前缀 `CUDA error: out of memory` 或 `CUDA out of memory`；Torch OOM 异常类型本身不足以识别 CUDA，普通 host OOM 不重启。

组报告包含每次 request/log/report 路径、原始错误、READY/GO 时间、函数是否进入、实际启动墙钟和每批预算。旧 worker 时间包含启动和等待；新增 `operation_*` 区间只表示算法执行。全部启动文件保留在 `subject/scripts/<operation>.<hemi>.startup-XX.*`；request 中私有影像目录在组结束后清理，仅供参数追溯。组墙钟包含等待。双侧成功并完成文件审计后才发布输出。

## API、CLI 与资源协议

本参数属于内部半球调度函数；现有 recon-all Python API 和 CLI 接口不变。CLI 示例及原软件命令沿用 recon-all 主文档。固定设备、TF32/精度、allocator 缓存策略和线程预算不变，不 reset 父 API CUDA 上下文。总显存需按父进程与所有 worker 合计检验，READY 持有上下文本身也占资源，不能把标量分配作为 20 GB 峰值证明。

## 验证记录

本次 CPU 回归覆盖 READY 屏障、一次启动 OOM 后成功、算法/后同步 OOM 禁止重试、缺失报告、期限取消和资源等待关闭。模拟测试仅验证调度，真实 annotation 输入精度、总峰值和启动成功率须由独立 GPU 验证确认。

实现：`src/fnit/recon_all/hemisphere_worker.py`、`hemisphere_parallel.py`。上游错误边界：[PyTorch 2.5.1 CUDAStream.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDAStream.cpp)、[NVIDIA CUDA primary context API](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)。
