# CPU persistent 平均器安全交付

## 当前范围

生产候选：`fc2abc94966150906f4f3421f23aab235b64dd8d`，基于 `a713f87571db2b9f7a8c3c344f3479bd72dc57be`，独立分支 `cpu-average-persistent-20261004`。仅新增自有 CPP 与懒加载模块，并在原 CPU wrapper 中尝试该后端；原有序 NumBa、三跳 BFS、完整 `RegistrationGradientAverager` 类 AST 与基线相同，CUDA 文件逐字节相同。SynthSR、FNIRT 和其他生产文件未修改。

本目录记录安全实现与局部合同测试。协调任务负责冻结这些源码，再运行同一真实完整左半球的 CPU ABBA 与实际 H100 GPU 回归；当前没有把原型输出或局部测试改标为正式候选通过。

2026-10-05 实际 H100 GPU 算子回归已通过，见 [gpu.public.json](gpu.public.json)。完整真实梯度的 1/16/256 轮按旧→新→新→旧执行，输出 SHA 与输入不变性相同；两臂逐调用 allocated 25,273,344 B / reserved 46,137,344 B，相同七次峰值；CPU helper 导入清单为空。GPU UUID、实际源码与输入 SHA 已保存。首次 JIT 共用进程及缓存，不能作为 CUDA 冷启动提速证据。

正式完整 CPU 对照冻结于 `7deb4de541890807e8b8052361f03ec330eba9f3`，按旧→新→新→旧、相同八核亲和性、各自空编译缓存运行。`compare_cpu_abba.py` 在四次完整运行结束后核对冻结全部源文件、优化轨迹、逐点坐标、有序面、几何、实际 CPP 后端及线程预算，并与此前同输入官方输出核对。当前仍在运行；完整配准结论不由平均算子或原型门代替。

## 实现与边界

- Linux x86-64，原生 FP32，至少 8,192 点、至少 16 轮自动尝试 CPP；其他调用保留原 NumBa。没有新增低精度或 64 位替代路径。
- 原 wrapper 决定线程预算，调用 CPP 时传入实际 NumBa mask。OpenMP 使用局部 `num_threads`，没有硬限 8，也没有全局 OpenMP setter。原 `finally` 恢复 NumBa mask。
- 原邻接次序、FP32 累加与乘法、双缓冲和逐轮 barrier 保留。编译采用 `-O3 -std=c++17 -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared`；不使用 `-march=native`。
- 输入或倒数有 NaN/Inf 时执行原 NumBa。CPP 在调用线程与每个工作线程检查 MXCSR FTZ/DAZ/rounding；非默认政策返回状态 6，原 NumBa 接续，不修改任何浮点环境。
- 懒编译使用既有 Conda C++ 编译器，无 `shell=True`。源码、flags、编译器路径/版本/真实二进制 SHA、平台/ABI 共同确定 cache key。
- 私有 0700 缓存目录、私有无链接 lock/artifact、进程锁、原子产物与 ABI/字节 SHA 核查。最多构建 60 s；锁竞争、ENOLCK、EINTR 的重试共用 15 s deadline；超时只终止自身编译进程组。
- 同 key/cache 的编译失败本进程只尝试一次。`backend_info()` 提供当前线程最近一次实际后端及原因，支持 coordinator 区分真正 CPP 与回退；`.so` 的大小/SHA 在同 stem `.json` 中。

## 局部测试

本地 Linux x86-64 / Python 3.11，GNU C++ 11.4，Torch 2.5.1，NumBa 0.67。完整匿名绑定见 [implementation.public.json](implementation.public.json)。

```bash
PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
python -m pytest \
  tests/recon_all/test_average_cpu_cpp.py \
  tests/recon_all/test_average_cpu_parallel.py -q
```

结果：**32 passed，23.92 s**。其中 29 项新测试、3 项原线程测试。覆盖 1/2/4/8 预算、16/17/256 轮、非连续输入、原逐位结果、输入不变、新连续输出、mask 恢复、unsupported/compiler/短调用回退、failed build once、compiler/flag cache 失效、安全目录/锁、manifest 损坏、编译超时、两个进程一次构建，以及 ENOLCK 成功重试/期限耗尽。

特殊位模式在独立子进程测试，避免改变父进程工作池：默认政策下 subnormal、正/负零 CPP 与原 NumBa 位模式相同；显式 flush denormal 时确认 NumBa 回退；NaN payload 与正/负 Inf 输入逐位保留原路径结果及输入。退出前恢复调用线程政策，子进程结束销毁其工作池。

这些合成数据是合同、安全与错误路径测试，不是性能 benchmark。没有重新跑整套 2k 测试、官方 registration 或完整 recon-all。当前 Conda 最低 NumBa 0.59 的全新安装和干净 wheel/sdist 安装没有在此目录重测。

## 已有真实原型证据

已现场读取 coordinator 的两个匿名 JSON。原型 CPP SHA `8fb858ea0caee073b3af01d6787a883a50b32c3b0d6678cdcd5c18b629e5e9d6` 与正式 CPP SHA `a6dcc287830b60379b75545943ec5e019f2f7274c9266610177c849f05a3cc89` 不同。

真实 ds000114 v1.0.2 完整 LH 119,451 点、238,898 面：原型 0/1/16/256/16,384 轮与 ordered NumBa ABBA 逐位相同；完整球面配准的 seed/轨迹/坐标/面/geometry 与保存 accepted/native 对照相同。平均累计 48.069→20.127 s；API 345.714→448.403 s，未修改刚体段 49.294→196.200 s。共享节点的非相邻两次完整运行不能支持整体提速。

正式源码的四次完整 CPU 接受轨迹与实际 GPU 门由 coordinator 单列；不要用以上原型 SHA 或时钟替代。新功能七节文档见 [MRIS_REGISTER_AVERAGING.md](../../../src/fnit/recon_all/MRIS_REGISTER_AVERAGING.md)。

## 打包交接

coordinator 维护共享 `pyproject.toml` / `MANIFEST.in`。需要在 setuptools `fnit` package-data 添加 `recon_all/*.cpp`，并在 sdist manifest 添加 `recursive-include src/fnit/recon_all _average_cpu_persistent.cpp`（或等效 `*.cpp`）。`.py` 已由包发现规则覆盖，`.md` 已由既有 `recon_all/*.md` 覆盖。只发布自有 `.cpp`，不发布缓存 `.so` 或私密运行状态。

功能文档与本目录随后单独提交，生产源码在 `fc2abc94` 保持冻结。
