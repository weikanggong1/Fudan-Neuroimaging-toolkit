# CUDA 区域并行 benchmark（2026-10-08）

本记录验证 `segment_4_subregions(..., parallel_regions=True)` 是否在不改变输出空间和标签语义的前提下缩短端到端时间。它只读取原始 T1、FNIT 图谱和声明的 SynthSeg 权重，不读取官方结果参与生产拟合。

## 固定条件

- 数据：公开 ds000114 snapshot1.0.2，`sub-01` 原始 T1，SHA-256：`2bbe4430223d9b9ce47d4dab3ff398a69fefbb481cbf3bd136e07f363a73919c`。
- 图谱：FNIT 亚区 atlas tree SHA-256：`13828e6208e114ea6eb5d0704d4cf99658b000859c624c07aada46af62a89c5c`。
- 权重：FNIT SynthSeg/SynthSeg+ 权重 tree SHA-256：`2793f1dfc9e866722105a73c0aee0395a65503c75a455f4d1c119b1f9f64830c`。
- 硬件：NVIDIA H100 PCIe 80 GB，驱动 535.216.03，PyTorch 2.5.1，CUDA 11.8，Torch/Numba/BLAS 外层线程预算 4。
- 精度：默认 float32/TF32；并行 mesh overlap 阶段统一 FP32，未启用 FP16/BF16。
- 计时：新进程从 API 调用前开始，包含预处理、GPU 同步、合并和输出读写；驱动显存由 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 200` 采样。

## 实测结果

| 模式 | wall | compute | 保存 | 驱动显存峰值 | 输出标签 SHA-256 |
|---|---:|---:|---:|---:|---|
| 串行 | 798.496 s | 797.912 s | 0.494 s | 18,936 MiB / 19,855,835,136 bytes | `750d91a424fdb7619e2d93d437e011ecab634fed93d6824141b2d62b49dd2701` |
| 分批并行（worker=2） | 625.168 s | 624.562 s | 0.519 s | 18,936 MiB / 19,855,835,136 bytes | `30fcbe67c344abdfbb0350d99dcb48d762e89fba9ac938053fe7cb085f4c7439` |

相同节点仍有其他 GPU 进程，以上是实测观察值，不能外推为空载吞吐。分批策略为第一批 `thalamus + brainstem`，随后分别运行左右海马/杏仁核；相对串行 wall 缩短 21.69%。将丘脑与左海马/杏仁核直接并发的诊断达到 23,224 MiB，超过 20,000,000,000 字节，未采用。

并行与串行输出的 shape/affine 相同；721/10,223,616 个原生体素不同，整体标签 Dice 为 0.995692。结构级 Dice 为：脑干 0.999186、丘脑 0.996412、左海马/杏仁核 0.982290、右海马/杏仁核 0.990102。并行只改变调度和并行阶段的 FP32 数值策略，未改变标签编码、坐标空间、体积定义或合并顺序。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 200 > gpu.csv &
fnit segment-4-subregions --i sub-01_T1w.nii.gz --o out/subregions_native.nii.gz \
  --atlas-root /path/to/atlases --synthseg-weights /path/to/weights \
  --device cuda:0 --threads 4 --parallel-regions --max-parallel-regions 2 \
  --output-dir out --save-highres
```

正式输出目录必须为空；串行对照去掉 `--parallel-regions`，并使用另一空目录。`benchmark_parallel.py` 也可用于 API 级计时。报告中的 `parallel_batches`、`region_schedule`、`peak_gpu_scope` 与每区结果保存在 `parallel_benchmark.json`/`report.json`。

## 代码范围

区域 recipe 内部仍保持串行；共享预处理只执行一次。CUDA worker 为每个独立 recipe 建立 stream，完成后先把 labels、posterior、prior、vertices 和 Gaussian 参数移回 CPU，再释放该批显存。最终在固定 canonical 顺序合并。默认参数保持 `parallel_regions=False`，因为不同 GPU、驱动和图谱配置需要重新验证显存与整体指标。
