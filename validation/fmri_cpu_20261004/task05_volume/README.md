# Volume 与 helper 的完整真实数据 CPU 验证

## 1. 功能与流程

本目录保存隔离的 benchmark 适配器和聚合报告。范围包括完整 volume、FEAT core、Gaussian 高通／强度缩放、可选 slice timing、T1w 采样参考、Otsu mask 和旧版 motion-only 重采样。输入清单、每个参数及参考程序见 [准备清单](PREPARATION.md)；生产实现用法和完整结果见 [helper 官方对照](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md)。

```mermaid
flowchart LR
    A[私有完整真实输入与冻结源码] --> B[同核同线程串行执行]
    B --> C[正常读写的 API 时间]
    B --> D[完整输出与来源记录]
    D --> E[计时外全网格数值审核]
    C --> F[匿名聚合报告]
    E --> F
    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000;
```

## 2. Python 调用、输入与输出

生产 Python 示例及所有输入／输出参数见 [helper 文档第二节](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#2-python-调用输入与输出)。benchmark 私有 JSON 清单包含完整原始 BIDS、固定中间影像、合法权重与模板的服务器路径；路径和个体影像不进入公共报告。公共 JSON 仅包含尺寸、dtype、SHA、时间和聚合误差。

## 3. 命令行调用

```bash
# 数值分析不重新运行生产或原版算法；使用已完成的完整调用。
python validation/fmri_cpu_20261004/task05_volume/compare_helpers_v2.py \
  --run-root /absolute/path/private/task05_volume \
  --source /absolute/path/frozen_baseline \
  --inputs /absolute/path/private/inputs.json \
  --float-input /absolute/path/private/raw_bold_float32.nii.gz \
  --prior-report /absolute/path/private/comparison_v1.public.json \
  --output /absolute/path/private/helper_float32_protocol.public.json
```

`--run-root` 包含 baseline_v3、reference_fixed_v2、candidate_stc_matched_v5 和 reference_stc_float32_v5 的原回执；`--source` 为未修改基线；`--inputs` 定位实际完整影像及元数据；`--float-input` 为无损数值转换的完整浮点控制；`--prior-report` 绑定原 int16 协议报告；`--output` 必须为新文件。`--stc-candidate-folder` 和 `--stc-reference-folder` 可显式指定新的同核 STC 对照目录。脚本核对实际输入 SHA、整个冻结 Python 源码、核预算、网格、TR、所有数值及 ignored 帧；同时完整比较新、旧 FNIT STC 的值和影像头网格，保留 13 项完整官方比较及三项旧协议指标。

## 4. 原软件调用

完整 FSL temporal、AFNI Fourier STC 和 NiWorkflows sampling-reference 命令见 [原软件协议](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#4-原软件调用)。完整 volume 的原版参考使用固定 fMRIPrep 25.2.4 SIF，默认关闭 STC、禁用 recon-all／surface，生成 T1w 和 MNI preproc；它没有 FNIT 的后续 ICA-AROMA、高通、额外混杂回归。整链时间按这些范围分别报告。

原版 volume 的私有输入清单另声明 `official_fs_license`，只将合法许可文件的路径传入容器。适配器按清单选择相同 session 和 task；容器内的真实 payload 与外层启动器必须同时退出 0，不对 setuid 启动器使用 host strace。四个最终 FNIT CPU 队列各在同一核组配对 baseline／candidate；原版完整链保留其独立核组。它们在同节点采用相同 1／8 线程预算，原版与 FNIT 整链的物理核组不同。

## 5. 最新精度、时间与脑图

[完整 helper 结果和已有公开图](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#5-完整真实数据精度时间与脑图)列出 CPU1／8 的完整 API、全网格误差与来源边界。[最终 13 项完整聚合报告](helper_float32_protocol_20261005.public.json)已完成：强度缩放与采样参考逐值相同；STC float32 RMSE 约 1.9×10⁻⁵，ignored 帧保持原值，新、旧 FNIT 完整数据和影像头一致。首轮 STC 输出类型不同；[旧协议报告](helper_int16_protocol_20261005.public.json)保留原 int16 取整差异，不代替 float32 数值控制。

[完整 volume 执行基线](baseline_execution_20261005.public.json)记录基线四组 first/cache-call 已完成。最新 main 已改变 robust BOLD reference，该旧记录不作为当前整链验收。本轮新个体影像与脑图不公开。

## 6. 最近记录

- 2026-10-04：固定完整真实输入、CPU 分组、冻结源码和原程序；修复队列锁覆盖导入与校验，保留竞争中的无效计时。
- 2026-10-05：13 项最终完整 helper 比较完成，全部正常调用、来源、实际输入、同核预算、完整数值与头网格已核对；三项原始 int16 STC 协议保留。float32 原版控制和同核 FNIT first/warm 已通过，未改变生产默认。
- `compare_helpers.py` 与旧 int16 报告保存首轮事实；`compare_helpers_v2.py` 增加实际输入重哈希、完整浮点转换证明、预算和时序元数据门槛，不覆盖旧报告。

## 7. 原实现与参考文献

见 [helper 原实现和参考文献](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#7-原实现与参考文献)。原程序仅供验证使用；FNIT 生产不依赖 FSL、AFNI、NiWorkflows 或 fMRIPrep。
