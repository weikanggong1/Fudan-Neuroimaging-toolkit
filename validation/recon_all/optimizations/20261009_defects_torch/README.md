# 完整双侧缺陷投射：2026-10-09

本目录只记录冻结同输入的完整 `mri_label2vol --defects` 替换，包含读取、传输、
左侧清零、右侧累积与压缩写出。不是拓扑GA，也不是原始T1整例。
模块SHA为 `807b123d7e1ac47021864dfe5b486fa1ca3e9dd6eecbef792d469fe8552d12e5`；
实际测试版本以各JSON的 `code_commit` 和 `source_sha256` 为准，不能当作main结果。

| 同输入测试 | 原生中位数/s | Torch CPU/s | Torch GPU/s | 解码结果 |
|---|---:|---:|---:|---|
| [sub01 CPU](sub01_cpu.json) | 2.615 | 2.794 | 未测 | 0差异 |
| [sub01 GPU](sub01_gpu.json) | 2.628 | 3.021 | 2.761 | 0差异 |
| [sub07 GPU](sub07_gpu.json) | 3.411 | 3.293 | 3.190 | 0差异 |

两例所有非背景标签Dice为1，最大/P99误差0，affine/dtype相同。
原生两次重跑的解码标签相同；随机颜色表与命令元数据使MGZ原始字节不同。
Torch标签名/编号相同，调色板确定性；不得声称严格文件SHA复现。
GPU API在两例分别慢约5.1%和快6.47%，不据此设置默认或宣布稳定整例收益。

## 资源与失败范围

H100、PyTorch2.5.1/CUDA11.8、4线程、默认TF32，不使用半精度。
投射本身使用有序逐分量乘法，无TF32矩阵计算。完整试验顺序为原生、CPU、GPU、
GPU、CPU、原生，CUDA同步明确目标设备。context初始化、哈希和评分单列于API之外。

sub01 GPU allocated/reserved峰294,460,416/360,710,144字节，整进程占用未采样。
sub07同期目标设备进程树采样峰912,261,120字节，allocated/reserved峰
295,558,144/360,710,144字节；计划0.1秒采样，实际最大间隔3.034秒。
样本完整保存在JSON，外部任务占用单列，未把不同时间峰值相加。

开发v1–v5诊断暴露并修复许可证环境传递、overlay格式、输入字节序、
Fortran布局和MGH字段块90字节/数据起点284字节混淆。此后sub01 v6/v7、sub07 v9
通过。sub07 v8及后续新进程GPU单测首次极小分配出现OOM，未计为通过；
没有证据确认根因。v9在主线程初始化CUDA后启动监测线程，本次运行成功，
未添加生产重试/CPU回退。完整失败原始日志留在服务器独立诊断目录。

本地不能导入Torch，完成语法检查。远端CPU规则测试6项通过；GPU规则单测初始
OOM，真实GPU完整API两例通过。缺少tifffile的基础环境接线失败见
[原失败日志](wiring_tests.txt)，完整recon环境中接线先7/7、最终9/9通过：
[最终日志](combined_wiring_tests.txt)。这些检查不代替整例或干净安装。

## 复现

```bash
# 输入是既有被试的自产冻结前置文件；输出必须是新诊断目录，不修改被试。
python tools/benchmark_recon_defects_torch.py \
  --subject /data/subject \
  --binary /opt/fnit-conda/bin/mri_label2vol \
  --assets /data/fnit-assets \
  --output /data/benchmark-defects-new \
  --device cuda:0 \
  --threads 4 \
  --code-commit ACTUAL_COMMIT_AND_MODULE_SHA
```

`subject`需有orig、双侧orig.nofix、defect_labels和nofix.cortex.label；`binary`
为固定源码Conda构建参考程序，`assets`是声明的资产；其余参数分别为新输出目录、
目标GPU、线程预算、实际源码身份。参考程序的个人许可证由外部环境配置，
不读取内容、不写入报告。全部输入和程序哈希保存在JSON。
详细API、空间、失败语义及原软件链接见 [中文功能页](../../../../docs/recon_all/DEFECTS_TORCH.md)。

严格复现：解码标签通过、调色板字节不同；优化无退化：此同输入范围通过；
整体指标等效：未判定。原始T1整例、10例该阶段和隔离安装仍未完成。
