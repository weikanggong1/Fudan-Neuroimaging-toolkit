# robust-register prepared/M0 实际捕获（2026-10-06）

## 1. 功能简介

在同一已保存moving/fixed MGZ输入对上（来自既有目标掩膜与右侧atlas前处理），比较冻结 FreeSurfer 源码观察前缀与原 FNIT 实验前缀，停在 prepared/M0 完成、pyramid 开始之前。v4 已完成受控比较：prepared 两侧数组、全部几何字段及 Rsrc/Rtrg 逐位一致；质心和 M0 有约 1.9e-13 至 4.1e-13 的 Double 微差。

本轮完成的是这个观察边界的控制。原安装二进制内部轨迹、完整 robust-register 等价性及端到端速度尚未验收，正式 CPU gate 保持 17/20。产品和 GPU 路径改动均为 0。

后续[CPU保序质心控制](../centroid_serial_cpu_probe_20261006/README.md)复用本轮保存的准备图像和SDK返回值，6个Double质心已逐位匹配。这是独立内核结果，未重算本报告的M0；下表的v4原质心/M0差异保持。候选的完整初始化及默认配准接入仍待验证。

## 2. Python 调用、输入与输出

输入为原固定moving/fixed MGZ对，使用 CPU、rigid 模式和平移初始化。输入身份、原数学源码、AST tap 和 protocol 全程保留；v4 的新增桥只显式导入原 registration 模块并核其身份。

| 输入/观察输出 | 意义与格式 |
| --- | --- |
| moving / fixed | 既有前处理产生的反射atlas与目标mask；MGZ，由各自前缀读取 |
| prepared source / target | 原空间准备后的完整数组，little-endian float32，X 最快；每侧 428,220 B |
| geometry | 两侧实际 shape/type/RAS、outside 值及 15 个保存的 float32 几何字段 |
| centroid | 每侧一次原返回值，3 个 float64 |
| M0 | pyramid 前的初始矩阵，4×4 float64，行优先 |
| Rsrc / Rtrg | 已有 source/target reslice 状态，4×4 float64；候选实际 Float target_pull 以 Double 精确表示 |
| 公共 manifest | 发布聚合差异、身份和诊断时长，不含 MRI、坐标、质心或矩阵数值 |

本报告没有新增公共计算 API。读取结果：

```python
import json
from pathlib import Path

# 读取已完成控制的聚合结果，不执行图像处理或配准。
report_directory = Path("validation/robust_register/prepared_m0_capture_20261006")
manifest_path = report_directory / "manifest.public.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(manifest["precision"]["M0"])
```

`different_words` 是 IEEE 字级差异数，`bits_exact` 表示逐位相同，`max_absolute_from_actual_receipt` 是原控制保存的最大绝对差。未执行的 pyramid/optimizer 参数沿用冻结调用，不构成本轮结果。

## 3. 命令行调用

在本报告目录读取聚合状态：

```bash
python -c 'import json; from pathlib import Path; print(json.loads(Path("manifest.public.json").read_text())["status"])'
```

v4 为一次 candidate-only 组，native 0、编译 0、candidate 1；原 native 使用 v2 已保存且通过身份核验的结果。每版观察组单独授权一次，没有自动重试。

## 4. 原软件对应

原功能对应 `mri_robust_register` 的空间准备与平移初始化。完整原调用形式为：

```bash
mri_robust_register --mov moving.mgz --dst fixed.mgz \
  --lta rigid.lta --mapmovhdr rigid.header.mgz --sat 50
```

该命令包含完整配准；本轮两臂分别停在既定观察边界。原参考来自冻结源码 commit `d932c45` 的独立 observer，源码身份和实际 binary 身份分列于清单。v2 实际加载的 23 个 ELF provider 集合及每项大小、SHA 均匹配计划，v4 复用这些原回执。

旧 native outside 字段实际为 8 字节 Double。v3 起仅接受其正/负零，通过整数符号位形成 4 字节 Float32 比较字段；非零或非有限值禁止使用此桥，原 JSON 和数组保持原字节。

Schur guard 的未进入结论来自绑定唯一退出守卫、原 native exit 0 及无固定 abort 标记，没有动态计数器；完整 Schur/LAPACK ABI 尚未验收。

## 5. 实际精度、耗时与可视化

每侧 prepared 为 107,055 个 float32 字。完整数组逐字节相同，source/target 各 **0 个差异字**；shape、dtype、layout、RAS、outside 比较字段及全部保存几何字段相同。

| 实际 Double 观察输出 | 不同字数 / 总字数 | 最大绝对差 | 逐位相同 |
| --- | ---: | ---: | --- |
| source centroid | 3 / 3 | 1.9184653865522705e-13 | 否 |
| target centroid | 3 / 3 | 2.2026824808563106e-13 | 否 |
| M0 | 3 / 16 | 4.121147867408581e-13 | 否 |
| Rsrc | 0 / 16 | 0 | 是 |
| Rtrg | 0 / 16 | 0 | 是 |

按比较字段顺序，首个不同字段为 source centroid。两臂相关 Double 元数据均有限，每侧 centroid 原调用一次，观察器没有重算质心。prepared 已相同，微差保留在质心/M0 返回状态中；具体计算顺序的原因尚未证明，也没有评估后续 optimizer 的影响。

| 原始诊断计时 | 秒 |
| --- | ---: |
| v2 保存的原 native 前缀 | 0.251042 |
| v4 candidate entry | 1.810702 |
| v4 candidate 前缀 | 2.063693 |
| v4 group worker | 2.963292 |
| v4 supervisor | 3.053844 |
| v4 outer | 3.118625 |

这些时间包含各自观察、导入、序列化和控制开销，不能用于完整 MRI benchmark 或加速比。v4 本地可恢复的 21 份原文本逐项大小/SHA 一致；最终回执记录 340 项绑定资源与 9 份自有源码保持，8 个原进程均退出、树无残留、共用锁释放、peer 索引保持。全部 12 个控制冻结文件另经独立现场 SHA 核验。

本轮 pyramid、halfway、QR/IRLS、optimizer、完整 API、最终 MGH/LTA、GEMS 和 GPU 均为 0。脑图输出为 **0**；没有可发布脑图，不能据聚合数字构造脑图。

## 6. 最近更新与 benchmark 记录

| 版本 | 实际结果与保留记录 |
| --- | --- |
| v1 | 原 native 因 FREESURFER_HOME 缺失 exit 255，candidate 0；FAIL 保留 |
| v2 | 增加 SDK 环境入口；原 native exit 0、runtime 23 核验完成，outside 的 8B/4B 协议门失败，candidate 0 |
| v3 | native 0；封存已保存 native 并做正/负零字段桥，candidate 1 在惰性模块尚未导入处 exit 1 |
| v4 | 保持 v3 原 8 份代码，新增显式模块导入桥；native 0、candidate 1 exit 0，完成上述同输入观察比较 |
| 后续独立质心控制 | 仅对保存准备图像计算串行质心，6个Double返回值逐位同；原运行库观察门失败保留，追加元数据核验不重跑数学；完整M0未执行 |

v1/v2/v3 原失败和旧 native 原字节均保留。可恢复原文本核验数分别为 16、18、19、21；私密数组仅在原控制中核验，本独立审查没有打开数组或重跑数学。v4 控制结束后 capture/compile 权限关闭，正式全门仍为 17/20。

## 7. 参考文献与原代码

- [冻结 FreeSurfer robust-register 原代码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register)。
- [原软件文档](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register)与[软件许可](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)。报告没有发布上游源码或二进制。
- Reuter M, Rosas HD, Fischl B. *Highly accurate inverse consistent registration: a robust approach*. NeuroImage (2010)，[DOI](https://doi.org/10.1016/j.neuroimage.2010.07.020)。
