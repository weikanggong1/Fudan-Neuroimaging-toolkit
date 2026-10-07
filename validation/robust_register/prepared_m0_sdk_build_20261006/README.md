# Robust SDK Schur 守卫 v2：实际 compile/link 诊断（2026-10-06）

本报告截至独立构建。随后[实际准备态/M0对照](../prepared_m0_capture_20261006/README.md)已完成：图像/几何逐位同，M0有Double尾差；本报告构建计数保持。

## 1. 功能简介

本轮为独立 SDK 的 prepared/M0 观察器编译一个自有 Schur 边界守卫，再复用原九对象完成一次 link。两条命令均 exit0，静态 ELF 审查通过。新观察器未运行，M0 capture 为 0。

流程：冻结源码 → 编译一个 guard 对象 → 原九对象加 guard 链接 → 静态符号及 provider 审查 → 保存回执。guard 若被调用，只写固定 stderr 并退出86；它不读取参数、不返回数学结果。

## 2. Python 调用与输入输出

输入为冻结的自有源码、原九对象及原静态/共享 providers；编译过程没有影像输入。输出为新 guard 对象、观察器 binary、linkmap、依赖与执行回执。公共报告只公开摘要和计数，不提供 SDK 代码或二进制文件。

本次没有新增生产 Python 接口。读取公共结果示例：

```python
import json
from pathlib import Path

# validation_directory：构建诊断的公共报告目录。
validation_directory = Path("validation/robust_register/prepared_m0_sdk_build_20261006")
# validation_manifest：读取实际 compile/link 和静态审查摘要。
validation_manifest = json.loads(
    (validation_directory / "manifest.public.json").read_text(encoding="utf-8")
)
print(validation_manifest["build"]["command_exit_codes"])  # [0, 0]
```

## 3. 命令行调用

本阶段没有新增生产 CLI。以下命令仅格式化公共结果，参数为 manifest 路径：

```bash
python -m json.tool validation/robust_register/prepared_m0_sdk_build_20261006/manifest.public.json
```

观察器运行和 M0 capture 需另行验收，当前没有这些执行结果。

## 4. 原软件对应

参考为 FreeSurfer 的独立 SDK `mri_robust_register`，原九对象及原 tap 保持不变。守卫保留原 `zgees_` 的完整参数类型序列，包括 long 整数指针；其作用仅为 Schur 前控制边界，不能作为 Schur 实现。

原完整 Schur 的整数/Fortran ABI 尚未验收。这个 SDK 参考也不能称安装 binary 的内部 trace。原仓库见[FreeSurfer mri_robust_register](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register)。

## 5. 实际结果、精度与耗时

| 实际构建或静态门 | 结果 |
| --- | ---: |
| 新 guard 对象编译／link | 1／1，均 exit0 |
| 原对象重编译 | 0；复用9个 |
| guard 唯一静态强定义 | 1 |
| 实际 guard 依赖 header | 175 |
| 静态 provider records／主程序 NEEDED | 23／13 |
| 匹配的版本化符号要求 | 2,114 |
| 缺失 NEEDED／未解析强符号 | 0／0 |
| 未解析 optional weak imports | 80 |

实际 guard 对象的外部调用为 `write` 与 `_Exit`；ELF/linkmap 将守卫绑定到新对象。静态 provider 闭合在本次记录的解析条件下通过，运行时 provider maps 尚未验证。23是静态 provider 记录数，不能写成运行时已加载数量。

| 构建诊断范围 | 耗时 |
| --- | ---: |
| guard 编译 | 0.364124 s |
| link | 0.263927 s |
| worker | 0.771657 s |
| supervisor／outer | 1.045895／1.116869 s |

这些仅为构建诊断耗时，不能作为配准 speedup。没有新 MRI/M0 精度结果；既有正式 CPU 门仍为17/20。本轮没有运行新 binary，guard entry count 未测，不能填0；完整 Schur ABI、完整 robust 兼容性及实际 M0 均未通过验收。图像输出为0，不能构造脑图。

## 6. 更新与 benchmark 记录

- v1 的九对象编译成功，原 link 因缺少 `zgees_` 失败；旧日志及 FAIL 保留，本轮不覆盖。
- v2：完成一个自有 guard compile 和一次 link，无自动重试；15份原始日志文本 SHA 核验一致，源码及129项资源绑定保持原值，受控进程全部退出，锁和资源释放门通过。
- 本阶段产品修改为0。构建通过只形成后续 M0 控制的准备条件，不能替代真实 capture 或完整 Schur/LAPACK 验收。

## 7. 参考文献与原实现

- [FreeSurfer 原 SDK mri_robust_register](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register)。
- Reuter M, Rosas HD, Fischl B. *Highly accurate inverse consistent registration: a robust approach*. NeuroImage. 2010;53:1181–1196。
