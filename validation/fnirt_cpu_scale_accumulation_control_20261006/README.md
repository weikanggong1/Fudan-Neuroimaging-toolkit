# FNIRT CPU scale / SSD 累加控制：一次保存状态结果

## 1. 功能简介

唯一有限控制自然退出0。当前保存moving状态的scale梯度在串行Double累加后只改变 `3.197442310920451e-14`，SSD只改变 `2.1316282072803006e-14`；同一串行和的两种最终factor分组逐位相同。本状态的归约顺序变化不足以构成保存native scale差异的主要来源，候选未接入生产。

本控制仅恢复保存点的4个成员：`fixed`、`state_residual`、`state_mask`、`scale`。先复现原baseline bits，再把同一对FP32产品按Z/Y/X串行累加；自身候选全部完成后，才读保存native totalg末项8字节。没有重新采样moving、计算Jte、H或PCG。来源、输入、产品、精度旗标和影像头前后保持一致。结果见[实际标量与调用计数](RESULT.public.json)，原[准备叶](../fnirt_cpu_scale_order_prepare_20261006/README.md)和服务器冻结源码保持原字节。

```mermaid
flowchart LR
    A[来源和4成员守卫] --> B[同一对FP32产品]
    B --> C[baseline逐位门]
    C --> D[一次Z/Y/X Double扫描]
    D --> E[同一和的两个factor分组]
    E --> F[native末项8字节后验比较]
    F --> G[前后绑定和cleanup记录]
```

## 2. Python调用、输入与输出

这是自有有限诊断，没有新增FNIT公共API。下例只读已保存的标量报告，不恢复checkpoint或重复计算：

```python
import json
from pathlib import Path

scalar_report_directory = Path("validation/fnirt_cpu_scale_accumulation_control_20261006")
scalar_control_result = json.loads((scalar_report_directory / "RESULT.public.json").read_text())
serial_scale_comparison = scalar_control_result["own_scalar_comparison"]["serial_source_scale_vs_baseline"]
print(serial_scale_comparison["absolute_difference"])  # 当前保存状态的归约差：3.197442310920451e-14
print(scalar_control_result["calls"])  # 4成员、各1产品/基线/串行pair及1末项读取
```

| 输入 | 格式、意义与实际读取 |
| --- | --- |
| `fixed` | CPU FP32 `[24,28,24]`，缩放前raw Ref；与此前保存官方Ref核对过，当前恢复值SHA再次绑定 |
| `state_residual` | CPU FP32同形状，当前保存moving的warp减缩放Ref；直接恢复，不重新采样 |
| `state_mask` | CPU bool同形状，有效体素mask；count严格为14341 |
| `scale` | CPU FP64零维标量，绑定保存点身份；未优化或修改 |
| 私有checkpoint | NPZ共68成员；整体大小与SHA核对，只读上述4成员，保留记录的stride并恢复到独立CPU storage |
| baseline metadata | 已有stage1和moving控制JSON，提供count、SSD和scale的FP64值/hex门；projection summary、native state仅作原始输入哈希绑定 |
| saved Ref头 | gzip首348字节；确认 `[24,28,24]`、FP32及三条有符号对角空间轴，X为内层正向扫描 |
| saved native totalg | FP64 `[1177]`，整体9416字节SHA先绑定；仅自身控制完成后读取最后8字节作scale后验对照 |
| 来源与runtime | 23个源码、7个输入整体哈希；已有Conda解释器、prefix、Torch2.5.1与NumPy1.26.4按冻结身份核验；无新增依赖 |

输出包括baseline/serial/native后验标量的数值、FP64 hex与`absolute_difference`，实际调用数、source/input/operand SHA，精度旗标、资源、退出和清理记录。公开文件只含上述元数据；原始checkpoint、影像和完整官方源码不发布。[原始worker summary](run/summary.public.json)保持服务器原始JSON字节，[controller摘要](run/controller.public.json)去除私有绝对命令路径并绑定原始receipt SHA。

| 参数 | 本次意义与值 |
| --- | --- |
| `--root` | 既有FNIT canonical根；源码读取`repo/`，保存输入读取已有`runs/` |
| `--workspace` | 新叶`workspaces/smri_cpu_20261004/remaining_20261006/fnirt-scale-accumulation-control-v1`；只使用冻结自有代码 |
| `--run` / `--output` | 共同叶`runs/…/fnirt-scale-accumulation-control-v1`；controller指定run，child在新建`scalar/`写JSON；目录0700、文件0600 |
| `--canonical-main-commit` | 实际绑定`7ff215ee86c49414b2fa6156fdf6769aa54d00f8`，入队、取锁后和退出时再次核验 |
| `--mode` | 唯一外部`enqueue`启动一个内部`controller`；once claim永久保留，拒绝重派 |
| `--approved-scale-control` | 根任务逐文件审核后授权本次有限范围；本记录不是下一次运行授权 |
| CPU与期限 | 8物理核 `[32,36,40,44,48,52,56,60]`；共同CPU锁等待120秒、worker60秒、controller300秒、地址空间8GB |
| 精度与锁生命周期 | FP32产品、FP64累加，CPU无autocast/CUDA/FSL DSO；已有TF32旗标不变。child继承共同锁FD，父finally记录未回收状态并仅关闭自身FD |

## 3. 命令行调用

报告核验只用标准库，不读数组或运行科学worker：

```bash
python validation/fnirt_cpu_scale_accumulation_control_20261006/verify_report.py
```

下面是唯一实际派发形式的审计示例，once claim已存在，不再次执行。变量名说明完整，实际私有路径与解释器身份保存在原始receipt：

```bash
FNIT_SERVER_ROOT=/path/to/FNIT  # 既有canonical入口
FNIT_SCALE_WORKSPACE="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-scale-accumulation-control-v1"
FNIT_SCALE_RUN_DIRECTORY="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-scale-accumulation-control-v1"
FNIT_EXISTING_ENV_PREFIX="$(readlink -f "$FNIT_SERVER_ROOT/envs/default")"  # 保留原环境prefix
"$FNIT_EXISTING_ENV_PREFIX/bin/python" -B "$FNIT_SCALE_WORKSPACE/controller.py" \
  --root "$FNIT_SERVER_ROOT" --workspace "$FNIT_SCALE_WORKSPACE" \
  --run "$FNIT_SCALE_RUN_DIRECTORY" \
  --canonical-main-commit 7ff215ee86c49414b2fa6156fdf6769aa54d00f8 \
  --mode enqueue --approved-scale-control
```

本次controller118665、worker118666自然退出0，均已回收。cleanup没有发送TERM/KILL、没有错误，`unreaped_child_retains_lock=false`。终态后对同一共同inode5185616833的`LOCK_EX|LOCK_NB`取得成功并关闭检查FD，见[锁释放记录](run/lock_release.public.json)。六INDEX锁以共同25秒deadline更新本叶；其他任务字段和文件权限保持。

## 4. 原软件调用与源码定义

这两项是FNIRT内部标量，没有独立原软件CLI；完整注册调用见[TorchFNIRT功能说明](../../docs/fnirt/README.md)。本次没有启动原软件。依据已安装FSL6.0.7.4源定义，SSD在FP32 residual平方后按Z/Y/X累入Double再除count；scale使用FP32 Ref×residual及masked Double扫描，factor在累加后应用。审计行号与SHA见[源码审计](../fnirt_cpu_scale_order_prepare_20261006/SOURCE_AUDIT.public.json)：`fnirt_costfunctions.cpp:885–898,992–993`、`intensity_mappers.cpp:424–427`、`newimagefns.h:644–658`。

自有worker从当前FNIT `evaluate`的SSD表达式和既有FSL-order控制的scale表达式提取AST，产品只执行一次，baseline和串行扫描共享这些不可变FP32产品。不复制原软件源、头、库或二进制。

## 5. 最新精度、运行时间与可视化

输入为同一真实保存状态。baseline SSD与scale均通过FP64逐位门，随后才进入候选。

| 当前保存状态标量 | 原baseline | Z/Y/X串行Double | 绝对差 |
| --- | --- | --- | --- |
| full scale g | 10.452025165741516 | 10.452025165741547 | 3.197442310920451e-14 |
| SSD | 60.329756040354546 | 60.329756040354525 | 2.1316282072803006e-14 |
| 最终scale factor分组 | `2*(-sum/count)` | `(2/count)*(-sum)` | 0，FP64 hex均`69a8afd76fe72440` |

仅在自身控制完成后，保存native末项读得 `10.452032835660868`。它与当前baseline相差 `7.669919352082388e-6`，与串行结果相差 `7.669919320107965e-6`，变化仅 `3.197442310920451e-14`。同点native Robj/ScaledRef/Mask未保存，因此这份结果控制的是当前保存moving状态的归约方式，不定位native输入缓存的唯一首差。

此前“官方保存moving替换＋FNIT采样”得到scale `10.452035382537394`，对应残差 `2.546876526565711e-6`；其残差数组未单独保存。本次没有重建该输入，这一归约结论不解释那项残差。原[轴projection负结果](../fnirt_cpu_rhs_projection_control_20261006/README.md)和完整注册失败均保持原结论。

本次controller记录区间为 **1.849982秒**，包含锁等待、校验、子进程与后置检查，终点位于最终receipt字段构建；worker记录 **1.568003秒**（开始至summary字段构建），共同锁等待 **0.010546秒**，最大RSS319860KiB。完整enqueue至进程回收未独立计时。产品pair、baseline pair、串行pair及native末项读取各1次，未分别计时。这是唯一保存状态诊断观察，没有重复配对或整例提速结论。未生成新的warp或分割，所以无新增脑图；既有真实脑图和完整注册比较见功能说明页。

23源码、7输入、7operand、10冻结payload、Ref348字节头、所有精度旗标前后相同；CUDA未初始化、FSL DSO为空。sampler、Jte、bending、H、diag、PCG、native程序、raw MRI和GPU调用均0。两个baseline门与原始退出码、cleanup和锁生命周期都通过；生产实现没有替换。

## 6. 最近更新与benchmark记录

- `3dc99bef`：修正README实际输出项，child继承共同锁FD，标准库AST守卫通过；冻结`cb3b5827`、manifest`599ef02d`。唯一控制随后完成，上述标量、原始receipt与退出均保留，没有重试。
- `a8316940`：有限worker/controller和准备清单提交；三项标准库清理mock通过，修复finally清理错误丢失receipt的风险，尚未运行科学控制。
- `ddc8bb3a`：只读scale/SSD来源审计与准备；已有moving/projection结果不改标成本次数据。
- 上一真实控制：轴projection替换未改善总gradient，scale逐位不变；同点native缓存仍缺失。

本次serial与factor分组的变化极小，保留该排除结果，不重复这一假设。后续若需检查输入缓存或系数Jte，另立有限计划和授权。原冻结源码、环境及私有输出保持原身份；共享说明和main由根任务统一整合。

## 7. 参考文献、原实现与许可

成熟实现：[registration.py](../../src/fnit/fnirt/registration.py)、[spline.py](../../src/fnit/fnirt/spline.py)；自有有限worker与边界见[原准备叶](../fnirt_cpu_scale_order_prepare_20261006/README.md)。

原软件：[FSL FNIRT代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007)。

本次只公开自有说明、标量和哈希元数据，遵循[第三方声明](../../THIRD_PARTY_NOTICES.md)。没有新增外置权重、模板或依赖；私有checkpoint、MRI、凭据、完整官方源及二进制不进入本报告。
