# 局部 inverse 的实图结果与剩余原因

本次只分析已保存结果与固定源码。两次配准 API 已执行；之后只补做原先未开始的保存输出评分，没有重复配准、改变阈值或再采样诊断。生产默认和成熟 inverse 没有修改。

## 已直接证明

1. 固定 ITK 4.13.2/VNL Float inverse 的项顺序与成熟通用 cofactor 不同；六个原保存矩阵的 18 项 C++ 源定义合同通过。新刚性 API 确实调用局部函数 **26 次**，模块 SHA 为 `41eeb53d…`。这不是仅换一个未使用的参数。
2. 新刚性和仿射的四个 MGH/LTA 均已保存，SHA 与旧 B 不同；源体素/shape/dtype、LTA shape 与自身 header 几何检查通过。
3. 原 20 项门仍为 **17/20**。仿射 warp relL2 从 `2.5881264790108407e-5` 到 `2.556642902132947e-5`，约降低 1.22%，仍高于 `1e-5`；两阶段支持集差仍各 1 体素。刚性 warp relL2 从 `2.4091169053125693e-6` 到 `2.5840940291760602e-6`，仍在原门内。不能把所有指标称为改善。
4. 新最终 LTA/组合矩阵仍与原软件有微差，133 点 RMS/max 仍通过原门。两阶段 MGH 字段仍分别 11/13 exact，差异仍在方向和中心字段；不是最终脑区标签或厚度的验收。
5. 评分采用原 B 的 sampler 和成熟 inverse，双方原始源体素、目标网格、chunk/dtype 和采样函数相同。改变局部配准 inverse 后，没有同时改变评分算法。

## 保存输出的几何与 warp

评分定义仍为：

`warp = resample(original_values, target_shape, old_inverse(saved_moving_geometry) @ target_geometry)`。

因此已保存几何仍有微差，经同一采样函数得到剩余 warp 差异。该关系有实际输出/源码证据；它不能单独指出最早在哪一层或哪一次更新分叉。

源码包含原 half-voxel 支持、整数 `FEQUAL` 分支和 Float 存储，提供了微小坐标变化影响非零判定的机制。本次评分只保存支持集差的**数量**，没有保存差异体素位置、两侧值或邻域。仍不能证明它发生在与旧 B 相同的位置，不能指定某个 shortcut 为根因，也不能把那一个值称为已量化的极小尾数。

固定阈值二值 mask 的体素数/交集/Dice 是另一组统计；相同计数不等于逐体素分割相同。官方使用 `--mapmovhdr`，本实验没有独立官方重采样 warp，所以不把共享 sampler 评分冒称官方插值器的同矩阵比较。

## 这次排除了什么，尚未排除什么

局部 inverse 实际进入配准，并使输出变化；但原三项失败没有消失。这证明**仅替换这处 Float 运算顺序不足以通过本例验收**，并未证明它在安装 binary 内逐指令相同，或其他阶段已经逐点一致。

原 B 与本次仍包含源码审计所列的潜在数值边界：Double 质心求和、Schur/Double inverse 库、VNL Float QR 与 Torch QR/reduction、每轴 Float 图像准备、更新累积及 MGH 保存。公开的实际源码位置见 [原 B 审计](../rigid_affine_20261006/prepared/SOURCE_AUDIT.md)。原软件内部 same-A/b、各层图像、初始质心、halfway/internal Double matrix 的轨迹未在本次捕获。不能仅凭最终差异把 QR、Double 或保存格式定为唯一 bug。

新仿射输入也继承自身新刚性 MGH，和官方自己的刚性 MGH 存在微差；这不是两种 affine solver 从同一内部状态开始的严格对照。共享评分已保留实际保存 header，未做交换变量评分、目标标签反拟合或最终输出修补。

## 报告序列化失败单列

原 finite controller 的刚性 phase RC0；仿射 phase 在四个输出都已保存后，写 JSON 时遇到 `np.int32` shape 的 `TypeError`，原 controller RC1。`atomic_report` 先序列化再创建文件，所以没有截断的仿射 JSON 可回读。

本次只把现有输出交给原冻结 scorer；原控制器没有改标为成功。仿射内部 API 时钟、counter 数值、flags receipt 和 RSS 均 **NA**，不从 source 预期或进程时钟重建。成功持久化的刚性 counter=26、flags 门通过，score 的 flags 和 19+518 前后源/输出门通过；仍不能宣称原全部 runtime 门已经验证。

[metadata_shape_fix.patch](metadata_shape_fix.patch) 只展示后续报告字段的一行显式 `int` 适配，未作用于原冻结源码。局部 JSON 对象合同复现该类型错误并验证值/类型/roundtrip；没有重新运行任何 MRI。

## 最小后续定位建议（本次未进行）

优先取得一个真实同状态的最早中间数组或小矩阵，再比较原软件与 FNIT 的坐标/运算边界，区分准备差、Double inverse/halfway、QR/reduction、更新与保存。支持集问题需先保存那一个差异体素的坐标和两侧值，再判断是否为边界或近整数分支；当前数量字段不能给出这一证据。

这些建议不改变原 `1e-5`/support0 门，也不将本例失败推广为最终 ROI 一定失败。当前不接入生产、不做另一例注册、完整 GEMS 或 GPU benchmark。
