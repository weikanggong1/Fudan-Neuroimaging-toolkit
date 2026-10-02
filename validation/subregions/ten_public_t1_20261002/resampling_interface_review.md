# 分区统计与脑图接口只读核对

## 结论与当前证据

未发现本次冻结的统计/绘图接口中会必然造成 Dice 失真的坐标反向、右侧重复编号、遗漏 ROI、空标签记为满分或显示像素替代原网格统计问题。此核对没有重新读取影像，没有运行拟合，也没有修改冻结脚本。

已完成的 10 例真实 FNIT raw 数值审计提供以下依据：全部 native 输出与各自真实 T1 的 shape 一致，affine 最大绝对差均为 **0**；native 和四种 HR 结果的实际标签编号均属于声明的 atlas 支持集。`sub-02` 的真实 shape 是 `(156,256,256)`，其余样本是 `(256,156,256)`，两份脚本都通过完整 affine 使用物理空间，没有写死 1.3 mm 所在轴。实际 HR spacing 为脑干/丘脑 0.5 mm，双侧海马/杏仁核约 0.3333300054 mm。只读元数据复核见 [resampling_actual_raw_metadata_checks.json](resampling_actual_raw_metadata_checks.json)。

这些依据验证 FNIT 输入/输出与自身编号约定，**尚不验证当前 10 例官方输出的真实跨方法 Dice**。

## 逐项检查

| 项目 | 实际实现与检查 | 结果 |
|---|---|---|
| Native 官方 → raw | Analyzer 的 target 是该病例公开 T1 的 `(shape, affine)`；plot 使用同一 target、同一 `order=0, mode='constant', cval=0`。Nibabel 的映射为 `inverse(official_affine) @ raw_affine`，不是反向乘法，也没有使用 FreeSurfer 的 tkregister 坐标替代 `.affine`。 | 未发现方向/轴序接口错误。 |
| 同阶段 native | Target 是该病例 fresh `norm.mgz`，FNIT stage 本身须与实际输入网格一致；它与 raw 原网格分开报告。 | 未混合两种分辨率。 |
| HR 共同网格 | 以该结构官方 HR affine 为基准，计算官方、FNIT raw、FNIT stage 图像全部角点在其网格中的范围；对范围向外取整数，平移 `A[:3,:3] @ integer_lower`，原三列轴向/spacing 保持不变。两种 FNIT 模式用同一 union 网格。 | 保留官方整数相位，无分数平移搜索；没有根据 Dice 优化坐标。 |
| 背景裁切 | 统计 helper 在共同网格之后，只移除全部方法均为背景的外部区域；任一方法的假阳性都会进入并集。 | 不会通过裁切删掉不匹配体素。 |
| 右侧 HA 编号 | Analyzer 和 plot 都只给官方 `hippo-amygdala-right` 非零标签加 `10000`；FNIT 已使用全局编号，offset 为零。真实 canonical metadata 的 28 个右侧 HA 标签都与对应左侧 ID 恰好差 `10000`。 | 未发现重复加 offset 或给背景加 offset。 |
| 110 ROI | Canonical 包含脑干 4、丘脑 50、海马左右各 19、杏仁核左右各 9，共 110 个唯一 ID。统计按各 ROI 正确的官方来源图选择标签；其他 atlas/背景不进入该 ROI。 | 未发现缺失一侧或将整幅 aseg 当作目标分区。 |
| Dice/NA | 每个 ROI 用 `2*intersection/(official_count+fnit_count)`。双方为空为 NA；一方为空且另一方非空为 0。Confusion matrix 保留背景与前景之间的不一致，不将背景真阴性加入 Dice。 | 未把双方空标签记为 Dice=1。 |
| 宏平均 | 每例先在该例网格上计算 ROI/家族/ALL110 指标；跨受试者的 mean/median/SD 对有定义的病例等权汇总，不汇总全部病例体素后当作宏平均。输出同时保留 planned=10/9、defined 数和 NA 数。 | 分母明确；NA 没有被伪造为 0 或 1。 |
| 脑图与统计 | Plot 在显示重格式化之前，逐 110 ROI 重新验证官方/FNIT/交集计数和原网格 Dice 与 analyzer 相同。显示采用同一 raw 数组的局部 RAS 轴向重格式化；两种标签都最近邻，灰阶窗与切片一致。标题 Dice 仍是原网格统计，不在 0.6 mm 显示像素上重新计算。 | 图像空间与统计空间区分清楚。 |
| 中位/最差选择 | Plot 重新计算 ALL110 官方体积加权 Dice，并核对 `case_rows`、ranking eligibility 和 analyzer 的 median/worst。完整 raw 模式不因 stage 模式失败被丢弃；10 个计划病例和无法排名的原因全部公开。 | 未发现按外观选病例或隐去失败的接口问题。 |

## 必须按定义解释的指标

1. **官方体积加权 Dice 并非等权 110 ROI 平均。** 小核权重小；官方原网格没有该核、但 FNIT 有假阳性时，该核的 Dice 是 0，但其官方权重也是 0。这个加权指标本身不会惩罚这种单侧假阳性。因此必须同时查看逐 ROI、`mean_label_dice` 和 `micro_label_dice`，不能只用总体加权分数描述全部细核精度。
2. **NA 的跨受试者均值分母是有定义的受试者数。** 若某 ROI 只有 8 个有定义结果，其均值是这 8 个值的均值，planned 仍为 10，另报告 2 个 NA；不能将该值表述为 10 个有效测量的均值。失败病例也保留计划行和原因。
3. **ALL110 是按独立 ROI 计数的标签指标。** 官方四个结构图分别提供各 ROI，不先用人为优先级合并成一幅互斥标签图。其总体加权/微平均指标不能改称整脑前景 Dice。
4. **原始 T1 网格的离散化是测量条件。** 该队列原图约 1×1.3×1 mm，最近邻映射可能让很小的核只剩几个体素或消失；这不是绘图生成的分数。原网格和 HR 结果应分别保留。

## 尚待 fresh 官方结果完成的核对边界

- 当前审阅不读取尚未完成的官方影像，不能保证其实际文件头/体素内容均满足预期。Native 统计以文件 `.affine` 为物理空间真值；完整结果阶段仍须核对官方 FSvoxelSpace 网格与其 fresh `norm` 的关系并查看真实脑图。
- HR 是整数相位的 union 扩张，未额外加入 registration；当前代码没有单独断言“官方 HR 在其整数扩张网格重采样前后每个标签计数完全守恒”。正常内部 ROI 不应受 FOV 扩张影响，但靠图像边界的非零标签及浮点边界取值仍需最终结果检查。这里没有发现本轮实际丢失标签的证据，不能据此归因任何尚未计算的低 Dice。
- Plot 与 analyzer 使用相同的 nibabel 重采样接口；它的 110 计数复核可发现统计/绘图不一致，但无法独立证明上游官方文件头绝对正确。最终官方输出审计和真实空间图仍承担这一验证。

## 冻结身份

- `analyze_cohort.py`：`ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc`。
- `plot_cohort.py`：`f135af099a67d543bfbefbbabbe8a71f1c6ba044133a7b5767d8368787e5b4ab`。
- 统计 helper：`beef69670191f89ebf3a0d71395c274b54b7c61d02ed096abebebd7a5172f842`。
- 真实 raw 审计：`820774cb5af76224b427f7495d8bdcbfc75f87473e3d95d3b5018c6fdb476059`。
- gpucw1 使用 nibabel 5.4.0、NumPy 1.26.4、SciPy 1.11.4。本机 nibabel 为 5.4.2，但本次检查的 `resample_from_to` 与 `MGHHeader.get_affine` 函数源码哈希和服务器完全一致，分别是 `e3bf6d6c195711b3a72faf9a7854567154d45243e7065f716465da1d8ded5cb7`、`3c30492d3d88dd0bc7295c49490a8d71be7a783bbda7ed4cdc6a8a9d33011855`。该检查只读取库代码与版本字符串。
