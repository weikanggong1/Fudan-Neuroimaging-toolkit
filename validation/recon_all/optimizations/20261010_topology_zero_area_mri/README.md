# 首次真实拓扑候选的零面积诊断修复

## 1．功能与范围

复用 `topology_mri_likelihood.patch_mri_likelihood()`，修正成熟子函数在接入完整 GA 时发现的接口 bug：原程序的首次重连候选尚未刷新面积字段，99 个修改顶点与 138 个面面积全部为零。MRI 两个主似然仍有限，但面积加权诊断按源算法计算 0/0。旧函数错误地抛异常，导致合法的中间候选无法继续评分。本次保留原 IEEE 诊断与位模式，不填零、不改变主似然、候选几何或最终质量门。

## 2．输入、输出与 Python 调用

完整参数及 surface RAS/mm、面积 mm²、MRI tkregister RAS→体素约定见 [MRI 子项说明](../../../../docs/recon_all/TOPOLOGY_MRI_LIKELIHOOD.md)。输入包括当前有序几何/法向/面积、完整修改索引、同输入 uint8 brain、已在初始统计生成的全局联合 PDF。候选不根据新的 marked 状态重建全局 PDF。返回两个有限主项、面积诊断、`*_defined` 布尔值及源 float32 `*_bits`；JSON 未定义诊断编码为 null，保留 `0xffc00000`，不是误差为零。

## 3．复现命令

```bash
# 初始统计先通过现有 FNIT 函数生成全局 PDF；全目录必须是新的。
python tools/benchmark_recon_topology_mri_likelihood.py \
  --capture benchmark/reference/statistics.bin \
  --brain benchmark/input/brain.mgz \
  --output benchmark/new-initial \
  --threads 4 \
  --capture-role initial-statistics

# 重连候选复用上一步自产 PDF，而不接收参考输出标量。
python tools/benchmark_recon_topology_mri_likelihood.py \
  --capture benchmark/reference/patch_statistics.bin \
  --brain benchmark/input/brain.mgz \
  --output benchmark/new-patch \
  --threads 4 \
  --capture-role retessellated-patch \
  --joint-pdf-input benchmark/new-initial/candidate.npz
```

各路径分别为固定源码同输入检查点、同一 uint8 MRI、新输出及初始自产 PDF。`threads` 为总 CPU 线程预算，默认 4；affinity 由运行者固定。脚本保存候选后才比较原标量/PDF；失败保留报告并抛异常。二进制检查点和影像仅留在授权 benchmark 存储，不回灌生产。完整当次 driver、源码逐文件 SHA 和原参考收据见 `reports/`。

## 4．原软件对照

固定 FreeSurfer `d932c45b7941662ea380a05efef580568b98d41a` 的 `mris_fix_topology -ga -seed 1234 -threads 4`，只在隔离 benchmark 中运行。观察器在首个完整 fitness 调用处捕获；它是参考诊断，不是完整 GA 运行。两次参考的初始统计、候选统计、体积输入、brain 和 fitness 收据 SHA 均相同。

## 5．真实数据精度与耗时

公开 ds000114/sub-07/ses-test 左半球，上游为已完成原始 T1 链的 `79a41cdd` 冻结几何。同一主机 Xeon Platinum 8369B，四核 affinity 120–123，NumPy/Numba CPU；未使用 GPU，无新依赖。实际候选源 SHA 为 `e2867b8f3faf939cd8d7e057a62c5dd0de71e9c9b7783a4692709681e9b2207a`；工作树快照 HEAD `9adaaff50f175c3cd75bc9b8f1da73af664b9e3e`，576 个实际上传文件全部按 SHA 验证，报告没有将其称为干净提交。

| 冻结阶段 | 初始完整统计 | 首个重连候选 |
| --- | ---: | ---: |
| 修改顶点/面 | 108,538 / 216,722 | 99 / 138 |
| 面/顶点主似然不同数 | 0 / 0 | 0 / 0 |
| 面/顶点面积诊断 | 有限，差 0 / 0 | 两项均原 NaN，float32 bits 一致 |
| 联合 PDF / MRI 矩阵不同数 | 0 / 0 | 0 / 0 |
| 自有重复调用 | 全字段一致 | 全字段一致，包括 NaN bits |
| 首次完整 MRI API / 热调用秒 | 0.425138 / 0.115265 | 0.061067 / 0.060614 |
| 读取检查点/MRI并写候选 NPZ 的脚本段秒 | 1.760388 | 1.447977 |

候选 MRI 主值 `face_ll=-7.894798278808594`、`vertex_ll=-7.215606212615967`。11 个边界契约在本地和 CFFF 通过。driver 的校验、契约、两段加载/计算/读写合计 9.879783 秒，从工具导入后开始计，不包括解释器启动/导入及最终 COMPLETE 写入；各 API 秒数不包含完整 GA。参考观察器包含其余几何拟合、体积等步骤，不能据此计算 MRI 加速比。未测原始 T1 整例、指标等效或 600 秒目标；此修复也未改变 TF32/半精度策略。

## 6．最近记录

原 v5 的 `c3932110…` 初始完整统计通过收据保留；它不能覆盖本次零面积候选。新增 `error_model="numpy"` 使用原顺序 IEEE 除法，主项仍必须有限。未定义诊断单独记状态和位模式；NaN 的 max/P99 误差为 null，不能写成 0。候选完整输入与输出分别绑定 SHA，更新没有放宽主似然、坐标、标签或最终网格验收标准。

## 7．参考

- [固定 MRI 子项源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)：面/顶点 log-likelihood 和 GA 完整 fitness。
- Fischl B, Liu A, Dale AM. Automated manifold surgery. IEEE Transactions on Medical Imaging. 2001. [DOI](https://doi.org/10.1109/42.906426)。
