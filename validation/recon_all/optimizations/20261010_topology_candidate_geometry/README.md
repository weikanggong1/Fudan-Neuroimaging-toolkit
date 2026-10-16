# 首个拓扑候选的连续几何真实回归

本报告只验证一个完整候选的动态重连、25次平滑、40次MRI匹配和法向。
它不是GA搜索终点、原始T1完整运行或整体等效验收。候选入口与具名参数见
[中文七节说明](../../../../docs/recon_all/TOPOLOGY_CANDIDATE_GEOMETRY.md)。

## 实际输入与版本

- 公开数据：ds000114，sub07 ses-test左半球；上游为79a41cdd原始T1链。
- 初始状态：110534顶点、216722面、650530有序邻接项、4723完整边。
- 域：67 inside、61 border、125完整hull；本次all模式，无overlap表。
- 仅retess before、同一uint8脑MRI和四项明确冻结均值参与生成。
  两份smooth/after参考只在全部候选NPZ写出后读取。
- geometry模块实际SHA：18b75b4b273dedfb83cea8e8c011dbccc41e989b61d8f59bc9495f542d05f2b0。
- 调用已提交HEAD依赖4b20a92f…；未提交wrapper89d7423f…不属于必要依赖。
  两个导入函数AST相同，新独立HEAD-only完整回归依然0bitdiff。
- 固定参考源码：FreeSurfer d932c45b7941662ea380a05efef580568b98d41a；
  仅在独立benchmark执行只读观察器。原程序、观察器及输入SHA在报告中。

## 精度、时间及边界

accelerator-b同主机，CPU8–11、总4线程、CUDA隐藏；共享负载，没有GPU计时。

| 范围 | 冷(s) | 重复(s) | 实测一致性 |
|---|---:|---:|---|
| 重连和完整面发现 | 6.091473 | 4.052848 | 完整有序216860面及边状态相同 |
| 25次平滑 | 1.111635 | 0.403258 | 全110534坐标0bitdiff |
| 40次MRI匹配 | 1.466715 | 0.613045 | 全坐标0bitdiff，P99/max=0mm |
| 完整面和所选顶点法向 | 1.000890 | 0.533573 | 99个所选顶点法向0bitdiff |
| 完整API，含内部校验 | 9.705629 | 5.642997 | 全8项拓扑数组字节相同 |
| NPZ写出，另计 | 0.933853 | 0.932518 | 两次自产数组重复一致 |

加载0.128497秒；一次进程包含两次API、导入/加载、读写和事后比较，墙钟
21.933309秒。参考观察程序5.321940/5.144005秒包含上游与捕获，范围不同，
不能给此候选API加速倍数。GPU峰值、整例耗时及隔离环境未测，不能写0。
650882邻接项、650580入射项和角点、selectedV/F与有序面全部逐位一致。

![实际MRI叠加与候选形变](candidate_geometry.png)

图是实际MRI体素Z=49的叠加（69个邻近切片顶点），中间是候选实际形变，
最大4.053288mm；右侧同输入参考误差0mm。它不证明最终形状或质量合格。

## 复现与产物

以下命令使用自己的授权隔离数据路径，全部参数显式；均值仅为本次冻结先验。

```bash
# 读取唯一before；候选保存后才读取两份参考，output必须为空且不存在。
python tools/benchmark_recon_topology_candidate_geometry.py \
  --before /workspace/benchmark/retess.before.bin \
  --mri /workspace/inputs/brain.mgz \
  --white-mean 85.10910034179688 \
  --gray-mean 62.694217681884766 \
  --k1-mean=-0.09168962389230728 \
  --k2-mean=-0.03310907259583473 \
  --smooth-reference /workspace/benchmark/match.smooth.bin \
  --after-reference /workspace/benchmark/match.after.bin \
  --output /workspace/diagnostics/new-candidate
```

- `reports/geometry_v2/report.json`：可调用API当前真实全数组比较及完整计时。
- `reports/geometry_v2/process.json`：包含解释器导入、读写和比较的进程墙钟。
- `reports/geometry_v1`：提取API之前同算法链回归，单独保留旧绑定。
- `reports/dependency_head`：仅替换成Git HEAD依赖的完整重复门。
- `reports/match_only_v3`：已有smooth→match函数同输入连续及冻结回归。
- `reports/*observer*`：两次独立参考的程序/捕获/构建哈希与时间。
- `reports/PUBLIC_EXPORT_MANIFEST.json`：白名单数值叶保真和文件SHA。
- `local_contracts.json`：四个本地输入/有序incident合同，不替代MRIbenchmark。

原MRI、NPZ、源码构建树、内部命令和完整收据只留授权目录。公开JSON只含
安全指标和哈希；所有指标数值/类型保真，脱敏前原收据SHA保留。没有发布
原程序、私人数据、许可证或服务器地址。

## 当前未覆盖

当前候选入口使用完整冻结基底、边表及校准均值；尚未在本接口自产base/DVS、
全局prior、SFF与五项fitness、GA种群/交叉/变异/停止和最终质量。
当前全网格行/集合构建每候选仍有成本，不能据此假设完整GA高效。
下一步用自产base及一次/defect校准接完整候选链，独立验证最终连通、流形、
自相交、解剖形状及原始T1整例；指标等效为not_assessed。
