# pial 逐面对约束投影

## 1．功能简介

`resolve_projected_intersections` 是显式实验接口，为原完整SOAP停在固定点的自产pial尝试逐相交面对分离。它使用Python/NumPy和SciPy约束求解，复用FNIT成熟的Numba CSR候选与源/F64两套碰撞谓词；不调用原版阶段程序，不替换默认SOAP。

每次暂时保持对方面坐标，尝试其两个法向半空间和三个边外半空间。移动面的0/1/2邻接环通过最小位移投影调整；解析Jacobian同时约束全部受影响面和顶点。显式的可转动平面模式同时优化单位法向与偏移，五个方向作为初值。量化到float32后重新按完整谓词验证，仅接受两套阳性面对都严格减少且没有新面对的候选。

```mermaid
flowchart LR
 A[自产末态与明确固定域] --> B[全量两套阳性面对]
 B --> C[逐面对及0/1/2邻接环]
 C --> D[五种预定半空间约束投影]
 D --> E[float32量化后的全受影响面检查]
 E --> F[两套面对严格子集]
 F --> G[固定点及完整全局复检]
 G --> H[保留通过或残余失败结果]
```

## 2．Python调用与全部输入输出

```python
from fnit.recon_all.place_surface_residual_projection import resolve_projected_intersections

candidate_vertices, diagnostics = resolve_projected_intersections(
    vertices=pial_terminal_vertices,  # (N,3) float32，自产清理末态，surface RAS/mm
    faces=ordered_faces,  # (F,3)整数，顶点编号与顺序保持
    fixed_vertices=ripped_vertices | ~cortex_mask,  # (N,)bool，固定在进入时坐标
    white_vertices=own_white_vertices,  # 自产同网格(N,3)，只检查结构，不读官方参考
    max_displacement_mm=1.0,  # 相对进入时坐标的累计位移上限
    minimum_area_ratio=0.2,  # 全部受影响面相对初始面积下限
    clearance_mm=0.002,  # 三个面角点到分离半空间的目标距离
    max_sweeps=4,  # 最多四次完整面对遍历
    optimizer_iterations=100,  # 每次局部SLSQP最多100次迭代
    diagnostic_pair_limit=None,  # None处理全部；1冻结初始首面对，不补入其他面对
    projection_mode="fixed_halfspaces",  # 固定五种半空间；另一个显式模式单独验证
)
```

| 参数 | 结构、默认值与限制 |
| --- | --- |
| `vertices` | 必选有限NumPy float32 `(N,3)`，surface RAS，毫米；不会原位修改。 |
| `faces` | 必选整数`(F,3)`，编号在`[0,N)`且每面三个不同顶点；全部有序面保持。 |
| `fixed_vertices` | 必选bool`(N,)`，调用者合并rip及全部cortex外点，保留进入函数时全部坐标元素；不重新pin到white。 |
| `white_vertices` | 必选有限同形状浮点`(N,3)`自产white，保持同索引空间；当前投影不使用其坐标选择方向。 |
| `max_displacement_mm` | 正有限数，默认1mm，累计上限；逐次位移不各自重新获得1mm预算。 |
| `minimum_area_ratio` | `(0,1]`，默认0.2，每个受影响面的面积相对初始值；原法向点积还必须为正。 |
| `clearance_mm` | 正有限数，默认0.002mm；属于候选投影参数。 |
| `max_sweeps` | 正整数默认4；每轮按确定的排序面对遍历，bool无效。 |
| `optimizer_iterations` | 正整数默认100；每次SLSQP预算，bool无效。 |
| `diagnostic_pair_limit` | 默认None完整处理；正整数冻结初始完整阳性集合按编号排序的前几个面对，接受后不补入其他面对。返回`limited_initial_pairs`及`scope=limited_diagnosis`，不是生产完整清理。 |
| `projection_mode` | 默认`fixed_halfspaces`保持固定方向；`free_separating_plane`以同五方向初始化可转动单位法向和偏移。对方三个角点全部在另一侧，顶点目标最小位移、方向正则权重`1e-6`、单位法向误差≤`1e-9`，原质量门不变。其他字符串无效。 |

返回新float32`(N,3)`坐标和JSON字典。字典包含参数、每次投影状态/迭代/耗时、受影响顶点/面数量、全部拒绝原因与接受记录、两套面对前后计数、全局面对集合/固定域复检、位移均值/P99/最大值、函数完整墙钟。`passed_zero_intersections`仅表明两套自相交检查为零；`failed_residual`保留剩余失败，不能当作生产成功。`overall_metric_equivalence`为`not_assessed`。

SLSQP使用F64和解析Jacobian；固定目标精度`ftol=1e-9`、可行性余量`1e-9`只决定是否进入真实量化后复检，不放宽面朝向/面积/位移/两套碰撞谓词。无GPU计算，不更改TF32或启用半精度。非法输入、参数或退化初始面抛ValueError；局部更新与完整全局检测不一致或固定点变化抛RuntimeError。上述候选算法参数不属于整例指标等效标准。

## 3．命令行与冻结阶段验证

不新增生产CLI。复用同七输入benchmark工具，协议明确指定`independent_projected_pair_solver_v1`或`independent_free_plane_pair_solver_v1`与相应模式，参数需与本轮预先固定规则一致。初始单面对定位`diagnostic_pair_limit=1`和完整处理`null`各自保留新输出目录、协议与源码SHA。

```bash
# 给四核独立进程；包含导入、哈希、读写、候选、全部QC和距离。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python tools/benchmark_recon_pial_residual_solver.py \
  --frozen-source frozen-fnit/src \
  --overlay-source projected-overlay/fnit/recon_all \
  --diagnostic-case own-terminal/01-cli-enabled \
  --input-hashes seven-input-hashes.json \
  --protocol predeclared-projected-protocol.json \
  --output-directory new-projected-diagnostic \
  --distance-module reference_surface_distance.py \
  --threads 4
```

七输入、benchmark各参数、输出形状与失败行为见[共用阶段工具说明](PIAL_LOCAL_RESIDUAL_SOLVER.md#3命令行与完整阶段诊断)。候选保存并完成QC后可读取已有隔离参考，禁止读取参考来选择移动、参数或步长。新目录保存`candidate.private.npy`、`candidate.pial`、`report.json`、完成收据和独立冷CLI记录；影像、坐标和私人路径不发布。

## 4．对应原软件

原完整清理是固定`mris_place_surface --pial`内部的`MRISremoveIntersections`等步骤，没有等价的独立CLI。本投影是独立算法，不声称逐字节复现；原顺序实现继续保留。[原固定点诊断](PIAL_RESIDUAL_CLEANUP_DIAGNOSTIC.md)与本算法分开报告。

## 5．精度、耗时与真实数据

本轮真实单面对候选已完成，残余仍为32面/35顶点，坐标全元素不变。首面对只有一个可动角点，预定15种投影中12次固定角点半空间不满足，3次求解约束不可行。随后只读几何确认：唯一允许的法向分离至少需要1.652002mm，超过执行前固定的1mm累计预算；其他四个半空间排除固定角点。它只解释该候选族在首面对失败，不能证明任意分离算法无解。没有据此放宽质量门。

最小定位完整API300.798299秒、冷CLI301.682361秒、求解函数112.324403秒，单子进程峰值RSS5,998,170,112字节。全部source/F64面对集合复检一致，7,890固定点零移、有序面/闭合/非流形和面朝向保持；完整同索引距离0，双向点到三角面最大约`5.17e-17 mm`。无参考输入、无GPU计算，显存未测为null。[最小定位报告](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/projection_min_report.json)与[冷CLI](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/projection_min_cli_time.json)保留全部失败。

最小协议SHA`b5186bf9e6b5858c5f6b4428b6d38922be33bee3be81da6a91b2a70f8680a424`，投影helper SHA`5785f33f4ffe769c28eff75308caade554c47f9fa74b51c49280c638ed3d01e7`。随后同代码、相同约束的完整40面对阶段已经完成；仅将预定`diagnostic_pair_limit`设为null，完整协议SHA`5c685eae560f1d68a27dfd2044948587fa63a1f437b94c1388b9a66346dfdf85`。

完整候选source/F64面对80/40→68/34，相交面32→29、标记顶点35→32，2次合格接受。但仍有残余，质量失败。只改变2顶点、6坐标元素；均值位移`9.9522e-6 mm`、全体P99=0、最大0.626105mm，改变顶点P99=0.623042mm。完整双向点到三角面最大分别0.297078及0.130203mm；不能只用很小的全体均值描述局部变化。7,890固定点全元素不变，有序面、闭合/连通/非流形结构保持，零新增面对、零初始法向翻转，最小面积比0.474522。white/pial相互穿越仍未测，不可接为默认生产修复。

完整候选API430.546099秒（最终报告写出前）、冷CLI431.716168秒、求解函数232.368419秒、单子进程峰值RSS6,002,274,304字节。共享节点单次阶段时间，没有整例加速结论。[完整报告](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/projection_full_report.json)保留1654次候选的全部接受与拒绝，另有[冷CLI收据](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/projection_full_cli_time.json)。

可转动平面v1有限诊断source/F64面对80/40→44/22，相交面32→19、标记顶点35→24，2次接受、改变2顶点。最大同索引位移0.576458mm，改变顶点P99为0.572961mm；完整双向点到三角面最大0.554600/0.039372mm。7,890固定顶点全元素不变、无新面对、零法向翻转，最小面积比0.779105，仍有残余，质量失败。API321.564687秒（最终报告写出前）、冷CLI322.702184秒、求解119.137527秒、RSS5,990,699,008字节；显存未测为null。[v1报告及范围审计](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/free_plane_per_sweep_v1_report.json)保留全部候选。

该v1的实验接口存在范围bug：`diagnostic_pair_limit=1`实际每轮重新取当前首面对，四轮中尝试了三个不同面对，并非冻结初始一个面对。原源码、协议、结果完整保留；旧固定半空间最小例因零接受而没有补入其他面对，完整模式None也不受影响。现已修复为冻结初始面对并增加两个互不相连交叉组的回归，确认接受后不会补入另一组。它不影响成熟SOAP或生产默认路径。修正后初始单面对v2绑定helper SHA `47831ec2aa9de39934bb17df0f0352e2d398fc114227375ff02ba3c859df7440`、协议SHA `d6605a7e6de3c0167eee29a85d725f057673fea9140cf07fdb23f2afb67bd65f`，真实回归结果单独记录，不重标v1。

修正后的v2已完成，报告确认只尝试初始首面对`25917/132048`、1次接受，改变1顶点/3元素。source/F64面对80/40→48/24，相交面32→20、标记顶点35→25，仍未归零，质量失败。7,890固定顶点全元素不变、全部有序面不变、无新增面对或初始法向翻转，最小面积比0.932124。同索引最大位移及改变点P99均为0.576458mm，全体P99=0；完整双向点到三角面最大0.554600/0.039372mm。首面对的一次更新同时移除了同一活动顶点关联的其他面对；这不改变只选择一个初始目标的范围。

v2求解131.622137秒、API338.466451秒（最终报告写出前）、冷CLI339.830810秒，单子进程峰值RSS5,981,442,048字节。实际冻结包基于`5d1cdb5ef7e9c50932e63f2361c8640d66ed8384`及逐文件SHA，四核同节点共享负载，不作为整例加速证据。[完整v2报告](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/free_plane_initial_min_v2_report.json)、[冷CLI](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/free_plane_initial_min_v2_cli_time.json)、[完成收据](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/free_plane_initial_min_v2_complete.json)与[硬件/精度](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/hardware_context.json)保留失败和全部接受记录。GPU显存为null；white/pial穿越、其他半球/被试及空目录整例未测。

之前完整方向候选v1/v2仍32面/35顶点、零合格候选，详见[完整失败和枚举回归](../../validation/recon_all/optimizations/20261011_pial_local_residual_solver/README.md)。本投影继续保持同一固定域、1mm/20%与双谓词无新增规则。异常局部及图像哈希见该页，最小定位没有改变几何，仍对应相同局部图。没有整例或整体指标等效结论。

## 6．最近版本

- 新增兼容默认关闭的 `crossing_guard=False`。显式开启时完整检查受影响面，拒绝新增 white–pial 内部穿越或非内侧壁精确接触，最后重做全量集合检查；初态非零与末态零穿越分别记录。数学语义、真实只读回归和独立候选范围见 [white–pial 检查](WHITE_PIAL_CROSSING_QC.md)。默认投影几何行为不改，尚未作为生产接线。
- 修复接受行复制前缺少 `decision="eligible"` 的报告字段；不改变候选坐标、接受顺序或质量约束。

- 原SOAP：保留完整原算法及同输入重复性证据。
- 局部方向候选v1/v2：失败全部保存；v2只复用成熟CSR，面对集合真实回归一致。
- 投影v1：五个分离半空间、完整受影响面约束、解析Jacobian和独立候选，先运行一个真实面对，再决定完整阶段与生产接入。测试4/5子测试通过8.35秒，属于模拟合同。
- 可转动单位分离平面v1：保留原固定域和全部质量门，有限诊断减少相交但失败；原范围bug与冻结源码保留。合同5/6子测试通过8.54秒，不作为真实结果。
- 初始面对限制修复v2：冻结初始集合、接受后不补入其他组。合同6/6子测试通过7.73秒；真实首面对回归减少32→20相交面但失败，另绑定代码与协议，不接生产默认。

## 7．源码与参考

- [FreeSurfer固定曲面形变源码](https://github.com/freesurfer/freesurfer/blob/d932c45/utils/mrisurf_deform.cpp)。
- [SciPy SLSQP接口](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。
- Fischl B. FreeSurfer. *NeuroImage* 62,774–781 (2012). [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
