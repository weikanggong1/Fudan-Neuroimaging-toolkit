# pial 局部残余候选：完整失败与枚举回归

## 验证范围

公开真实T1的同七前置自产pial末态，95,045顶点、190,086面。不是原始T1连续整例。原SOAP与两次同七输入参考坐标全元素相同，但源谓词和独立F64都残余32相交面、35顶点；严格复现不能代替网格质量通过。

本轮显式候选不会调用原版阶段程序、读取参考选取移动、改变标签或替换默认SOAP。rip及全部cortex外点冻结在实际进入坐标；原末态cortex外已有954个坐标元素不同于white，不重新pin。

## 运行前固定规则

[protocol.json](protocol.json) SHA-256为`827f8868ec090b40bad17ba9e21c36d2650fecf1a234f1175c50b4ca473d8ad8`。累计位移≤1mm，受影响面面积≥初始20%、初始法向点积>0；两套阳性面对都严格减少且不能新增。0/1/2邻接环、两种方向及七个步长比例均提前固定。它们属于算法参数，不是整例指标等效阈值。

## 实测结果

| 项目 | v1 | v2 CSR |
| --- | ---: | ---: |
| source/F64 相交面，前→后 | 32→32 / 32→32 | 32→32 / 32→32 |
| source有向/F64无向面对，前→后 | 80→80 / 40→40 | 80→80 / 40→40 |
| 接受候选 / 坐标差异元素 | 0 / 0 | 0 / 0 |
| 完整求解函数，秒 | 335.525912 | 131.461099 |
| 阶段API，最终报告写出前，秒 | 591.221743 | 343.467511 |
| 冷CLI，秒 | 未采集 | 344.770774 |
| 单诊断进程峰值RSS，字节 | 5,979,979,776 | 5,969,301,504 |

42次候选拒绝原因：面朝向38、两检测未严格减少3、新面对1。两版保留失败：`execution_status=complete_diagnosis`、`residual_self_intersection_quality=failed`。7,890固定顶点全元素零变化、有序面不变；结构为一连通分量、Euler=2、边界/非流形边、非流形顶点链、边朝向冲突均零；最小面积比1。逐索引均值/P99/最大位移为0，完整双向点到三角面距离最大`5.16696640684342e-17 mm`（距离计算浮点余量）。

v2真实完整面对回归先顺序比较旧v1与已成熟CSR：80有向source与40无向F64面对的完整数组和SHA分别完全相同，调用177.686519与58.771429秒。单次顺序、不同时间共享负载，不作为ABBA速度结论。

![完整网格背景及全部异常局部](residual_failed.png)

图像为自产pial的矢状投影，完整32个相交面和35个顶点均显示。v1/v2坐标SHA与原末态相同，采用已核验的同一面对工件。[绘图收据](residual_failed_figure.json)绑定完整几何、阳性面对工件、脚本和PNG哈希；不是凭投影重叠判断3D相交。

## 版本与机器可读证据

- [v1完整报告](v1_report.json)、[完成收据](v1_complete.json)、[源码绑定](v1_source_binding.json)：helper SHA `058edba6cbf02c215c1b1b09bf050a66c35f8a9332f89dfa54b71dfa6f713e87`。
- [v2完整报告](v2_report.json)、[完成收据](v2_complete.json)、[源码绑定](v2_source_binding.json)、[面对集合回归](v2_pair_preflight.json)、[冷CLI](v2_cli_time.json)：helper SHA `d2d92cef03db4b6b3c9e05ad2c63b5b729dd921317034d4a89b754ee1e485631`；CSR marker SHA `752210294df00382a72e38157daced69a04893de0d305e6a8d3ceb4434384d1f`。
- 输入/依赖/协议/原始私有报告/完成收据SHA全部记录。公开文件只保留数值、版本及哈希；坐标、影像、服务器路径、地址和凭据不发布。
- 原v1合同测试为[contracts.json](contracts.json)/[XML](contracts.xml)；CSR版本为[contracts_csr.json](contracts_csr.json)/[XML](contracts_csr.xml)。模拟合同不替代真实benchmark。

CLI计时工具以新解释器调用一次子进程；包含加载、传输、QC及读写。RSS采用该新计时进程的RUSAGE_CHILDREN高水位，不代表连续多次API调用的单次RSS或同时父子占用。显存未采集为null。两轮候选采用同一四核预算；原生参考只复用既有重复性记录并在候选保存后作诊断。

## 未完成

独立约束投影的单面对真实定位已完整失败并保留：[完整报告](projection_min_report.json)、[收据](projection_min_complete.json)、[冷CLI](projection_min_cli_time.json)、[冻结协议](projection_min_protocol.json)、[源码绑定](projection_min_source_binding.json)。source/F64仍32面/35V，0接受、固定点及全部坐标零移，冷CLI301.682361秒。首面对的5种半空间×3环中12次固定角点不可行，3次约束不可行；唯一允许法向分离至少1.652002mm，超过原定1mm预算，这只说明该候选族失败。

完整40面对候选已经完成：[报告](projection_full_report.json)、[完成收据](projection_full_complete.json)、[冷CLI](projection_full_cli_time.json)、[协议](projection_full_protocol.json)、[源码绑定](projection_full_source_binding.json)。source/F64面对80/40→68/34，相交面32→29、35→32标记顶点，2次合格接受。但质量仍失败，不接生产。最大逐顶点位移0.626105mm、改变顶点P99为0.623042mm，完整双向点到三角面最大0.297078/0.130203mm；固定元素差异0、初始法向翻转0、最小面积比0.474522、无新增面对。冷CLI431.716168秒，RSS6,002,274,304字节。投影[4/5子测试](projection_contracts.json)不代替真实运行。可转动平面另一个候选按新协议做最小定位，原质量门不变。

可转动平面v1[有限诊断](free_plane_per_sweep_v1_report.json)、[完成收据](free_plane_per_sweep_v1_complete.json)、[冷CLI](free_plane_per_sweep_v1_cli_time.json)已完成，质量仍失败：面对80/40→44/22、相交面32→19、标记点35→24，2次接受改变2点。最大位移0.576458mm、改变点P99=0.572961mm、全体P99=0，双向点到三角面最大0.554600/0.039372mm。固定点零变化、原法向翻转零、最小面积比0.779105、零新增面对。冷CLI322.702184秒、RSS5,990,699,008字节。

**范围修正：**v1的`diagnostic_pair_limit=1`每轮重新取当前首面对，实际尝试三个不同面对，不能称为初始单面对回归。公开报告附完整范围审计，原[协议](free_plane_min_protocol.json)与[源码绑定](free_plane_min_source_binding.json)保留。当前实验helper修复为冻结初始前缀，不补入其他面对；None完整处理不受影响，生产SOAP也未改。[合同6/6子测试](projection_initial_limit_contracts.json)确认接受后另一交叉组保持原样，7.73秒，不代替真实数据。修正后v2采用相同求解参数与质量门、新[协议](free_plane_initial_min_v2_protocol.json)及[源码绑定](free_plane_initial_min_v2_source_binding.json)，真实回归单独报告。

修正后初始单面对v2[完整报告](free_plane_initial_min_v2_report.json)、[收据](free_plane_initial_min_v2_complete.json)、[冷CLI](free_plane_initial_min_v2_cli_time.json)确认只选择`25917/132048`、1次接受，改变1点/3元素。source/F64面对80/40→48/24、相交面32→20、标记点35→25，仍质量失败；一次顶点移动同时减少其关联面对，没有补入第二个目标。最大位移及改变点P99为0.576458mm，全体P99=0，双向点三角面最大0.554600/0.039372mm。固定点全元素零变、有序面保持、零新增面对、零法向翻转、最小面积比0.932124。求解131.622137秒、API338.466451秒（报告写出前）、冷CLI339.830810秒、RSS5,981,442,048字节。冻结包绑定`5d1cdb5e`与helper `47831ec2aa9de39934bb17df0f0352e2d398fc114227375ff02ba3c859df7440`，同一四核预算及实际[硬件/精度记录](hardware_context.json)保留；不声称整例加速或整体等效。

只读[几何深度报告](geometry_report.json)已经完成：40/40面对均为非共面真实交线重叠，长度0.005004984–1.040136417mm，互平面两侧穿透深度0.117694335–1.147011675mm。35标记顶点无精确重复，全网格0零面积面；涉面面积0.123101665–3.673744072mm²，共线比0.013832443–0.637125208。不是仅近接触或精确共线，但结论仅限旧数字拓扑网格，不能替代新GA自产网格验证。冷CLI2.104223秒，RSS101,175,296字节，完整[协议](geometry_protocol.json)、[源码绑定](geometry_source_binding.json)、[计时](geometry_cli_time.json)和[最终合同](geometry_contracts_final.json)各自保留。分类不豁免任何原质量失败，详见[中文功能页](../../../../docs/recon_all/PIAL_RESIDUAL_GEOMETRY_DIAGNOSTIC.md)。

white/pial相互穿越、其他半球/被试、空目录整例、完整指标等效与600秒目标均未验证；本报告不提供整例提速结论。原默认SOAP路径未切换。

上述所有残余候选来自旧数字拓扑网格，不是新的完整独立GA上游。当前停止扩大候选参数，保留完整失败；新GA自产网格由协调流程另测。未以1mm、20%等候选参数建立整例等效标准。

完整具名调用、参数、坐标与失败说明见[中文功能页](../../../../docs/recon_all/PIAL_LOCAL_RESIDUAL_SOLVER.md)。
