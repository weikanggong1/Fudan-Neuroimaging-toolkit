# sphere finish：同输入完整CPU/GPU与标记行候选回归

## 范围与结果

这是公开ds000114 sub-07、803aec50自产smoothwm的固定阶段诊断。v1重新运行实际完整smoothwm→inflation→standard sphere链，在清理入口透明捕获带SHA的检查点；v2复用该检查点，不将最终sphere冒充入口。候选不读取官方结果或修改参考。

- v1 LH CPU/GPU/GPU/CPU四次均完整1001轮，坐标、文件、几何头、逐轮SOAP/marked/负面/投影和停止状态相同。CPU82.024885/53.788186s，GPU9.472575/7.090744s。
- v2 LH完整CPU dense、CPU marked、GPU dense、GPU marked及反向共八次，所有坐标、文件和共同逐轮状态严格相同。GPU dense中位数7.814067s，marked4.355450s，阶段缩短44.26%。首次marked包含CSR首次JIT；只读轨迹开销均包含。
- RH四次与八次均保持原零轮早退、严格相同；首次CUDA初始化成本不能隐藏或套用LH速度比。
- 原最终LH的47个FP64径向负面及0.0559604954mm²面积保持相同，RH为0。没有新翻折，但既有缺陷未修复；这里只检查径向朝向/面积，未完成三维相交或整体指标等效验收。
- 8项CPU/GPU合同通过4.78s；模拟网格合同与真实benchmark分开。

阶段时间包含完整计算、传输与只读轨迹，读写另列；v1还列入包含采样的完整读/算/写墙钟。首次捕获链含native inflation、JIT和入口写出，LH207.393407s、RH132.265688s。这里没有新的原始T1整例，不构成600秒目标或新官方完整链验收。

## 文件和源码绑定

- `v1/{lh,rh}/summary.json`：完整实际入口捕获、CPU/GPU ABBA、逐轮轨迹、网格质量、源/程序哈希和采样。
- `v2/{lh,rh}/summary.json`：dense/marked CPU/GPU各两次完整回放及全部精度、逐轮与资源结果。
- `SUMMARY_METRICS.csv`：24次回放的紧凑指标；空字段表示该版本未测或未知，不代表0。
- `v1/FROZEN_SOURCE.json`、`v2/SOURCE_PATCH_MANIFEST.json`与`input_manifest.json`：实际冻结源码及公开自产输入的逐SHA清单。
- `v1/controls/launch_*.log`：算法完成日志与两次仅启动失败的日志；失败记录保留。
- `v2/controls/unit_v2.xml`及`launch_v2.log`：8项合同与完整执行日志。
- `hardware_observation.json`：同认证主机的运行后硬件观察；各测试实际线程、亲和性、精度和源码由其原始JSON记录。
- `COLLECTION_MANIFEST.json`、`public_export_manifest.json`和`PUBLIC_RECEIPT.json`：原始/公开文件SHA和脱敏校验。原始收据私有保留；114,594个数值、布尔或null字段核验保持相同。

v1实际调度SHA为`be1e044e2db43a4d59c9b6752f997643a4d7e5f9749ba6b30b464b37b557002b`；v2标记模块SHA为`608db8a10022b88ec16f9439b4b2da5eebe342707cc1280d13232c2ce02f84f9`，显式finish接口SHA为`ea000ac45f4008b1635c8c7e04af161ac937758e97cbd0ab980af3b8b04b15fc`。后来文本提交不重标实际测试源码。

## 资源解释

同A100-SXM4-80GB、CPU64–67、总4线程、Torch2.5.1/CUDA11.8、默认TF32开启，无半精度；共享节点观察。v2目标卡同期上界594,542,592字节，所有计算进程上界585,105,408字节；树归属未知为null。GPU allocated/reserved是allocator记录，CPU行0不等于整进程零显存。

名义采样0.5s，v2实际最大间隔3.219151s，v1为6.228536s；零失败查询。短循环的采样不保证连续峰值，也不把不同时间/阶段峰值相加。没有在无预装软件的物理隔离环境执行完整重建。

公开包只有收据和说明，不包含MRI、surface/NPZ、模板、权重、许可证或凭据。复现参数与全部输入输出见[专页](../../../../../../docs/recon_all/SPHERE_FINISH_TORCH_20261009.md)；脚本见[入口捕获](../../benchmark_sphere_finish.py)和[标记候选](../../benchmark_sphere_finish_marked.py)。
