# 完整首缺陷显式距离配对

本目录保存两个全新进程的完整首缺陷GA回归安全出口。两次计算都从同一自产nofix、qsphere、brain MRI准备输入，固定seed 1234、默认10/10/−1控制，CPU92–93共2线程、CUDA隐藏。serial结束后才启动parallel，各自Numba缓存独立。原MRI、NPZ、日志和完整私有收据保留服务器。

## 实际完成门

父进程和两个child均为EXIT0/COMPLETE。每次495次搜索评分加最终1次，共29条代际轨迹（索引0–28）。81数组的dtype、shape、NPY字节和4个完整NPZ文件SHA相同；496项float64评分与完整trace及最终scalar关联。1362381次实际RNG区间/值序列SHA相同。

两份控制按类型逐项JSON相等，不证明浮点符号零或JSON文本字节一致。两次落盘后才读取旧query参考：其40字段末态、8字段search和8项可比控制相同；旧RNG只有次数，没有逐draw比较。旧第一候选没有对应成本次最终候选。

| 实测秒 | serial | parallel |
|---|---:|---:|
| fresh child CLI | 1101.111005 | 1188.950796 |
| search及最终写回API | 1033.454756 | 1119.051656 |

本次parallel CLI慢7.9774%、API慢8.2826%。这是一组依次执行的共享节点配对，没有稳定负载或独占CPU证明，不能归因或外推整例提速。父driver wall 2301.832474秒和pair内部wall 2292.062257秒单列，不当作外部fresh parent CLI。API默认仍是serial。

## 冻结与报告

- 科学源来自Git `eb164824de5dec324f38a42626037f56bd040eb7`；原清单SHA `692de9f0a53c81cd28163fb637185ed7f7f6fe14e4281da06a33cf87275fc779`，71具名模块加根包初始化共72科学文件逐Git字节核验，没有工作树科学overlay。
- [metrics.json](metrics.json) 是服务器白名单出口，715个数值/布尔/null叶及类型保真；[EXPORT_RECEIPT.json](EXPORT_RECEIPT.json) 和 [COMPLETE.json](COMPLETE.json) 绑定其实际SHA。
- [source_bindings.json](source_bindings.json) 区分计算、诊断入口、安全出口、绘图及旧query子集版本；[setup_history.json](setup_history.json) 保留准备和只读检查错误，未重算算法。
- [contracts.json](contracts.json) 是出口正负向合同结果。小文件合同用于来源、类型、完整产物和拒绝门，不是MRI benchmark。
- [PUBLIC_MANIFEST.json](PUBLIC_MANIFEST.json) 逐文件绑定本页、工具、合同和公开产物SHA/大小；它不自列自身。

## 真实图

![自产最终候选与原MRI](figures/final_candidate_t1.png)

两条轮廓是本次serial/parallel完全相同的最终候选，切片由selected vertices自动确定。仅使用MRI tkreg的F64逆矩阵显示，没有拟合、配准或网格重排；这不是原生参考误差，也不是全九缺陷最终网格QC。

![共享节点实际计时](figures/observed_timings.png)

[PLOT_RECEIPT.json](figures/PLOT_RECEIPT.json) 绑定MRI SHA、显示矩阵、实际tkreg方向L/I/A、切片与PNG SHA。原绘图v1固定体素轴标题把轴位/冠状位互换；本页v2仅修标题，GA和metrics未改。只读安全导出7.041405秒、v2绘图1.877418秒独立计时，不加入GA墙钟。

完整中文七节说明及具名调用见 [功能页](../../../../docs/recon_all/TOPOLOGY_GENETIC_DISTANCE_FULL.md)。本页只证明第一个默认缺陷的自有serial/parallel一致；全九缺陷、原生逐候选winner、最终整网格质量与整体脑区等效均未在本页评估。
