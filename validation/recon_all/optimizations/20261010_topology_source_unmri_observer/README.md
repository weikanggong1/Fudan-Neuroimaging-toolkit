# 首个真实拓扑候选：完整 fitness 与距离输入

本目录保存独立固定源码观察器、去敏数值收据及可复现参数。
检查点只用于冻结同输入阶段回归，不用于 FNIT 生产输入，不代表完整 GA。
中文参数、格式、空间和失败边界见
[七节说明](../../../../docs/recon_all/TOPOLOGY_FIRST_FITNESS_DIAGNOSTICS.md)。

## 当前完成范围

- 公开 ds000114 sub07 ses-test 左侧，来自 `79a41cdd` 原始 T1 自产链。
- 固定源码 d932c45b7941662ea380a05efef580568b98d41a；实际源码、编译 argv、
  程序、三项输入、原始 T1 和全部参考检查点 SHA 在公开机器可读收据中。
- 完整 firstfit 两次参考的 12 个原 float32 TP 字段和最终 double 分数重复。
  五个被消费子项有限，fitness −97.760777950286865；四个面积诊断实际 0/0，
  原位模式 0xffc00000。不是未赋值字段，也不参与最终合成。
- 距离 v2 两次参考的完整最终图、XYZ、CSR、法向和伪法向重复；原 OMP
  buffer 追加顺序不同，二进制 SHA 不同。按 fno 匹配的输入全相同；原件不排序。
- v3 / FNITDST2 两次真实参考补齐算法几何，原始值与此前 source unMRI 距离图
  逐元素和字节相同；6 项解析契约在声明环境全部通过。
- 实际 metricProperties TU 类型探针确认 sqrt/acos(float) 和两次除法均
  float32；宏 MIN/MAX 的 double 常量在 float32 alpha 赋值前参与表达式。
  通用 triangle fixture 和两次 stdout 重复证据独立绑定；探针在固定源码主机执行。
- 旧距离格式只含 distance 存储头。实际 bbox/xSURF 使用 mri_defect 的另一
  套 origin/scale，新格式显式补齐两者；旧检查点和失败证据保留。

`build_volume_observer.py`、`build_full_fitness_observer.py`、
`build_distance_observer.py`、`build_numeric_type_probe.py` 只生成隔离 benchmark 程序；不安装为生产后端。
单一 `load_volume_capture`、`load_statistics_capture`、`load_distance_capture`
解析器复用已有规则，不读取官方结果来修补候选。

## 公开数据与验证边界

`reports/` 只含去敏 JSON；`PUBLIC_EXPORT_MANIFEST.json` 核验每个公开文件 SHA，
同时保留原始收据 SHA 和所有数值/布尔/null 叶。MRI、NPZ、二进制程序、
许可证及服务器标识保留在授权环境，不进入 GitHub。

firstfit v1 的非法 `-nan` JSON 与 distance v1 的编译失败不会被改写为成功。
参考带观察器的耗时包含上游计算及写入；不是未插桩原程序单独内核耗时。
本次 CPU8–11/4线程、GPU隐藏，不是 GPU显存达标或十分钟整例证据。

未完成全缺陷动态 retessellation、完整 mutation/crossover/controller 与最终
原始 T1 整例验收；整体指标等效 `not_assessed`。另一个模块的子项同输入通过
不自动改变这些范围。最终网格质量门保留，不因中间诊断 NaN 放宽。
