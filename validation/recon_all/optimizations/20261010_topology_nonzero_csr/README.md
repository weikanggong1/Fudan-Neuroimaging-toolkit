# 真实非零相交：自产 ORIGINAL + 压缩有序 CSR

## 结果与范围

公开 ds000114 历史3a左半球，**自产 pre-GA 母板**压缩后124723顶点、
246985面、741916有序邻接项。两槽同版本；此前CURRENT双检查均0，
本次ORIGINAL独立source/F64检测均2相交面、6标记顶点。

完整 v2 文件API两次各执行一个周期、完整100SOAP：2→0。源和独立
F64复检均0；坐标/逐轮SHA/有序面/几何头/全部非坐标字节/自身文件
重复严格通过，最大/P99重复误差0 mm。输入原文件不变。

| 调用 | 含观察、读写的API/s | 计算/s | 独立双复检/s | 100SOAP/s |
|---|---:|---:|---:|---:|
| 首次 | 126.954754346 | 84.595859363 | 41.608238209 | 0.357238507 |
| 重复 | 117.323496864 | 76.775938502 | 40.378487521 | 0.029452088 |

冷双调用CLI256.350624749秒，包含解释器启动、导入、加载和诊断读写，
不代表一次API或整例。峰RSS1,035,692 KiB；Intel Xeon Platinum8369B
2.90GHz、亲和性8–11/线程预算4、共享主机、GPU隐藏。Python3.11.16、
NumPy1.26.4、Numba0.67.0、nibabel5.4.2。完整API主要耗时为有向
全网格标记，不能把0.03秒SOAP当完整修复速度。

这不是完成全部25缺陷的最终母板，不是完整GA或recon-all，不是生产
ORIGINAL提前清理。原版同输入一致性、完整网格质量、独立环境安装
及整体指标等效均未评估。仅自身重复，不宣称原版精度或提速。

## 文件与版本

- [count.json](count.json)：ORIGINAL真实非零输入的两个独立计数及时间。
- [v1.json](v1.json)：原完整计算和自身重复保留。新诊断器将big-endian
  faces直接与native字节比较、将输出头与初始文本序列化前对象比较；
  两个flag False却未纳入printed passed，单列`checker_schema_error`。
- [v2.json](v2.json)：新冻结完整双调用，面按整数值校验，几何头与
  实际生成输入比对，并严格比较全部非坐标字节。初始几何尾部保留
  同源nofix原字节；上述合同全部True。成熟清理算法不改。
- [figure.json](figure.json)：同源MRI空间、输入/绘图SHA和位移统计。
- [numeric_preservation.json](numeric_preservation.json)：服务器先白名单
  导出，所有选择的数值/布尔/null叶及类型保持不变，原/公开JSON SHA。
- [PUBLIC_MANIFEST.json](PUBLIC_MANIFEST.json)：本次实际源码、文档和
  指标文件的SHA及大小；完整私有轨迹、执行命令、环境和数据仅留服务器。

实际v2新工具SHA `a95ebdc33f8fd3a0220a04d287ff30c1c39109f986711b599c45869ac1890a93`。
源码冻结时属于未提交验证代码，按完整文件SHA绑定；已提交成熟
cleanup/SOAP分别来自`15683c1b…`/`ca8d970f…`，实际调用模块SHA逐项
记录，不用后续仓库HEAD改标旧运行。新工具未新增数值内核或依赖。
只导出指标、程序SHA和实际派生PNG，未导出原始MRI/许可证/私有路径。

## 脑图

![六个真实非零修复点](nonzero_original_csr.png)

输入→输出6个顶点改变，最大位移0.665353455 mm、均值0.000020669 mm；
全网格P99为0，不能据此隐藏局部变化。蓝/橙分别为初/末态，同源MRI
tkreg逆变换叠加于最大位移附近±8 voxel。该图显示实际修复形变，
不是与官方的误差或全网格解剖质量证明。

## 复现

详见 [中文七节说明](../../../../docs/recon_all/TOPOLOGY_NONZERO_CSR_REPAIR.md)。
使用新的空输出目录、同一自产compact和几何源、固定四线程、GPU隐藏；
不读取参考after补坐标。保留原停滞、最佳恢复和严格双复检。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python tools/benchmark_recon_topology_nonzero_csr.py \
  --compact-state /data/own/compact.private.npz \
  --input-surface /data/own/lh.orig.nofix \
  --source-manifest /data/frozen/SOURCE_MANIFEST.json \
  --output /data/runs/nonzero-original-csr \
  --threads 4 \
  --repeats 2
```
