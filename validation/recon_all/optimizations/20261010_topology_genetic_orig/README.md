# 完整自产premesh之后的标准orig验证

本目录绑定新增独立工具、合同及实际服务器预检/排队回执。工具复用
已有标准三轮remesh和完整ORIGINAL相交修复，不新增几何算法、不调用
原版程序，也不读取参考。生产默认未变。

## 已完成与未评估

- 三项本地文件合同及六个子测试通过，6.36秒；验证完成回执、报告/
  表面SHA、无原程序/参考的父语义、两个相交结果绑定同一输出SHA，
  以及成功/失败回执。四面体只用于文件与控制合同，非真实benchmark。
- 实际服务器全依赖`--help`预检退出0，4.910129秒，四线程/CPU120–123、
  CUDA隐藏。75项冻结文件逐SHA及大小验证，实际工具SHA
  `ad008be0cbdb17730e6dc3041e7dfbd4acdfefd69a35caf68d7d9b534a3c4bad`。
- 后台等待当前完整九缺陷GA的成功退出；随后仍独立验证父COMPLETE、
  报告SHA和premesh文件SHA，才能从其新产物开始完整标准orig阶段。
  等待期间不产生候选表面，不改父源码/输入/迭代策略。
- 当前回执只证明工具预检和排队。真实连续premesh→orig精度、耗时、
  最终网格、官方对照、完整138项、显存/隔离安装及指标等效均未评估。

## 版本与报告

[local_contracts.json](reports/local_contracts.json)保留实际测试源码SHA、
JUnit SHA及合同范围。[launch_receipt.json](reports/launch_receipt.json)是
该时刻预检/等待收据，不代表之后仍等待或已经完成。
[runtime_source.json](reports/runtime_source.json)记录75项实际冻结文件、
基线提交`2474392bf7aa76a1ea1778dae389a31537724621`、父冻结清单及驱动SHA。
后续提交不冒充此运行的源码版本。

运行文件夹名使用2026-10-10的启动日期；在北京时间2026-10-11仍继续
等待与评估，日期不更改输入和源码版本绑定。服务器完整命令、源码、
原始影像和阶段产物保留私有任务目录，公开报告仅含指标与哈希。

## 复现与接口

全部具名参数、输入输出及空间、失败行为、原软件命令和真实验证边界
见[中文七节说明](../../../../docs/recon_all/TOPOLOGY_GENETIC_ORIG.md)。
后续必须从本次完整GA实际新结果运行，不能以手工补写回执或官方网格
启动。标准remesh与最终清理的计时含加载和读写；外层CLI另包含导入。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python tools/benchmark_recon_topology_genetic_orig.py \
  --premesh-directory /data/own/complete-genetic-premesh \
  --hemisphere lh \
  --output /data/runs/new-genetic-orig \
  --threads 4 \
  --remesh-scalar-storage numpy
```

同源脑图在[完整首缺陷报告](../20261010_topology_genetic_search/README.md)，
本工具的最终orig脑图尚未生成。没有用旧图替代新最终输出。
