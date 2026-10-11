# 冻结 pial 残余相交诊断（2026-10-11）

这是公开 T1 派生的单半球冻结清理诊断。三个实际子进程均退出 0，网格质量仍失败；没有修改四轮放置、rip、pin、谓词、100 SOAP/轮或停止规则。

| 验证 | 本次实测 |
| --- | --- |
| 网格 | 95045 顶点、190086 有序面 |
| 完整源清理 | 415→32 面，24 轮 / 2400 SOAP；659.150075 秒 |
| 终态额外完整清理 | 32→32 面，16 轮 / 1600 SOAP；66.595758 秒；坐标 0 改变 |
| 源 / 独立 F64 检测 | 均 32 面、35 顶点；标记顶点集合相同 |
| 实际 40 对相交面 | 双方全 rip 0、单方全 rip 15、双方各含可动顶点 25 |
| 同输入前置 white | 源检测 0 面、0 顶点 |
| 两次原生参考 | 七前置 SHA 相同；坐标逐位相同、有序面相同 |
| 两段候选输出 / 原终态 | 285135 坐标元素逐位相同 |
| 整体等效 / 原始 T1 整例 | 本诊断未评价 / 未运行 |

原生参考的表面文件 SHA 不同，几何完全相同；旧 v3 收据记录了创建时间注释差异。本次重新核对已有参考的输入和解码几何，未重新执行原程序。v4 原始 T1 失败流程没有发布 LH pial，本诊断未证明 v3 和 v4 终态 SHA 同一。

## 产物和版本

- [清理完整轨迹与严格差异](reports/cleanup_report.json)、[面对计数与 white 检查](reports/pairs_report.json)。
- [诊断摘要](reports/summary.json)、[实际退出码收据](reports/completion_receipt.json)、[接收端 SHA/大小](reports/receiver_receipt.json)。
- [源码、工具、参考程序和运行边界](reports/source_binding.json)、[脑图收据](reports/figure_receipt.json)。
- [中文七节说明](../../../../docs/recon_all/PIAL_RESIDUAL_CLEANUP_DIAGNOSTIC.md)。

清理工具部署快照为 `ebc52ddae64ad83e82b526d9bda88f1b700a38b0`，两项 overlay 加历史冻结依赖按实际六模块 SHA 绑定。原参考程序 SHA 为 `78b64b7395aa0db0592ab6912fc026128b221c56d9f802db225fa59c18ceda44`。当前提交仅记录公开聚合数据和工具，不重标旧运行。

CPU 四线程，亲和性 88–91；CUDA 隐藏。完整诊断墙钟 1064.999621 秒，面对及 white 复检 122.860783 秒；这些均含观察/读写，不能作为四轮 pial、原生性能配对或整例提速。峰值内存未测，不写为零。原生重复性不代表这批网格通过质量检查。

## 复现

按中文说明的三个工具命令在新目录原位读取已授权冻结输入。七输入哈希由 `reports/cleanup_report.json` 的 `input_sha256` 给出，快照保留在原任务目录。候选保存后才读取参考坐标。几何快照、完整面对明细、私有日志和服务器配置不公开。

无需影像即可从公开收据重新生成面对分类和完成/质量状态：

```bash
python validation/recon_all/optimizations/20261011_pial_residual_cleanup/summarize.py \
  --report-directory validation/recon_all/optimizations/20261011_pial_residual_cleanup/reports \
  --output /tmp/fnit-pial-residual-summary.json
```

这是报告复算，不能替代真实清理运行。脚本只消费已有聚合 JSON，不调用脑影像程序或读取 MRI。

![完整 pial 及相交区域放大](figures/residual_surface.png)
