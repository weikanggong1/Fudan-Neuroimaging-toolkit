# 完整GCA缓存隔离实测收据

两例公开ds000114冻结自产nu/brainmask，同A100节点四线程；算法`a756fffb`，
采样`f57484f1`，worker/driver冻结补丁逐文件SHA见`gca_cached_v2/SOURCE_MANIFEST.json`。
不读取官方图像生成候选；旧Torch/独立Conda原生仅用于事后诊断。
不是原始T1整例或整体指标等效验收。无影像、许可证内容或凭据。

- `gca_cached_cfff_20261009_v2/sub06-*/report.json`、`sub07-*/report.json`：
  未初始化和已初始化CUDA父API，共四次完整LTA优化回归精确；完整墙钟、
  子阶段、实际precision、输入/源码哈希、父状态、同期显存样本。
- `gca_cached_v2/sub07-reference.json`：旧64块/CPU求逆完整API和两次独立
  Conda原生，输入完全相同，原生重复精确；sub-07新旧/原生矩阵精确。
- `gca_complete_v4/sub06.json`：缓存关闭，原生/旧Torch/批量求逆的六次完整
  配对；原生和新旧各自重复精确。sub-06旧新对原生仍有既有矩阵差异。
- `gca_cached_cfff_20261009_v1`：保留初版比较器混比API原矩阵与舍入LTA的
  诊断记录。v2比较实际写出LTA，没有改变算法或放宽阈值。
- `PUBLIC_EXPORT_MANIFEST.json`：原私有JSON SHA与公开副本SHA对应；全部数值/
  布尔/null叶子保持，私有路径和主机名替换为角色名。原始收据私下保留。

目标GPU进程归属无法确认，进程树峰值为null；整卡上界另列，不能把0当零显存。
GPU3缓存关闭配对和GPU4隔离验证是不同窗口，不据两者直接计算稳定提速倍数。
本轮21项接口/隔离失败保持/求逆契约在Python3.11真实环境通过，日志另留私有。
安装为已有环境迁移副本，干净主页安装和无预装软件整例尚未验证。

具名调用、全部参数、坐标和精度、计时范围、复现命令及原代码参考见
[GCA隔离说明](../../../../docs/recon_all/GCA_ISOLATED_TORCH_20261009.md)。
