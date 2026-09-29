# recon-all 源码入口

`run_recon_all_python(t1=..., subject_dir=..., weights_dir=..., assets_dir=..., device="cuda:0", threads=4, native_bin_dir=None)` 执行固定单 T1 标准流程；`fnit-recon-all` 是其命令行入口。`run_recon_all_python_batch(jobs=..., weights_dir=..., assets_dir=..., devices=..., threads=..., native_bin_dir=None)` 按输入顺序返回多个被试的报告。

安装、全部参数、坐标空间、输出文件结构和验收边界见[用户说明](../../../docs/recon_all/README.md)。固定文件清单位于 [expected_outputs.py](expected_outputs.py)，并与 138 项比较器共用。
