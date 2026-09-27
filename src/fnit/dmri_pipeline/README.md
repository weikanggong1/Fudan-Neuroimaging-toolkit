# dMRI pipeline 模块

公开的单被试接口是 DMRIPipeline。它依次执行可选 TOPUP、EDDY、DTIFIT、
AMICO-NODDI，再进入 TBSS/FNIRT 或 T1+tensor MMORF 配准分支。两条分支写出
相同命名的九张标准空间参数图。输入、输出、显式参数 Python 示例、单被试命令行、
UKB 官方对应步骤和真实数据验证见
[docs/dmri_pipeline/README.md](../../../docs/dmri_pipeline/README.md)。
