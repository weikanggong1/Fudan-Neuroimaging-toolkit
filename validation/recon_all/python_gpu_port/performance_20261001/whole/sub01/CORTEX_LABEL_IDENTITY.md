# 本轮受控 cortex 标签索引核对

`cortex_label_identity_readonly.py` 是本次实际发送给 headcw 的只读诊断源码。
输入固定为本轮受控 FP32 基线和 1b8c36d 候选的四份 `cortex.label`，先要求
两次整例执行完成；只用 nibabel 读取顶点索引，不读取影像、不执行模型或
相交搜索。输出 JSON 记录文件 SHA、索引 shape/dtype、唯一索引数、规范化
little-endian int64 索引数组 SHA 和相等性；标签原文件不传回本地。

这些索引属于相应半球的 surface 顶点顺序，无空间长度单位。顺序数组和
集合都相同，且双侧标签文件 SHA 也相同。标签中已有重复索引，报告保留
shape 与唯一数；检查没有去重或改写输入。皮层子集使用索引的成员关系。
缺文件、读取失败或整例未完成会抛异常，没有独立的 FreeSurfer 等价命令。

复现时在 headcw 的已声明 FNIT Conda 环境执行该文件，将 stdout 保存为
新 JSON。`gpu_control_cortex_label_identity.json` 另记录本地添加的
`read_only_script_sha256`；它与本文件 SHA 对应。复测 UTC 可以不同，比较
的是标签文件/索引数组 SHA 和相等性。精度汇总只在这些索引和三角几何均
实测相同时，推得受控基线的 cortex 子集相交结果与候选检查相同；不把该
推论说成基线重新执行了质量搜索，也不引用历史 TF32 的 e036 结果。
