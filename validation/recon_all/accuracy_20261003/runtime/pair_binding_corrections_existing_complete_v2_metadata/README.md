# 两例已完成比较的输入绑定元数据纠正

状态：两例 metadata_correction_verified。仅顶层 input_sha256 改正；原18阶段数值和图表哈希不变，没有重新比较或GPU计算。原strict 138为checked138、passed2，不能把元数据审计通过称为精度通过。

工具：独立 startup_pair_binding_tools_v4，归档SHA 25710b84310503a70f15f645de24fb7b623c85bda9d2fa739aa36d1ace4525a1。创建时 audit 为未提交补丁，完整工具文件SHA见plan/summary，未冒用4c084版本号。

两例旧错误值均为8aa986c658663d2c26fa482a2ce303c4eecc28fbd69e4f6e9bace438be058200，即冻结官方manifest最后一个程序SHA。重新读取真实原始输入、cohort、两侧config/launch/completion、资源清单和程序文件后，六份冻结回执审计只发现这一字段变化。

|病例|真实输入SHA|verify墙钟秒（含等锁）|audit秒（不含等锁）|audit等锁秒|
|---|---|---:|---:|---:|
|ds000030_sub-10159|0cb8a28928917b452b516e4e4694b3b79eb96b4c558f47ac63603035d0072e8b|12.448385|15.534274|0.000640|
|ds000030_sub-10171|6f7587d216cc979940f9ecde0dbe9dd4546cb2f2ca8158cf7b80946c3fc67af0|11.893871|13.507470|0.000732|

verify-only仅有verify_binding阶段；这些CPU文件核验耗时不属于生产算法时间。双方均使用/tmp/fnit-shared-benchmark.lock。整体官方等效仍为not_assessed。

旧v3两例verify成功但audit因baseline角色限制返回1，旧队列和失败日志原样保留；本归档另存小型证据副本。原比较配置/绑定/checkpoint亦只保存字节副本用于核验，没有改写原目录。
