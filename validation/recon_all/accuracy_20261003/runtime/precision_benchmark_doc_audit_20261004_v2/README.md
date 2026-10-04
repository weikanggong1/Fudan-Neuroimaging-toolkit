# 精度候选说明v2独立审计

94项轻量核验全部通过，原63项重新执行；新增核对proper穿越296/197 vs441/269、双方singlecomponent/nonmanifold0及sphere/sphere.reg negative0。proper计数未与非proper或同索引接触合并。

Python示例五个具名Path参数与summarize实际keyword-only签名一致，输入目录存在，输出路径不存在；仅AST审核，未调用汇总函数。汇总计时字段在末尾独占JSON写出之前取值，文档明确不含写出。

旧1429审计未改。本v2绑定最新doc/summary/script/quality原报告SHA；未做CPU4测试、影像计算、GPU或算法修改。
