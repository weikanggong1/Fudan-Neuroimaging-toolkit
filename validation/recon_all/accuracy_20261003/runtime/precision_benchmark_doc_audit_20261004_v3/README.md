# 精度候选说明v3独立审计

178项轻量核验全部通过，原63项重新执行；sub-06核对proper穿越296/197 vs441/269、双方singlecomponent/nonmanifold0及sphere/sphere.reg negative0；这段negative0只属于sub-06。新增sub-07另核对候选sphere/sphere.reg左49/56、右0/0，官方左58/26、右27/35，proper穿越265/227 vs240/213；并保留近6mm方向和质量未判定的范围。proper计数未与非proper或同索引接触合并。

Python示例五个具名Path参数与summarize实际keyword-only签名一致，输入目录存在，输出路径不存在；仅AST审核，未调用汇总函数。汇总计时字段在末尾独占JSON写出之前取值，文档明确不含写出。

旧v2审计未改。本v3新增sub07数值、方向、质量及全部本地链接核对。原46档案清单逐项核验；本v3绑定最新doc/summary/script/quality原报告SHA；未做CPU4测试、影像计算、GPU或算法修改。
