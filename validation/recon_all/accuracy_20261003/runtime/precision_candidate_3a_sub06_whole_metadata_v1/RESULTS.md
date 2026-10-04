# 3a sub-06 原始 T1 整例完成元数据

观察时间：2026-10-04T13:29:17.548981+00:00。原目录只读；这是候选3a的真实CLI整例，不是8f结果。

源码：3a0c9aba6321b4981fd8174b4b191515459aa38b；源包SHA 03cc806fb449a8620c8caa78b1af86dfaff6b64ab9e9e7c5210cf12184aa7f31。外置工具commit fe653dfb273d639b8077fa96b2d6e3296cd3fb29，工具包SHA db6abf0323ed05446d751d014a506c23fc993498b2d20edaeb9f04bdb6b2b5e4；实际脚本副本和SHA一并保存。

完成UTC 2026-10-04T13:13:57.848641+00:00，exit/child exit均0；138/138齐全。LH130416/RH132688顶点，white/pial自交均0，网格验证passed。

入口墙钟 2948.558977915s；内部 API 总墙钟 2945.172643721s；pipeline 2939.520808352s；monitor墙钟 2948.631078388s。内部API时钟不等于独立API整例入口，本次invocation=cli。

自身父子进程同期采样峰 13570670592 B，1402次，请求间隔2s，最大采样间隔4.622375769s，查询失败0。整卡峰另列summary，含其他任务，不能与自身峰相加；这不是连续峰值。缓存关闭时allocator计数不可用/零不代表真实显存为零。

真实startup report 10份；pre-callable失败 2次；算法进入 8次。每个worker的attempt、failure_phase、operation_entered见summary及原报告；不能套用8f全部首次启动成功的事实。

准备评估config SHA b7c5bd990ecc90b542c87323cfa1872cae1bf3246b8ea887a8f478e229ca0942，输出在本次观察时存在：False。collector未启动评估；数值比较/整体官方等效均未评估。

## 分阶段时钟

|阶段|seconds|function_seconds|
|---|---:|---:|
|input_talairach|35.192241364|35.191936274|
|n4|126.748182816|126.747923054|
|nu|6.826579438|6.826232760|
|T1_normalize|83.915458715|83.915181456|
|brainmask|0.902694091|0.902506873|
|SynthSeg|17.080089821|17.079892504|
|mri_em_register|121.957368578|121.957207701|
|mri_ca_normalize|32.730386852|32.729595726|
|mri_cc|16.446978633|16.446573339|
|brain_second_normalize|67.759280410|67.758997428|
|entowm|1.760211264|1.759957899|
|ants_denoise|27.323822933|27.323459649|
|mri_segment|62.013359161|62.013136131|
|mri_edit_wm_with_aseg|26.524273257|26.524003843|
|wm_pretess|9.040024159|9.039606484|
|wm_fix_ento|1.452933115|1.452705370|
|wm_fix_acj|1.910571943|1.910329486|
|mri_fill|71.498891374|71.498463467|
|filled_auto_checkpoint|0.004371744|0.004301318|
|mni_aux|9.679119154|9.678792319|
|mni_nonlinear|59.933979237|59.933725773|
|brain_finalsurfs|7.428869124|7.428563095|
|surface_hemisphere_group|935.993045705|935.992554763|
|defects_lh|1.470232201|1.469998355|
|defects_rh|1.341829956|1.341553650|
|register_hemisphere_group|269.798444245|269.798055320|
|annotation_hemisphere_group|175.806731060|175.806236667|
|finish_surface_hemisphere_group|460.026805255|460.025918714|
|finish_metrics_lh|30.275636142|30.275456204|
|finish_metrics_rh|30.484662921|30.484484458|
|exvivo_annotations|2.386851052|2.386593581|
|jacobian_lh|0.042586396|0.042390423|
|contrast_lh|1.844761873|1.844519054|
|jacobian_rh|0.044954926|0.044774875|
|contrast_rh|1.858318320|1.858053302|
|ribbon|4.902314388|4.902051349|
|relabel_hypointensities|4.499492986|4.499277473|
|aseg_ribbon_fix|6.405026881|6.404702433|
|project_aparc_volumes|21.279899092|21.279615292|
|project_wmparc|5.180408583|5.180133637|
|brain_volume_stats|2.830798581|2.830199426|
|aseg_stats|14.975396072|14.975056391|
|wmparc_stats|12.054053551|12.053833809|
|stats_lh_aparc|9.805012124|9.804789418|
|stats_lh_aparc.a2009s|0.376790924|0.376623937|
|stats_lh_aparc.DKTatlas|0.190374077|0.189964332|
|stats_lh_aparc.pial|9.480785732|9.480546665|
|stats_lh_BA_exvivo|0.092164234|0.092001412|
|stats_lh_BA_exvivo.thresh|0.087914267|0.087690991|
|stats_lh_w-g.pct|0.055967632|0.055814281|
|stats_lh_curv|3.128444991|3.128200994|
|stats_rh_aparc|9.824094258|9.823805504|
|stats_rh_aparc.a2009s|0.311131258|0.310907343|
|stats_rh_aparc.DKTatlas|0.180381349|0.180181576|
|stats_rh_aparc.pial|7.184152098|7.183809536|
|stats_rh_BA_exvivo|0.095189900|0.095009743|
|stats_rh_BA_exvivo.thresh|0.092141998|0.091919765|
|stats_rh_w-g.pct|0.082063170|0.081917662|
|stats_rh_curv|2.654660896|2.654381751|
|mesh_validation|115.232935600|115.232527133|

表中60个顺序父 stages[] 可相加，总计2930.506141879s。worker组内部阶段、surface metrics内部指标及MNI子步骤属于嵌套计时，不能在父阶段之外重复相加。父阶段之和也不等于pipeline或API总墙钟；未剖析残差、校验及线程设置/恢复按各自作用域另列。原pipeline及worker报告保存完整字段。

归档限JSON/CSV/log、公开验证脚本和3个绑定生产脚本；不包含MRI、surf、影像资产、权重或license内容。资源库存只保留已有路径/SHA声明，不读取指向的数据。
