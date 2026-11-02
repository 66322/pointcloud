# 实验名称 / run_id

- 更新时间与时区；最新观测事件时间（未知写未知）：
- 来源：用户服务器回执 / 直接服务器产物 / 本地合成测试 / 独立重跑：
- 状态及适用时刻：pending / running / completed / failed / blocked / inconclusive：
- 目的、可证伪假设和唯一变化因素：
- 对照run_id、训练预算、评价条件（clean / 相机子集 / 偏移）：
- 实际执行代码commit、有效配置、快照来源（与记录提交commit分开）：
- 数据划分、train/val帧数、帧清单来源、类别/标签定义：
- 初始checkpoint来源、严格加载与DINO冻结审计：
- seed / epoch / 单模型batch / 优化器与调度 / GPU（仅0/1）/ TTA：
- 精确命令、nohup父PID/日志、独立输出目录：
- 最终mIoU/mAcc/OA、逐类与分组；未知字段为null：
- 计划迭代、成功更新、AMP跳步；推算项标明推算：
- 耗时/显存及计时范围；未知不估造：
- 失败与依赖阻断、重启/重排、旧输出保留位置：
- 能支持的观察、尚不能支持的结论：
- 下一步与仍需验证的问题：
- 证据文件链接及尚缺文件：

按records/experiments/<run_id>/归档，分析放records/analyses/，同时更新docs/HANDOFF.md、EXPERIMENT_MASTER.md和必要的路线状态。没有完整服务器原件时使用reported_results.json、summary.reported.json等明确名称，不伪造export_record的status/provenance。

不例行要求数据/权重/点数组SHA；不把6019帧当6019次独立训练重复；不以训练loss取代最终验证结果。新模型/特征/投影/TTA设置使用新输出目录。
