# 首批四组同预算微调：已完成

- 批次：finetune_stage1_20261101_r02。
- 用户回传根status=completed，UTC结束2026-11-01T16:50:19.435656+00:00，即北京时间2026-11-02 00:50:19。
- 数据：28130条train索引；单模型单卡batch6、2 epoch、每轮4688步、共9376次计划迭代。
- 同一兼容checkpoint初始化；DINO冻结/eval，原3D网络、解码器、图像投影和分类头继续训练。seed=20261101；峰值学习率2e-4，block组0.1倍。
- 每组只评价最后checkpoint：完整6019帧、原作者TTA与SemSegTester。原始有效配置尚待服务器完整导出。

| 方法 | mIoU % | 相对训练control pp | 成功optimizer更新 | 推算AMP跳步 |
|---|---:|---:|---:|---:|
| control | 82.20 | 0.00 | 9371 | 5 |
| gate | 82.21 | +0.01 | 9372 | 4 |
| mean | 82.47 | +0.27 | 9371 | 5 |
| camera_dropout | 82.04 | -0.16 | 9371 | 5 |

推算AMP跳步=9376次已完成计划迭代−report中的成功更新；并非已拿到train/summary.json的原始字段。mAcc/OA/逐类数据未回传，不填估计值。

## 已有观察

mean相对同预算control +0.27pp；gate +0.01pp；camera_dropout正常输入−0.16pp。control相对历史82.48%低0.28pp，当前配方未提高历史clean最佳精度。camera_dropout的鲁棒性需看同缺失条件评价，见[接续批次](../multi_route_20261101_r01/README.md)。

## 来源与交接

[用户报告摘录](receipts/completion_20261102.txt)、[结构化记录](reported_results.json)、[CSV](comparison.csv)。助手未连接服务器，尚未完成独立重跑或完整产物审计。首步回执的源版本为81ec641，完整运行快照commit待接收；本次归档commit不作为实验执行commit。

如需追加完整服务器导出，使用本目录下全新的server_export_日期时间子目录，保留当前回执记录。旧四组已完成，不重启。
