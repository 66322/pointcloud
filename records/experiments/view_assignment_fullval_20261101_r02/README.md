# 原权重视图赋值推理对照

- 批次：view_assignment_fullval_20261101_r02；已完成，2026-11-01 19:38:55（北京时间）。
- 目的：不重新训练，检查重叠视图的选择/聚合是否影响完整验证结果。
- 原兼容权重、同一6019帧val、原TTA；已有基线82.48%。
- mean：82.64% mIoU、90.06% mAcc，较基线+0.16pp；first：82.55%、89.94%，+0.07pp。
- 来源：[末帧与completed回执](../../analyses/view_assignment_r02_last_frame_20261101.md)。启动版本报告为3f0003d。
- [机器可读结果](reported_results.json)、[对照表](comparison.csv)。未知OA保留null/空白。

这是用户回传的服务器证据整理，不是助手直接读取服务器或重新评测。当前目录不是完整export_record导出；尚缺最终逐类JSON、OA、runtime及完整有效配置。单次小幅差异不证明稳定提升。已有结果保留，不追加重复完整val。
