# 实验轻量证据与回执

完整服务器输出留在runs/；这里按run_id保存可随Git同步的小型依据。两台电脑接手先读[交接记录](../../docs/HANDOFF.md)和[实验总表](../../docs/EXPERIMENT_MASTER.md)。

## 当前结果入口

| 目录 | 已归档的证据 |
|---|---|
| [view_assignment_fullval_20261101_r02](view_assignment_fullval_20261101_r02/README.md) | 原权重mean82.64、first82.55；完成回执与结果表 |
| [finetune_stage1_20261101_r02](finetune_stage1_20261101_r02/README.md) | 四组两轮微调最终mIoU、成功更新、completed回执 |
| [multi_route_20261101_r01](multi_route_20261101_r01/README.md) | 11组clean、4项缺相机结果、2组失败、分时进度和重启接续；两份完整逐类JSON |
| env_baseline_audit_20261101 | 历史环境核查；不是新方法结果 |
| coverage_overlap_range_pilot100_20261101 | 历史诊断，标签相关数值受映射错误影响，[更正](../analyses/diagnostic_mapping_correction_20261101.md) |

## 文件含义

- reported_results.json / comparison.csv：根据用户回传整理；未知为null或CSV空白。
- jobs/<job_id>/summary.reported.json：用户直接粘贴的完整最终JSON，原值保留，来源由该批README说明。
- receipts/：短小原始报告/日志摘录，不含完整大日志。
- progress_snapshots.json：每份状态的观察时刻/来源；旧Counter不冒充最新计数。
- README：目的、对照、已观察结果、限制、故障和下一步。

以上reported文件不是完整export_record产物，不生成虚假的原始status.json/provenance.json/queue.json。证据层级保持“用户服务器回执”，不声称助手直接查看服务器、完成独立重跑或具备多seed支持。

## 追加服务器产物

已有目录放回执后，export_record要求目标目录不存在。将后续真实导出放入同一批次下全新的server_export_日期时间子目录，保留已有记录；不要覆盖回执或删除目录。

只允许冻结已结束的run/单项。当前campaign尚在运行，不能整批export；可导出已完成单项。整批结束后还须处理.interrupted_归档目录被导出器遍历的问题，见[恢复记录](../analyses/campaign_restart_20261102.md)。

保留配置/执行commit、指标/逐类、必要日志摘录、更新与运行成本；不提交权重、逐点预测、原始数据、凭据和大日志。不例行计算文件/权重SHA；新大下载或实际疑似损坏时才按需核对。
