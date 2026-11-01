# 实验轻量证据

沿用遥感项目的记录方式：此目录保存可随 Git 同步的冻结依据。
服务器原始输出留在 runs/；原始模型、数据和旧复现缓存留在原位置。

每个目录使用能解释作用的 run_id，例如：

- env_baseline_audit_20261101
- coverage_overlap_range_pilot100_20261101
- mean_vs_default_pilot100_comparison_20261101

保留 README、status/provenance、指标JSON/CSV、关键评测行与哈希；
不提交checkpoint、逐点预测、原始数据或完整大日志。

截至初始化提交，这里没有新的服务器实测结果。
