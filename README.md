# Pointcloud — DITR 改进实验

以 GitHub main 为两台开发电脑和服务器的共同版本，用户在服务器手动执行命令。

**实验目的、进度、路线与结论统一入口：**
[实验总表](docs/EXPERIMENT_MASTER.md)

**本轮先执行：**
[服务器运行手册](docs/RUNBOOK.md)中的“第一次执行：只做环境核对与 100 帧离线诊断”。

当前状态：已整理 nuScenes 16 类、PTv3 + DINOv2-L 的既有复现记录：
完整 val 6019 帧，mIoU **82.48%**。来源为用户提供的复现说明；
原始服务器日志和检查点尚待本仓库的 preflight / smoke 重新核对。
作者 83.1% 仅列参考，差距 -0.62 个百分点尚未解释。

| 目录 | 用途 |
|---|---|
| docs/ | 总计划、运行手册、基线事实与数据流 |
| configs/ | 实验定义；机器路径另放被忽略的 paths.local.json |
| pointcloud_lab/ | 独立诊断、运行适配器、统计及记录导出 |
| scripts/ | 固定流程入口 |
| tests/ | CPU 统计与数据对齐测试；服务器补充 torch 测试 |
| runs/ | 服务器完整运行输出，不进 Git |
| records/experiments/ | 审核后的轻量实验依据，进入 Git |

第一批：旧预测分组诊断 → 单帧默认推理 → 同 100 帧默认/首视图/均值干预。
当前没有新增训练结果，也没有已证实的改进分数。

所有新输出使用唯一 run_id；重复目录直接报错。数据、权重与旧预测不搬动。
研究依据来自 [DITR](https://github.com/RWTHVision/ditr)，固定上游
f66d1dadb82e97ded7750567a043e4783305b01c。
