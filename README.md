# Pointcloud — DITR改进实验

GitHub main是两台电脑和服务器的共同版本。服务器命令由用户手动执行。

**换电脑先读：[交接记录](docs/HANDOFF.md) · [实验总表](docs/EXPERIMENT_MASTER.md) · [完整路线](docs/EXPERIMENT_MATRIX.md)**

最新依据：用户2026-11-02回传的服务器输出；最近带时间的完成事件为北京时间14:15:23，随后日志确认下一批已启动。这是回执快照，不是实时监控。

| 实验 | 当前已确认结果 |
|---|---|
| 历史完整val基线 | 82.48% mIoU，沿用 |
| 原权重推理mean / first | 已完成：82.64% / 82.55% |
| 首批四组两轮微调 | 已完成：control82.20、gate82.21、mean82.47、camera_dropout82.04 |
| 多路线两轮clean验证 | 11组有结果，valid_pool/valid_both训练失败 |
| 缺相机验证 | control 3/1/0路=60.79/49.20/36.80；camera_dropout 3路=73.13 |
| 最近启动任务 | GPU0 camera_dropout 0路；GPU1 camera_dropout 1路 |
| 十轮训练与4px偏移评价 | 尚无最终结果回传；valid_both十轮已被依赖失败阻断 |

固定三相机条件下camera_dropout相对control +12.34pp，正常输入−0.16pp。当前两轮clean最佳mean82.47%，未超过历史82.48%；单seed/固定子集结果不作稳定泛化结论。

## 当前操作

旧四组及推理r02已经完成。多路线r01机器重启后的两项验证已恢复完成，并自动接续。保持现有队列，不重复启动同名批次。

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
tail -n 12 runs/_launch/multi_route_20261101_r01/resume_console.log
~~~

日志与完整产物在服务器runs/，GitHub不自动镜像实时进度。恢复边界与查看命令见[队列手册](docs/CAMPAIGN_RUNBOOK.md)；机器重启的实际处理见[恢复记录](records/analyses/campaign_restart_20261102.md)。仅git pull不会更改已保存的任务代码快照。

## 结果与目录

- [推理r02](records/experiments/view_assignment_fullval_20261101_r02/README.md)
- [四组微调r02](records/experiments/finetune_stage1_20261101_r02/README.md)
- [多路线r01](records/experiments/multi_route_20261101_r01/README.md)：JSON、CSV、逐类回传、失败和进度快照。
- [结果解释与校核](records/analyses/results_interpretation_20261102.md)
- [双电脑同步约定](docs/GITHUB_SYNC.md)

| 目录 | 用途 |
|---|---|
| docs/ | 交接、实验问题/协议、操作手册 |
| configs/ | 实验定义；本机路径放ignored的paths.local.json |
| pointcloud_lab/ | 实验适配器与汇总代码 |
| runs/ | 服务器完整输出，不进Git |
| records/experiments/ | 按run_id组织的轻量依据，进入Git |
| records/analyses/ | 结论边界、故障与恢复分析 |
| tests/ | 本地开发检查，不要求服务器每次重跑 |

接受已有82.48%基线；新路径只确认一次可运行后直接实验，不恢复20/100帧关卡或例行哈希扫描。旧离线诊断81.7190%/84.3997%受标签20映射错误影响，不能当有效基线或提升，见[更正](records/analyses/diagnostic_mapping_correction_20261101.md)。

原始DITR、数据、权重、旧预测保持原位置。基于[DITR](https://github.com/RWTHVision/ditr)。
