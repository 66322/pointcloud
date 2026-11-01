# Pointcloud — DITR改进实验

GitHub main是两台电脑和服务器的共同版本。用户手动执行服务器命令。

**当前直接做实验：** [运行命令](docs/RUNBOOK.md) · [实验总表](docs/EXPERIMENT_MASTER.md)

[完整路线与实施状态](docs/EXPERIMENT_MATRIX.md) · **[无人值守接续队列](docs/CAMPAIGN_RUNBOOK.md)**：
旧四组继续；新入口已实现9组短训、6组十轮训练及配对压力评价。
现在启动一次后自动等待旧批次、复用完成项、两卡接续，方法失败不停止其他独立任务。
新入口服务器待用户启动；显式r/u分支、选择性蒸馏等仍未实现。

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4
bash scripts/start_campaign.sh multi_route_20261101_r01 \
  --after runs/finetune_stage1_20261101_r02
~~~

已有nuScenes 16类、PTv3+DINOv2-L完整val基线为82.48% mIoU。
代码/数据/权重已准备好；新入口只确认一次单帧可运行，就直接进行实验。
不重复默认基线，不设20/100帧关卡，不例行做文件/权重哈希检查。

当前两项正式对照：GPU0有效多视图均值、GPU1首有效视图，完整val，自动汇总。
用户末帧回执：mean累计mIoU 82.64%（+0.16pp），first 82.55%（+0.07pp）。
最终status.json及report.md已确认completed；OA/逐类轻量归档待接收，单次小幅提升先作候选线索。

推理r02两项已遍历完验证集，读取现有汇总即可，不要重复启动。
提速入口已加入有序I/O/CPU预取、同帧冻结DINO特征复用及真实墙钟进度；
H100上的等价性和加速幅度尚待新入口首次单帧确认及后续正常运行日志。
该推理r02使用启动时的旧代码，不为测速度重跑完整val。
训练入口已就绪，见[训练命令](docs/TRAINING_RUNBOOK.md)和[实验总表](docs/EXPERIMENT_MASTER.md)。
首批为同预算的control/gate/mean/camera_dropout，各2 epoch，两个GPU自动排队，
训练r01的优化器类型适配错误已修复。训练r02于20:49通过H100首次更新检查，
20:50回执确认control/GPU0、gate/GPU1均到140/9376，mean/dropout随后自动排队。
当前训练继续，不要重复启动。以下启动命令仅供尚未启动本批次时使用，默认nohup：

~~~bash
bash scripts/start_training_experiments.sh finetune_stage1_20261101_r02
~~~

启动信息、PID和父进程日志位于runs/_launch/<run_id>/；
实际实验产物仍在runs/<run_id>/。

| 目录 | 用途 |
|---|---|
| docs/ | 实验总表、命令与基线说明 |
| configs/ | 实验定义，本机路径放ignored的paths.local.json |
| pointcloud_lab/ | 实验代码与结果汇总 |
| runs/ | 完整服务器输出，不进Git |
| records/experiments/ | 轻量实测结果，进入Git |
| tests/ | 开发时验证代码；不要求服务器每次重跑 |

旧离线诊断的原始标签20映射曾出错，已修正；旧81.7190%不作为新基线。
详见[更正说明](records/analyses/diagnostic_mapping_correction_20261101.md)。
原始DITR、数据、权重、已有预测保持原位置，新实验输出独立保存。
基于[DITR](https://github.com/RWTHVision/ditr)。
