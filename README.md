# Pointcloud — DITR改进实验

GitHub main是两台电脑和服务器的共同版本。用户手动执行服务器命令。

**当前直接做实验：** [运行命令](docs/RUNBOOK.md) · [实验总表](docs/EXPERIMENT_MASTER.md)

已有nuScenes 16类、PTv3+DINOv2-L完整val基线为82.48% mIoU。
代码/数据/权重已准备好；新入口只确认一次单帧可运行，就直接进行实验。
不重复默认基线，不设20/100帧关卡，不例行做文件/权重哈希检查。

当前两项正式对照：GPU0有效多视图均值、GPU1首有效视图，完整val，自动汇总。
代码已准备，暂无这两项的真实改进分数。

~~~bash
python -m pointcloud_lab.run_views --paths configs/paths.local.json \
  --out runs/view_assignment_fullval_20261101_r02
~~~

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
