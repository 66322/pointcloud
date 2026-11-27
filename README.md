# 点云论文 · DITR 实验记录

本项目从冻结 DINOv2 图像特征注入 PTv3 的 DITR 出发，考察视觉信息能否提高 nuScenes-lidarseg 的三维逐点分割精度。两台电脑和服务器以 GitHub `main` 同步研究计划、代码和轻量结果。

| 要做的事 | 从这里开始 |
|---|---|
| 接手项目，查最新回执 | [交接记录](docs/HANDOFF.md) · [实验总表](docs/EXPERIMENT_MASTER.md) |
| 看本轮问题、对照和证据 | [基于 DITR 的实验计划](docs/research/20261126_proposal_experiments.md) · [论文与代码的数据流](docs/BASELINE_AND_DATAFLOW.md) |
| 运行或预览 | [双卡运行手册](docs/PROPOSAL_RUNBOOK.md) · [正常六路默认配置](configs/training/proposal_20261126.json) |
| 找旧实验、失败和逐类结果 | [路线索引](docs/EXPERIMENT_MATRIX.md) · [11 月 8 日五组十轮](docs/RESOURCE_EXPERIMENT_RUNBOOK.md) · [实验回执目录](records/experiments/README.md) |
| 换电脑、同步回执 | [GitHub 同步约定](docs/GITHUB_SYNC.md) |

## 当前安排

默认使用一个共同种子 20261101，比较正常六路输入时的原 DITR 与有效视角均值，各十轮，每个最终权重评一次完整 6,019 帧。已有两轮结果和匹配的服务器产物优先使用。需要一起跑旧路线时，可选合并队列加入原来的四种 dropout，共享同一个 control。局部对应有独立的两轮筛查；质量门控、时序补偿、蒸馏和跨库泛化根据先期证据安排。三个开题想法各自接受检验。

| 已有观察 | 可比较的数值 |
|---|---:|
| 同一原权重，原赋值 → 有效视角均值 | 82.48% → 82.64% |
| 两轮微调，正常六路，control → mean | 82.20% → 82.47% |
| 两轮微调，固定三路，control → camera_dropout | 60.79% → 73.13% |
| 上一行同一对模型，正常六路 | 82.20% → 82.04% |

以上为既有服务器回执，十轮和新筛查的成绩待回传。[原权重推理](records/experiments/view_assignment_fullval_20261101_r02/README.md) · [四组微调](records/experiments/finetune_stage1_20261101_r02/README.md) · [缺失实验](records/experiments/multi_route_20261101_r01/README.md)

`configs/` 存实验定义，`pointcloud_lab/` 存训练评测适配器，`scripts/` 存后台入口。服务器由用户执行，仅用 GPU 0、1。机内路径放在忽略提交的 `configs/paths.local.json`，数据、权重和逐点预测保存在服务器。
