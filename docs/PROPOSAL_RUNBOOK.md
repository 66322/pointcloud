# 实验启动与回执

服务器根目录为 `/home/jovyan/zzx/pointcloud`。GitHub `main` 同步代码、配置和结果摘要。以下是可选的运行方式，先根据实际队列选择一项。显卡使用 0、1，每卡一次执行一个独立任务。[研究问题与比较规则](research/20261126_proposal_experiments.md)

| 选择 | 训练及评测上限 | 实际比较 |
|---|---:|---|
| [默认正常融合](../configs/training/proposal_20261126.json) | 2 训 + 2 评 | 20261101，十轮。原四级解码注入的 control / mean，正常六路完整验证。 |
| [与旧实验合跑](../configs/training/fusion_with_legacy_10ep.json) | 6 训 + 21 评 | 同种子、十轮。一个 control 和 mean，加原来的四种 dropout。六组评正常输入，原五组另评固定 3、1、0 路。该队列已经包含默认两组。 |
| [对应误差筛查](../configs/training/proposal_alignment_pilot.json) | 3 训 + 9 评 | 20261101，两轮。原多级 control、解码后固定中心 p_center、相同位置的局部候选 p_offset。三组均评 clean、−3°、+3°。 |

上限按新建队列计算。匹配的原生队列产物从已有记录复用。三个入口都只用一个训练种子。完整验证按相同的最后权重、6,019 帧和 TTA 运行。单种子记录差值，后续复核只选有依据的候选和直接对照。外部天气和其余开题原型可按研究问题另列配置。[历史 14 模型规划与旧运行手册](https://github.com/66322/pointcloud/blob/5a79e3cf2787c6beeb1e1a00b82c35d579ec140b/docs/PROPOSAL_RUNBOOK.md)

## 预览和启动

保留已验证的 `configs/paths.local.json`。下列预览只读计划，不用数据和 GPU。

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4
python -m pointcloud_lab.campaign --plan configs/training/proposal_20261126.json --preview
python -m pointcloud_lab.campaign --plan configs/training/fusion_with_legacy_10ep.json --preview
python -m pointcloud_lab.campaign --plan configs/training/proposal_alignment_pilot.json --preview
~~~

两卡可用，只验证正常六路时运行这一条。输出名使用新批次 ID。

~~~bash
bash scripts/start_proposal.sh clean_fusion_10ep_r01
~~~

如果也打算做旧四种 dropout，选用下面这条合并入口。调度器在同一队列内向两卡派发，直接对照共享同一份权重。

~~~bash
bash scripts/start_proposal.sh fusion_with_legacy_10ep_r01 --with-legacy
~~~

两种入口选其一。若服务器有上一批在跑，或要读取已完成的十轮产物，在所选命令末尾加 `--after runs/实际批次名`。该目录应是服务器实际存在的上一批。例如已运行过 11 月 8 日的五组，可用 `--after runs/resource_stage1_20261108_r01`。同一对 GPU 用一个活动控制器。新原生队列等上一批终态后，核对训练种子、预算、数据、最后权重和评价条件，按匹配情况补缺。对照缺少有效产物时才重新执行。已有独立的 11 月 1 日 4px、门控、局部注意力等任务保留自己的快照。[旧队列与恢复](CAMPAIGN_RUNBOOK.md)

单独检验局部对应时选下面的入口，两卡已有批次时同样加 `--after`。这批的三组从同一原权重出发，共用按样本固定的训练抽样，50% 为 clean、25% 为 −3°、25% 为 +3°，图像暗化、点云稀疏和相机掩码均不加入这次额外扰动。其 control 的数据配方有别于正常融合队列，需要独立训练。

~~~bash
bash scripts/start_proposal.sh alignment_pilot_2ep_r01 --alignment
~~~

## 先核查历史图像能新增多少覆盖

这个 CPU 诊断可以与训练独立使用。读取关键帧点云、相机标定、图像尺寸和原始 `sample.json`，通过 `sample.prev` 找同场景前一关键帧，把当前点变换到前一帧 LiDAR 坐标系，再投到历史相机。历史相机曝光必须早于当前 LiDAR。默认从 `data_root/raw/v1.0-trainval/sample.json` 读取元数据，其他布局用 `--samples 实际文件`。可传 `--limit N` 或已有 `--tokens` 文件分析同一批点。[覆盖统计代码](../pointcloud_lab/view_coverage.py)

~~~bash
cd /home/jovyan/zzx/pointcloud
run=coverage_triple_a_persistent_r01
mkdir -p "runs/_launch/$run"
nohup python -u -m pointcloud_lab.view_coverage --condition triple_a --outage persistent \
  --out "runs/$run" > "runs/_launch/$run/console.log" 2>&1 < /dev/null &
echo $! > "runs/_launch/$run/pid"
~~~

临时丢帧取 `--outage current-only` 并换新输出名。持续掉线屏蔽同一组历史相机，当前单帧丢失则允许历史六路。输出 `coverage.json` 包含同帧其余视角能接手的点数、可配对帧中当前全部存活相机均看不到的点数、过去帧新增覆盖数、分母及覆盖率、历史年龄范围。没有上一帧的场景起点单列，分母为零时比例记 null。结果表示静态场景假设下的视锥覆盖；实际可用性和分割收益需再检验。时间邻近的模型仍是待验证的方案。

## 核对结果并同步

三种 GPU 启动器共用 nohup、独立运行目录、日志和代码快照。下面把 `run` 设为实际批次。

~~~bash
run=clean_fusion_10ep_r01
cat "runs/$run/status.json"
tail -n 20 "runs/_launch/$run/console.log"
cat "runs/$run/comparison.csv"
~~~

`comparison.csv` 的同条件差值以百分点计。正常融合看 mean 减同预算 control；缺相机看相同保留子集；对应筛查看 `seed_deltas.csv` 中 p_offset 相对 p_center 的差。固定索引 [0,2,4] 的旧三路成绩与按相机名定义的掩码分别注明。恢复已有控制器用 `bash scripts/resume_campaign.sh "$run"`，先区别控制器退出与整机重启，并保留原现场。复用的权重和成绩在新报告中标来源，不能算独立重复。

终态导出再同步。对应筛查还可用 `python -m pointcloud_lab.proposal_record --run "runs/$run" --snapshot "records/experiments/$run/snapshots/新时间戳.json"` 保存中途快照。旧的三种子批次如已开始，沿用自己的快照和已保存队列。

~~~bash
python -m pointcloud_lab.export_record --run "runs/$run" --record "records/experiments/$run/final"
git pull --ff-only
git add -- "records/experiments/$run" docs/HANDOFF.md docs/EXPERIMENT_MASTER.md docs/EXPERIMENT_MATRIX.md
git commit -m "Record results for $run"
git push origin main
~~~

核对原始回执后再更新交接和总表。记录种子、初始化、预算、数据划分、相机名或扰动、执行提交、复用来源、最终指标和失败原因。大权重、预测和完整日志留服务器。CPU 覆盖分析可单独提交轻量摘要，注明采样范围和分母。
