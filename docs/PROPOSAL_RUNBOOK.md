# 开题新增实验运行手册

在服务器现有 DITR 环境运行。先读[实验规划](research/20261126_proposal_experiments.md)，再看[机器可读矩阵](../configs/training/proposal_20261126.json)。本批为新训练和新评测，历史 82.48% 基线沿用。所有新方法、共同增强和最终指标都由本仓库的运行时适配器接入。

## 双卡启动

保留服务器原有 `configs/paths.local.json`。其中 `ditr_root`、`config`、`weight`、`data_root`、`info`、`hf_cache` 沿用已验证值。若要一并做外部验证，在同一 JSON 加入 `semanticstf_root`，值取服务器上直接包含 `calib/` 和 `val/val.txt` 的实际绝对路径。该键只在启用外部验证时使用。[原服务器路径示例](../configs/paths.server.example.json)

```bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4

# 有 SemanticSTF 时的整批入口。GPU 0、1 各运行一个独立任务。
bash scripts/start_proposal.sh proposal_20261126_r01 --with-stf
```

仅运行 nuScenes 时省略 `--with-stf`。如两卡已有旧控制器，用同一条启动命令附加 `--after runs/实际旧批次名`，在旧队列终态后接续。这两种写法按实际情况选择一次。启动器通过 `nohup` 后台运行、断开标准输入、重定向输出，并记录 PID。调度器在每次派发前检查卡上计算进程。旧批次和新批次各自从自己的代码快照运行。

单种子可传 `--seed 20261101`；预注册的三个种子合在同一批次可直接得到完整的跨种子统计。以下命令只预览任务数，不读数据、不占卡。

```bash
python -m pointcloud_lab.campaign \
  --plan configs/training/proposal_20261126.json --preview --with-stf
```

| 工作量 | 全部三个种子 |
|---|---:|
| nuScenes 训练，每种子 14 个模型 | 42 |
| nuScenes 正常输入，每种子 14 个模型 | 42 |
| 外参扰动，每种子 7 模型 × 5 条件 | 105 |
| 质量退化，每种子 7 模型 × 4 条件 | 84 |
| 相机缺失，每种子 5 模型 × 9 条件 | 135 |
| SemanticSTF，另选 5 模型 × 3 种子 | 15 |

默认每个训练为 30 轮、batch 6、每轮 4,688 次计划迭代，最后权重进入评测。nuScenes 每个条件覆盖完整 6,019 帧和原 TTA；STF 每次覆盖完整 val 250 帧，以同一份预测计算四种天气。无壁钟超时。任务数与实际耗时分开记录，两个 GPU 的可用显存、存储吞吐和新路径运行时间需由服务器实测。完整权重、日志与逐点预测保存在服务器，启动前可用 `df -h /home/jovyan/zzx` 查看空间。

## 实际处理

| 项目 | 定义 |
|---|---|
| 共同训练 | 每个样本按种子、轮次及索引固定。50% 正常、15% 在五种外参处理中等概率选一、15% 在四种质量处理中等概率选一、20% 随机缺相机。缺失时等概率选保留数 0–5，再均匀选具体相机。各组严格加载同一份原兼容 checkpoint，重新建立优化器 |
| 外参 | 在相机坐标系左乘局部 y 轴误差旋转，或沿局部 x 轴平移。用相机内参重投影并重算有效掩码。评测在 ±1°、±3°、+0.20 m 五个固定条件下逐一进行 |
| 图像、点云质量 | 原 RGB 中央 1/2 高 × 1/2 宽区域乘 0.2，或 31×31、σ=8 高斯模糊。距离 ≥20 m 且相对原始 LiDAR +x 方位在 ±60° 内的点按帧固定随机保留约 25%。联合条件共用相同索引 |
| 候选与拒配 | 冻结 DINO 语义特征与 RGB token 细节相加。正常路径五个候选，主点固定，另四个由已编码点特征在短、长两个尺度上修正；双线性采样。语义相似度、稀疏 z-buffer 和学习的图像对数深度参与评分。启用拒配时加入零视觉值的空匹配，剩余视觉质量直接决定注入幅度 |
| 图像深度监督 | 最近正深度作为有支持 token 的稀疏目标。对标定未扰动且当前可用的训练视角使用 Smooth L1，对数米尺度，损失权重 0.1。测试时深度头仅用图像输入。片段内的深度缓冲仍是 LiDAR 投影代理 |
| 质量与双向残差 | 图像使用亮度、对比度、梯度、拉普拉斯响应；点云用 1 m 体素内点数、局部位置离散程度及距离。P→I 按图像 token 汇聚点消息，再从更新后的网格采样得到 I→P 残差。共享门对同尺寸的两项取均值，质量盲对照将 8 个质量量置零 |
| 缺失与蒸馏 | 主视角失效时，在物理相邻且可见的视角中选更接近成像中心的一路，使用三候选和相对朝向/基线门。无观测时按配置返回点特征或局部均值/最大值后备特征。p_distill 对原可见而学生走后备的有标签点使用全视角在线教师，温度 2、KL 权重 0.5；教师目标停止梯度 |
| 相机开销 | 新模型在评测中只编码被保留的视角，零相机和 p_lidar 跳过 DINO。原 DITR 对照仍走原图像编码路径。本批在完整数据上模拟故障，原始读取、解码和点投影仍发生。端到端耗时含数据、TTA 与写盘 |
| STF 外部验证 | 读五列 float32 点与 uint32 标签低 16 位；0、20 及未映射目标忽略。采用 HDL64-S3 到左目外参加 R 与 P 投影，再将点骨干输入的水平朝向对齐到 nuScenes 前相机标定。单前视相机独立输入；全程无目标域微调或模型选择 |

保留相机的物理环序为 FRONT、FRONT_RIGHT、BACK_RIGHT、BACK、BACK_LEFT、FRONT_LEFT。`triple_a` 保留 FRONT、BACK_RIGHT、BACK_LEFT，`triple_b` 保留其余三台，`only_*` 一次保留指定的一台，`none` 为零台。旧三路结果使用历史索引子集，阅读时应分别列出各自的相机定义。

## 查看运行和接续

```bash
run=proposal_20261126_r01
cat "runs/_launch/$run/pid"
tail -n 20 "runs/_launch/$run/console.log"
cat "runs/$run/status.json"
cat "runs/$run/summary.json"
```

训练或评测日志位于 `runs/<run_id>/logs/<job_id>.log`，作业保存独立的有效配置与结果。控制器意外退出后，以 `bash scripts/resume_campaign.sh "$run"` 接管已有队列；它采用仍存活的子进程并跳过已完成作业。整机重启后，已经失去子进程且没有完成凭据的任务会标记失败，其依赖标记阻断，其余独立任务继续。保留中断产物，按[已有恢复记录](../records/analyses/campaign_restart_20261102.md)区分训练断点与结果回执。查看源代码时使用运行目录中的 `code/snapshot.json` 确定执行提交。

## 同步实验记录

运行期间可为同一批次保存多份带时间的快照。每份快照读取同一个已落盘的 `queue.json`，保存观察时间、队列更新时间、各作业的种子、状态、已完成分数。终端旧 PID 只描述对应观察时刻。

```bash
run=proposal_20261126_r01
stamp=$(date -u +%Y%m%dT%H%M%SZ)
python -m pointcloud_lab.proposal_record --run "runs/$run" \
  --snapshot "records/experiments/$run/snapshots/$stamp.json"
```

整批终态后导出轻量证据，包括负结果、失败和阻断项。执行提交取自启动快照，记录提交由 Git 另行生成。核对文件后同步同一分支。

```bash
run=proposal_20261126_r01
python -m pointcloud_lab.export_record \
  --run "runs/$run" --record "records/experiments/$run/final"
git pull --ff-only
git add -- "records/experiments/$run"
git commit -m "Record proposal experiment $run"
git push origin main
```

| 文件 | 读法 |
|---|---|
| plan.json、selection.json、seed_*_protocol.json、provenance.json | 已选择的种子、完整矩阵、训练配置及执行提交 |
| comparison.csv | 每个种子、模型、条件的 mIoU/mAcc/OA 与同条件 DITR 差值。稀疏条件的干净损失按完全相同的原始点子集计算 |
| seed_deltas.csv、contrasts.csv | 预注册直接对照，同种子内先平均指定条件，再汇总种子。样本标准差至少需两个完整配对；完整分母同时列出 |
| jobs/<job_id>.json | 最终逐类 IoU、标签数、路由/距离或天气分层、耗时显存、模型审计或失败原因。路由分层用相同名义可见性定义；门强度另按 TTA 片段访问计数 |
| tokens.json、stf_tokens.json | 顺序固定的验证帧标识。跨库主指标固定九类，分组另报并集非空类别 mIoU 和支持数 |

研究结论、限制和人工回执补到该批说明，并一同更新[交接记录](HANDOFF.md)、[实验总表](EXPERIMENT_MASTER.md)及[路线表](EXPERIMENT_MATRIX.md)。本次本地开发记录另存于[代码验证记录](../records/analyses/proposal_validation_20261126.md)。
