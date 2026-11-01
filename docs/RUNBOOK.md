# 服务器手动执行与 GitHub 回传

本项目由用户手动执行服务器命令。所有新文件位于 /home/jovyan/zzx/；
只使用 GPU 0/1。旧 DITR 源码、原始数据、权重和预测缓存保持原位置。

## 第一次执行：只做环境核对与 100 帧离线诊断

以下是 Linux 服务器命令，使用已有 conda 环境，不重装依赖。
如果仓库已克隆，进入现有 pointcloud 目录执行 git pull --ff-only，
不要重复 clone 或覆盖已有目录。

~~~bash
cd /home/jovyan/zzx
git clone https://github.com/66322/pointcloud.git pointcloud
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4
cp -n configs/paths.server.example.json configs/paths.local.json
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -v
bash scripts/run_stage1.sh preflight env_baseline_audit_20261101
~~~

先查看 runs/env_baseline_audit_20261101/preflight.json：

- 文件路径存在；旧预测数量应为 6019。
- 基线日志行应支持已有的 82.48%/90.03%/94.63%。
- SHA、dirty 状态如与记录不一致，先记录实际情况；不自动重置。
- preflight 不加载权重，不会声称 strict/freeze 已验证。
- 如需显卡信息，单独运行：

~~~bash
nvidia-smi -i 0,1 --query-gpu=index,name,memory.total,memory.used --format=csv
~~~

核对 baseline_predictions 指向最终 *_pred.npy，它们必须是原始点顺序。
已记录的 SemSegTester 输出会做 index 累加及必要的 inverse 回映；
若来自自定义 eval_subset.py，需先确认相同含义。长度一致只是必要条件。

确认后执行：

~~~bash
bash scripts/run_stage1.sh diagnose-pilot coverage_overlap_range_pilot100_20261101
~~~

运行只读旧缓存、LiDAR、标签和图像头部尺寸，不使用 GPU、不修改旧数据。
产物集中在 runs/coverage_overlap_range_pilot100_20261101/：

- report.md：分组概览。
- summary.json / confusions.json：指标及混淆矩阵。
- per_class.csv：各分组逐类 IoU、真值样本数、union。
- frame_manifest.json：每帧计数与数组指纹。
- tokens.json：后续配对实验必须使用的完全相同 100 帧。
- provenance.json / status.json：来源和完成/失败状态。

若任一步失败，保留该目录，不在原目录重跑；修复后使用新 run_id 后缀 r02。
不应通过删除目录来绕过防覆盖检查。

## 第一次回传：先回传这两项

~~~bash
python -m pointcloud_lab.export_record \
  --run runs/env_baseline_audit_20261101 \
  --record records/experiments/env_baseline_audit_20261101
python -m pointcloud_lab.export_record \
  --run runs/coverage_overlap_range_pilot100_20261101 \
  --record records/experiments/coverage_overlap_range_pilot100_20261101
~~~

阅读两个导出目录，补充 README 中的目的、结果和问题，检查 provenance 中
路径/主机名是否适合公开。导出程序不会执行 git，也不会带走大权重或预测。

~~~bash
git pull --ff-only
git add records/experiments/env_baseline_audit_20261101 records/experiments/coverage_overlap_range_pilot100_20261101
git commit -m "Record baseline audit and 100-frame coverage diagnostic"
git push origin main
~~~

另一台电脑仅需 git pull --ff-only 查看相同记录。
不要 git add .；不要用 force push。若拉取遇到本地修改，先提交自己明确要同步
的记录再拉取，不使用 reset --hard 清理。

## 第二步：完整离线诊断

pilot 对齐和计数通过后：

~~~bash
bash scripts/run_stage1.sh diagnose-full coverage_overlap_range_fullval_20261101
~~~

完整结果应覆盖6019帧，固定16类的mIoU应与已有82.48%接近（原记录已舍入）。
若不一致，先查点序、GT映射、索引和缓存来源；不修改协议追分。

## 第三步：一帧默认模型 smoke

生成 1 帧清单，与 100 帧采用同一 seed，保证嵌套。
**先做此项，不先排整夜训练。**

~~~bash
python -m pointcloud_lab.select_tokens --paths configs/paths.local.json \
  --limit 1 --out runs/manifests/pilot1_seed20261101.json
python -m pointcloud_lab.evaluate --paths configs/paths.local.json \
  --experiment configs/experiments/default_assignment.json \
  --tokens runs/manifests/pilot1_seed20261101.json --expected-frames 1 \
  --gpu 0 --out runs/default_assignment_smoke1_20261101
~~~

这里调用原作者 SemSegTester，保留完整 TTA。需要：

- strict=True 加载成功；任何不匹配直接失败，不跳过参数。
- model_audit.json 中 DINO trainable elements=0、training flag=false。
- test.log 完成、result/ 下恰有一个 *_pred.npy。
- registered DINO 模式是加载兼容权重所需设置；后续若要训练仍需独立审计优化器。

当前适配器只支持单GPU评测。--gpu 限制为0或1，设置物理设备后才导入torch。
它不修改已验证的上游源文件，只在当前进程替换图像特征赋值函数。
默认设置HF_HUB_OFFLINE=1，缺少缓存时失败并记录，不自动下载。
若离线缓存位置与原环境不同，先核对 paths.local.json/hf_cache 与 HF_HOME，
不要通过重下载整个数据或重装环境处理。

## 第四步：同 100 帧的默认/首视图/均值干预

smoke通过后执行，同一路径不存在才能启动。以下可串行执行；若要并行，
须手动分配0/1，先确认显存，避免两项同时挤到同一GPU。

~~~bash
python -m pointcloud_lab.evaluate --paths configs/paths.local.json \
  --experiment configs/experiments/default_assignment.json \
  --tokens runs/coverage_overlap_range_pilot100_20261101/tokens.json \
  --gpu 0 --out runs/default_assignment_pilot100_20261101

python -m pointcloud_lab.evaluate --paths configs/paths.local.json \
  --experiment configs/experiments/first_visible_camera.json \
  --tokens runs/coverage_overlap_range_pilot100_20261101/tokens.json \
  --gpu 0 --out runs/first_visible_camera_pilot100_20261101

python -m pointcloud_lab.evaluate --paths configs/paths.local.json \
  --experiment configs/experiments/mean_visible_cameras.json \
  --tokens runs/coverage_overlap_range_pilot100_20261101/tokens.json \
  --gpu 1 --out runs/mean_visible_cameras_pilot100_20261101
~~~

推荐通过 tmux 进入会话后运行命令，断开网页不影响任务。
不提供自动占用GPU、自动后台启动或批量训练队列。

每项使用全新 save_path，原始6019帧缓存绝不放入新save_path。
若更换权重、投影、聚合、TTA或帧清单，使用新的run_id。
运行输出的 model_audit/provenance/runtime/status 和 test.log 会保存在各目录。

## 第五步：对新预测做相同诊断并比较

以默认和均值为例；首视图按相同方式做：

~~~bash
python -m pointcloud_lab.diagnose --paths configs/paths.local.json \
  --tokens runs/coverage_overlap_range_pilot100_20261101/tokens.json \
  --predictions runs/default_assignment_pilot100_20261101/result \
  --out runs/default_assignment_pilot100_analysis_20261101 \
  --confirm-original-point-order

python -m pointcloud_lab.diagnose --paths configs/paths.local.json \
  --tokens runs/coverage_overlap_range_pilot100_20261101/tokens.json \
  --predictions runs/mean_visible_cameras_pilot100_20261101/result \
  --out runs/mean_visible_cameras_pilot100_analysis_20261101 \
  --confirm-original-point-order

python -m pointcloud_lab.compare \
  --reference runs/default_assignment_pilot100_analysis_20261101 \
  --candidate runs/mean_visible_cameras_pilot100_analysis_20261101 \
  --out runs/mean_vs_default_pilot100_comparison_20261101
~~~

比较工具要求 index SHA、原始点/GT指纹、token顺序和各组真值支持一致。
先比较旧缓存诊断 vs 新默认诊断，确认默认模型没有漂移，再看干预差值。

100帧的缺失类别会影响固定16类mIoU，因此同时输出present-class指标与样本数。
没有scene-map时不生成置信区间；不把数百万点当独立样本来制造很窄区间。
后续可提供 --scene-map（JSON: lidar_token→scene_token），至少两个场景才做
场景簇bootstrap。当前初筛不以置信区间作为继续/停止的唯一条件。

## 归档新的结果

对每个已完成/失败的run重复 export_record，再审核 README、git add 指定记录、
commit/push。同时更新 docs/EXPERIMENT_MASTER.md 对应行和结论。
保留失败原因，不挑最好结果。

原始运行位于 runs，不进入Git；稳定的轻量依据位于records。
同一结果目录只归档一次，后续复跑使用新run_id，不覆盖历史证据。
