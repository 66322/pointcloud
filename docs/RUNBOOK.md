# 当前实验执行手册（精简流程）

按用户2026-11-01要求：接受已有82.48%完整val基线；新入口只跑通一帧，随后直接正式实验。
不再重复preflight、20/100帧试验、全量旧缓存统计或文件/权重哈希扫描。

## 当前r02已运行

用户终端已确认单帧通过，GPU0/1的两项6019帧正式推理实验已启动。
这是已有权重的融合策略对照，不是训练。当前任务继续运行，不重复启动。

本轮是前台启动。如果需要转后台，在原终端按Ctrl+Z，看到Stopped后执行：

~~~bash
bg %+
disown -h %+
jobs -l %+
~~~

这会恢复任务并禁止Bash在退出时向该job转发SIGHUP；它不改变已打开的终端
输出文件描述符，不等价于从一开始nohup并重定向。若要最稳妥，本轮保留网页。
后续长任务统一使用下方nohup启动器。

## 后续启动方式：默认nohup

以下供需要启动新批次时使用，当前r02运行中不要重复执行：

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --rebase
conda activate pointcept-torch2.5.0-cu12.4
bash scripts/start_view_experiments.sh view_assignment_fullval_20261101_r03
~~~

脚本自动完成：

1. GPU0上运行一次单帧mean聚合，确认新入口可用。
2. 成功后直接并行跑完整val：GPU0多视图均值、GPU1首有效视图。
3. 沿用兼容权重、作者配置与TTA，由原作者SemSegTester计算最终指标。
4. 自动汇总两项结果，与已验证的82.48%比较；不额外重跑默认模型。

只有单帧失败才停止并保留报错，不追加20/100帧关卡。后续同一入口已跑通可用
--skip-smoke。每次使用新的输出名r02/r03，不覆盖已有结果。

此批是无需训练的融合策略对照。策略可能提高或降低精度；完整结果用于决定
下一步是否训练质量加权/局部对应/门控/模态dropout，而非预先认定方法有效。

## 保存位置

所有新实验集中在：

/home/jovyan/zzx/pointcloud/runs/view_assignment_fullval_20261101_r02/

- smoke_console.log：唯一一次单帧检查的终端输出。
- mean_visible_cameras/：GPU0完整实验，包括test.log、summary.json、预测。
- first_visible_camera/：GPU1完整实验。
- *_console.log：对应实验完整终端输出。
- comparison.csv、summary.json、report.md：自动汇总。
- status.json：批次状态；运行中/完成/失败均记录。

旧DITR和数据路径不变；新代码只在独立进程中替换特征赋值，不覆盖旧代码。
只使用GPU0和1。nohup启动器会将stdin接到/dev/null、stdout/stderr写入文件，
并在runs/_launch/<run_id>/中保存console.log和pid。普通网页断连不再依赖终端输出。
脚本打印启动信息后即可返回提示符。已有run/launch目录会拒绝再次启动。

例如新批次r03的父进程日志位于：
runs/_launch/view_assignment_fullval_20261101_r03/console.log。
实际推理日志和结果仍在runs/<run_id>/，当前r02的路径如下。

在另一个终端查看进度（可选）：

~~~bash
cd /home/jovyan/zzx/pointcloud
tail -n 10 runs/view_assignment_fullval_20261101_r02/mean_visible_cameras_console.log
tail -n 10 runs/view_assignment_fullval_20261101_r02/first_visible_camera_console.log
~~~

r01因timm额外读取DINO预训练缓存失败，已修复。现在构建结构后直接严格加载完整兼容权重，无需下载或迁移缓存。若再次失败，入口会直接显示日志最后30行。

## 结束后同步结果

~~~bash
python -m pointcloud_lab.export_record \
  --run runs/view_assignment_fullval_20261101_r02 \
  --record records/experiments/view_assignment_fullval_20261101_r02

git add records/experiments/view_assignment_fullval_20261101_r02
git commit -m "Record full-val view assignment experiments"
git pull --rebase
git -c credential.helper= push origin main
~~~

一次导出会带上两项子实验的轻量结果，不带预测和权重，不例行计算哈希。
完成后把report.md发给我或告知已上传即可，失败同样记录。

## 统计错误更正

旧离线诊断把原始标签20误映射为motorcycle，而DITR应将它ignore。已修正。
旧100帧84.3997%及全量81.7190%不作为有效改进指标或新基线。
保留旧产物，不要求重新跑旧诊断。基线仍为已有原作者评测流程得到的82.48%。
详见records/analyses/diagnostic_mapping_correction_20261101.md。

## 最小记录要求

记录实验配置、代码commit、seed、数据划分、最终指标、日志和独立输出目录。
只保留必要的形状/标签检查、严格权重加载和DINO冻结；不设SHA一致性门槛。
大文件新下载或真实疑似损坏时，才按需核对checksum。
鉴权问题见[GITHUB_SYNC.md](GITHUB_SYNC.md)。

作业控制依据：[Bash Signals](https://www.gnu.org/s/bash/manual/html_node/Signals.html)；
[GNU nohup](https://www.gnu.org/s/coreutils/manual/html_node/nohup-invocation.html)。
