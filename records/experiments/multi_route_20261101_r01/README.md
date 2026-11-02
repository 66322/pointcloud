# 多路线队列：2026-11-02回执归档

最近带时间的已完成事件：北京时间2026-11-02 14:15:23。随后父日志确认两项新评价START；精确查询时刻未提供。本目录是用户回传记录，不能代替服务器实时queue.json，也不是完整服务器导出。

## 实际范围

计划50个节点：旧四组8个训练/评价节点已复用；新增9组两轮训练、6组十轮训练、27次完整验证。两轮13种方法中11组已完成clean验证，2组训练失败；缺相机9项中已收到4项最终mIoU。三项4px偏移和十轮结果尚未回传，valid_both十轮被依赖失败阻断。

## 两轮正常输入结果

| 方法 | 处理变化 | 验证问题 | mIoU % | 相对两轮control pp |
|---|---|---|---:|---:|
| control | 原赋值/加法继续训练 | 分离额外训练预算本身的影响 | 82.20 | 0.00 |
| gate | 四级图像注入的恒等初始化门控 | 学习注入强度是否优于原加法 | 82.21 | +0.01 |
| mean | 有效多视图特征均值 | 重新训练适配均值后是否有收益 | 82.47 | +0.27 |
| camera_dropout | 逐相机 p=0.2 删除对应证据 | 正常精度与缺相机能力的取舍 | 82.04 | -0.16 |
| center_select | 图像中心质量代理 top-1 | 固定质量选择是否优于原选择/均值 | 82.46 | +0.26 |
| center_weighted | 中心质量代理加权 | 固定质量加权是否优于等权均值 | 82.42 | +0.22 |
| bilinear | 对齐后的双线性取样 | 离散 nearest 取样是否限制融合 | 82.36 | +0.16 |
| local_average | 3×3 有效图像 token 邻域均值 | 无参数局部上下文是否有益 | 81.86 | -0.34 |
| local_attention | xyz/strength 查询的3×3局部注意力 | 学习局部对应是否优于原取样/局部均值 | 82.21 | +0.01 |
| valid_mask | 投影后无观测点零注入 | 无对应图像支路的偏置是否影响结果 | 82.06 | -0.14 |
| valid_pool | 图像池化排除无效项 | 无效零值是否压过有效负特征 | 训练失败 | — |
| valid_both | 有效性 mask 与有效池化组合 | 两种处理是否需要共同传播 | 训练失败 | — |
| whole_camera_dropout | p=0.2 整组六相机证据丢弃 | 显式整组缺失训练是否优于逐相机 dropout | 82.41 | +0.21 |

旧四组是复用，不算新增独立训练重复。七组新增成功训练的optimizer更新数、mAcc/OA和逐类结果尚未回传。

## 相机缺失结果

| 模型 | 正常6相机 | 固定3相机 | 固定1相机 | 0相机证据 |
|---|---:|---:|---:|---:|
| control | 82.20 | 60.79 | 49.20 | 36.80 |
| camera_dropout | 82.04 | 73.13 | 最近回执运行中 | 最近回执运行中 |
| whole_camera_dropout | 82.41 | 最近完整快照待执行 | 最近完整快照待执行 | 最近完整快照待执行 |

三相机=[0,2,4]，单相机=[0]，零相机=[]，由固定协议定义；不是全部组合均值。camera_dropout三相机相对同条件control +12.34pp；正常输入−0.16pp。其clean→3相机降幅8.91pp，control为21.41pp；降幅差12.50pp与同条件差12.34pp是不同量。

完整回传的两份16类结果已保存：

- [control零相机](jobs/robust_control_cam0/summary.reported.json)：mIoU36.80%、mAcc44.59%、OA85.42%。
- [camera_dropout三相机](jobs/robust_camera_dropout_cam3/summary.reported.json)：73.13%、81.12%、93.68%。

零相机证据不是单独训练的纯LiDAR模型；OA受类别占比影响，不能取代mIoU。

## 失败与恢复

valid_pool、valid_both都在首次训练loss检查失败，错误为Non-finite first training loss；尚无首次成功更新。具体算子/根因未定位，不能记为算法无收益。阻断screen_valid_pool_clean、screen_valid_both_clean、medium_valid_both_train及medium_valid_both_clean；独立任务继续。

用户报告10点多机器重启。两项中断评价的旧输出和日志已改名归档，重新排队后从头计算；已有完成项未重跑。恢复控制器PID59334。两项恢复验证已在14:15结束，随后GPU1启动camera_dropout单相机（PID100312），GPU0启动零相机（PID100432）。PID仅代表该回执时刻，不能据此直接终止进程。

[恢复详细记录](../../analyses/campaign_restart_20261102.md)、[原始摘录](receipts/server_events_20261102.txt)、[分时进度快照](progress_snapshots.json)。

## 数据与解释

[结构化结果](reported_results.json)、[CSV](comparison.csv)、[描述性校核](../../analyses/results_interpretation_20261102.md)。当前最明确的观察是固定三相机条件下dropout有12.34pp收益；单seed结果不能证明所有缺失组合或稳定泛化收益。所有两轮clean结果均未超过历史82.48%基线，mean82.47%最高。

尚缺：有效配置/运行快照commit、绝大多数逐类summary、各训练AMP统计、runtime/显存和后续任务最终结果。未知字段留null，不从其他模型/条件补值。
