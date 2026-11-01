# 训练r02：H100首次更新通过（2026-11-01）

来源：用户粘贴的服务器日志，源版本81ec641；助手没有SSH。
本记录截至北京时间20:50:40，不代表实时状态或整批完成。

运行目录：/home/jovyan/zzx/pointcloud/runs/finetune_stage1_20261101_r02/
父队列PID3975913；control/GPU0 PID3975987，gate/GPU1 PID3975988。
train索引28130条，batch6、每轮4688步、2轮共9376次计划迭代，协议未改。

## H100实见回执

~~~text
[2026-11-01 20:49:34,484 INFO train.py line 235 3975987] First actual update passed: forward/backward connected, DINO frozen/eval; saving checkpoint
[2026-11-01 20:49:34,483 INFO train.py line 235 3975988] First actual update passed: forward/backward connected, DINO frozen/eval; saving checkpoint
[2026-11-01 20:50:39,823 INFO train.py line 251 3975987] Train control: [140/9376] loss=0.16671 wall/step=1.023s wait/step=0.013s ETA=157.5min
[2026-11-01 20:50:40,280 INFO train.py line 251 3975988] Train gate: [140/9376] loss=0.16667 wall/step=1.031s wait/step=0.017s ETA=158.7min
~~~

两项都在首次检查后打印了step1并继续至140，说明首次checkpoint写入也已返回成功。
GPU训练的模型/优化器、读取/传输、前向/反向及DINO冻结路径已通过首批实际运行。
尚未收到完整epoch、最终训练summary、逐类val或最终mIoU；不记录方法已提升。
140是日志的训练迭代计数，成功optimizer更新/AMP跳步总数以最终summary为准。

## 耗时解读

| 项目 | control | gate |
|---|---:|---:|
| 20步时刻 | 20:49:44.809 | 20:49:44.761 |
| 140步时刻 | 20:50:39.823 | 20:50:40.280 |
| 最近120步每步秒 | 0.45845 | 0.46266 |
| 140步日志累计每步秒 | 1.023 | 1.031 |
| 140步日志累计阻塞取数每步秒 | 0.013 | 0.017 |

首步日志约80秒，包括首次编译、数据进程启动、检查与首次checkpoint。
现有ETA用累计均值，尚未摊薄该开销，所以157–159分钟不能当成稳态剩余时长。
以20:50的9236剩余步和上述最近速度计算，训练本身约余70.6/71.2分钟。
若后续速度保持，大约22点左右结束这两项训练；此估计不包含最终验证及后续两组。
不是对整条队列的完成时间承诺，不预报尚未测量的完整val耗时。

日志未显示持续的大量取数阻塞；不能据此推断具体GPU利用率或独立GPU算子耗时。
两组初期loss接近与恒等初始化设计相符，不能用训练loss差判断研究收益；
最终比较同预算control与候选的原版完整val指标。

## 当前决定

继续原nohup队列，不重启、不改batch/seed/学习率，不新增检查关卡。
首组各自训练完成后会自动评价最终checkpoint，再领取mean/dropout。
只有文档与回执更新，没有更改运行中的代码快照。
