# r02末帧回执：多视图均值与首视图（2026-11-01）

证据来源：用户粘贴的服务器控制台末帧日志；助手没有SSH。
运行目录：/home/jovyan/zzx/pointcloud/runs/view_assignment_fullval_20261101_r02。
执行版本为r02启动时的3f0003d，尚未使用后来的有序预取与DINO缓存优化。

## 原始回执

~~~text
[2026-11-01 19:38:54,083 INFO test.py line 291 3885331] Test: 597351ed877f4a75bdaeaef94d38c0e7 [6019/6019]-34720 Batch 0.817 (0.810) Accuracy 0.8868 (0.9006) mIoU 0.5624 (0.8264)
[2026-11-01 19:38:49,072 INFO test.py line 291 3885332] Test: 597351ed877f4a75bdaeaef94d38c0e7 [6019/6019]-34720 Batch 0.888 (0.803) Accuracy 0.8887 (0.8994) mIoU 0.5647 (0.8255)
~~~

首行为mean/GPU0，次行为first/GPU1，与启动记录对应。

## 已能读取的结果

| 方法 | 帧数 | 累计全局mIoU % | 与已有基线差值pp | 累计mAcc % |
|---|---:|---:|---:|---:|
| 已有默认赋值基线 | 6019 | 82.48 | 0 | 90.03 |
| mean_visible_cameras | 6019 | 82.64 | +0.16 | 90.06 |
| first_visible_camera | 6019 | 82.55 | +0.07 | 89.94 |

源码确认：m_iou=mean(sum_intersection/(sum_union+eps))，因此括号内值是全局
累计统计；括号外0.5624/0.5647只是当前帧的present-class mIoU。
Accuracy括号内是逐类平均准确率mAcc，不是整体点准确率OA。
两项单进程各自完整val，末帧累计值与最终汇总使用同一交并集计数定义；
以上保留日志显示的四位小数精度，不推断更多位数。

## 结论与接续

- 两种推理干预相对已有基线均有小幅正向差值，mean优先保留。
- 这是单次、未经训练的结果。随机初始体素采样及基线历史执行差异可能影响小差值，
  不能据此宣称稳定提升、统计显著或已成立论文贡献。
- 不追加重复完整val、多级小样本关卡或额外SHA检查。读取已有report/status、
  逐类summary并导出轻量记录即可；继续准备同预算训练对照、门控与独立dropout路线。
- 尚未接收到最终Val result行、OA、status.json、report.md或轻量导出。
  此文件只记录已见末帧证据，不代替真实运行目录的完整导出，也不声称最终清理/退出已确认。
- 不从Batch平均0.810/0.803推算端到端总耗时，该计时排除了loader等待。

指标定义依据：[原SemSegTester](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/engines/test.py#L274)。
