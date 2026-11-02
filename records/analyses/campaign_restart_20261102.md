# 多路线队列：机器重启与接续记录

日期：2026-11-02，Asia/Shanghai。来源：用户报告和粘贴的服务器命令/日志；助手未连接服务器。

## 中断前

10:24:42快照：reused8、completed16、failed2、blocked4、running2、pending18，共50节点。GPU0 robust_control_cam0（PID188329）正在第3931/6019帧，GPU1 robust_camera_dropout_cam3（PID192877）正在第2919/6019帧。

用户报告“10点多机器重启”，精确重启时刻未提供，不能用10:24:42代替重启时间。nohup不能使进程跨机器重启存活。

## 原恢复入口的边界

scripts/resume_campaign.sh使用该run保存的代码/路径/计划并加控制器OS锁。原AdoptedProcess只观察活进程；死进程若没有completed子状态，返回失败。原execute_queue不会把机器中断自动重置为pending；所以仅执行原resume命令会把未完成节点记录为failed，而不是逐帧续算。

这也是为什么本次先进行一次用户执行的恢复准备，而没有修改正在用的实验代码快照。

## 用户实际执行的恢复准备

1. 锁定同一run的controller.lock，若控制器仍运行则拒绝。
2. 仅选择queue.json中running节点，核对对应进程已退出；若发现训练中断则停止，不套用验证恢复逻辑。
3. 确认输出位于本run/jobs、日志位于本run/logs；备份queue为queue.before_reboot_时间戳.json。
4. 若子状态和完整summary已完成则保留并登记completed；否则将两项未完成评价的输出/日志改名为.interrupted_时间戳，保留旧预测与记录。
5. 清理旧PID/命令/阶段时间等字段，把这两项设为pending，原子写回queue。
6. 在准备成功后执行bash scripts/resume_campaign.sh multi_route_20261101_r01。nohup、stdin断开、stdout/stderr重定向、GPU0/1由脚本设置。

恢复准备在本地模拟队列检查过重新排队、完成项保留、终态保留、重复执行、活任务拒绝、中断训练拒绝、双控制器锁。该检查不是服务器模型测试。服务器实际运行由下列回执支持。

## 实际回执与结果

- 用户终端：已重新排队robust_control_cam0、robust_camera_dropout_cam3。
- 新父控制器PID59334；GPU0子PID59338，GPU1子PID59343。
- 12:39:36：两项分别正在83/6019、80/6019帧；12:41:34为209/6019、207/6019，证明确实重新推进。
- dropout三路于14:15:03.374 End Evaluation，最终73.13% mIoU；control零路于14:15:23.358结束，最终36.80%。
- 父日志随后DONE两项，GPU1启动robust_camera_dropout_cam1（PID100312），GPU0启动robust_camera_dropout_cam0（PID100432）。START行无精确时间，不能补造时间戳。

两项验证从头计算，没有承诺从原3931/2919帧精确恢复；旧完成训练/验证均保留，8个旧四组节点继续复用。原valid_pool/both失败与四项blocked未清除，未自动重试。

## 交接注意事项

- [.interrupted_归档与新日志](../experiments/multi_route_20261101_r01/receipts/server_events_20261102.txt)的确切服务器目录后缀尚未回传，只记录实际命名规则，不编造目录名。
- 恢复后查看runs/_launch/multi_route_20261101_r01/resume_console.log，原console.log不会记录新控制器的后续事件。
- 当前export_record.py遍历jobs目录，会尝试导出带status的.interrupted_旧目录；这些目录若仍为running会导致整批导出拒绝。整批导出前需按queue当前引用选择目录或排除归档，保留旧产物，不删预测绕过检查。
- 若未来在十轮训练时机器再次中断，需要按本组model_last恢复optimizer/scheduler/scaler/RNG/批次位置；不能把本次仅验证重排当作训练恢复实现。

[分时快照](../experiments/multi_route_20261101_r01/progress_snapshots.json) · [结果目录](../experiments/multi_route_20261101_r01/README.md)
