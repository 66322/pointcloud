# 两台电脑的当前交接记录

更新日期：2026-11-02，Asia/Shanghai。根据用户回传，最近带时间的完成事件为14:15:23；随后父日志确认新任务START，精确查询时刻未提供。本文件不是实时监控。

## 接手先做

1. git pull --ff-only，读取本文件和[实验总表](EXPERIMENT_MASTER.md)。本地有未提交修改先保留并检查，不覆盖另一台电脑的提交，不force push。
2. 服务器运行由用户手动执行；不要根据本机测试/旧PID宣称当前服务器状态。旧推理r02和四组微调r02已经完成。
3. 保留multi_route_20261101_r01现有队列。其代码/配置已经快照；文档更新不会改变在跑任务，不重复启动整批。

## 最近确认的服务器状态

| 项目 | 最新收到的证据 |
|---|---|
| 原权重推理r02 | completed：mean82.64、first82.55 |
| 四组微调r02 | 11月2日00:50:19 completed：control82.20、gate82.21、mean82.47、dropout82.04 |
| 多路线r01 clean | 11组结果；valid_pool、valid_both首次loss非有限而失败 |
| control缺相机 | 3/1/0路=60.79/49.20/36.80 |
| camera_dropout缺相机 | 3路73.13；1路/0路在最后回执已启动 |
| 恢复父控制器 | PID59334；日志runs/_launch/multi_route_20261101_r01/resume_console.log |
| GPU0最后确认任务 | robust_camera_dropout_cam0，PID100432 |
| GPU1最后确认任务 | robust_camera_dropout_cam1，PID100312 |
| 后续 | 整组dropout缺相机、三项4px、五个可执行十轮训练尚无最终分数；valid_both十轮blocked |

PID只用于识别该回执，不能只凭旧PID操作或终止进程。完整结果见[多路线目录](../records/experiments/multi_route_20261101_r01/README.md)。

## 当前可以写入的观察

- 两轮mean相对训练control +0.27pp，中心top-1 +0.26pp；全部两轮clean结果未超过历史82.48%。
- 三路camera_dropout73.13%相对同条件control60.79%高12.34pp；clean仅低0.16pp。限单seed、固定子集，尚无1/0路和整组dropout最终结论。
- valid_pool/both是执行失败，不是负精度结果；共同有效池化路径待定位，未修复，不自动重试。

## 下一次接手的工作

1. 收到新回执后补camera_dropout 1/0路，再补whole_camera_dropout 3/1/0路，所有条件和control对齐。
2. 保留后续4px和十轮安排；不因两轮没涨点取消预先固定的较长预算任务。
3. 如进行代码修复，优先定位valid_pool/both非有限loss；修复后新批次补跑，不修改已验证上游DITR或运行中快照。
4. 补运行快照commit、有效配置、所有整体/逐类summary、AMP更新/跳步、耗时/显存。未接收字段继续写null。
5. 整批导出前处理归档目录边界：现export_record会遍历jobs下所有带status的子目录，可能碰到.interrupted_旧目录并拒绝导出；应按当前queue引用导出或排除归档。不要删除旧预测来绕过。已完成单项可单独导出。

## 查看当前任务（服务器只读）

~~~bash
cd /home/jovyan/zzx/pointcloud
cat runs/multi_route_20261101_r01/status.json
tail -n 12 runs/_launch/multi_route_20261101_r01/resume_console.log
python - <<'PY'
import json
from pathlib import Path
from collections import Counter, deque
run = Path("runs/multi_route_20261101_r01")
jobs = json.loads((run / "queue.json").read_text())["jobs"]
print(dict(Counter(j["status"] for j in jobs)))
for j in jobs:
    if j["status"] == "running":
        print(j["id"], "GPU", j.get("gpu"), "PID", j.get("pid"))
        with Path(j["log"]).open(errors="replace") as f:
            print("".join(deque(f, maxlen=3)))
PY
~~~

父日志主要记录任务切换，帧进度看子日志。只恢复控制器与整机重启的处理不同，见[恢复记录](../records/analyses/campaign_restart_20261102.md)。不能直接把旧running字段当进程仍存活。

## 更新位置

- docs/EXPERIMENT_MASTER.md：做什么、验证什么、结果与未回答问题。
- docs/EXPERIMENT_MATRIX.md：实现/执行/失败/待验证路线。
- records/experiments/<run_id>/：reported_results.json、CSV、用户原始摘录、逐类JSON、带时间的进度快照。
- records/analyses/：描述性解释、失败与恢复。
- 每次交接同时更新本文件；公开GitHub仅放轻量证据。完整产物留服务器，凭据不进Git。同步步骤见[GITHUB_SYNC](GITHUB_SYNC.md)。
