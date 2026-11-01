# 同预算训练：服务器执行

已有推理r02已完成：mean82.64%，first82.55%。无需重复评测。
本页启动的是从现有兼容checkpoint开始的真实训练，不是再次推理或环境诊断。
本地Torch 2.5.1 CPU共29项测试通过，含原版优化器/调度器接口回归。
r01在优化器构造时因配置类型适配失败，已修复；
r02已于20:49完成H100首次更新及checkpoint保存，20:50两组均到140/9376。
当前control/gate训练继续，mean/dropout会自动接续，不要重复启动。

## 启动

以下仅供尚未启动本批次时使用，已运行的r02无需执行。

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --rebase
conda activate pointcept-torch2.5.0-cu12.4
bash scripts/start_training_experiments.sh finetune_stage1_20261101_r02
~~~

默认协议configs/training/stage1.json：

| 方法 | 唯一处理变化 | 初始checkpoint | epoch | 单卡batch |
|---|---|---|---:|---:|
| control | 原赋值/原加法，继续训练 | 原兼容checkpoint | 2 | 6 |
| gate | 四级图像注入处恒等初始化门控 | 同上 | 2 | 6 |
| mean | 有效视图均值赋值 | 同上 | 2 | 6 |
| camera_dropout | 训练时每相机p=0.2丢弃对应证据 | 同上 | 2 | 6 |

GPU0先control，GPU1先gate；每项训练完成释放进程，然后该卡跑一次最终完整val，
再自动领取mean/dropout。只使用GPU0/1，不需要手动开四个终端。
主比较是各候选与同预算训练后的control之差；82.48%仅作历史参考。
首轮2 epoch是找方向的真实训练预算，不能据此断言已收敛或方法已稳定有效。

## 保存位置及进度

所有新结果位于：
/home/jovyan/zzx/pointcloud/runs/finetune_stage1_20261101_r02/

- control/train、gate/train、mean/train、camera_dropout/train：config、训练日志、
  trajectory.jsonl、first_step.json、summary.json、model/model_last.pth。
- 各方法val/：原作者完整6019帧TTA结果、逐类summary、runtime和预测。
- 根目录*_train_console.log、*_val_console.log：各阶段终端输出。
- 根目录comparison.csv、report.md、summary.json：自动汇总，差值以训练control为主。
- code/、protocol.json、paths.json：启动时的小型代码/配置快照，后续git pull不改变排队任务。
- runs/_launch/finetune_stage1_20261101_r02/：父队列console.log及pid。

~~~bash
cd /home/jovyan/zzx/pointcloud
tail -n 20 runs/_launch/finetune_stage1_20261101_r02/console.log
tail -n 10 runs/finetune_stage1_20261101_r02/control_train_console.log
tail -n 10 runs/finetune_stage1_20261101_r02/gate_train_console.log
~~~

每20次训练步打印实际每步耗时、阻塞取数耗时、loss及ETA。首次模型加载、数据进程
启动和DINO编译需要时间，前几步ETA会偏大，以后续实际日志为准，不预报固定总时长。
当前实现的ETA使用从启动起的累计均值，因此140步时仍受首步约80秒开销影响。
按20:50回执最近120步约0.46s/步估，当前两项训练余下约71分钟，
只作为维持该速度的条件估计，不包含最终完整val及后续mean/dropout。

nohup已接好stdin/stdout/stderr并记录PID，普通网页断开不会依赖终端存活。
不要因没看到新的终端输出重复启动；新批次必须使用新目录名。
如某项失败，队列只终止自己创建的其他训练/验证进程，保留checkpoint与错误日志。

## 训练设置与数据

从同一个完整兼容checkpoint严格加载所有原参数，再加入末层零初始化门控。
DINO保持注册、冻结且eval，父模型.train()不会重新打开其dropout。
max_lr=2e-4，block组0.1倍；沿用作者AdamW、weight_decay、损失、
增强、Mix3D和AMP配置，OneCycle按两轮步数重新计算，不使用原checkpoint的optimizer。
没有逐epoch完整val，统一评价最后一个checkpoint，不挑best。

训练DataLoader默认4个spawn进程/卡，prefetch=1、persistent worker和pinned memory。
传输用无损NumPy管道，主进程还原相同dtype的Tensor，避免大Tensor占用/dev/shm配额。
增强和Mix3D按seed/epoch/样本索引固定，调整进程数不会改换一套随机输入；
相机dropout用独立RNG，不扰动3D增强或模型drop-path。
训练禁用推理图像特征缓存；最终val才使用同帧精确输入缓存。

直接调用原NuScenesDataset读取训练split。缺train索引或文件时会在真实加载时明确报错，
只补缺项；不要求事先全量扫描、哈希检查、额外小样本验证或下载第二个数据集。
batch6若不适合实际显存，应统一修改协议再用新批次名启动，不静默给某一方法降batch。
AMP成功更新次数和跳步数分别保存，分析时不能把名义步数当作成功更新次数。
本批只评价正常输入，camera_dropout的鲁棒性结论还需后续针对性相机缺失实验。

## 失败后续跑单个训练

每项首次成功更新及每轮结束保存到它自己的model/model_last.pth，包含optimizer、
scheduler、scaler、批次位置及RNG。支持从本实验目录续跑，不能用于套用别组checkpoint。
确认旧队列及占用该卡的本项目进程已退出后，单项续跑示例：

~~~bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m pointcloud_lab.train \
  --paths runs/finetune_stage1_20261101_r02/paths.json \
  --protocol runs/finetune_stage1_20261101_r02/protocol.json \
  --method control --gpu 0 \
  --out runs/finetune_stage1_20261101_r02/control/train --resume \
  > runs/finetune_stage1_20261101_r02/control_resume_console.log 2>&1 < /dev/null &
~~~

此命令只续跑control训练，不自动复活已退出的整条队列。
重新评估或接续队列应根据保留状态安排，避免重复完成的组；
不能把单项训练完成误认为整批训练与评价均完成。

## 完成后导出

~~~bash
python -m pointcloud_lab.export_record \
  --run runs/finetune_stage1_20261101_r02 \
  --record records/experiments/finetune_stage1_20261101_r02
git add records/experiments/finetune_stage1_20261101_r02
git commit -m "Record paired DITR fine-tuning experiments"
git pull --rebase
git -c credential.helper= push origin main
~~~

仅导出小型配置、首次梯度记录、轨迹、指标及状态；不上传checkpoint、逐点预测或完整日志。
