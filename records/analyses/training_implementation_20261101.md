# 训练入口交付与r02收尾（2026-11-01）

## 已收到的服务器结果

r02最终status为completed，UTC结束于2026-11-01T11:38:55.907847+00:00。
用户粘贴report确认mean82.64%（+0.16pp）、first82.55%（+0.07pp），各6019帧。
完整逐类/OA文件仍在服务器；已有回执见view_assignment_r02_last_frame_20261101.md。
不重跑r02或旧基线，不因为浮点显示尾数重新计算预测。

## 新训练协议

configs/training/stage1.json固定：四项control/gate/mean/camera_dropout，
各2 epoch、单卡batch6、seed20261101、max_lr2e-4，block组0.1倍、gate隐藏32，
相机丢弃p0.2。保留作者损失/AdamW weight_decay/训练变换/Mix3D/AMP；
这是同预算checkpoint微调筛选，不是从头复现作者50 epoch。
实际optimizer成功次数与AMP跳步次数独立记录，不能伪称二者完全等于名义步数。

模型与checkpoint：

- 原兼容权重完整strict加载，保留DINO模块注册名，禁止多一次Hub预训练下载。
- DINO冻结且永久eval，父train()不重新打开其dropout。
- gate直接在四个原SerializedUnpooling增加lab_gate并替换局部forward，原参数名不迁移。
  1+tanh gate末层零初始化，初始原函数不变；添加门控不扰动原模型CPU RNG。
- mean单独使用已实现的有效视图均值；camera_dropout单独改对应mask且用私有RNG，
  eval时关闭，不与gate叠加，不把同时多处变化当单模块收益。
- 第一批真实训练更新完成梯度/冻结检查并保存checkpoint，然后继续；没有独立试训练pass。
- 每项只评价最后checkpoint一次完整原TTA，不在每epoch额外验证或选best。
  评估gated checkpoint时先重建门控结构再strict加载全部权重。

数据与执行：

- 实际训练加载时直接读取原NuScenesDataset的train索引/文件，缺项由报错指出，
  不扫描原始全集或做SHA门槛。
- 原几何/图像增强不删除，按seed/epoch/索引决定随机流，批次Mix3D也单独固定。
  NumPy/Python/CPU Torch状态在数据变换后恢复；模型CUDA RNG不被数据播种改写。
- 4个spawn读取进程/卡，prefetch1，persistent worker；无损NumPy经pipe传输，
  主进程恢复同dtype Tensor并pin，避免大Torch storage占用容器/dev/shm。
- GPU0先control、GPU1先gate，各自训练退出后开启最终val，再排mean/dropout。
  始终最多两个项目GPU进程，仅用0/1。第一个失败即停止本队列拥有的其余进程并保留产物。
- nohup、stdin=/dev/null、日志及PID已接好；每20步显示端到端耗时/取数等待/ETA。
- 启动时复制小型Python代码及配置到该run，排队期间git pull不会改换新任务的源文件或预算。
  只记录源commit，不做文件哈希、Git一致性阻断或数据/权重副本。
- 保存model_last含optimizer/scheduler/scaler、样本批次位置及RNG；单项可resume，
  不把单项恢复冒充整条队列恢复。底层CUDA/编译器的逐位确定性未经证明。

## 本地验证与边界

本地使用与服务器对应版本的Torch2.5.1 CPU完成28项测试，全部通过，无跳过。
覆盖原推理缓存（之前因无Torch跳过）、门控初始精确等价及反向、DINO永久冻结、
strict checkpoint多/缺key失败、dropout私有RNG和Mix3D偏移、无损数据传输、
0/2个进程的样本顺序/增强/Mix决策一致、两GPU训练后验证的排队与预算快照。
新增运行脚本通过bash语法和git diff检查。

H100上尚未启动训练，train数据可用性、完整Pointcept集成、显存与吞吐均待真实首批确认。
不把CPU模块测试写成H100训练已通过，不预先承诺收益或固定完成时间。
dropout只做clean val的首批结果不能证明相机退化鲁棒性。

源码依据：

- [原训练步、AMP与优化器逻辑](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/engines/train.py)
- [原四级图像注入](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/models/point_transformer_v3/point_transformer_v3m1_image.py)
- [原数据collate/Mix3D](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/datasets/utils.py)
- [原训练与测试配置](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/configs/nuscenes/semseg-pt-v3m1-0-image.py)
