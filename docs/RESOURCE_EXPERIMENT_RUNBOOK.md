# DITR 首批实验与资源执行安排

编制于 2026-11-08，Asia/Shanghai。代码从 GitHub main 的 5a9e16e 接续。
历史服务器证据截至 11 月 2 日回执，本文件的新增配置待用户在服务器执行。

## 首批可以启动的五组

共同协议为 seed 20261101、单模型单卡 batch 6、十轮、46,880 次计划迭代。
从原兼容权重重新开始十轮 OneCycle，保持原增强、损失、DINO 冻结和最后权重 TTA。

| 方案编号 | 入口方法 | 处理 | 直接比较 |
|---|---|---|---|
| N000 | control | 原 DITR 继续训练 | 正常输入和缺相机的同预算对照 |
| M0 | camera_dropout | 每台相机独立 p=0.2 掉线 | 对 N000，平均保留 4.8 台 |
| M1 | whole_camera_dropout | p=0.2 整组六台掉线 | 对 M0，保持相同期望可用数 |
| M11 | camera_dropout_half | 每台独立 p=0.5 | 对 M2，平均保留三台 |
| M2 | uniform_camera_dropout | k=0…6 等概率，再均匀选 k 台 | 对 M11，检验缺失分布的作用 |

每组一次正常 6,019 帧验证，加固定三路、一路、零路各一次完整验证。
首批上限为 **5 次训练＋20 次完整评测，共 25 个任务节点**。
三路仍是旧定义 [0,2,4]，一路 [0]。相机名称与输入顺序在本批有效配置中记录，
这三种条件仅用于衔接旧记录。全部 64 种命名组合在入选模型上另作 V17。
零路当前是屏蔽图像证据，尚会读图及编码，不能据此填写零相机部署时延。

旧 bilinear 在点到图像赋值处生效，随后影响各级注入，不能代替总方案中
只改 dec0 的 N08。本批因此保留上述五组，N08 待独立接好最细层入口后再跑。

旧队列中的十轮 control、whole_camera_dropout 若已完成且原始产物齐全，
新队列会复用匹配节点。旧两轮结果继续作为历史依据。
这批是总方案首轮 65 个配置中的五个，未一次性实现剩余网络路线。

## 两张卡与共享资源

| 资源 | 首批设置 | 后续调整依据 |
|---|---|---|
| GPU0、GPU1 | 每卡一个独立训练或评测进程，空闲后领下一节点 | 旧队列先完成。模型间不相互占同卡 |
| 每卡训练加载 | 4 个 spawn 进程，prefetch=1，persistent workers | CPU 配额少时降为 1–2。取数持续阻塞且 CPU 有余量时试 8 |
| worker 内部线程 | Torch/OpenCV 各 1，OMP/MKL/OpenBLAS 各 1 | 防止每个进程再开大量线程 |
| 解码缓存构建 | 默认 4 个 CPU 进程，两个 GPU 共用一个缓存目录 | 与训练同时构建时计入总 CPU 和内存预算 |
| 评测 | 每卡一个模型进程，两个任务并行。保留有序取帧、原 TTA 与同帧 DINO 缓存 | 先读已解码共享缓存，现有原始读取还可用 I/O 线程预取 |
| 主机内存 | 综合 MemAvailable、容器限制与当前使用量 | 缓存目标初选可用预算的 50%，另保留至少 16 GiB 或可用量的 25% |
| CPU 汇总与写盘 | 使用原评测器累计交并集，完成一次才导出小结果 | 逐点预测留服务器。归档不与关键训练抢大量 I/O |

CPU 配额按 cpuset/CPU quota 检查。两卡各 4 个 worker 加两个主进程，
应至少有约 12 核有效预算。主机显示很多 CPU/内存时，仍要看容器限额。
快照分别给出空闲时与训练并行时可分配的缓存构建进程数。
后者为零时，把解码构建排到 CPU 有余量的时段。
训练传输继续使用 NumPy 管道和主进程 pinned memory，避免大 Tensor 填满 /dev/shm。
大内存缓存通过文件映射与系统页缓存共享，worker 只保留当前样本和预取批次。

## 为什么先测分段耗时

11 月 1 日原始训练记录的最近 120 步约 0.458/0.463 秒每步，
累计等待取数约 0.013/0.017 秒每步。原记录尚不支持把全部 GPU 波动归于磁盘。
首次约 80 秒包含编译、加载进程启动和第一次 checkpoint，会拉高累计 ETA。

新入口在正式训练的第 21–40 次迭代采集 CUDA 事件，包含主机到显存传输、
前向/反向/优化器所在流的区间耗时。每 20 步记录最近窗口的端到端耗时和取数等待。
这 20 步属于正式训练。CUDA 流区间包含流上的空隙，不能当成纯算子时间之和。
训练摘要仍分开记计划迭代、成功更新和 AMP 跳步。

resource_stage1 启动器每 10 秒记录 GPU 利用率、显存、功耗、温度、
主机/容器内存、CPU 累计计数、缺页/交换计数和 I/O pressure。
利用率短时下跌只作定位线索，主要比较稳定窗口的秒/步或秒/帧。

| 实际读数 | 优先措施 |
|---|---|
| data_wait 持续占每步较大比例，磁盘或缺页压力高 | 共享原始帧缓存，查存储是否为网络盘，分配本地 SSD 和更多有效预取 |
| raw 文件已在内存，但 CPU 高且 JPEG 解码/投影耗时长 | 用下述已解码缓存，或在 CPU 预算内增加加载 worker |
| 等待很低，CPU 一核繁忙、GPU 呈短脉冲 | 查序列化/稀疏索引、Python 调度、同步点与 cfg.empty_cache，不继续盲增 worker |
| H2D 区间较高 | 查 pin_memory 是否生效、传输体积与 NUMA/PCIe 拓扑 |
| 显存接近上限或分配抖动 | 查有效 config、激活和点数变化。统一修改对照组的 batch，另开批次 |
| 并行后两卡均更慢 | 查总 CPU 配额、存储带宽、RAM 和容器压力，减少后台构建或预取 |

性能调整保持 batch、种子、样本、增强、损失与训练更新预算相同。
验证结果采用原评测器。调整 worker/prefetch 不改变已有按样本/epoch 设定的增强种子。

## 内存缓存如何使用

缓存调用原 NuScenesDataset.get_data，保存解码后的 uint8 图像、点云、标签、
相机投影和有效掩码，位于随机增强之前。两个 GPU 和所有 worker
读取同一份 .npy 文件，借助操作系统页缓存共用内存页。
读入后为本次增强建立可写副本，因此旋转、裁剪和颜色增强不会污染下一轮。
训练时继续实时计算 DINO 特征。

以原始 900×1600 RGB 图像为量纲例子，六路图像每帧约 24.7 MiB。
28,130 帧训练图像本身约 679 GiB，6,019 帧验证图像约 145 GiB，
另有点云、投影与标签。实际缓存大小和覆盖率以 manifest 为准。
服务器内存较大也需先算全量解码是否能容纳。

文件位于本地 SSD 时，系统可按内存压力自动回收缓存页。
若服务器已经提供属于本项目、容量充足的 RAM 文件系统，也可把缓存目录放在那里。
缓存目录独立于原始数据和 /dev/shm。新缓存开始写入后，完整成功的 manifest 才标为 ready。
构建失败保留输出供排查。读到已列入 manifest 的损坏文件时直接报错。

--max-gib 限制发布的缓存文件总量，未缓存的帧仍用原读取路径。
构建时最多另有 workers 个帧正在解码写入，磁盘和内存要为它们留出余量。
全部副本和 pinning 缓冲也计入容器内存，不能把目标缓存大小当成总峰值。

先生成资源快照，可同时查看旧任务的日志，不必再跑旧基线。

```bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4
CUDA_VISIBLE_DEVICES=0,1 python -m pointcloud_lab.resource_monitor \
  --out runs/resource_inventory_20261108.json
cat runs/resource_inventory_20261108.json
df -h /home/jovyan/zzx /dev/shm
findmnt -T /home/jovyan/zzx
```

如资源允许，先缓存重复读取的验证集。以下 64 GiB 是可修改示例，
使用前按快照给出的 combined_decoded_cache_target_bytes 分配 train/val 总预算。
该示例可能只覆盖部分验证帧，不声称足够容纳整套数据。

```bash
mkdir -p /home/jovyan/zzx/pointcloud/runs/_launch/cache_val_20261108
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m pointcloud_lab.raw_cache \
  --paths configs/paths.local.json --split test \
  --out /home/jovyan/zzx/pointcloud_cache/decoded_val_v1 \
  --max-gib 64 --workers 4 \
  > runs/_launch/cache_val_20261108/console.log 2>&1 < /dev/null &
echo $! > runs/_launch/cache_val_20261108/pid
```

缓存 ready 后，在忽略提交的 configs/paths.local.json 中设置
decoded_val_cache 为上述目录。训练缓存用 --split train 和新目录构建，
然后设置 decoded_train_cache。两个键均可省略。
路径、标签映射、原始数据或原始读取实现变更后，为缓存另建版本。
缓存命中仍按帧 token 校验，并核对数据根目录、类、图像开关、ignore 与标签映射。
对应扰动和图像退化在缓存读取后施加，再计算当次视觉特征。
缓存路径变化只影响新队列，已经保存的旧队列继续读取自己的路径快照。

## 启动首批

旧 multi_route 的实时状态还需服务器确认。下面的 --after 会等待其终态，
随后检查 GPU0/1 空闲并查找可复用结果。旧任务已有进程时继续原任务。
缓存构建为可选提速步骤，首次正式训练不依赖一次全量缓存构建成功。
缓存首次构建与冷读成本单列，以重复评测的稳态吞吐及总用时决定是否采用。

```bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4
bash scripts/start_resource_stage1.sh resource_stage1_20261108_r01 \
  --after runs/multi_route_20261101_r01
```

如旧队列的状态因机器重启仍停在 running，先按已有恢复记录处理。
不能通过重复启动首批绕过旧状态。
入口使用 nohup、断开 stdin、输出重定向、PID 和独立运行目录。
控制器退出会停止自己的资源采样，新入口首次真实更新做基本梯度和冻结检查。

如果快照显示 CPU 有效预算不足以支持两卡各 4 个加载进程，
在启动前统一修改 resource_stage1.json 的 workers，再记录新提交。
所有模型使用相同设置。若新增性能配置已启动，则在下一批应用。

启动后查看两个位置。

```bash
tail -n 20 runs/_launch/resource_stage1_20261108_r01/console.log
tail -n 20 runs/resource_stage1_20261108_r01/logs/screen_control_train.log
```

WAIT、START、first actual update、训练窗口、DONE 和失败都应有记录。
出现失败时只阻断依赖该模型的评测，其他独立节点继续。

## 后续具体次序

| 批次 | 需要接入的内容 | 首个成对实验 |
|---|---|---|
| 首批 | 本文五组，相机缺失分布与资源诊断 | N000/M0/M1，M11/M2 |
| 正常融合 | N08 仅 dec0 双线性，F1 语义查询与拒配，P1 部分旋转，I1 图像细化 | N000/N001，L0/L1，然后补 N000–N111 八格 |
| 对应 | A0 同扰动对照，A4 深度拒配，A5 全局纠偏 | A0/A4/A5，接好组合后 A6 |
| 退化 | 原始、仅坏图、仅坏点云、双坏的同帧版本 | R0/R4/R9，后接 R5/R6 |
| 缺失深化 | 纯点云解码头、同教师蒸馏、伪视觉开关 | M3/M4/M5，M3/M6/M7 |
| 联合与真实天气 | 三开关共用增强分布，跨库投影及类别映射 | J000–J111，K2/K3 与 T0/T1 |

其余 A/R/M/N 配置按总方案逐项补齐，首轮仍为每配置一个共同种子。
上述“需接入”项目前只存在研究设计和部分参考算子，不当作已可执行入口。
修复有效池化后，以新输出检查原失败方法的首次真实更新，保留旧失败记录。
它不是本文五组的前置依赖。

五组十轮共有 234,400 次计划迭代。按历史约 0.47 秒/步估算，
单算训练约 30.6 GPU 小时，两卡约 15.3 小时，实际复用项会减少。
完整评测时间用当前运行的 wall/frame 估计，20 次×6,019 帧×秒/帧÷3,600 得 GPU 小时。
排期把训练、评测和缓存构建分别计时，待首批窗口取得实测再更新 ETA。

## 本次防止重复故障的改动

| 历史问题 | 本次处理与验证边界 |
|---|---|
| timm 再读 DINO 缓存导致初始化失败 | 保留已有 checkpoint-only 结构构建和严格加载，不引入第二次下载 |
| optimizer 参数组是普通 dict，原构建器按属性读取失败 | 保留 ConfigDict 修复，检查原 optimizer/scheduler 集成测试 |
| valid_pool/both 首次 loss 非有限 | 将池化掩码的负无穷改为 dtype 有限最小值，全无效簇仍回零。测试训练态中间归约输入有限。CUDA 根因还需实测确认 |
| 非有限首次 loss 已经执行 backward | 将首次 loss 检查放到 backward 前，直接保留错误 |
| 机器重启后旧 running 字段误导接续 | 继续使用原恢复规则，旧 PID 只作历史记录。训练恢复保留 optimizer/scheduler/scaler/RNG/批次位置 |
| 导出器扫到 .interrupted 旧目录而拒绝 | 按当前 queue 引用选择产物，原归档继续保留 |
| 加载进程可能复制整套 RAM 数据 | 使用文件映射和系统共享页，缓存读取后仅为当前增强复制 |
| 替换读取入口后的绑定方法在 spawn 中回指自身 | 保存原类方法，真实 spawn 子进程测试覆盖缓存命中与未命中的读取 |
| 借用旧入口的名称但改动范围不符 | 首批排除全层 bilinear，N08 明确留给最细层实现 |

本地 CPU 检查与队列模拟只证明相应接口和控制逻辑。
本文件不包含新增服务器成绩或 GPU 利用率提升倍数。

实现依据可查 PyTorch 的 [DataLoader 文档](https://docs.pytorch.org/docs/2.15/data.html)
与 [性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。
服务器沿用原 PyTorch 2.5 环境，新增代码使用该环境已支持的 spawn、
pin_memory、自定义 batch 和 CUDA events 接口。
