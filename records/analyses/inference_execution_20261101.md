# DITR推理执行开销与训练接续（2026-11-01）

## 当前运行与证据边界

r02在GPU0/1分别执行mean/first完整val推理，无反向传播、优化器更新或新checkpoint。
该运行仍使用启动时加载的旧实现，不为工程提速而中断或重复启动。
这里只完成源码审查和本地开发验证，尚无H100优化后实测吞吐、利用率或提速倍数。

原作者SemSegTester构建模型、加载兼容checkpoint、遍历TTA/体素片段、累积softmax、
恢复点序、保存预测并汇总原版指标。本仓库在独立进程替换assign_image_feat，
所以是沿用原项目推理主体的融合干预；没有重新实现整条网络/评测。

## 确认的开销

- 旧入口DataLoader无后台取数：六路JPEG、标签/点云读取、投影、随机初采样、
  TTA复制/体素化和图像归一化完成后，模型才开始计算。
- 源码在每个fragment的网络forward重新调用get_image_feat。冻结DINO在相同RGB上
  的重复前向可避免，但3D分支、点图对应、TTA和概率累积仍须逐次计算。
- 原Batch计时在loader交付样本后开始；用户日志约0.81s不能代表端到端每帧耗时。
- 低GPU利用率与等待数据一致，但没有服务器profile，不能将全部时间归因于磁盘，
  也不能据瞬时利用率判断GPU故障。

## 参考与保留条件

参考遥感仓库rebuild-v1-protocol-20260915，读取时分支头c9e1341：
post_m2_a0_ipcfix_20261030.py的预取、pinned memory、等待/计算分开计时；
phase3_m2_runtime.py的常驻模型；resident_array_cache.py的只读且有界缓存思想。
没有照搬其数据SHA检查、训练精度、batch、裁剪或已冻结的研究协议。

DITR的测试transform中GridSample(mode='train', grid_size=0.025)仍会随机选点。
直接将DataLoader从0改为多进程会让worker重新播种，可能改变实际输入，因此未采用。
NuScenesDataset.get_data读取和投影本身不使用随机数，才允许并行原始帧I/O。

## 新实现

1. 两个线程预读原始帧，仅调用get_data；不并行运行随机采样/数据变换。
2. 一个后台生产线程按原索引顺序执行原transform、GridSample、TTA和collate。
   不更换或重新播种Python/NumPy RNG，不生成一套新的随机点云。
   DataLoader保留num_workers=0，iterator在主线程创建，保留其Torch base-seed消费位置。
   第一个实际帧同步完成，包括模型首次懒编译，随后重叠预取与推理。
3. 同一帧内，在eval/no-grad、同encoder/精度上下文且实际图像张量torch.equal时，
   复用DINO的feature map/cls token/patch_size。跨帧清空；输入变化重算；
   训练或梯度启用时直接绕过并清空，不把推理缓存用于随机增强训练。
   存储一个图像快照及一个输出，不建立全验证集GPU特征缓存。
4. 逐20帧输出端到端墙钟每帧耗时、阻塞取数时间、ETA；结束保存runtime.json。
   nohup入口保留，支持透传执行参数。--io-workers 0 --prefetch-frames 0
   --no-image-cache可恢复原执行方式。
5. 可把唯一一次新路径单帧预测与已有同帧预测直接比较；无额外旧模型推理，
   无SHA扫描，无完整val性能复跑。数值精度、TF32/AMP策略和所有TTA保持配置原值。

## 验证与训练接续

本地21项测试中16项通过、5项因本机无Torch而跳过；bash语法检查、git diff检查通过。
覆盖有序预取与原随机采样序列一致、异常传播/回收、帧边界、执行参数及已有流程。
Torch缓存测试覆盖输入/帧/encoder/AMP变化、梯度保留与训练禁用，尚未在本机执行；
服务器新入口首次一帧须实际运行，不能将本地检查表述为H100已验证。
之后直接用正常实验日志看吞吐，无20/100帧工程关卡。

当前训练入口和可学习模块尚未实现。下一步准备同预算继续训练对照、恒等初始化门控、
单独的模态dropout，按单因素比较；训练split在首次加载时确认。
一次前向/反向确认后进入训练，不追加旧基线重评测或全量旧缓存诊断。

源码依据：

- [原测试循环](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/engines/test.py)
- [原图像分支与融合](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/models/point_transformer_v3/point_transformer_v3m1_image.py)
- [原nuScenes配置](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/configs/nuscenes/semseg-pt-v3m1-0-image.py)
- [遥感读取/预取/等待计时](https://github.com/66322/remote_sensing_frequency/blob/c9e1341b96de36b90ae11f98a1c4db9bf37b969a/scripts/post_m2_a0_ipcfix_20261030.py)
- [遥感常驻运行](https://github.com/66322/remote_sensing_frequency/blob/c9e1341b96de36b90ae11f98a1c4db9bf37b969a/scripts/phase3_m2_runtime.py)
