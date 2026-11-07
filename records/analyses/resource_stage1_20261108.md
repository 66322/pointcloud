# 首批资源执行入口与本地验证

日期为 2026-11-08，Asia/Shanghai。基础代码为 GitHub main 5a9e16e。
服务器历史依据仍为 11 月 2 日用户回执，本次未连接 H100。

## 排定内容

N000/M0/M1/M11/M2 共五个十轮训练配置，每个一个 seed 20261101。
固定计划为五次训练、五次正常评测、十五次 3/1/0 路压力评测。
旧队列的同预算完整产物可复用，实际新增节点数由服务器上的 queue/reuse 记录确定。
旧 bilinear 改动全部注入层，与最细层 N08 的定义不同，未加入首批。

具体命令和后续批次见 [资源手册](../../docs/RESOURCE_EXPERIMENT_RUNBOOK.md)。

## 从旧失败中修补的内容

- 原 DINO checkpoint-only 初始化与 ConfigDict 参数组修复继续复用。
- 有效池化中的负无穷改为 dtype 有限最小值，全无效簇仍回零。
  训练态测试检查归约输入有限，CUDA 原始失败根因尚需新输出确认。
- 首次 loss 的非有限检查提前到 backward 之前。
- 导出器按当前 queue 引用选择输出，保留 .interrupted 目录。
- 缓存读取器保存原始类方法，避免 spawn 反序列化时绑定方法回指已替换入口。
- 首批启动器保存 nohup/PID/日志，并采样 GPU/主机资源。
  正式训练内记录数据等待、窗口吞吐和少量 CUDA 流区间。

## 本地验证

Windows，Python 3.12.14、PyTorch 2.5.0+cpu、NumPy 2.3.5。
52 项单元/集成检查通过，含原始 ConfigDict/optimizer/scheduler
CPU 更新检查、真实 spawn 缓存构建、缓存命中/未命中、增强副本隔离、
相机保留分布与独立 RNG、旧十轮复用而两轮不复用、中断目录排除。
原始池化/反池化测试使用固定上游 Python 类和 CPU CSR 替身。
Python 编译、Bash 语法和 git diff 空白检查通过。

本机另有 Python 3.13/PyTorch 2.7 环境，其旧 scheduler 的 verbose 参数接口不兼容。
验证已切换到 2.5 CPU 环境完成，服务器环境未因此升级或增加兼容补丁。
测试依赖和完整本地测试日志位于忽略提交的 external 目录。

尚未验证实际数据解码缓存的服务器覆盖率、CUDA 归约、训练收益和吞吐提升。
缓存构建成本、冷读与稳态时间在服务器分别记录，不能从本地 CPU 检查填写加速倍数。
