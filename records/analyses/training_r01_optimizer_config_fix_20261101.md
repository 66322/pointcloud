# 训练r01优化器配置类型修复（2026-11-01）

## 服务器回执

用户启动finetune_stage1_20261101_r01，源版本6906243。
父队列PID3964649；control/GPU0 PID3964729，gate/GPU1 PID3964730。
两个子进程于20:25:13均报告Totally 28130 x 1 samples in train set，
随后在utils/optimizer.py:31失败：

~~~text
param_group["lr"] = param_dicts[i].lr
AttributeError: 'dict' object has no attribute 'lr'
~~~

这确认train索引可读取、模型构造/严格权重加载与门控安装已走到优化器创建。
原始训练帧读取、前向/反向及参数更新尚未发生；不能记录为训练已通过或有效实验。
mean/dropout尚未排到执行。Albumentations keypoint提示不是此异常的停止点。
助手没有SSH；上述状态来自用户粘贴日志与对应代码控制流，未读取服务器最终status。

## 原因及修复

训练适配器把cfg.param_dicts重设为包含普通dict的list。原Config的赋值机制
不会递归将list里的dict转换为ConfigDict，而原优化器同时使用字典操作与
.lr/.keyword/.regex属性访问。此前组件测试未覆盖真实配置对象到原优化器的接口。

新增training_setup.configure_optimizer，显式构造原项目ConfigDict参数组。
保留max_lr、组比例、weight_decay、OneCycle曲线、seed、batch、epoch及所有处理定义，
不修改原DITR源文件，不增加服务器安装或数据检查步骤。

## 回归验证

新增test_upstream_optimizer，直接载入已配置DITR或本地ignored缓存中的原版
Config/ConfigDict、Registry、optimizer、scheduler Python源码，不导入CUDA扩展或加载模型。
隔离无关日志/路径辅助模块，使用服务器Linux配置解析分支。
测试复现旧dict的同一.lr异常，再验证修复后的配置dump、AdamW两组学习率/参数归属、
原OneCycle构造、一次CPU前向/反向/参数更新与冻结参数不变。
不下载源代码作为测试运行步骤，不把CPU集成测试写成H100训练成功。
本地Torch 2.5.1 CPU全套29项测试通过，无跳过；Python编译和git diff检查通过。

## 重新执行

r01使用独立代码快照，git pull不会替换其旧源码。保留r01失败记录，
修复拉取后以finetune_stage1_20261101_r02创建新快照，从同一个初始权重开始。
尚无成功训练更新，因此不使用--resume，也不重复已有推理或前期验证。
