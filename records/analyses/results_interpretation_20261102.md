## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（描述性校核与证据整理）
- Origin Date: 2026-11-02，Asia/Shanghai
- Verification Status: ANALYZED
- Version Label: pointcloud_receipt_review_20261102_v1
- Source: 用户粘贴的服务器report/status/日志，以及两份最终summary JSON；未直连服务器，未独立重跑。

## 数据范围与可复核量

[推理r02](../experiments/view_assignment_fullval_20261101_r02/README.md)、[四组微调r02](../experiments/finetune_stage1_20261101_r02/README.md)、[多路线r01](../experiments/multi_route_20261101_r01/README.md)。

| 描述性比较 | 数值 | 边界 |
|---|---:|---|
| 原权重推理mean−历史基线 | +0.16pp | 单次推理，体素采样/历史运行差异未量化 |
| 两轮mean−两轮control | +0.27pp | 单seed筛选，训练control82.20而非82.48 |
| 两轮center_select−control | +0.26pp | 与mean只差0.01pp，不能建立稳定排名 |
| camera_dropout三路−同条件control | 73.13−60.79=+12.34pp | 固定三路=[0,2,4]，并非全部子集/多seed |
| camera_dropout clean−control clean | 82.04−82.20=−0.16pp | 同时报告正常输入代价 |
| control clean→三路降幅 | 82.20−60.79=21.41pp | 与同模型clean比较 |
| camera_dropout clean→三路降幅 | 82.04−73.13=8.91pp | 与同模型clean比较 |
| 两模型降幅之差 | 21.41−8.91=12.50pp | 不是三路同条件差12.34pp |

收到的summary中16类IoU/accuracy的算术均值可在四舍五入精度内核对mIoU/mAcc；OA无法仅从各类accuracy推导，没有点数/混淆矩阵时不反推。完整val的mIoU是全局累计交并集后按类平均，不是逐帧mIoU均值。

没有p值、置信区间或多次独立训练结果，因此不制造显著性检验或标准差，不套用Cohen's d分级。总体解释标记CAUTION，含义是适用范围有限，不表示已回传数字错误。

## 11/11统计与解释风险检查

| 检查项 | 判断 | 本次处理 |
|---|---|---|
| 1 Simpson反转 | 未能评估完整分组趋势 | 两份逐类表属于不同模型/不同相机条件，不能当作配对逐类收益；其余逐类待回传 |
| 2 生态谬误 | 范围限制 | 不从总体mIoU推出所有帧、距离或小目标类别都改善 |
| 3 Berkson/选择偏差 | 固定子集限制 | 三路/单路索引固定，不推广到全部缺失组合 |
| 4 碰撞变量偏差 | 未见相关条件化证据 | 未引入按预测好坏筛帧/筛类别的分析；不能据此声称所有偏差已排除 |
| 5 忽略基率 | 需要区分OA和mIoU | control零路OA85.42但mIoU36.80，保留16类表，不用OA替代类别均衡表现 |
| 6 均值回归 | 有匹配control，仍有限 | 主比较用同预算control，历史82.48仅参考；单次小分差不当稳定收益 |
| 7 幸存者偏差 | 失败已完整记录 | valid_pool/both失败、四个blocked均保留；失败不填0分、不从清单删除 |
| 8 多处搜索效应 | 探索性筛选 | 保留13种定义及所有正负结果，不只报最高分；未宣称校正后的显著性 |
| 9 分析路径自由度 | 探索性协议有记录 | 保存两轮/十轮预先固定计划与重启过程；Git计划不冒充正式预注册 |
| 10 相关与因果混淆 | 控制实验范围内描述 | 有同预算处理比较，但不外推所有环境；实际AMP更新不完全相同且完整产物未审计 |
| 11 反向因果 | 对代码干预时间顺序不典型适用 | 处理在训练/评价前定义，未据结果倒造处理；仍须保留真实配置和执行来源 |

覆盖11/11是检查清单覆盖，不表示现有数据足以排除每一种偏差。

## 可写入当前交接的观察

1. 固定三路下逐相机dropout73.13%比control60.79%高12.34pp，同时clean低0.16pp；这是当前最清楚的已观察差异，1/0路和整组dropout待完整评价。
2. 两轮clean最佳mean82.47%，未超过历史82.48%；不能把相对训练control的+0.27pp写成刷新历史最佳。
3. gate和local_attention本次clean各+0.01pp，尚无稳定优势证据；短训结果不足以否定较长训练或整条机制路线。
4. valid_pool/both为首次loss非有限的执行失败，未完成首次有效更新；具体根因尚未定位。

## 重复性与缺失依据

- Method: not run；Verdict: CANNOT_VERIFY（没有独立服务器重跑）。
- 已核对的内容是数字/条件/来源的内部一致性，不是重现性通过证明。
- 尚需运行快照commit、有效配置、完整逐类/训练summary、AMP更新、耗时/显存；多seed/必要消融/更多子集与泛化属于后续结论验证，不作为当前继续队列的额外准备关卡。
- 机器重启后的两项完整评价是故障恢复，不算额外独立训练seed；中断前预测未当作最终分数。
