# DITR实验总表

更新日期：2026-11-02。来源是用户回传的服务器日志/报告/两份最终summary JSON；最近带时间的完成事件为北京时间14:15:23，随后确认下一批START。未直连服务器，不把回执中的running写成实时保证。

换电脑先读[HANDOFF](HANDOFF.md)。数值依据按批次存放在[records/experiments](../records/experiments/README.md)，运行手册中的历史命令不代表需要重跑。

## 共同协议与已有基础

- nuScenes lidarseg 16类，28130条train索引、完整val6019帧；PTv3+DINOv2-L，DINO冻结/eval。
- 历史完整val基线82.48% mIoU、90.03% mAcc、94.63% OA；作者83.1%仅参考。
- 两轮微调：单模型单GPU batch6，4688步/轮，共9376次计划迭代；seed20261101，max_lr2e-4、block组0.1倍；原AdamW、增强、损失、AMP和TTA。
- 每组从同一个原兼容checkpoint开始；只评价最后checkpoint一次，不逐轮完整验证或选best。原3D网络/图像投影/分类头继续训练，不只是训练新增模块。
- 十轮任务从原checkpoint重新开始，独立十轮OneCycle、46880次计划迭代；与十轮control比较，不能改旧两轮epochs后当同一实验续训。
- 成功更新/AMP跳步与计划迭代分开记录。所有运行只用GPU0/1、独立输出、nohup。
- 接受已有基线，新路径一次确认可运行后直接实验；不重复基线、20/100帧关卡或例行SHA检查。

旧100帧/全量离线诊断已结束，其84.3997%/81.7190%受原始标签20映射错误影响，撤回为有效指标。纯几何统计与标签相关统计分开看，不要求重跑旧诊断。[更正](../records/analyses/diagnostic_mapping_correction_20261101.md)

## 1. 无需训练的推理对照：已完成

| 方法 | 要验证的问题 | mIoU % | 相对历史基线 pp |
|---|---|---:|---:|
| BASE 原赋值 | 已有复现基准 | 82.48 | 0.00 |
| VIEW-MEAN | 有效多视图均值是否优于后视图覆盖 | 82.64 | +0.16 |
| VIEW-FIRST | 首有效视图选择是否影响重叠点预测 | 82.55 | +0.07 |

r02于2026-11-01 19:38:55 completed，原权重/原TTA/完整6019帧。mean/first的mAcc为90.06/89.94%，OA和逐类尚待回传。单次小幅差异保留为线索，不重跑完整val追分。[批次依据](../records/experiments/view_assignment_fullval_20261101_r02/README.md)

## 2. 同预算两轮训练与clean验证

旧四组finetune_stage1_20261101_r02已于2026-11-02 00:50:19 completed；其8个训练/验证节点被multi_route_20261101_r01复用。后者新增九组短训中七组完成、两组失败。因此13种定义中11组已有clean结果，复用不算独立重复。

| 方法 | 唯一处理变化 | 要验证的问题 | mIoU % | 相对两轮control pp |
|---|---|---|---:|---:|
| control | 原赋值/加法继续训练 | 分离额外训练预算本身的影响 | 82.20 | 0.00 |
| gate | 四级图像注入的恒等初始化门控 | 学习注入强度是否优于原加法 | 82.21 | +0.01 |
| mean | 有效多视图特征均值 | 重新训练适配均值后是否有收益 | 82.47 | +0.27 |
| camera_dropout | 逐相机 p=0.2 删除对应证据 | 正常精度与缺相机能力的取舍 | 82.04 | -0.16 |
| center_select | 图像中心质量代理 top-1 | 固定质量选择是否优于原选择/均值 | 82.46 | +0.26 |
| center_weighted | 中心质量代理加权 | 固定质量加权是否优于等权均值 | 82.42 | +0.22 |
| bilinear | 对齐后的双线性取样 | 离散 nearest 取样是否限制融合 | 82.36 | +0.16 |
| local_average | 3×3 有效图像 token 邻域均值 | 无参数局部上下文是否有益 | 81.86 | -0.34 |
| local_attention | xyz/strength 查询的3×3局部注意力 | 学习局部对应是否优于原取样/局部均值 | 82.21 | +0.01 |
| valid_mask | 投影后无观测点零注入 | 无对应图像支路的偏置是否影响结果 | 82.06 | -0.14 |
| valid_pool | 图像池化排除无效项 | 无效零值是否压过有效负特征 | 训练失败 | — |
| valid_both | 有效性 mask 与有效池化组合 | 两种处理是否需要共同传播 | 训练失败 | — |
| whole_camera_dropout | p=0.2 整组六相机证据丢弃 | 显式整组缺失训练是否优于逐相机 dropout | 82.41 | +0.21 |

control/gate/mean/camera_dropout均完成9376次计划迭代，成功optimizer更新分别9371/9372/9371/9371；由差值推算AMP跳步5/4/5/5。七组新增成功训练的更新数尚待train/summary.json。

normal-input观察：mean/中心选择/中心加权/整组dropout/bilinear分别较同预算control高0.27/0.26/0.22/0.21/0.16pp。control比历史基线低0.28pp；当前全部两轮clean结果未超过82.48%，不把相对训练control的恢复写成已刷新历史最佳。gate与local_attention各+0.01pp不足以建立稳定优势；短训负结果也不证明整条研究路线无效。

[四组微调证据](../records/experiments/finetune_stage1_20261101_r02/README.md) · [多路线证据/CSV](../records/experiments/multi_route_20261101_r01/README.md)

## 3. 缺相机配对验证：部分完成

目的：在同一相机子集下比较control、逐相机dropout、整组dropout，区分正常精度与缺失鲁棒性。

| 方法 | clean 6路 | 固定3路 | 固定1路 | 0路图像证据 |
|---|---:|---:|---:|---:|
| control | 82.20 | 60.79 | 49.20 | 36.80 |
| camera_dropout | 82.04 | 73.13 | 最近回执运行中 | 最近回执运行中 |
| whole_camera_dropout | 82.41 | 最近完整快照待执行 | 最近完整快照待执行 | 最近完整快照待执行 |

- 固定子集：3路=[0,2,4]、1路=[0]、0路=[]，不是全部组合均值。
- camera_dropout三路相对同条件control +12.34pp，clean−0.16pp。clean→三路降幅control21.41pp、dropout8.91pp，降幅差12.50pp。
- 当前结论限于本次训练与固定三路条件；1/0路及整组dropout未收到最终分数。
- 零相机证据不等于另训的纯LiDAR模型，也不能据仍读取/编码图像的实现宣称计算加速。
- 已收到两份完整16类summary：control零路36.80/44.59/85.42，dropout三路73.13/81.12/93.68（mIoU/mAcc/OA，%）。

[逐类结果及来源](../records/experiments/multi_route_20261101_r01/README.md)

## 4. 投影偏移评价：已排队，尚无结果

control/bilinear/local_attention三个两轮模型，各在resize后横向+4px条件下完整val。验证对应错位敏感性；与同样+4px的control比较，而非正常输入control。它是像素错位压力测试，不是SE(3)外参扰动。最近完整队列表中三项pending，之后没有新回执。

## 5. 十轮训练：已排队，尚无最终结果

预先固定control、gate、center_select、local_attention、valid_both、whole_camera_dropout，检验短训观察在较长相同预算下是否保留。除valid_both外的五项最近完整快照待执行；valid_both因短训失败，训练和评价均blocked。

晋级依据是短训与clean评价执行成功，不按两轮分数取消预定十轮路线。十轮不是已证明收敛的最终训练；各候选必须与十轮control比较，后续再对少量候选做必要多seed/消融/泛化验证。

## 6. 故障、机器重启和最近任务

- valid_pool与valid_both在首次loss检查报Non-finite first training loss，尚无首次成功更新。共同池化路径是排查范围，具体根因尚未定位；不是方法无收益。
- 依赖阻断4节点：screen_valid_pool_clean、screen_valid_both_clean、medium_valid_both_train、medium_valid_both_clean。其余独立任务继续。
- 用户报告11月2日10点多机器重启。两项中断验证旧输出/日志已保留归档，重新排队从头算；已有完成结果保留。
- 恢复父PID59334；dropout三路PID59343于14:15:03结束，control零路PID59338于14:15:23结束。
- 随后确认：GPU1启动robust_camera_dropout_cam1（PID100312），GPU0启动robust_camera_dropout_cam0（PID100432）。这是回执时刻状态，不是实时查询。
- 10:24快照counts为reused8/completed16/failed2/blocked4/running2/pending18；之后未收到完整queue.json，不把旧计数当最新。

[恢复记录与原恢复脚本边界](../records/analyses/campaign_restart_20261102.md)

## 7. 每条路线需要补齐的证据

| 路线 | 当前已回答的部分 | 仍要验证 |
|---|---|---|
| 输入选择/聚合 | 推理mean/first、训练mean/中心代理均已有分数 | 十轮中心选择、逐类/分组收益，必要时参数与预算匹配 |
| 局部对应 | bilinear/局部均值/局部注意力短训已有分数 | 配对4px、十轮注意力；额外参数容量消融；语义query/几何约束尚未实现 |
| 有效性传播 | valid_mask有clean结果 | 修复并新批次补跑valid_pool/both，再做单项/组合对照 |
| 融合门控 | 普通gate短训已有分数 | 十轮gate；显式可信度r/互补价值u仍未实现，不能用普通gate代替 |
| 鲁棒训练 | 固定三路dropout比control高12.34pp | 1/0路、整组dropout配对、必要多seed/更多子集，正常精度一并报告 |
| 完整方法结论 | 单seed筛选及部分压力评价 | 少量候选的充分训练/多seed/必要消融和泛化；不把6019帧当6019次训练重复 |

## 8. 记录与双电脑交接

每次新回执更新docs/HANDOFF.md和本表，在records/experiments/<run_id>/保留来源、日期/时区、配置/执行commit（未知写未知）、方法/预算/条件、最终指标、失败/恢复和下一步。完整16类JSON保持原回传值；展示表可四舍五入，原缺失字段保持null。

本次是用户服务器回执整理，不是完整export_record导出或独立重跑。尚缺多数逐类/有效配置/实际快照commit、更新与AMP统计、runtime和显存。记录提交commit不能冒充实验执行commit。

两机开发/服务器先git pull --ff-only；不force push。运行使用已保存代码快照，拉取文档不改变排队任务。大权重、预测、数据、大日志和凭据不提交。[同步约定](GITHUB_SYNC.md) · [描述性校核](../records/analyses/results_interpretation_20261102.md)
