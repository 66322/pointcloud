# 无人值守多路线接续队列

2026-11-01：代码已交付，服务器待用户执行下面的一次启动。
不修改、不重启finetune_stage1_20261101_r02，也不重复已有82.48%基线推理。

## 现在执行一次

~~~bash
cd /home/jovyan/zzx/pointcloud
git pull --ff-only
conda activate pointcept-torch2.5.0-cu12.4

bash scripts/start_campaign.sh multi_route_20261101_r01 \
  --after runs/finetune_stage1_20261101_r02
~~~

启动打印PID和日志位置后即可关闭网页。nohup、stdin=/dev/null、stdout/stderr日志已设置。
最开始显示WAIT/waiting_for_prior是正常的：控制器先等旧四组批次结束，
再检查GPU0/1空闲并接续。无需等到旧实验完成后再回来手动执行。
旧队列若正常写出failed，也会继续：只复用已完成结果，补跑缺失的短训/评价。
旧队列若被强制kill而状态一直停在running，接续器会保守等待，不能假装旧任务已退出。

## 实际数量

固定计划：[multi_route_queue.json](../configs/training/multi_route_queue.json)。
共50个任务节点，其中旧四组正常完成时复用8个节点，实际新增：

| 类别 | 数量 | 预算与用途 |
|---|---:|---|
| 两轮短训 | 9 | 独立筛选输入选择、取样、有效性与增强 |
| 十轮训练 | 6 | control/gate及四条预先指定路线，复核较长训练 |
| 正常输入完整val | 15 | 每个新训练最终权重评价一次，6019帧原TTA |
| 相机缺失完整val | 9 | 三个短训模型×保留3/1/0相机 |
| 投影偏移完整val | 3 | 三个短训模型×横向4像素偏移 |
| 合计实际新增 | 42 | 15个训练任务、27个评价任务，两卡自动滚动 |

任务按依赖就绪和列表顺序分配给空闲卡；某卡不必等另一卡同阶段结束。
只用GPU0和1，每模型单卡batch6。训练进程退出后，另开评价进程释放训练内存。
某方法失败会保留错误、阻断依赖它的任务，并继续其他独立方法。
不在OOM时静默降低batch，也不伪造完成分数。

## 新增九组短训

| 方法名 | 具体变化 | 对照与解释 |
|---|---|---|
| center_select | 按投影点离图像中心的归一化距离，选质量代理值最高的有效视图 | 主比较为同预算control/mean；first仅已有推理线索，不是物理遮挡可见性 |
| center_weighted | 用同一中心质量代理值加权有效视图 | 对比mean；固定规则，没有额外质量网络 |
| bilinear | 图像像素中心与patch中心对齐后双线性取样，边界复制 | 对比原nearest；原文中简单取样改法不保证有效，仅作必要对照 |
| local_average | 同分辨率3×3有效邻域平均，再按原last分配 | 局部上下文的简单规则对照，边缘不计补零 |
| local_attention | xyz/strength查询、3×3图像token键，学习局部注意力 | 残差系数从0开始；仍按原last选相机 |
| valid_mask | 有效对应标记跨池化传播，图像投影后无观测点严格零注入 | 单独检查投影分支偏置/归一化的影响 |
| valid_pool | 图像max pooling排除无效项，全无效簇回零 | 单独检查零值是否压过有效负特征 |
| valid_both | valid_mask+valid_pool | 与两个单项比较组合效果 |
| whole_camera_dropout | 训练时每样本p=0.2整组六相机证据丢弃 | 与现有逐相机p=0.2比较，期望丢弃相机数量相同 |

中心质量代理为q=exp(-2*r²)，r是投影点相对图像中心的归一化二维偏移。
两个中心策略都只使用有效投影；top-1相同分数时按相机顺序稳定打破平局。
所有方法均使用同一初始兼容checkpoint、数据、增强、损失、seed和两轮预算。
DINO冻结；其余原3D网络继续训练，门控/局部注意力新增参数另行训练。

local_attention当前查询来自输入xyz+strength，不是已经编码的3D语义特征。
键维度32、温度0.2，输出=nearest+tanh(alpha)*(attention-nearest)，alpha初始为0。
使用2048点分块及训练激活重计算，避免保留N×9×1024的全部候选值。
它与无参数local_average可作筛选比较；注意力机制和额外参数容量的最终归因
仍需要参数匹配对照。本批没有冒称完成多尺度语义query或表面几何约束。
valid_mask在投影后的mask不等于有效样本专用BatchNorm；无效输入仍参与原BN统计。

## 十轮任务已经固定，不等人回来挑

预先指定control、gate、center_select、local_attention、valid_both、whole_camera_dropout，
每项10 epoch，从同一个原预训练checkpoint重新开始，使用完整十轮OneCycle。
每轮4688个batch，十轮46880次计划迭代；统一与十轮control比较。
这不是将旧两轮checkpoint直接改epochs续训，也不根据验证集最高分临时挑方法。
只要求该方法短训及正常val执行成功；两轮分数下降不会自动取消其十轮任务。
十轮仍不是已经证明收敛的最终论文训练。

## 缺相机与偏移评价

使用短训control、camera_dropout、whole_camera_dropout，各评价3/1/0相机，另有正常6相机结果。
保留相机索引固定为：3路=[0,2,4]，1路=[0]，0路=[]。
三模型使用相同子集、帧、TTA和评测流程。固定子集含方向偏好，不能冒充全部相机组合均值。
0路表示没有点—相机证据；control原投影分支仍可能有偏置，不等于重训的纯LiDAR模型。
图像仍会读取/编码，不据此宣称掉相机时的计算加速。

短训control、bilinear、local_attention另在resize后图像坐标x方向统一+4像素下各评价一次。
越界对应会失效。这是像素错位压力测试，不是相机外参SE(3)扰动。
所有压力条件都包装在各模型自己的融合策略外，不把已训练的mean等改回last。
每个条件与同条件control比较；不混用正常输入、两轮、十轮的对照。

## 保存位置

代码仓库仍在/home/jovyan/zzx/pointcloud；原DITR、数据和权重目录不动。

~~~text
/home/jovyan/zzx/pointcloud/
  runs/_launch/multi_route_20261101_r01/
    pid
    console.log                         # 等待、启动、完成、失败事件
  runs/multi_route_20261101_r01/
    plan.json                           # 固定任务定义
    screen_protocol.json
    medium_protocol.json
    queue.json                          # 每个任务的状态/PID/依赖/错误/实际输出路径
    reuse.json                          # 明确列出复用的旧任务
    status.json
    report.md
    comparison.csv
    summary.json
    paths.json
    tokens.json
    code/                               # 启动时的小型代码/配置快照
    logs/<任务ID>.log                    # 每任务完整终端日志
    jobs/<任务ID>/                      # config、权重或验证预测、逐类指标等
~~~

复用项的实际产物仍在旧r02中，不复制大权重/预测；queue.json/source明确指回旧路径。
新代码、权重、预测都按任务分目录，禁止覆盖旧产物。
启动后再git pull不会修改已保存的子任务代码/计划。
所有检查都是必要的进程/小型协议/结果检查，不扫描数据和权重SHA。

按已见0.46s/步估算，新增两轮短训约10.8 GPU小时，十轮六组约36 GPU小时；
若速度相近，两卡仅新增训练约23.4小时。完整val、局部注意力额外开销、
旧队列等待均另计，宜留一到两天；实际时长以各任务日志为准。

## 查看与中断恢复

无需每隔几小时操作；想查看时：

~~~bash
cd /home/jovyan/zzx/pointcloud
cat runs/multi_route_20261101_r01/status.json
tail -n 15 runs/_launch/multi_route_20261101_r01/console.log
cat runs/multi_route_20261101_r01/report.md
~~~

控制器意外退出时，只对同一个批次执行：

~~~bash
bash scripts/resume_campaign.sh multi_route_20261101_r01
~~~

恢复使用该批保存的代码/路径/计划，接管仍存活的子进程观察状态；
不重复启动它们，不重跑已完成项目。OS锁防止同一批两个控制器并发。
失败的方法不会无休止重试，修复实现后应另建明确版本；其余任务仍会继续。

## 完成后同步轻量结果

终态completed或completed_with_failures都可导出；后者包含失败/依赖阻断记录。
训练/评价细节已经自动记录在runs，不需要逐个手工抄结果。
结束后一次导出、提交即可；队列不代替用户保存GitHub凭据。

~~~bash
python -m pointcloud_lab.export_record \
  --run runs/multi_route_20261101_r01 \
  --record records/experiments/multi_route_20261101_r01
git add records/experiments/multi_route_20261101_r01
git commit -m "Record multi-route DITR campaign"
git pull --rebase
git -c credential.helper= push origin main
~~~

本地CPU测试不代表H100实跑或涨点；服务器每方法第一次真实训练更新即完成
梯度/冻结检查，通过后直接继续，不追加单帧、20帧、100帧或旧基线复现关卡。
显式r/u可信度分支、选择性蒸馏、高分辨率区域和时序仍是后续任务，未伪装成已入本队列。
