# 基线事实、证据等级与数据流

日期：2026-11-01。

## 已有记录

来源：用户本轮提供的 DITR_改进实验_代码大模型提示词_v2_20261101.md。
该文档作为研究材料，其中历史环境和分数需要与服务器文件交叉核对。

- nuScenes full 已下载、派生 10-sweeps 索引已生成。当前源码实际只读取关键帧点云；
  索引名含 10sweeps 不等于已做时序融合。
- 任务：nuScenes lidarseg 16 类，PTv3 + DINOv2 ViT-L。
- 6019 帧 val：mIoU 82.48%、mAcc 90.03%、OA 94.63%。
- 作者相应 mIoU 83.1%；差 -0.62 pp 尚未解释。
- 性质：预训练检查点评测复现，不是从头训练复现。
- 原始/兼容权重约 1.7 GB；文档记载兼容权重 strict=True 匹配 859 项，
  四个 proj_image_feat 的 28 个编号条目做过适配。本仓库不重新转换权重，
  不使用 strict=False 隐藏失败；服务器 smoke 会再次验证。
- DINO 注册为 nn.Module 以支持权重加载，不代表已冻结。
  评测适配器显式 requires_grad_(False)、eval() 并输出审计。
- 原始缓存为最终类别，不含概率，不从类别缓存编造置信度。
- 21,356 个点—相机关系不等于 21,356 个不同可见点。

| 类别 | 已记录 IoU % |
|---|---:|
| barrier | 80.20 |
| bicycle | 66.68 |
| bus | 96.05 |
| car | 94.11 |
| construction_vehicle | 60.33 |
| motorcycle | 92.63 |
| pedestrian | 86.47 |
| traffic_cone | 75.33 |
| trailer | 73.88 |
| truck | 86.69 |
| driveable_surface | 96.75 |
| other_flat | 77.40 |
| sidewalk | 75.56 |
| terrain | 76.70 |
| manmade | 91.37 |
| vegetation | 89.57 |

舍入后的逐类数字仅供参考，不反向修改任何评测输出来贴合此表。

## 数据流与关键函数

| 阶段 | 典型形状 | 上游入口 | 核对事项 |
|---|---|---|---|
| LiDAR 关键帧 | N×5 原始；coord N×3、strength N×1 | datasets/nuscenes.py::get_data | 原始 ring 等不等于全部用作特征；不自动拼历史帧 |
| 六路相机 | 6×H×W×3 → B×6×3×378×672 | 数据加载和 ImageResize/Collect | resize/flip 同步投影坐标 |
| 投影对应 | image_coord N×6×2；image_mask N×6 | datasets/utils.py::points2image | 当前室外调用未提供深度图；frustum 并非无遮挡真值 |
| DINOv2-L | B×6×1024×27×48 | models/point_transformer_v3/utils.py::get_image_feat | patch=14；冻结；注册/保存行为明确 |
| 点级图像特征 | N×1024 | 同目录 assign_image_feat | 后视图覆盖前视图；无对应默认零 |
| PTv3 编码 | 多层 N_l×C_l | point_transformer_v3m1_image.py | 图像特征随层级 max pooling |
| 解码器注入 | 上采样、跳连、图像分支变换后相加 | SerializedUnpooling.forward | Linear/BN/GELU 后零输入未必仍为零 |
| 分割片段 | N_fragment×16 | 模型 seg_logits | 每片段 softmax 后按 index 累加 |
| 原始点预测 | N_original 整型类别 | engines/test.py::SemSegTester.test | 如有 origin_segment/inverse，先回映再存 *_pred.npy |

与已读上游提交一致；本次不是对服务器当前源码的实地审计。

## 对齐检查的能力边界

离线诊断直接读取原始 LiDAR、原始标签，经固定 32→16 标签映射后，
与 token 命名的最终缓存比较。严格检查形状、类别范围、帧唯一性和文件指纹。

**长度相同不足以证明点序正确。** 需用户依据已成功的 SemSegTester /
eval_subset.py 输出路径确认缓存为最终原始点顺序，再传
--confirm-original-point-order。不接受中间 voxel/fragment 预测作为原始点缓存。

原始模型 full val 的离线汇总应与 82.48% 舍入误差范围一致；不一致时先停下来查
点序、标签映射、数据索引及缓存来源。100 帧不需要也不应该匹配 82.48%。

## 复现与结果等级

1. reported：用户文档提供，待原始产物核对。
2. local_tested：工具在合成 CPU 数据上通过，不代表服务器通过。
3. server_verified：有服务器 manifest、日志、哈希和完整输出。
4. supported_method：公平训练、消融、多种子/泛化证据支持。

2026-11-01补充：用户终端回传支持服务器9项测试通过、preflight核对旧日志和
6019个预测、100帧旧缓存诊断完成。完整JSON/CSV因push 403暂未上传，
证据来源明确记为“服务器终端回传”，不冒充已从GitHub读取全部产物。
strict加载及DINO冻结审计仍待模型smoke；尚无新增改进方法结果。
