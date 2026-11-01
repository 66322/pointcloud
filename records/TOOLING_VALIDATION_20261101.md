# 工具开发机验证

日期：2026-11-01。地点：本地 Windows 开发机，非 H100 服务器。

命令：python -m unittest discover -s tests -v

- 9 项测试：7 项通过，2 项因本机缺少 torch 而跳过。
- 通过范围：ignore/缺失类别指标、无效ID/形状拒绝、确定性嵌套抽样、
  相机坐标反变换与视锥、场景bootstrap最低条件、
  合成原始点→标签→缓存→分组报告→导出端到端、
  已有目录保护、点序/长度不符拒绝、配对比较的输入指纹校验。
- 跳过范围：torch赋值顺序/均值、边界、batch及梯度测试；
  服务器首轮命令会补跑，不能把跳过写成通过。
- Python编译与 evaluate/diagnose 的 --help 入口通过。
- git diff --check 通过。
- 尚未执行：真实数据读取、strict权重加载、DINO冻结审计、
  单帧GPU前向、100帧及6019帧评测。

没有据此生成任何改进mIoU数值。
