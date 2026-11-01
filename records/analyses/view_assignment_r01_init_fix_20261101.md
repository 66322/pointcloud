# VIEW批次r01：DINO初始化失败与修复

日期：2026-11-01。依据：用户回传smoke_console.log，非代理登录服务器。

## 已观察到的错误

r01在构建DINOv2时，timm.create_model(pretrained=True)试图读取Hugging Face
预训练文件，但入口开启了HF_HUB_OFFLINE且未命中所需缓存，
抛出OfflineModeIsEnabled / LocalEntryNotFoundError。
失败发生在完整checkpoint加载之前，正式两项实验尚未启动。
Albumentations更新提示不是失败根因。

## 修复

- 完整兼容checkpoint已包含注册DINO分支参数，无需先额外加载timm预训练权重。
- 仅在本进程模型构建期间，将DINOv2的timm工厂调用改为pretrained=False。
- 随后仍由原作者加载器执行完整checkpoint的strict=True加载。
  如果缺少DINO参数，仍报错，不能用随机图像权重继续推理。
- 构建结束或异常时恢复原timm工厂，不改旧DITR源码，不迁移/删除缓存。
- 入口失败时直接显示日志尾部，减少额外复制日志步骤。

## 验证范围

本地新增回归测试覆盖：禁止额外DINO预训练读取，其他模型调用不变，
异常后工厂恢复，严格加载错误不被吞掉。
本地没有torch/timm服务器运行环境，未冒充H100前向成功。
服务器按相同入口启动新r02：一次单帧后直接完整实验，不添加验证关卡。

旧r01目录保留。使用r02避免覆盖失败日志。
