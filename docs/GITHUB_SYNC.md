# GitHub同步与两台电脑交接

2026-11-02用户约定：GitHub main保存最新代码、轻量实验结果与交接；两台电脑轮流开发。服务器由用户执行命令，完整产物仍留runs/，GitHub不是实时进度镜像。

## 每次接手

~~~bash
git status --short
git pull --ff-only
~~~

先读[HANDOFF](HANDOFF.md)、[EXPERIMENT_MASTER](EXPERIMENT_MASTER.md)、[EXPERIMENT_MATRIX](EXPERIMENT_MATRIX.md)。本地有修改先保留；如果分支分叉，检查双方提交并合并处理，不用force push或reset --hard覆盖另一台电脑。

服务器已有run使用启动时的code/计划/路径快照，git pull更新文档不会改变它；不要为同步文档重启实验。

## 新回执写到哪里

| 内容 | 文件位置 |
|---|---|
| 最新运行/完成/失败状态、接手动作 | docs/HANDOFF.md |
| 实验问题、对照、已有分数、待验证项 | docs/EXPERIMENT_MASTER.md、EXPERIMENT_MATRIX.md |
| 用户回传结果JSON/CSV、短日志、逐类JSON | records/experiments/<run_id>/ |
| 结论边界、故障、重启恢复 | records/analyses/ |

每次同时更新来源、日期/时区和观察时刻。保留所有正负结果，未知字段留null；旧running/PID/任务计数不当实时数据。报告中执行commit与本次记录commit分开，实际运行快照未知就写未知。

当前目录已有回执，完整export_record产物用同目录下全新的server_export_日期时间子目录。正在运行的队列不整批导出；机器重启归档目录的导出边界见[恢复记录](../records/analyses/campaign_restart_20261102.md)。

## 提交与同步

检查差异后，只加入本次修改的确切docs/records路径并提交。推送前再拉取：

~~~bash
git diff --check
git status --short
git pull --ff-only
git push origin main
~~~

若存在尚未推送的本地提交且main已被另一台电脑推进，保留双方提交、检查差异后完成合并再正常push；不要直接丢弃结果。无写权限时按下方处理，或使用当前已授权GitHub连接提交。不要在聊天或文件里保存token。

不提交数据、权重、逐点预测、完整大日志、凭据。每次结果同步不要求重新运行实验或重复export。

---

# HTTPS 403处理（故障排查）

适用：clone/pull成功，push提示Permission to 66322/pointcloud.git denied to 66322。
这说明当前写入鉴权未通过；单靠该输出不能确定是token仓库范围、写权限、
失效状态还是其他鉴权设置。公共仓库可匿名拉取，所以clone成功不证明能push。

实验结果已在服务器提交时，不需要重新跑实验、重新export或重新commit。
保留本地提交。不要reset --hard、clean或force push。

## 检查或创建细粒度token

在自己电脑浏览器打开GitHub Settings → Developer settings →
Personal access tokens → Fine-grained tokens。

- Resource owner：66322。
- Repository access：选中pointcloud；原来只授权remote_sensing_frequency的token
  不会自动获得pointcloud写权限。
- Repository permissions → Contents：Read and write。
- 检查有效期。生成后仅在服务器交互式Password提示中输入。

GitHub HTTPS命令行的Password位置输入个人访问token，不是网站登录密码。
不要把token放进remote URL、shell命令或实验文件，也不发送到聊天。

## 推送已有提交

~~~bash
cd /home/jovyan/zzx/pointcloud
git status --short
git log -2 --oneline
git remote get-url origin
git pull --ff-only
git -c credential.helper= push origin main
~~~

这里空credential.helper只影响这次命令，避免沿用旧helper中的凭据，
不会全局删除其他项目的凭据。用户名输入66322，Password输入新token，
输入时终端不显示字符属于正常行为。

若仍然403，核对token的所属账号、pointcloud是否被选中、Contents写权限及有效期。
只回传报错文字，不回传token。

本项目后续推送也可使用git -c credential.helper= push origin main，
每次在交互式提示中输入token；需要持久保存时再选择合适的凭据管理方式。

## Git提交身份提示

root身份自动配置不是403原因。若要规范后续提交，可仅在本仓库设置：

~~~bash
git config --local user.name "66322"
git config --local user.email "57258129+66322@users.noreply.github.com"
~~~

已存在实验提交无需为此amend或改写。

## 其他提示

- cp -n是可移植性提示，文件已存在时不会覆盖。新复制写法可用：
  test -f configs/paths.local.json || cp configs/paths.server.example.json configs/paths.local.json
- CSV的CRLF→LF提示是文本换行规范化，当前没有证据表明它改变指标数值。
- artifact_hashes记录的是服务器导出时的文件字节；Git可能规范化文本换行，
  跨机器逐字节核对需同时考虑Git blob与原始导出字节。JSON数值和运行来源仍需原样保留。

## 官方依据

- [管理个人访问token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [HTTPS远程仓库鉴权](https://docs.github.com/en/get-started/git-basics/about-remote-repositories)
