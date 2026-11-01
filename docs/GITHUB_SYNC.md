# GitHub同步：HTTPS 403处理

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
git pull --rebase
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
