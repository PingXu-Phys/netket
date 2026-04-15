# NetKet 仓库 Git 操作手册

## 0. 极简速查

先看这个，够用时就不用往后翻：

```powershell
cd D:\Seafile\PHD\NQS\NetKet\netket
git status
git add -- <path>
git commit -m "message"
git push
git pull --rebase
git fetch upstream
git rebase upstream/main
git branch -a
git branch -vv
git log --oneline --graph --decorate --all -n 20
git diff
git restore --staged <path>
git stash push -m "wip"
```

最常见的 3 个场景：

```powershell
# 1. 提交并推送当前改动
git add -- <path>
git commit -m "message"
git push

# 2. 先同步远程，再继续本地开发
git stash push -m "wip before pull"
git pull --rebase
git stash pop

# 3. 把上游 main 同步到当前分支
git fetch upstream
git rebase upstream/main
git push --force-with-lease
```

---
适用目录：`D:\Seafile\PHD\NQS\NetKet\netket`

当前仓库已知信息：
- 当前分支：`PingXu-withproposal`
- 默认个人远程：`origin = git@github.com:PingXu-Phys/netket.git`
- 上游远程：`upstream = https://github.com/netket/netket.git`
- 当前分支跟踪：`origin/PingXu-withproposal`

---

## 1. 进入仓库与查看状态

```powershell
cd D:\Seafile\PHD\NQS\NetKet\netket
```

查看当前状态：

```powershell
git status
```

常见用途：
- 看当前在哪个分支
- 看本地是否有未提交修改
- 看是否领先或落后远程
- 看是否存在未跟踪文件

更紧凑地显示状态：

```powershell
git status --short
```

查看当前分支：

```powershell
git branch --show-current
```

查看最近提交：

```powershell
git log --oneline -n 10
```

---

## 2. 处理未跟踪文件

例如你现在遇到的是未跟踪文件：

```text
log of run of kondoheisenberg
```

Git 中“追踪一个文件”的意思，是把它加入暂存区并纳入版本控制。

添加单个文件：

```powershell
git add -- "log of run of kondoheisenberg"
```

说明：
- 文件名里有空格时，建议加引号
- `--` 用来明确后面跟的是路径，而不是选项

添加后再看状态：

```powershell
git status
```

如果状态里显示：

```text
new file:   log of run of kondoheisenberg
```

说明这个文件已经被追踪，并且已经进入暂存区，等待提交。

如果不想追踪它，而是忽略它，可以写入 `.gitignore`。例如忽略这个文件：

```gitignore
log of run of kondoheisenberg
```

如果只是不小心 `git add` 了，想取消暂存：

```powershell
git restore --staged -- "log of run of kondoheisenberg"
```

---

## 3. 提交本地修改

提交前先看哪些内容已暂存：

```powershell
git diff --cached
```

提交：

```powershell
git commit
```

这会打开编辑器让你填写提交说明。

也可以直接在命令行里写提交说明：

```powershell
git commit -m "Add log file for KondoHeisenberg run"
```

建议：
- 一次提交只做一类事情
- 提交说明用英文短句更利于协作
- 动词开头，说明动作，例如 `Add`, `Fix`, `Refactor`, `Update`

示例：

```powershell
git add -- "log of run of kondoheisenberg"
git commit -m "Add log file for KondoHeisenberg run"
```

---

## 4. 查看本地改动

查看工作区改动：

```powershell
git diff
```

查看已暂存但未提交的改动：

```powershell
git diff --cached
```

查看某个文件的改动：

```powershell
git diff -- path\to\file.py
```

查看提交历史图：

```powershell
git log --oneline --graph --decorate --all -n 20
```

查看某个文件的历史：

```powershell
git log --follow -- path\to\file.py
```

查看某次提交具体改了什么：

```powershell
git show <commit-id>
```

---

## 5. 分支操作

查看本地分支：

```powershell
git branch
```

查看本地和远程分支：

```powershell
git branch -a
```

新建并切换到新分支：

```powershell
git switch -c my-new-branch
```

切回已有分支：

```powershell
git switch PingXu-withproposal
```

基于远程分支创建本地跟踪分支：

```powershell
git switch -c feature-x --track origin/feature-x
```

删除本地分支：

```powershell
git branch -d my-old-branch
```

强制删除本地分支：

```powershell
git branch -D my-old-branch
```

注意：
- 删除分支前先确认你不需要该分支上的未合并提交
- 当前分支不能删除自己

---

## 6. 本地与远程的互动

这个仓库当前有两个远程：

查看远程：

```powershell
git remote -v
```

典型输出含义：
- `origin` 是你自己的 fork 或主远程
- `upstream` 是上游项目 `netket/netket`

### 6.1 从远程获取最新信息

只更新远程引用，不改你当前文件：

```powershell
git fetch origin
git fetch upstream
```

作用：
- 更新 `origin/*` 和 `upstream/*` 的远程分支信息
- 不会自动修改当前工作区

看本地分支相对远程是领先还是落后：

```powershell
git status
```

或：

```powershell
git branch -vv
```

### 6.2 推送本地分支到远程

当前分支已有跟踪关系时：

```powershell
git push
```

如果是第一次把本地新分支推到 `origin`：

```powershell
git push -u origin my-new-branch
```

`-u` 的作用：
- 建立上游跟踪关系
- 以后可以直接用 `git push` 和 `git pull`

当前仓库中，`PingXu-withproposal` 已跟踪 `origin/PingXu-withproposal`，所以通常直接：

```powershell
git push
```

### 6.3 从远程拉取更新

最常用：

```powershell
git pull
```

它本质上相当于：

```powershell
git fetch
git merge
```

如果你偏好线性历史，可用 rebase：

```powershell
git pull --rebase
```

适用场景：
- 远程有人更新了同一分支
- 你想把远程改动先拿下来，再继续自己的工作

### 6.4 把上游仓库更新同步到自己的分支

先抓取上游：

```powershell
git fetch upstream
```

切到你要更新的分支：

```powershell
git switch PingXu-withproposal
```

把上游 `main` 合并进当前分支：

```powershell
git merge upstream/main
```

或者改成 rebase：

```powershell
git rebase upstream/main
```

两者区别：
- `merge` 会产生一次合并提交，历史更完整
- `rebase` 会重写你本地提交的基底，历史更直

如果 rebase 成功后要更新远程，通常需要：

```powershell
git push --force-with-lease
```

注意：
- 只有在你明确知道自己在改写分支历史时，才使用 `--force-with-lease`
- 不要随便对多人共用分支做强推

---

## 7. 合并与变基

### 7.1 merge

```powershell
git switch target-branch
git merge source-branch
```

例子：

```powershell
git switch PingXu-withproposal
git merge upstream/main
```

### 7.2 rebase

```powershell
git switch feature-branch
git rebase main
```

发生冲突时：
1. 手动编辑冲突文件
2. 标记已解决
3. 继续 rebase

```powershell
git add path\to\resolved_file
git rebase --continue
```

如果想放弃本次 rebase：

```powershell
git rebase --abort
```

---

## 8. 撤销与恢复

### 8.1 撤销工作区里尚未暂存的修改

```powershell
git restore path\to\file.py
```

撤销全部未暂存修改：

```powershell
git restore .
```

### 8.2 取消暂存

```powershell
git restore --staged path\to\file.py
```

### 8.3 回退最近一次提交，但保留文件内容

保留改动在工作区：

```powershell
git reset --mixed HEAD~1
```

保留改动在暂存区：

```powershell
git reset --soft HEAD~1
```

### 8.4 恢复某个历史版本的文件

```powershell
git restore --source=<commit-id> -- path\to\file.py
```

### 8.5 使用 reflog 找回误操作前的位置

```powershell
git reflog
```

然后：

```powershell
git reset --hard <reflog-id>
```

注意：
- `reset --hard` 会丢弃当前工作区改动，使用前务必确认

---

## 9. 暂存现场：stash

临时保存当前未提交修改：

```powershell
git stash push -m "wip before sync"
```

查看 stash 列表：

```powershell
git stash list
```

恢复最近一次 stash，并从列表删除：

```powershell
git stash pop
```

恢复但不删除：

```powershell
git stash apply
```

删除某个 stash：

```powershell
git stash drop stash@{0}
```

适用场景：
- 当前改了一半，但需要先切分支
- 先拉取远程更新，再继续手头工作

---

## 10. 远程分支与跟踪关系

查看每个本地分支跟踪哪个远程分支：

```powershell
git branch -vv
```

给当前分支设置上游：

```powershell
git branch --set-upstream-to=origin/PingXu-withproposal
```

第一次推送并顺便建立跟踪关系：

```powershell
git push -u origin PingXu-withproposal
```

删除远程分支：

```powershell
git push origin --delete my-old-branch
```

清理已在远程删除、但本地还保留引用的远程分支信息：

```powershell
git fetch --prune
git remote prune origin
```

---

## 11. 你这个仓库的常见工作流

### 场景 A：把一个新文件加入版本控制并推送到远程

```powershell
cd D:\Seafile\PHD\NQS\NetKet\netket
git status
git add -- "log of run of kondoheisenberg"
git commit -m "Add log file for KondoHeisenberg run"
git push
```

### 场景 B：先同步远程，再继续本地开发

```powershell
cd D:\Seafile\PHD\NQS\NetKet\netket
git status
git stash push -m "wip before pull"
git pull --rebase
git stash pop
```

### 场景 C：同步上游 `upstream/main` 到你的分支

```powershell
cd D:\Seafile\PHD\NQS\NetKet\netket
git fetch upstream
git switch PingXu-withproposal
git rebase upstream/main
git push --force-with-lease
```

### 场景 D：检查自己比远程多了哪些提交

```powershell
git log --oneline origin/PingXu-withproposal..HEAD
```

### 场景 E：检查远程比本地多了哪些提交

```powershell
git log --oneline HEAD..origin/PingXu-withproposal
```

---

## 12. 冲突处理基础

当 `merge` 或 `rebase` 出现冲突时，Git 会在文件里写入类似标记：

```text
<<<<<<< HEAD
当前分支内容
=======
另一侧内容
>>>>>>> other-branch
```

处理步骤：
1. 打开冲突文件，手工决定保留什么内容
2. 删除冲突标记
3. 保存文件
4. 标记为已解决

```powershell
git add path\to\resolved_file
```

如果是 merge：

```powershell
git commit
```

如果是 rebase：

```powershell
git rebase --continue
```

---

## 13. 查看配置与身份

查看用户名邮箱：

```powershell
git config --get user.name
git config --get user.email
```

设置全局用户名邮箱：

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

查看远程地址：

```powershell
git remote -v
```

把 `origin` 地址改成新的：

```powershell
git remote set-url origin git@github.com:PingXu-Phys/netket.git
```

---

## 14. 提交说明编辑器 nano 的基本操作

如果 `git commit` 打开 nano：
- 直接在最上面输入提交说明
- `Ctrl+O` 保存
- 按 `Enter` 确认文件名
- `Ctrl+X` 退出

推荐提交说明示例：

```text
Add log file for KondoHeisenberg run
```

如果想退出并放弃：
- `Ctrl+X`
- 如果提示保存，选 `N`

---

## 15. 建议与注意事项

建议：
- 先 `git status`，再做任何操作
- 推送前先看当前分支是不是你以为的那个分支
- 重要分支更新前先 `git fetch`
- 改写历史前先确认是否只有你自己在用该分支
- 强推时优先用 `--force-with-lease`，不要直接 `--force`

高风险命令：
- `git reset --hard`
- `git clean -fd`
- `git push --force`

这些命令会删除工作区内容或改写远程历史，使用前必须确认。

---

## 16. 一组最常用命令速查

```powershell
git status
git add -- <path>
git commit -m "message"
git push
git pull --rebase
git fetch upstream
git branch -a
git branch -vv
git log --oneline --graph --decorate --all -n 20
git diff
git restore --staged <path>
git stash push -m "wip"
git rebase upstream/main
```

