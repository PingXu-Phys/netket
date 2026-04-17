# optimization_overview

这份文档是 `orbital_optimization/` 目录的总领说明。

它只回答四个问题：

1. 这个目录最终想收敛成什么结构。
2. 为什么文件名要从 `optimize_no_*` 改成 `orbital_*`。
3. 每个模块最终负责什么。
4. 迁移时应该按什么顺序做，才能不把现有代码打断。

这份文档是总纲。

更细的实现拆分、接口设计、NetKet 使用边界、落地顺序，继续放在 [optimiaztion_engineering_plan.md](/D:/Seafile/PHD/NQS/NetKet/netket/orbital_optimization/optimiaztion_engineering_plan.md) 里。

---

## 1. 为什么要改名

当前代码主干虽然已经开始拆分，但文件名还带着很强的历史痕迹：

- `optimize_no.py`
- `optimize_no_core.py`
- `optimize_no_losses.py`

这里的 `no` 原本是 natural orbital 的缩写，在物理上没有问题，但在工程上有两个明显问题：

1. 它不直接表达模块职责。
   看到 `optimize_no_core.py`，并不能马上知道它是“公共优化骨架”；看到 `optimize_no_losses.py`，也不能马上看出它是“loss 工厂”。

2. 它把“物理背景”和“工程角色”混在了一起。
   我们现在更需要的是：
   - 文件名直接表达职责；
   - `NO` 作为物理概念留在变量名、注释、文档里；
   - 工程模块名则用更直白的 `orbital_*`。

所以新的命名原则是：

- 物理概念继续保留 `NO / natural orbital`；
- 文件与模块名称改成按职责命名；
- 让目录从“历史脚本集合”收敛成“职责清楚的模块集合”。

---

## 2. 目标结构

最终建议的主干结构如下：

```text
orbital_optimization/
  optimization_overview.md
  optimization_flow.md
  optimiaztion_engineering_plan.md
  README.md

  orbital_utils.py
  orbital_optimizer.py
  orbital_losses.py
  orbital_pruning.py
  orbital_sparse.py
  orbital_runner.py
  orbital_netket_eval.py      # 可选，后置

  orbital_legacy.py           # 兼容层，可选保留
  history/
```

这里的角色非常明确：

- `optimization_overview.md`
  总领文档。
- `optimization_flow.md`
  算法流程主纲。
- `optimiaztion_engineering_plan.md`
  工程实施细节。
- `README.md`
  当前主干说明。
- `orbital_utils.py`
  公共工具。
- `orbital_optimizer.py`
  公共优化骨架。
- `orbital_losses.py`
  loss 与 diagnostics。
- `orbital_pruning.py`
  连续阈值裁剪与稳定支撑发现。
- `orbital_sparse.py`
  固定支撑下的稀疏重优化。
- `orbital_runner.py`
  总调度入口。
- `orbital_netket_eval.py`
  NetKet benchmark 层。
- `orbital_legacy.py`
  兼容旧接口的历史壳。

---

## 3. 当前文件到目标文件的映射

建议按下面的映射来迁移：

| 当前文件 | 目标文件 | 说明 |
| --- | --- | --- |
| `optimize_no.py` | `orbital_legacy.py` | 旧版 monolithic baseline，最后压成兼容层或移入 `history/` |
| `optimize_no_core.py` | `orbital_optimizer.py` | 公共优化骨架 |
| `optimize_no_losses.py` | `orbital_losses.py` | loss 工厂与 metrics 评估 |
| `optimize_no_utils.py` | `orbital_utils.py` | 当前还没单独存在，但应先抽出 |
| `optimize_no_pruning.py` | `orbital_pruning.py` | 新增 |
| `optimize_no_sparse.py` | `orbital_sparse.py` | 新增 |
| `optimize_no_flow_runner.py` | `orbital_runner.py` | 新增 |
| `optimize_no_netket_eval.py` | `orbital_netket_eval.py` | 可选，后置 |

这张表的重点不是立刻改名，而是先把目标名字固定下来，避免后面边写边改，再次漂移。

---

## 4. 每个模块到底干什么

## 4.1 `orbital_utils.py`

这是最底层工具层。

它应该只放“积木”，不放“流程”。

适合放进去的内容：

- `_require_square_matrix`
- `_hermitian_part`
- `_resolve_distance_matrix`
- `_resolve_interaction_matrix`
- `extract_one_body_hopping_matrix`
- `extract_interaction_matrix`
- `extract_hamiltonian_terms`
- `rotate_one_body_matrix`
- `rotate_hamiltonian_to_basis`
- `rotate_hamiltonian_to_natural_orbitals`
- `build_occupation_blocks`

它不应该放：

- loss 定义
- pruning 逻辑
- sparse loop
- runner 调度

一句话：

> `orbital_utils.py` 负责提供公共工具，不表达策略。

## 4.2 `orbital_optimizer.py`

这是优化骨架层。

它负责：

- `prepare_base_context(...)`
- `NOCoreConfig` 对应的新公共 config
- block 参数初始化
- block rotation 组装
- Optax optimizer 创建
- 统一优化循环
- 最终结果打包

它不负责：

- 物理 loss 的具体公式
- 阈值 schedule
- 稀疏 mask 约束下的最终 refinement

一句话：

> `orbital_optimizer.py` 只回答“怎么优化”。

## 4.3 `orbital_losses.py`

这是目标函数层。

它负责：

- soft loss config
- barrier loss config
- original loss 的兼容 builder
- `build_soft_loss_module(...)`
- `build_barrier_loss_module(...)`
- `evaluate_metrics(...)`
- `history_entry(...)`

它不负责：

- 优化循环
- 裁剪顺序
- 总调度

一句话：

> `orbital_losses.py` 只回答“优化什么”和“怎么评估”。

## 4.4 `orbital_pruning.py`

这是连续阈值化层。

它负责：

- 管理 `tau_1 < tau_2 < ... < tau_K`
- 从当前 rotation 生成 support
- 做 top-`m` 保底和空行空列修补
- 记录每一轮 support
- 计算持久度矩阵
- 用统一 metrics 做 accept / reject

它不负责：

- 重写物理 loss
- 旋转完整 Hamiltonian 做重活
- 最终幺正稀疏 refinement

一句话：

> `orbital_pruning.py` 只回答“怎么连续剪枝并发现稳定结构”。

## 4.5 `orbital_sparse.py`

这是固定支撑下的重优化层。

它负责：

- 接收 `support / mask`
- 在支撑约束下优化 `V`
- 做 mask projection
- 做 polar retraction
- 输出 `V_sparse`
- 检查 `unitarity_error` 与 `mask_error`

它和 `orbital_optimizer.py` 不同。

原因是：

- `orbital_optimizer.py` 用的是 dense block 参数化；
- `orbital_sparse.py` 面对的是固定零结构约束；
- 两者的参数空间不同。

一句话：

> `orbital_sparse.py` 负责把发现到的结构变成真正可用的稀疏幺正矩阵。

## 4.6 `orbital_runner.py`

这是总调度层，也是以后最适合暴露给外部的入口。

它负责：

- 准备 `base_context`
- 选择 dense loss 类型
- 调用 dense optimize
- 调用 pruning schedule
- 调用 sparse refinement
- 可选调用 NetKet benchmark
- 汇总最终结果

它不应该：

- 自己写 loss
- 自己实现底层优化器
- 自己重写 support 构造规则

一句话：

> `orbital_runner.py` 负责把整条 orbital optimization 流程串起来。

## 4.7 `orbital_netket_eval.py`

这是后置模块，不是当前第一优先级。

它负责：

- 接过 rotated Hamiltonian
- 构建 `MCState`
- 运行 `VMC` 或 `VMC_SR`
- 输出 benchmark 结果

重点是：

- 它属于 NQS benchmark 层；
- 不属于 basis loss loop 本体。

一句话：

> `orbital_netket_eval.py` 只做基底评估，不参与轨道基优化本体。

## 4.8 `orbital_legacy.py`

这是兼容层。

它的使命不是继续长大，而是逐步缩小。

它可以暂时承担：

- 旧接口兼容
- 对外保持老函数名可用
- 内部转调新的 `utils + optimizer + losses + runner`

最终它有两个结局：

1. 保留为薄兼容壳。
2. 或完全抽干后移入 `history/`。

一句话：

> `orbital_legacy.py` 负责兼容，不负责未来新功能。

---

## 5. 推荐的迁移顺序

这里最重要的不是“改名”，而是“保证每一步改完后代码还能跑”。

推荐顺序如下：

### Step 1：先新增 `orbital_utils.py`

先把现在 `optimize_no.py` 里被复用的纯工具抽出来。

这一轮的目标不是重写逻辑，而是切断反向依赖。

优先迁出的函数包括：

- `_hermitian_part`
- `_require_square_matrix`
- `_resolve_distance_matrix`
- `_resolve_interaction_matrix`
- `build_occupation_blocks`
- `extract_one_body_hopping_matrix`
- `extract_interaction_matrix`
- `extract_hamiltonian_terms`
- `rotate_one_body_matrix`
- `rotate_hamiltonian_to_basis`
- `rotate_hamiltonian_to_natural_orbitals`

### Step 2：让 `core / losses` 改为依赖 `orbital_utils.py`

把现在：

- `optimize_no_core.py -> optimize_no.py`
- `optimize_no_losses.py -> optimize_no.py`

改成：

- `orbital_optimizer.py -> orbital_utils.py`
- `orbital_losses.py -> orbital_utils.py`

这一步完成后，旧大文件就不再是主干依赖。

### Step 3：重命名并稳定主干

当依赖已经切干净后，再把主干文件正式收敛到：

- `orbital_optimizer.py`
- `orbital_losses.py`

此时保留旧文件名只作为兼容壳，避免一次性大面积改导入。

### Step 4：新增 `orbital_pruning.py`

这时开始写真正缺失的第二步流程：

- 阈值 schedule
- support 构造
- 持久度统计
- accept / reject

### Step 5：新增 `orbital_sparse.py`

在固定支撑下做最终 refinement。

第一版策略明确采用：

- gradient step
- mask projection
- polar retraction

先把流程跑通，不追求最花哨的参数化。

### Step 6：新增 `orbital_runner.py`

最后把 dense -> pruning -> sparse 串成一个总入口。

### Step 7：把 `optimize_no.py` 压成兼容层

等新主干稳定后，把旧版文件改成：

- 仅保留兼容入口；
- 或直接转调新模块；
- 不再添加新功能。

---

## 6. 命名迁移时要特别注意什么

### 6.1 不要把“改名”和“重构”一次做完

最稳的方式是：

- 先抽工具；
- 再改依赖；
- 再引入新文件名；
- 再写 pruning / sparse / runner。

如果一边改名、一边抽函数、一边改接口，很容易把问题搅在一起。

### 6.2 不要再把新功能写回 `optimize_no.py`

这是最重要的边界。

从现在起：

- `pruning` 不回写旧文件；
- `sparse` 不回写旧文件；
- `runner` 不回写旧文件。

否则目录会重新变回大泥球。

### 6.3 `NO` 这个物理概念可以继续保留

文件名不再写 `no`，不代表自然轨道这个概念被删掉。

它依然可以存在于：

- 文档说明
- 变量命名
- 注释
- 函数参数
- 数学公式

改变的是工程命名，不是物理内容。

---

## 7. 最终想达到的状态

最终希望把这个目录收敛成下面这种关系：

```text
orbital_utils
    -> orbital_optimizer
    -> orbital_losses

orbital_optimizer + orbital_losses
    -> orbital_pruning
    -> orbital_sparse

orbital_optimizer + orbital_losses + orbital_pruning + orbital_sparse
    -> orbital_runner

orbital_legacy
    -> 兼容调用新主干，或退出主干
```

这时整个工程会有三个优点：

1. 依赖方向清楚。
   不再由旧大文件反向控制主干。

2. 新功能好放。
   稀疏化、裁剪、benchmark 都有明确归属。

3. 以后重命名或替换实现更安全。
   因为每层职责已经固定，不会牵一发动全身。

---

## 8. 一句话总结

新的命名方案不是为了好看，而是为了把这个目录从：

> 围绕 `optimize_no.py` 的历史脚本集合

收敛成：

> 以 `orbital_utils + orbital_optimizer + orbital_losses + orbital_pruning + orbital_sparse + orbital_runner` 为主干的清晰工程结构。