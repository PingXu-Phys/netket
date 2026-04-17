# orbital_optimization README

这份 README 只解释当前目录里最核心的三个文件：

- `optimize_no.py`
- `optimize_no_core.py`
- `optimize_no_losses.py`

如果只记一句话，可以记成：

> `optimize_no.py` 是旧的总装版和工具来源，`optimize_no_core.py` 负责公共优化骨架，`optimize_no_losses.py` 负责不同 loss 的定义与评估。

---

## 1. 三个文件的总体分工

### 1.1 `optimize_no.py`

这是最早的 monolithic baseline。

它里面同时放了几类东西：

- 配置类 `NOOptimizationConfig`
- 输入检查、矩阵处理、距离矩阵/相互作用矩阵处理
- Hamiltonian 提取和旋转工具
- occupation block 构造
- JAX loss 定义
- Optax 优化循环
- 对外的一站式入口

因此它的特点是：

- 功能最全
- 历史包袱也最重
- 现在仍然是一些公共工具函数的来源

换句话说，这个文件目前还不能直接删，因为 `core` 和 `losses` 还在从这里借工具函数。

### 1.2 `optimize_no_core.py`

这是从 `optimize_no.py` 里抽出来的“公共优化骨架”。

它不关心你到底用的是 soft loss、barrier loss，还是以后新的 sparse loss。它只负责回答下面几个问题：

- 怎么准备优化所需的基础上下文
- 怎么把 occupation blocks 变成 block-local 参数
- 怎么把参数组装成 post-NO rotation
- 怎么建立 Optax optimizer
- 怎么跑统一的优化循环
- 怎么在优化结束后生成旋转后的 Hamiltonian 和统一结果字典

因此，这个文件是“算法外壳”，不是“物理目标本身”。

### 1.3 `optimize_no_losses.py`

这是 loss 层。

它负责：

- 定义不同 loss 对应的 config dataclass
- 构造 soft loss module
- 构造 barrier loss module
- 保留原始 optimize_no loss 的兼容 builder
- 提供 `evaluate_metrics(...)`，让外层代码能在给定一个 rotation 后统一拿到诊断信息

因此，这个文件回答的是：

- 优化目标是什么
- 每个指标怎么计算
- dense 阶段不同策略之间如何切换

它不负责优化循环本身。

---

## 2. 三者如何衔接

最简单的理解方式是三层：

1. `optimize_no.py`
   提供历史 baseline 和仍在复用的底层工具。
2. `optimize_no_core.py`
   提供“拿到 loss 之后怎么优化”的通用骨架。
3. `optimize_no_losses.py`
   提供“这个 loss 到底长什么样”的物理定义与评估。

当前推荐的调用方向是：

```text
prepare_base_context(...)        # core
    -> build_xxx_loss_module(...)  # losses
    -> optimize_with_loss(...)     # core
```

也就是：

- `core` 先准备好 NO 基、occupation blocks、hopping / interaction / distance 等基础上下文；
- `losses` 再根据 soft / barrier / original 构造对应的 loss module；
- `core` 最后接过这个 loss module，跑统一优化循环。

---

## 3. 分别看每个文件里最关键的内容

## 3.1 `optimize_no.py` 里最值得知道的内容

这个文件最重要的几类函数是：

### A. 数据检查与基础工具

例如：

- `_hermitian_part`
- `_require_square_matrix`
- `_resolve_distance_matrix`
- `_resolve_interaction_matrix`
- `_validate_no_reference_data`

这部分负责把输入数据整理干净，防止后面 loss 或 Hamiltonian 旋转时出错。

### B. 物理量与诊断计算

例如：

- `occupation_phase_space`
- `interaction_footprint`
- `pair_structure_metrics`
- `orbital_locality_metrics`
- `evaluate_basis_metrics`

这部分负责算复杂度、occupation 偏离、局域性等物理指标。

### C. Hamiltonian 处理

例如：

- `extract_one_body_hopping_matrix`
- `extract_interaction_matrix`
- `extract_hamiltonian_terms`
- `rotate_hamiltonian_to_basis`
- `rotate_hamiltonian_to_natural_orbitals`

这部分负责从 NetKet Hamiltonian 里提取出单体和相互作用信息，并把 Hamiltonian 旋转到新基底。

### D. 原始优化主线

例如：

- `build_occupation_blocks`
- `_loss_terms_jax`
- `build_no_loss_fn`
- `evaluate_no_loss`
- `optimize_no_from_context`
- `optimize_no_basis`

这部分就是旧版的一站式 dense orbital optimization。

所以一句话概括：

> `optimize_no.py` 是“旧的一体化实现 + 现在还在复用的工具仓库”。

---

## 3.2 `optimize_no_core.py` 里最关键的内容

### A. `NOCoreConfig`

这是公共优化设置，不绑定具体 loss。

主要包括：

- block 划分阈值：`filled_tol`、`empty_tol`、`degeneracy_tol`
- optimizer 选择：`optimizer_name`
- 学习率、裁剪、步数、日志频率
- 是否只允许实轨道：`real_orbitals`
- 旋转 Hamiltonian 时的 cutoff

这说明 `core` 只关心“怎么优化”，不关心“优化什么”。

### B. `prepare_base_context(...)`

这是整个新结构里最关键的入口之一。

它负责：

- 读取和检查 `rdm`
- 计算自然轨道 `natural_orbitals`
- 计算自然占据 `natural_occupations`
- 可选截取 `active_indices`
- 提取或接收 `hopping_matrix`
- 提取或接收 `interaction_matrix`
- 解析 `distance_matrix` 和 `site_positions`
- 根据 occupation 构造 `occupation_blocks`

最后返回一个统一的 `base_context` 字典。

这个字典就是后续 `losses` 和 `runner` 的基础输入。

### C. `_init_block_params` / `_block_unitary` / `_build_post_no_rotation`

这一组函数定义了当前 dense 优化的参数化方式：

- 每个 occupation block 一个生成元
- 生成元反对称或反 Hermitian
- 通过矩阵指数得到 block rotation
- 最后把所有 block rotation 组装成完整的 `post_no_rotation`

所以当前 dense 阶段的约束是自动满足幺正性的。

### D. `create_optimizer(...)`

把公共 Optax 逻辑统一封装起来。

这样不同 loss 可以共用同一套 optimizer 创建过程。

### E. `evaluate_with_loss(...)`

给一个现成的 `post_no_rotation` 或参数，调用 loss module 做一次评估。

这对后面 pruning / sparse 阶段很重要，因为它允许你复用同一套 metrics 评估逻辑，而不用把物理公式再写一遍。

### F. `optimize_with_loss(...)`

这是新的主骨架。

它负责：

- 初始化参数
- 调用 `loss_module["build_loss_fn"]()` 得到可微 loss
- 用 `jax.value_and_grad + optax` 跑优化循环
- 定期记录 `history`
- 输出最终 rotation、metrics、rotated Hamiltonian 等统一结果

所以一句话概括：

> `optimize_no_core.py` 是“拿着某个 loss module 去跑优化”的通用引擎。

---

## 3.3 `optimize_no_losses.py` 里最关键的内容

### A. 配置类

当前这个文件里主要有三类配置入口：

- `NOSoftOptimizationConfig`
- `NOBarrierOptimizationConfig`
- `coerce_optimize_config(...)` 对旧版 `NOOptimizationConfig` 的兼容入口

这说明 loss 层已经开始从旧版一体化配置，拆成更清楚的多策略配置。

### B. 通用辅助函数

例如：

- `_pair_weights_np/jax`
- `_weighted_ratio_np/jax`
- `_interaction_np/jax`
- `_occupancy_np/jax`
- `_spread_np/jax`
- `_structure_jax`
- `_barrier_np/jax`

这部分负责把不同 loss 都会用到的数学积木拆出来。

### C. `build_soft_loss_module(...)`

这是当前最适合做 dense 第一阶段默认入口的 loss builder。

它构造的目标大致是：

- hopping complexity
- interaction complexity
- occupancy soft penalty
- locality regularizer

并返回一个 `loss_module` 字典，里面至少包含：

- `config`
- `loss_context`
- `build_loss_fn`
- `evaluate_metrics`
- `history_entry`

### D. `build_barrier_loss_module(...)`

这是更强约束版本。

相比 soft 版，它额外强调：

- barrier 形式的 occupation 偏离控制
- active/frozen mixing 的 barrier
- structure metric 的混合目标

因此它更偏“约束强化型优化”，而不是最小可行默认版。

### E. `build_optimize_loss_module(...)`

这是旧版 `optimize_no.py` loss 的兼容封装。

它的作用不是代表未来最优结构，而是让新 `core` 还能接上旧版损失函数。

所以一句话概括：

> `optimize_no_losses.py` 是“把物理目标包装成可被 core 调用的 loss module”的地方。

---

## 4. 为什么现在还保留这三个文件

原因很实际：

### 4.1 不能只留 `core + losses`

因为 `core` 和 `losses` 现在仍然从 `optimize_no.py` 借工具函数，比如：

- 基础矩阵检查
- Hamiltonian 提取与旋转
- occupation block 构造
- 原始 loss 与诊断逻辑的一部分

所以 `optimize_no.py` 现在还不是纯历史文件。

### 4.2 也不能继续只往 `optimize_no.py` 里加东西

因为那样会把已经拆出来的结构重新塞回大文件，后面会越来越难维护。

因此现在最合理的状态是：

- `optimize_no.py` 暂时保留，作为 baseline 和工具来源
- `optimize_no_core.py` 继续承担优化主骨架
- `optimize_no_losses.py` 继续承担 loss 层
- 新的 `pruning / sparse / runner` 文件直接接在 `core + losses` 后面写

---

## 5. 后续新代码应该往哪里放

如果接下来要写新的 sparse flow，建议遵守下面的边界：

### 放进 `optimize_no_core.py` 的内容

只放公共优化机制，例如：

- 公共 context 准备
- 公共参数验证
- 公共 optimizer 建立
- 公共结果打包

### 放进 `optimize_no_losses.py` 的内容

只放目标函数和指标定义，例如：

- 新 loss config
- 新 loss builder
- 新 diagnostics evaluator

### 不要再放回 `optimize_no.py` 的内容

新的 pruning、mask、sparse refinement、runner 都不要再写回 `optimize_no.py`。

更合理的新增文件是：

- `optimize_no_pruning.py`
- `optimize_no_sparse.py`
- `optimize_no_flow_runner.py`

---

## 6. 最简 mental model

如果你之后隔一段时间回来，只需要记住下面这个图：

```text
optimize_no.py
    = 旧版全集成实现 + 仍在复用的工具仓库

optimize_no_core.py
    = 通用优化引擎
    = 负责 prepare context / build rotation / run optax / package result

optimize_no_losses.py
    = loss 工厂
    = 负责 soft / barrier / original 的目标和 metrics
```

再压缩一句：

> `optimize_no.py` 管历史和工具，`optimize_no_core.py` 管怎么优化，`optimize_no_losses.py` 管优化什么。