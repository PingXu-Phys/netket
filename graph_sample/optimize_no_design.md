# `optimize_no` 架构与接口设计

> 文档定位：
> 这份文件不再只是原版 `optimize_no.py` 的接口说明，而是整个 `optimize_no` 系列的统一设计文档。
> 它同时覆盖：
>
> - 当前三套实现的共同骨架
> - 计划中的“公共优化骨架 + 可插拔 loss 模块”重构方向
> - 对外接口应如何保持稳定

---

## 1. 设计目标

当前目录里已经存在三个代码版本：

- `optimize_no.py`
- `optimize_no_soft.py`
- `optimize_no_barrier.py`

它们的**硬分块、块内旋转参数化、优化循环、Hamiltonian 旋转、结果打包**高度相似；
主要差异集中在：

- loss 的定义
- loss 依赖的参考量如何准备
- loss breakdown / diagnostics 如何组织

因此，下一步设计目标不是继续平行复制新文件，而是把结构明确拆成两层：

1. **输入输出与优化骨架层**
2. **loss 构建层**

也就是：

> 同一套优化框架，挂接不同的 loss 模块。

---

## 2. 当前三套实现的关系

### 2.1 共同部分

三者都保留了相同的基础流程：

1. 从 `rdm` 得到自然轨道和自然轨道占据数
2. 根据 `filled_tol / empty_tol / degeneracy_tol` 做硬分块
3. 仅允许块内旋转，不同块之间不混合
4. 用 JAX/Optax 对块参数做优化
5. 输出：
   - `site_to_optimized_orbital`
   - `optimized_rdm`
   - `optimized_diag_occupations`
   - `optimized_hamiltonian`（若给定 `H`）

这部分就是未来应抽出来的**公共骨架**。

---

### 2.2 差异部分

三者真正不同的主要是：

#### 原版 `optimize_no.py`

- 多目标并列加权
- `C_t` / `C_int` 主要依赖结构代价
- interaction 相空间用旋转后的 `diag(gamma_rot)`
- locality 用 IPR / participation + spread 混合

#### `optimize_no_soft.py`

- 最小版
- 主目标改成 weighted smooth-L1 norm
- interaction 相空间固定用初始 NO occupation
- locality 只用 spread

#### `optimize_no_barrier.py`

- barrier 版
- 主目标为 norm + structure 混合
- occupation 保护改成 barrier
- 可选 active/frozen barrier
- locality 只用 spread

也就是说：

> 三者不是三套不同的优化器，而是同一优化器骨架上的三种 objective 设计。

---

## 3. 目标架构

建议重构为下面的文件结构：

### 3.1 公共骨架层

文件建议：

- `optimize_no_core.py`

职责：

- 输入预处理
- NO 基底准备
- 硬分块
- 块内参数化
- JAX/Optax 优化循环
- Hamiltonian 旋转
- 输出结果打包

它不应该知道“当前是哪一种 loss”，只知道：

- 如何生成 post-NO rotation
- 如何调用给定的 loss module
- 如何记录历史与结果

---

### 3.2 loss 层

文件建议：

- `optimize_no_losses.py`

职责：

- 构造不同 loss 的参考量
- 提供 JAX loss 计算
- 提供可读 diagnostics / loss breakdown

建议在同一文件里实现三类 loss module：

- `original`
- `soft`
- `barrier`

如果后续又加新版本，只需要新加 loss module，不必再复制整个优化器骨架。

---

### 3.3 对外兼容层

文件保留：

- `optimize_no.py`
- `optimize_no_soft.py`
- `optimize_no_barrier.py`

职责：

- 保持各自当前的公开函数名不变
- 在内部把调用转发到 `optimize_no_core.py + optimize_no_losses.py`

这样做的目的，是让外部使用者不必改调用代码。

---

### 3.4 当前已经落地的三层分工

经过当前这轮重构，设计上已经不只是“未来目标”，而是有一部分已经落地：

- [optimize_no_core.py](D:/Seafile/PHD/NQS/NetKet/netket/graph_sample/optimize_no_core.py)
- [optimize_no_losses.py](D:/Seafile/PHD/NQS/NetKet/netket/graph_sample/optimize_no_losses.py)
- [optimize_no_soft.py](D:/Seafile/PHD/NQS/NetKet/netket/graph_sample/optimize_no_soft.py)
- [optimize_no_barrier.py](D:/Seafile/PHD/NQS/NetKet/netket/graph_sample/optimize_no_barrier.py)

它们现在分别承担下面三层职责：

#### `optimize_no_core.py`

定位：

- 优化骨架层

负责：

- `prepare_base_context(...)`
- 硬分块
- 块内参数化
- 统一优化循环
- Hamiltonian 旋转
- 公共结果打包

一句话理解：

> `core` 决定“怎么优化”。

#### `optimize_no_losses.py`

定位：

- loss 选择层

负责：

- 不同 loss 的配置校验
- 不同 loss 的参考量准备
- 不同 loss 的 JAX loss 构造
- 不同 loss 的 diagnostics / history 格式

当前已经有的 builder：

- `build_optimize_loss_module(...)`
- `build_soft_loss_module(...)`
- `build_barrier_loss_module(...)`

一句话理解：

> `losses` 决定“优化什么”。

#### `optimize_no_soft.py` / `optimize_no_barrier.py`

定位：

- 对外包装层

它们现在的主要作用已经不是各自维护一整套独立优化器，而是：

- 保留用户熟悉的公开入口名
- 保留各自配置类型和参数语义
- 自动把调用转发到 `core + losses`

也就是说，这两个文件现在主要负责：

1. 接收用户输入
2. 调 `prepare_base_context(...)`
3. 挂接对应的 `build_*_loss_module(...)`
4. 调 `optimize_with_loss(...)`

一句话理解：

> `soft.py / barrier.py` 是面向用户的专用入口与适配层，不再是两套完全独立的底层实现。

#### `optimize_no.py`

当前状态：

- 仍然保留原版独立实现
- 还没有完全迁入 `core + losses`
- 但原版对应的 loss 已经可以在 `optimize_no_losses.py` 中单独构造

所以现在的真实状态是：

- `soft` 和 `barrier` 已经进入“三层结构”
- 原版 `optimize_no.py` 还处于“旧入口 + 新 loss 已可独立构造”的过渡状态

---

## 4. 两层接口划分

这次重构应明确分成两类接口：

### 4.1 骨架层接口

骨架层只负责“输入输出与优化”，不负责定义 loss。

建议公共接口：

#### `prepare_base_context(...)`

用途：

- 统一准备优化需要的基础对象
- 只准备与 loss 无关、或所有 loss 都共用的对象

应包含：

- `rdm`
- `natural_occupations`
- `natural_orbitals`
- `occupation_blocks`
- `hopping_matrix`
- `hopping_no`
- `interaction_matrix`
- `distance_matrix`
- `site_positions`
- `H`
- `spin_symmetric`

#### `create_optimizer(core_config)`

用途：

- 根据统一优化配置生成 Optax 优化器

#### `optimize_with_loss(base_context, loss_module, core_config)`

用途：

- 运行统一优化循环
- loss 的细节完全由 `loss_module` 提供

返回：

- 公共结果字段
- loss-specific 的最终分解项
- history

#### `evaluate_with_loss(base_context, loss_module, ...)`

用途：

- 不做优化，只评估某个 post-NO rotation 的 loss 与 diagnostics

---

### 4.2 loss 层接口

loss 层只负责“loss 怎么定义”。

每个 loss module 建议统一实现下面三个函数：

#### `prepare_loss_context(base_context, loss_config)`

用途：

- 准备该 loss 所需的参考量

例如：

- `pair_weights`
- `reference_phase_space`
- `reference_interaction`

#### `loss_terms_jax(post_no_rotation, base_context, loss_context, loss_config)`

用途：

- 返回可 JIT 的总 loss
- 同时返回 JAX 版分项

返回建议：

- `total_loss`
- `terms`

#### `evaluate_loss_metrics(post_no_rotation, base_context, loss_context, loss_config)`

用途：

- 返回 Numpy/可读版 diagnostics
- 用于日志、最终结果和人工分析

---

## 5. 配置拆分设计

当前一个 config 往往同时包含：

- 优化器参数
- 硬分块参数
- loss 参数

这不利于后续扩展。

建议拆成：

### `NOCoreConfig`

负责：

- `filled_tol`
- `empty_tol`
- `degeneracy_tol`
- `optimizer_name`
- `learning_rate`
- `weight_decay`
- `gradient_clip`
- `n_steps`
- `log_every`
- `hamiltonian_cutoff`
- `real_orbitals`

### `NOOriginalLossConfig`

负责原版 loss 的参数：

- `lambda_occupancy`
- `lambda_hopping`
- `lambda_interaction`
- `lambda_locality`
- `structure_metric`
- `distance_power`

### `NOSoftLossConfig`

负责 soft 版参数：

- `alpha_hopping`
- `alpha_interaction`
- `lambda_occupancy`
- `lambda_locality`
- `distance_power`
- `distance_weight_strength`
- `smooth_eps`

### `NOBarrierLossConfig`

负责 barrier 版参数：

- `alpha_hopping`
- `alpha_interaction`
- `eta_hopping`
- `eta_interaction`
- `mu_gamma`
- `delta_gamma`
- `mu_active_frozen`
- `delta_active_frozen`
- `barrier_tau`
- `lambda_locality`
- `structure_metric`
- `distance_power`
- `distance_weight_strength`
- `smooth_eps`

---

## 6. 函数归属建议

这一节说明哪些函数应该保留在骨架层，哪些应该并入 loss 层。

### 6.1 应保留在骨架层的函数

这些函数与具体 loss 无关：

- `build_occupation_blocks`
- `_init_block_params`
- `_build_post_no_rotation`
- `rotate_one_body_matrix`
- `rotate_hamiltonian_to_basis`
- `rotate_hamiltonian_to_natural_orbitals`
- `extract_hamiltonian_terms`
- `extract_one_body_hopping_matrix`
- `extract_interaction_matrix`

它们应该归入 `optimize_no_core.py` 或相邻的基础工具模块。

---

### 6.2 应放到 loss 层的函数

这些函数与 loss 的定义直接相关：

- `occupation_phase_space`
- `pair_structure_metrics`
- `weighted_smooth_ratio`
- `interaction_footprint`
- `occupancy_cost`
- `spread_locality_cost`
- `active_frozen_cost`
- `smooth_barrier`

它们应统一放到 `optimize_no_losses.py`，供不同 loss module 复用。

---

### 6.3 应变成薄包装的函数

这些高层接口应继续保留名字，但内部不再直接实现全部逻辑：

#### 原版

- `prepare_no_optimization_context`
- `build_no_loss_fn`
- `evaluate_no_loss`
- `optimize_no_from_context`
- `optimize_no_basis`

#### soft

- `prepare_soft_optimization_context`
- `build_soft_loss_fn`
- `evaluate_soft_loss`
- `optimize_soft_from_context`
- `optimize_soft_basis`

#### barrier

- `prepare_barrier_optimization_context`
- `build_barrier_loss_fn`
- `evaluate_barrier_loss`
- `optimize_barrier_from_context`
- `optimize_barrier_basis`

这些函数对外接口可以保持不变，但内部应改成：

- 调 `prepare_base_context`
- 调 `prepare_loss_context`
- 调 `optimize_with_loss`

---

## 7. 对外接口兼容原则

重构时应遵守下面三条：

### 7.1 公开函数名先不变

不立刻要求用户把：

- `optimize_no_basis(...)`
- `optimize_soft_basis(...)`
- `optimize_barrier_basis(...)`

改成新的统一入口。

### 7.2 返回字段先不变

现有调用方很可能依赖这些键：

- `site_to_optimized_orbital`
- `rotation_in_no_basis`
- `natural_occupations`
- `natural_orbitals`
- `occupation_blocks`
- `final_loss_terms`
- `optimized_rdm`
- `optimized_diag_occupations`
- `optimized_hamiltonian`

这些字段在重构后应保持一致。

### 7.3 先重构内部，不先改外观

也就是说：

- 第一步重构内部组织结构
- 第二步再考虑是否提供统一新 API

---

## 8. 统一结果对象建议

虽然对外仍返回 dict，但内部应明确有一套统一结果结构。

建议公共字段：

- `config`
- `optimizer_name`
- `natural_occupations`
- `natural_orbitals`
- `occupation_blocks`
- `initial_metrics`
- `final_metrics`
- `final_loss_terms`
- `history`
- `rotation_in_no_basis`
- `site_to_optimized_orbital`
- `mode_rotation`
- `optimized_hamiltonian`
- `optimized_rdm`
- `optimized_diag_occupations`
- `optimized_hopping_matrix`
- `interaction_footprint`
- `effective_scattering`

loss-specific 项只放进：

- `final_loss_terms`
- `metrics`
- `history`

这样对外结构最稳定。

---

## 9. 建议的实现步骤

### 阶段 1：抽公共骨架

先建立 `optimize_no_core.py`，迁出：

- block 相关逻辑
- 优化循环
- Hamiltonian 旋转
- 结果打包

这一阶段不改数学定义，只改文件组织。

当前状态：

- 已完成

---

### 阶段 2：抽 loss 公共工具

建立 `optimize_no_losses.py`，迁出：

- `occupation_phase_space`
- `pair_structure_metrics`
- `weighted_smooth_ratio`
- `interaction_footprint`
- `spread_locality_cost`
- `active_frozen_cost`
- `smooth_barrier`

当前状态：

- 已完成
- 另外已经补入 `build_optimize_loss_module(...)`

---

### 阶段 3：先迁 soft 和 barrier

原因：

- 它们重复最多
- 结构也更清晰
- 先合并它们，风险最低

完成后，`optimize_no_soft.py` 和 `optimize_no_barrier.py` 应变成薄包装。

当前状态：

- 已完成
- 这两个文件现在已经主要承担“对外包装层”的职责

---

### 阶段 4：最后迁原版

原版 `optimize_no.py` 的 diagnostics 更老、更杂，最后再迁最稳妥。

这时再把：

- `prepare_no_optimization_context`
- `build_no_loss_fn`
- `optimize_no_basis`

也统一改成薄包装。

当前状态：

- 尚未完成
- 但原版 loss 已经能通过 `build_optimize_loss_module(...)` 接入公共层

---

### 阶段 5：补验证

至少需要做：

1. 重构前后原版 loss 数值一致性检查
2. 重构前后 soft loss 数值一致性检查
3. 重构前后 barrier loss 数值一致性检查
4. 返回字段兼容性检查

---

## 10. 现在这份 design 对应的后续动作

按照本设计，后续代码工作应按下面顺序进行：

1. `optimize_no_core.py` 已创建
2. `optimize_no_losses.py` 已创建
3. `soft` 与 `barrier` 已重构为包装层
4. 下一步若继续收口，再重构原版 `optimize_no.py`
5. 然后做数值一致性验证与文档补全

也就是说：

> 先统一输入输出优化骨架，再统一 loss 构建。

这就是整个 `optimize_no` 系列接下来的主重构方向。

---

## 11. 三种调用方式

这一节给出当前建议的实际调用方式。

原则上：

- 普通使用者优先调用各自的高层入口
- 只有在你明确想手动拼接 `core + losses` 时，才直接调用公共层

---

### 11.1 原版 `optimize_no.py`

适用场景：

- 你想保留原版 objective
- `interaction` 相空间希望跟随 `diag(gamma_rot)` 动态变化
- 你依赖原版 locality 与 diagnostics

典型调用：

```python
from netket.graph_sample.optimize_no import (
    NOOptimizationConfig,
    optimize_no_basis,
)

config = NOOptimizationConfig(
    lambda_occupancy=1.0,
    lambda_hopping=1.0,
    lambda_interaction=1.0,
    lambda_locality=0.1,
    filled_tol=0.05,
    empty_tol=0.05,
    degeneracy_tol=0.02,
    learning_rate=1.0e-2,
    n_steps=200,
)

result = optimize_no_basis(
    H=H,
    rdm=rdm,
    hopping_matrix=hopping_matrix,
    interaction_matrix=interaction_matrix,
    distance_matrix=distance_matrix,
    site_positions=site_positions,
    config=config,
    spin_symmetric=True,
)
```

主要入口：

- `prepare_no_optimization_context(...)`
- `build_no_loss_fn(...)`
- `evaluate_no_loss(...)`
- `optimize_no_basis(...)`
- `optimize_no_from_context(...)`

---

### 11.2 `optimize_no_soft.py`

适用场景：

- 你想先跑一个最小版
- 主目标想直接压 `hopping / interaction` 的 norm
- occupation 只作为一个轻量 soft regulariser

典型调用：

```python
from netket.graph_sample.optimize_no_soft import (
    NOSoftOptimizationConfig,
    optimize_soft_basis,
)

config = NOSoftOptimizationConfig(
    alpha_hopping=0.5,
    alpha_interaction=0.5,
    lambda_occupancy=0.05,
    lambda_locality=0.0,
    filled_tol=0.05,
    empty_tol=0.05,
    degeneracy_tol=0.02,
    learning_rate=1.0e-2,
    n_steps=300,
)

result = optimize_soft_basis(
    H=H,
    rdm=rdm,
    hopping_matrix=hopping_matrix,
    interaction_matrix=interaction_matrix,
    distance_matrix=distance_matrix,
    site_positions=site_positions,
    config=config,
    spin_symmetric=True,
)
```

主要入口：

- `prepare_soft_optimization_context(...)`
- `build_soft_loss_fn(...)`
- `evaluate_soft_loss(...)`
- `optimize_soft_basis(...)`
- `optimize_soft_from_context(...)`

---

### 11.3 `optimize_no_barrier.py`

适用场景：

- 你希望把 occupation 保护写成“允许一定偏离，再加墙”
- 你想把主目标写成 `norm + structure` 的混合
- 你还想可选地限制 active/frozen mixing

典型调用：

```python
from netket.graph_sample.optimize_no_barrier import (
    NOBarrierOptimizationConfig,
    optimize_barrier_basis,
)

config = NOBarrierOptimizationConfig(
    alpha_hopping=0.5,
    alpha_interaction=0.5,
    eta_hopping=0.5,
    eta_interaction=0.5,
    mu_gamma=5.0,
    delta_gamma=0.02,
    mu_active_frozen=0.0,
    lambda_locality=0.0,
    filled_tol=0.05,
    empty_tol=0.05,
    degeneracy_tol=0.02,
    learning_rate=1.0e-2,
    n_steps=300,
)

result = optimize_barrier_basis(
    H=H,
    rdm=rdm,
    hopping_matrix=hopping_matrix,
    interaction_matrix=interaction_matrix,
    distance_matrix=distance_matrix,
    site_positions=site_positions,
    config=config,
    spin_symmetric=True,
)
```

主要入口：

- `prepare_barrier_optimization_context(...)`
- `build_barrier_loss_fn(...)`
- `evaluate_barrier_loss(...)`
- `optimize_barrier_basis(...)`
- `optimize_barrier_from_context(...)`

---

### 11.4 低层组合调用：`core + losses`

这个接口更适合：

- 做实验
- 手动替换 loss module
- 验证不同 loss 在同一骨架上的表现

当前已经可用的公共 loss builder 有：

- `build_optimize_loss_module(...)`
- `build_soft_loss_module(...)`
- `build_barrier_loss_module(...)`

典型写法：

```python
from netket.graph_sample.optimize_no_core import (
    prepare_base_context,
    optimize_with_loss,
)
from netket.graph_sample.optimize_no_losses import (
    NOSoftOptimizationConfig,
    build_soft_loss_module,
)

config = NOSoftOptimizationConfig()

base_context = prepare_base_context(
    H=H,
    rdm=rdm,
    hopping_matrix=hopping_matrix,
    interaction_matrix=interaction_matrix,
    distance_matrix=distance_matrix,
    site_positions=site_positions,
    core_config=config,
    spin_symmetric=True,
)

loss_module = build_soft_loss_module(base_context, config=config)

result = optimize_with_loss(
    base_context=base_context,
    loss_module=loss_module,
)
```

如果你想切换成别的 loss，只需要替换这里：

```python
loss_module = build_optimize_loss_module(base_context, config=config)
loss_module = build_soft_loss_module(base_context, config=config)
loss_module = build_barrier_loss_module(base_context, config=config)
```

也就是说：

- 骨架层决定“怎么优化”
- loss module 决定“优化什么”

---

### 11.5 现在应该怎么选

如果你只是正常使用，建议：

1. 用 `optimize_no.py` 当原版基线
2. 用 `optimize_no_soft.py` 做最小版探索
3. 用 `optimize_no_barrier.py` 做约束更明确的版本

如果你在做开发或重构，建议：

1. 先调 `prepare_base_context(...)`
2. 再挂一个 `build_*_loss_module(...)`
3. 最后统一走 `optimize_with_loss(...)`
