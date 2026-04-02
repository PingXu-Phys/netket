# `optimize_no.py` 接口说明

这份文件对应同目录下的 `optimize_no.py`。当前公开接口可以分成六层：

- 配置
- 读取 Hamiltonian / occupation
- 基底变换
- 诊断量
- 损失函数
- 优化

---

## 1. 配置

### `NOOptimizationConfig`

这是统一配置入口，主要控制：

- 损失函数权重：
  - `lambda_occupancy`
  - `lambda_hopping`
  - `lambda_kondo`
  - `lambda_locality`
- 占据块划分：
  - `filled_tol`
  - `empty_tol`
  - `degeneracy_tol`
- 优化器：
  - `optimizer_name`
  - `learning_rate`
  - `weight_decay`
  - `gradient_clip`
  - `n_steps`
  - `log_every`
- 结构代价：
  - `structure_metric`
  - `distance_power`
- Hamiltonian 旋转截断：
  - `hamiltonian_cutoff`

---

## 2. 读取 Hamiltonian 和 occupation

### `extract_hamiltonian_terms(H, cutoff=None, normal_order=True)`

用途：

- 读取 `FermionOperator2nd` 的显式项
- 看常数项、总项数、每个项的算符串、每阶项的数量

返回：

- `constant`
- `n_terms`
- `counts_by_order`
- `terms`

每个 `term` 的格式：

- `operators`: `((mode, dagger), ...)`
- `weight`
- `order`

### `extract_one_body_hopping_matrix(H, spin_symmetric=True, cutoff=None, ...)`

用途：

- 从 Hamiltonian 中抽出一体 hopping 矩阵
- 作为后续 `C_t` 的输入

### `inspect_occupation_structure(rdm=..., occupations=..., config=...)`

用途：

- 看自然轨道占据数
- 自动分近满 / 活性 / 近空块
- 生成相空间因子

返回：

- `occupations`
- `natural_orbitals`（如果是从 `rdm` 推出来的）
- `occupation_blocks`
- `phase_space`
- `active_weight`

### `build_occupation_blocks(occupations, ...)`

用途：

- 单独根据占据数做块划分

这在你想手工检查 block 结构时更方便。
前置条件：

- `occupations` 需要按降序排列；函数按相邻占据数连续性做分块


---

## 3. 基底变换

### `rotate_one_body_matrix(matrix, site_to_orbital)`

做一体矩阵旋转：

\[
M' = U^\dagger M U.
\]

### `rotate_hamiltonian_to_basis(H, site_to_orbital, spin_symmetric=True, cutoff=None)`

用途：

- 按给定一体基底旋转整个 `FermionOperator2nd` Hamiltonian

这是“给定任意轨道变换，如何旋转 Hamiltonian”的主接口。

### `rotate_hamiltonian_to_natural_orbitals(H, rdm, ...)`

用途：

1. 从 `rdm` 提取自然轨道
2. 把 Hamiltonian 直接转到 NO 基底

返回：

- `natural_occupations`
- `natural_orbitals`
- `rotated_hamiltonian`

这是“如何根据自然轨道变换 Hamiltonian”的直接接口。

---

## 4. 诊断量接口

这些函数本身不做优化，只做度量。

### `occupation_phase_space(diagonal_occupations)`

定义：

\[
\Xi_{ab}=n_a(1-n_b)+n_b(1-n_a).
\]

它只编码占据结构给出的低能相空间。

### `kondo_tensor_from_site_rotation(site_to_orbital, exchange_profile=...)`

定义：

\[
K_{i,ab}=J_i U_{ia}^\ast U_{ib}.
\]

### `kondo_footprint(site_to_orbital, exchange_profile=...)`

定义：

\[
\Lambda_{ab}=\sum_i |J_i U_{ia}^\ast U_{ib}|.
\]

它只看 Hamiltonian footprint，不看占据相空间。

### `pair_structure_metrics(weights, distance_matrix=None, power=2.0, metric=\"hybrid\")`

用途：

- 度量一个轨道对权重矩阵是否更集中、更带状、或者衰减更快

支持：

- `participation`
- `decay`
- `hybrid`

### `orbital_locality_metrics(site_to_orbital, site_positions=None)`

用途：

- 用 IPR / participation ratio / optional spread 度量轨道是否过度延展

### `evaluate_basis_metrics(rdm, site_to_orbital, ...)`

这是最完整的静态诊断接口。

它会一次性返回：

- `occupation_offdiag_cost`
- `diag_occupations`
- `phase_space`
- `rotated_hopping`
- `hopping_metrics`
- `kondo_footprint`
- `effective_scattering`
- `kondo_metrics`
- `locality_metrics`
- `loss`

也就是说，如果你只想“评估某个基底好不好”，直接用它。

---

## 5. 损失函数接口

### `prepare_no_optimization_context(...)`

这是最重要的准备接口。

用途：

- 统一准备优化所需的全部对象
- 把 `rdm`、NO 基底、块结构、hopping、exchange、distance、positions 等整理成一个 `context`

返回的 `context` 里最关键的是：

- `natural_occupations`
- `natural_orbitals`
- `occupation_blocks`
- `hopping_matrix`
- `hopping_no`
- `exchange_profile`
- `distance_matrix`
- `site_positions`
- `initial_metrics`

如果你手工传入 `natural_occupations` / `natural_orbitals`，它们必须与同一个 `rdm` 一致：

- `U^\dagger rdm U` 需要近似对角
- 其对角元需要与 `natural_occupations` 一致

### `build_no_loss_fn(context)`

用途：

- 返回一个真正的 JAX loss function
- 输入是 block-unitary 参数
- 输出是标量 loss

对应损失函数：

\[
\mathcal L(V)
=
\lambda_\gamma \mathcal C_\gamma
+ \lambda_t \mathcal C_t
+ \lambda_K \mathcal C_K
+ \lambda_{\rm loc} \mathcal C_{\rm loc}.
\]

其中：

- `C_gamma`: `1-RDM` 非对角代价，也是保持 NO/block 结构的正则项
- `C_t`: 旋后 hopping 的结构代价
- `C_K`: `W_eff` 的结构代价
- `C_loc`: 轨道局域性正则项

如果你自己在外层做迭代，建议把它包装成：

- `jax.jit(jax.value_and_grad(loss_fn))`

### `evaluate_no_loss(context, post_no_rotation=... / params=...)`

用途：

- 不做优化，只评估一个给定 post-NO rotation 下的 loss
- 或者直接评估某组 block 参数对应的 loss

返回：

- `loss`
- `loss_terms`
- `post_no_rotation`
- `site_to_optimized_orbital`
- `metrics`

---

## 6. 优化器接口

### `create_optax_optimizer(config)`

用途：

- 根据配置生成标准优化器

当前支持：

- `adam`
- `adamw`
- `sgd`

### `optimize_no_from_context(context, optimizer=None)`

用途：

- 你自己先准备好 `context`
- 再单独控制优化流程

### `optimize_no_basis(...)`

最高层的一步到位接口。

它会：

1. 准备 NO 基底与上下文
2. 构造 loss
3. 调用 `optax` 做优化
4. 返回最终旋转、最终 loss、最终指标、以及可选的旋后 Hamiltonian

### `optimize_no_basis_from_rdm`

这是 `optimize_no_basis` 的别名入口。

---

## 7. 最小调用模板

### 7.1 只检查 Hamiltonian / occupation

```python
import jax

from netket.graph_sample.optimize_no import (
    extract_hamiltonian_terms,
    inspect_occupation_structure,
)

term_info = extract_hamiltonian_terms(H)
occ_info = inspect_occupation_structure(rdm=rdm)
```

### 7.2 直接转到 NO 基底

```python
from netket.graph_sample.optimize_no import rotate_hamiltonian_to_natural_orbitals

no_data = rotate_hamiltonian_to_natural_orbitals(H, rdm)
H_no = no_data["rotated_hamiltonian"]
```

### 7.3 自己拿到 loss 再交给优化器

```python
import jax

from netket.graph_sample.optimize_no import (
    NOOptimizationConfig,
    prepare_no_optimization_context,
    build_no_loss_fn,
    create_optax_optimizer,
)

config = NOOptimizationConfig()
context = prepare_no_optimization_context(
    H=H,
    rdm=rdm,
    exchange_profile=J,
    config=config,
)
loss_fn = build_no_loss_fn(context)
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
optimizer = create_optax_optimizer(config)
```

### 7.4 直接优化

```python
from netket.graph_sample.optimize_no import optimize_no_basis

result = optimize_no_basis(
    H=H,
    rdm=rdm,
    exchange_profile=J,
)
```

---

## 8. 最核心的量

这个文件里最重要的联合诊断量仍然是：

\[
\Lambda_{ab} = \sum_i |J_i U_{ia}^\ast U_{ib}|,
\qquad
\Xi_{ab}=n_a(1-n_b)+n_b(1-n_a),
\]

\[
W_{ab}^{\rm eff} = \Lambda_{ab}\Xi_{ab}.
\]

它同时编码：

- Hamiltonian footprint
- occupation phase space

所以它是“占据数结构与哈密顿量稀疏性 / decay”结合得最直接的 hardness indicator。
