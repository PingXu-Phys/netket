# `optimize_no.py` 修改建议（综合版）

优先级：**P1 必改 → P2 建议 → P3 低优先**

---

## P1 必改

### 1. 主优化循环缺少 `jax.jit`

**位置**：第 1229 行

```python
# 当前（未 jit 编译 loss-and-grad 主路径，优化阶段明显偏慢）
loss_and_grad = jax.value_and_grad(loss_fn)

# 修复
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
```

同样，`build_no_loss_fn` 返回的 `loss_fn` 也应在文档中提示用户套 `jax.jit`。

---

### 2. 缺少 `rdm` 与 `natural_occupations` / `natural_orbitals` 的一致性校验

**位置**：`optimize_no_basis` 和 `prepare_no_optimization_context` 的参数验证处

当前代码只检查 shape，不检查语义一致性。若用户手工传入一组 shape 正确但与 `rdm` 不一致
的 `natural_occupations` / `natural_orbitals`，优化会静默进行，结果难以解释。

建议增加，并优先使用相对容差：
```python
gamma_no = natural_orbitals.conj().T @ rdm @ natural_orbitals
offdiag = gamma_no - np.diag(np.diag(gamma_no))
offdiag_ratio = np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(rdm, ord="fro"), EPS)
if offdiag_ratio > rel_tol:
    raise ValueError("natural_orbitals do not diagonalize rdm within tolerance.")

diag_occ = np.diag(gamma_no).real
allowed_diff = atol + rel_tol * max(np.max(np.abs(diag_occ)), np.max(np.abs(natural_occupations)), 1.0)
if np.max(np.abs(diag_occ - natural_occupations)) > allowed_diff:
    raise ValueError("natural_occupations inconsistent with diag(U^dagger rdm U).")
```

---

## P2 建议

### 3. `build_occupation_blocks` 的接口契约不够显式

**位置**：第 725 行

函数内部依赖相邻元素单调性做块划分，但公开接口接受任意 `occupations` 数组，也没有文档
注明前置条件。标准路径（`natural_orbitals_from_rdm` 返回降序本征值）通常不会出错，但用户
手工传入乱序数据时会静默产生错误的块划分。

建议：要么加入排序检查，要么在函数 docstring 中明确写明"要求降序输入"。

---

### 4. 给 context 增加统一的字段校验

**位置**：`build_no_loss_fn` / `evaluate_no_loss` 的入口

当前函数直接信任外部 context，字段错误会延迟到 JAX 计算阶段才爆出，定位差。建议增加一个
内部校验函数，至少检查：

- `natural_orbitals.shape == (n, n)`
- `natural_occupations.shape == (n,)`
- `distance_matrix.shape == (n, n)`
- `occupation_blocks` 不重不漏地覆盖 `0..n-1`

---

### 5. `kondo_footprint` 注释符号统一

**位置**：第 249 行注释

注释写的是 `sum_i |J_i U_ia U_ib|`，但实现用的是 `U_ia^* U_ib`。取绝对值后结果等价，
但建议统一为 `sum_i |J_i U_ia^* U_ib|`，与 `kondo_tensor_from_site_rotation` 保持一致。

---

### 6. 补充三条关键文档说明

- `build_occupation_blocks` 要求输入 `occupations` 按**降序**排列；
- `C_gamma` 是保持 NO/block 结构的**正则项**，而非主优化目标；`lambda_occupancy` 过大
  会使优化锁死在 NO 基底附近；
- 用户自行使用 `build_no_loss_fn` 时，推荐用 `jax.jit(jax.value_and_grad(loss_fn))`。

---

## 已澄清的点

### A. 不采纳 `C_gamma` 的 JAX / NumPy 归一化不一致

对 Hermitian `1-RDM`，`||rdm||_F^2` 与 `sum_i n_i^2` 本来就严格相等，因此这不是单独的修复项。
真正需要防的是：外部手工传入了与 `rdm` 不一致的 `natural_occupations` / `natural_orbitals`。

### B. `optimize_no_from_context` 保持 P3 即可

它确实会重复做一部分轻量准备工作，但不会重新对 `rdm` 做 NO 对角化，也不会在已给定
`hopping_matrix` 时重新从 `H` 抽取 hopping。因此这条保留为结构整理项即可，不必前置。

---

## P3 低优先

### 7. `build_occupation_blocks` 定义顺序

函数定义在第 725 行，但第 605、689 行已调用。运行不报错，但建议把定义移到前面。

### 8. `_structure_cost_jax` / `pair_structure_metrics` 中 `abs` 冗余

调用方已传入 `|...|²`（非负实数），内部再取一次绝对值无害但冗余，可删除。

### 9. `optimize_no_from_context` 的轻微重复

解包 context 后传回 `optimize_no_basis`，会重复执行 `build_occupation_blocks`、
`hopping_no` 旋转、`initial_metrics` 计算。不是性能瓶颈，但结构上可整理。

### 10. `_block_unitary` 中 `expm` 对大块的效率

对 active space 较大（> 20 个轨道）的块，`jax.scipy.linalg.expm` 较慢。小规模无需处理；
大规模可考虑 Cayley 变换替代。

---

## 优先级汇总

| # | 问题 | 优先级 |
|---|------|--------|
| 1 | 主循环缺少 `jax.jit` | **P1** |
| 2 | 缺少 `rdm` / NO 数据一致性校验 | **P1** |
| 3 | `build_occupation_blocks` 接口契约不显式 | P2 |
| 4 | context 字段统一校验 | P2 |
| 5 | `kondo_footprint` 注释符号 | P2 |
| 6 | 三条关键文档补充 | P2 |
| 7 | 函数定义顺序 | P3 |
| 8 | `abs` 冗余调用 | P3 |
| 9 | `optimize_no_from_context` 轻微重复 | P3 |
| 10 | `expm` 大块效率 | P3 |
