# KondoHeisenberg 约束问题说明

## 1. 当前问题

当前 `KondoHeisenberg.py` 与 `KondoHeisenbergChain.py` 的 Hilbert 空间写法，在 `J_K != 0` 时与所选约束不兼容。

当前代码做法是：

- 电子部分使用 `nk.hilbert.SpinOrbitalFermions(..., n_fermions_per_spin=(N_up, N_dn))`
- 局域自旋部分使用 `nk.hilbert.Spin(..., total_sz=local_total_sz)`
- 然后用张量积组合成总 Hilbert 空间

对应代码位置：

- `Hamiltonian/KondoHeisenberg.py` 中 `build_kondo_heisenberg_hilbert(...)`
- `Hamiltonian/KondoHeisenbergChain.py` 中 `build_kondo_heisenberg_chain_hilbert(...)`

## 2. 实际运行时表现

已经做过本地运行验证。

### 2.1 能成功的步骤

下面这些步骤本身可以通过：

- 构造 `H, hi = KondoHeisenberg(...)`
- 创建 `nk.sampler.MetropolisHamiltonian(hi, hamiltonian=H, ...)`
- 创建 `nk.vqs.MCState(...)`

### 2.2 失败的步骤

但一旦真正应用哈密顿量，就会报错：

- `H.to_sparse()` 失败
- `vs.expect(H)` 失败

报错核心信息是：

```text
States do not fulfill constraint.
```

### 2.3 对照实验

在相同参数下把 `J_K=0` 以后：

- `H.to_sparse()` 可以通过
- `vs.expect(H)` 可以通过

因此问题不是 `TensorHilbert`、`EmbedOperator` 或几何构造本身，而是 `J_K` 项和当前约束之间的冲突。

## 3. 根因分析

### 3.1 Kondo 项包含电子自旋翻转

当前代码中，Kondo 耦合写成各向同性形式：

\[
H_K = J_K \sum_i \mathbf{s}_i \cdot \mathbf{S}_i
\]

在代码里具体展开为：

- 电子自旋算符：
  - `s_z = 0.5 * (n_up - n_dn)`
  - `s_plus = c^dag_up c_dn`
  - `s_minus = c^dag_dn c_up`
- 局域自旋算符：
  - `S_z`
  - `S_plus`
  - `S_minus`
- 耦合项：
  - `s_z S_z`
  - `s_plus S_minus`
  - `s_minus S_plus`

对应代码位置：

- `Hamiltonian/KondoHeisenberg.py` 第 301-320 行附近
- `Hamiltonian/KondoHeisenbergChain.py` 第 187-201 行附近

### 3.2 为什么 `n_fermions_per_spin` 不再守恒

`n_fermions_per_spin=(N_up, N_dn)` 的意思是把电子 Hilbert 空间限制在固定 `N_up` 和固定 `N_dn` 的子空间里。

但：

- `s_plus = c^dag_up c_dn` 会把一个 `down` 电子变成 `up` 电子
- `s_minus = c^dag_dn c_up` 会把一个 `up` 电子变成 `down` 电子

所以在 `J_K != 0` 时：

- `N_up` 单独不守恒
- `N_dn` 单独不守恒

因此 `n_fermions_per_spin` 对 full Kondo Hamiltonian 不是合法约束。

### 3.3 为什么 `local_total_sz` 单独约束也不对

同理，局域自旋部分中的 `S_plus` 和 `S_minus` 也会改变局域自旋自己的总 `S^z`。

所以在 `J_K != 0` 时：

- 电子单独的 `S^z_e` 不守恒
- 局域自旋单独的 `S^z_{loc}` 不守恒
- 但两者之和通常守恒

守恒的是：

\[
S^z_{\text{total}} = S^z_e + S^z_{loc}
\]

而不是两部分各自的磁量子数。

## 4. 这是不是 NetKet 的问题

不是 NetKet 的 bug。

这属于“约束是否与 Hamiltonian 保持同一个不变子空间”这一物理和数学一致性问题。

NetKet 文档对 constraint 的要求很明确：

- 如果 Hamiltonian 不保持你施加的约束子空间，那么这个 constraint 就不能直接用于该 Hamiltonian。

当前错误是 NetKet 在运行时正确地把这个不一致检查出来了。

如果它不报错，反而意味着哈密顿量的部分连通态被错误地丢弃，模型会被悄悄改坏。

## 5. 当前代码在什么条件下是可运行的

### 5.1 可运行情况

当前写法在下面条件下可以工作：

- `J_K = 0`

因为这时没有电子自旋翻转，也没有电子与局域自旋之间的互相翻转，当前约束不会被破坏。

### 5.2 不可运行情况

当前写法在下面条件下会出问题：

- `J_K != 0`

这时 full Kondo 项中的自旋翻转过程会把系统带出当前约束子空间。

## 6. 如何修改

下面有两条不同方向的修法，取决于物理模型目标。

### 方案 A：保留 full Kondo 物理

适用于你想保留真正的各向同性 Kondo 耦合：

\[
J_K \mathbf{s}_i \cdot \mathbf{S}_i
\]

这时应该：

- 电子部分只固定总电子数 `n_fermions`
- 不再使用 `n_fermions_per_spin`
- 不再单独使用 `local_total_sz`
- 如果要固定磁量子数，应改为约束联合总磁量子数 `S^z_e + S^z_{loc}`

这意味着需要为 `TensorHilbert` 写一个自定义 constraint，约束条件类似：

- 电子总粒子数固定
- 电子与局域自旋的联合总 `S^z` 固定

这是物理上正确的做法。

### 方案 B：保留“每个 spin 的电子数固定”

如果你的核心目标是：

- 继续固定 `N_up`
- 继续固定 `N_dn`

那就不能保留 full Kondo 的自旋翻转项。

这时必须把 Kondo 耦合改成不改变电子自旋分量的形式，例如只保留纵向项：

\[
H_K^{\mathrm{Ising}} = J_K^z \sum_i s_i^z S_i^z
\]

或者更一般的各向异性版本，但要求横向翻转项关闭：

- 保留 `s_z S_z`
- 去掉 `s_plus S_minus`
- 去掉 `s_minus S_plus`

这样 `N_up` 和 `N_dn` 才会守恒，`n_fermions_per_spin` 才合法。

## 7. 推荐修改方向

如果目标是物理上标准的 Kondo-Heisenberg 模型，推荐采用方案 A：

- full Kondo 项保留
- 改用总电子数约束
- 如有需要，再加联合总 `S^z` 的自定义约束

如果目标是为了数值实现简单，或者你明确只研究无电子自旋翻转的近似模型，才建议方案 B。

## 8. 对当前文件的直接结论

当前以下两个文件都存在同样的问题：

- `Hamiltonian/KondoHeisenberg.py`
- `Hamiltonian/KondoHeisenbergChain.py`

具体结论是：

- `J_K = 0` 时当前写法可运行
- `J_K != 0` 时当前写法与 `n_fermions_per_spin` / `local_total_sz` 约束不兼容
- 若继续保留 full Kondo 项，则必须改约束设计
- 若继续保留 `n_fermions_per_spin`，则必须改 Hamiltonian 形式

## 9. 后续可执行修改

如果继续修改代码，推荐按下面顺序进行：

1. 先决定物理模型要保留 full Kondo，还是要保留固定 `N_up, N_dn`
2. 若保留 full Kondo，则重写 Hilbert 构造接口：
   - 支持 `n_fermions`
   - 取消 `n_fermions_per_spin`
   - 取消单独 `local_total_sz`
   - 可选加入联合总 `S^z` 自定义 constraint
3. 若保留固定 `N_up, N_dn`，则把 Kondo 项改为 Ising 型或无横向翻转型
4. 最后重新验证：
   - `H.to_sparse()`
   - `nk.sampler.MetropolisHamiltonian(...)`
   - `vs.expect(H)`

## 10. 一句话总结

当前报错的根本原因不是 NetKet 写法技巧问题，而是：

> full Kondo 自旋翻转项不守恒电子分自旋粒子数，因此不能和 `n_fermions_per_spin` 这种约束同时使用。
