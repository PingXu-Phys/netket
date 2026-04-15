# `run_kondoheisenbergchain_transformers_site_benchmark.py` 的系统与哈密顿量说明

本文档说明 `my_run/run_kondoheisenbergchain_transformers_site_benchmark.py` 这个最小 benchmark 中，系统的 Hamiltonian 是怎样由参数决定的。

## 0. 速查版：公式 + 参数表

如果只想先看最核心的系统定义，可以直接看这一节。

### 0.1 总 Hamiltonian

\[
H = -t \sum_{(i,j)\in nn} \sum_{\sigma}
\left(c^{\dagger}_{i\sigma} c_{j\sigma} + c^{\dagger}_{j\sigma} c_{i\sigma}\right)
+ J_K \sum_i \left[
 s_i^z S_i^z + \frac{1}{2}s_i^+ S_i^- + \frac{1}{2}s_i^- S_i^+
\right]
+ J_1 \sum_{(i,j)\in nn} \left[
 \frac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+) + \frac{\delta_z}{4} S_i^z S_j^z
\right]
+ J_2 \sum_{(i,j)\in nnn} \left[
 \frac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+) + \frac{\delta_z}{4} S_i^z S_j^z
\right]
\]

这里：

- `nn` 是最近邻键
- `nnn` 是次近邻键
- `t`, `J_K`, `J1`, `J2`, `delta_z`, `Lx`, `pbc` 直接决定系统 Hamiltonian
- `n_fermions`, `joint_two_sz` 决定 sector / 初始化，不改 Hamiltonian 公式

### 0.2 当前 benchmark 的参数表

| 参数 | 当前值 | 进入哪里 | 作用 |
| --- | --- | --- | --- |
| `Lx` | `4` | 几何 / `nn` / `nnn` | 4 个 site 的链 |
| `pbc` | `False` | 几何 | 开边界 |
| `t` | `1.0` | `H_fermion` | 电子最近邻 hopping |
| `J_K` | `1.0` | `H_Kondo` | 同 site 的 Kondo 耦合 |
| `J1` | `1.0` | `H_local(nn)` | 局域自旋最近邻交换 |
| `J2` | `0.0` | `H_local(nnn)` | 当前关闭，不进入 Hamiltonian |
| `delta_z` | `1.0` | `H_local` 的 `S^z S^z` 项 | 当前是各向同性 Heisenberg 规范 |
| `n_fermions` | `4` | Hilbert sector | 固定总电子数，当前是 half filling |
| `joint_two_sz` | `0` | 初始 joint sector | 从联合总 `2S^z=0` 扇区开始采样 |

### 0.3 当前 benchmark 的一句话版本

当前跑的是一个 `Lx=4`、开边界、half-filling 的 1D full Kondo-Heisenberg chain：

- 电子部分有最近邻 hopping `t=1`
- 每个 site 上有 Kondo 耦合 `J_K=1`
- 局域自旋链只有最近邻交换 `J1=1`
- `J2=0`，所以没有次近邻自旋交换
- `delta_z=1`，因此局域自旋交换写成各向同性形式

### 0.4 哪些常见参数不改 Hamiltonian

下面这些参数不会改系统 Hamiltonian，只会改采样、ansatz 或优化流程：

- `joint_sampler`
- `models`
- `d_model`, `n_heads`, `n_layers`, `mlp_ratio`
- `driver`, `optimizer`, `learning_rate`, `diag_shift`, `momentum`
- `n_iter`, `n_samples` 以及其他 runtime 参数

参考文件：

- `Hamiltonian/KondoHeisenberg_constraint_notes.md`
- `Hamiltonian/HamiltonianFilesOverview.md`
- `Hamiltonian/KondoHeisenbergChain.py`
- `Hamiltonian/KondoHeisenbergChainSpinFermion.py`
- `my_run/run_kondoheisenbergchain_transformers.py`

## 1. 这个 benchmark 实际调用的是哪个系统

调用链是：

1. `run_kondoheisenbergchain_transformers_site_benchmark.py` 先显式写出参数：
   - `Lx, t, J_K, J1, J2, delta_z, pbc`
   - `n_fermions, joint_two_sz`
2. 然后它调用 `run_kondoheisenbergchain_transformers.py`
3. 这个 runner 在 `build_system(...)` 里调用 `KondoHeisenbergChainSpinFermion(...)`
4. `KondoHeisenbergChainSpinFermion(...)` 底层再调用 `KondoHeisenbergChain(...)`

因此，这个 benchmark 真正使用的系统 Hamiltonian，就是 `Hamiltonian/KondoHeisenbergChain.py` 里构造的 1D Kondo-Heisenberg chain Hamiltonian。

## 2. 这个系统有哪些自由度

对每个空间格点 `x = 0, 1, ..., Lx-1`，系统里有两类自由度：

- 一个自旋费米子轨道，也就是 conduction electron
- 一个局域自旋 `S=1/2`

所以图上看起来像“两条链”，但物理上不是两条独立的空间腿，而是同一组 1D site 上叠着两个 sector：

- 电子 sector
- 局域自旋 sector

总 Hilbert 空间是它们的张量积：

```text
joint_hilbert = fermion_hilbert x local_spin_hilbert
```

## 3. 几何是怎样由 `Lx` 和 `pbc` 决定的

底层几何由 `kondo_heisenberg_chain_geometry(Lx=Lx, pbc=pbc)` 给出。

它会生成两组键：

- `nn`：最近邻键，用在电子 hopping `t` 和局域自旋最近邻耦合 `J1`
- `nnn`：次近邻键，用在局域自旋次近邻耦合 `J2`

当前 benchmark 取值是：

```python
Lx = 4
pbc = False
```

因此几何是开边界 4-site chain，对应键为：

```text
nn  = [(0, 1), (1, 2), (2, 3)]
nnn = [(0, 2), (1, 3)]
```

这意味着：

- 电子 hopping 只发生在 `0-1, 1-2, 2-3`
- `J1` 只作用在 `0-1, 1-2, 2-3`
- `J2` 只作用在 `0-2, 1-3`

如果改成 `pbc=True`，这些键会按周期边界绕回去，Hamiltonian 的连接关系也会一起改变。

## 4. Hamiltonian 的三部分

底层 `KondoHeisenbergChain(...)` 把总 Hamiltonian 写成三部分：

```text
H = H_fermion + H_local + H_Kondo
```

### 4.1 电子 hopping 项 `H_fermion`

代码里它沿 `nn` 键，对两个自旋分量都加入最近邻 hopping：

\[
H_{\mathrm{fermion}} =
-t \sum_{(i,j)\in nn} \sum_{\sigma=\uparrow,\downarrow}
\left(c^{\dagger}_{i\sigma} c_{j\sigma} + c^{\dagger}_{j\sigma} c_{i\sigma}\right)
\]

因此：

- `t` 决定 hopping 强度
- `Lx` 和 `pbc` 决定有哪些 `nn` 键

当前 benchmark 里：

```python
t = 1.0
```

所以这部分就是标准的最近邻电子跃迁项，系数为 `-1.0`。

### 4.2 局域自旋链项 `H_local`

代码把 `J1` 和 `J2` 都写成局域自旋链上的 XXZ 型交换：

\[
H_{\mathrm{local}} =
\sum_{(i,j)\in nn} J_1 \left[
\frac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+) + \frac{\delta_z}{4} S_i^z S_j^z
\right]
+ \sum_{(i,j)\in nnn} J_2 \left[
\frac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+) + \frac{\delta_z}{4} S_i^z S_j^z
\right]
\]

因此：

- `J1` 决定最近邻自旋交换强度
- `J2` 决定次近邻自旋交换强度
- `delta_z` 只乘在 `S_i^z S_j^z` 这一部分上，用来控制各向异性
- `Lx` 和 `pbc` 决定 `nn` / `nnn` 键集

当前 benchmark 里：

```python
J1 = 1.0
J2 = 0.0
delta_z = 1.0
```

所以当前局域自旋部分的含义是：

- 最近邻自旋链耦合打开
- 次近邻自旋链耦合关闭
- `delta_z = 1`，因此最近邻自旋项是各向同性 Heisenberg 规范下的写法

也就是说，当前 benchmark 中 `J2` 那一整块实际上不进入 Hamiltonian。

### 4.3 同站点 Kondo 耦合项 `H_Kondo`

代码在每个 site 上定义电子自旋算符：

\[
s_i^z = \frac{1}{2}(n_{i\uparrow} - n_{i\downarrow}), \qquad
s_i^+ = c^{\dagger}_{i\uparrow} c_{i\downarrow}, \qquad
s_i^- = c^{\dagger}_{i\downarrow} c_{i\uparrow}
\]

以及局域自旋算符：

\[
S_i^z = \frac{1}{2}\sigma_i^z, \qquad S_i^+, \qquad S_i^-
\]

然后 Kondo 项写成：

\[
H_{\mathrm{Kondo}} = J_K \sum_i
\left[
s_i^z S_i^z + \frac{1}{2}s_i^+ S_i^- + \frac{1}{2}s_i^- S_i^+
\right]
\]

因此：

- `J_K` 决定 onsite Kondo exchange 强度
- 它把电子 sector 和局域自旋 sector 在同一个 site 上耦合起来
- 这是一份 full Kondo 项，因为它既包含纵向 `s^z S^z`，也包含横向 spin-flip 项

当前 benchmark 里：

```python
J_K = 1.0
```

所以每个 site 上都存在单位强度的各向同性 Kondo 耦合。

## 5. 当前 benchmark 的总 Hamiltonian 可以怎样读

把当前参数代进去：

```python
Lx = 4
t = 1.0
J_K = 1.0
J1 = 1.0
J2 = 0.0
delta_z = 1.0
pbc = False
```

那么当前 benchmark 的 Hamiltonian 可以读成：

\[
H =
- \sum_{(i,j)\in \{(0,1),(1,2),(2,3)\}} \sum_{\sigma}
\left(c^{\dagger}_{i\sigma} c_{j\sigma} + c^{\dagger}_{j\sigma} c_{i\sigma}\right)
+ \sum_{i=0}^{3}
\left[
s_i^z S_i^z + \frac{1}{2}s_i^+ S_i^- + \frac{1}{2}s_i^- S_i^+
\right]
+ \sum_{(i,j)\in \{(0,1),(1,2),(2,3)\}}
\left[
\frac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+) + \frac{1}{4} S_i^z S_j^z
\right]
\]

这里没有 `J2` 项，因为当前 `J2 = 0`。

## 6. 哪些参数真的改变 Hamiltonian，哪些不改变

### 6.1 直接改变 Hamiltonian 的参数

下面这些参数直接进入系统 Hamiltonian：

- `Lx`
- `t`
- `J_K`
- `J1`
- `J2`
- `delta_z`
- `pbc`

其中：

- `Lx, pbc` 主要先改几何和键集
- `t, J_K, J1, J2, delta_z` 再改各项系数和结构

### 6.2 不直接改 Hamiltonian，但会改 Hilbert 扇区或初始化的参数

下面这些参数不改算符表达式本身，但会改系统所在的 sector 或初始化方式：

- `n_fermions`
- `joint_two_sz`

#### `n_fermions`

`n_fermions` 固定的是电子总数，不是 Hamiltonian 的 coupling constant。

它决定：

- 电子 Hilbert 空间只保留固定总粒子数的子空间

当前 benchmark 里：

```python
n_fermions = 4
```

对 `Lx = 4` 的自旋费米子链来说，这对应总电子数为 4，也就是常说的 half filling。

#### `joint_two_sz`

`joint_two_sz` 不进入 `KondoHeisenbergChain(...)` 的 Hamiltonian 构造公式。

它的作用是在 runner 里通过 `seed_joint_sz_sector(...)`，把采样初态放到指定的联合总磁量子数扇区：

\[
2S^z_{\mathrm{total}} = 2S^z_{\mathrm{electron}} + 2S^z_{\mathrm{local}}
\]

当前 benchmark 里：

```python
joint_two_sz = 0
```

所以它的作用是：

- 让采样从联合总 `2S^z = 0` 扇区开始
- 但它本身不改变 Hamiltonian 的项和系数

## 7. 为什么这里不用 `n_fermions_per_spin` 和 `local_total_sz`

这和 `KondoHeisenberg_constraint_notes.md` 的结论一致。

因为当前 benchmark 取的是：

```python
J_K = 1.0
```

而 full Kondo 项里有：

- `s_i^+ S_i^-`
- `s_i^- S_i^+`

这些项会翻转电子自旋，也会翻转局域自旋，因此：

- 单独的 `N_up` 不守恒
- 单独的 `N_dn` 不守恒
- 单独的局域自旋总 `S^z` 也不守恒

所以在 `J_K != 0` 时：

- 不应该固定 `n_fermions_per_spin`
- 不应该单独固定 `local_total_sz`

当前 benchmark 用的是更合理的方案：

- 只固定总电子数 `n_fermions`
- 需要时再用 `joint_two_sz` 指定联合总磁量子数扇区

## 8. `joint_sampler='hamiltonian'` 与 Hamiltonian 本身的关系

当前 benchmark 写的是：

```python
joint_sampler = 'hamiltonian'
```

这表示采样 proposal 按 full Hamiltonian 的连通性来走，也就是使用完整 Kondo-Heisenberg operator 的连接关系。

但这不会修改 Hamiltonian 本身。

它只影响：

- Monte Carlo proposal 如何选下一步候选态
- 是否显式加入额外的 proposal bias

同理，如果以后改成 `hamiltonian-with-proposal`，改变的也是 proposal 规则，而不是系统算符本身。

## 9. `site-token`、Transformer 参数、优化器参数与 Hamiltonian 的关系

下面这些参数也都不改变系统 Hamiltonian：

- `models = ['site-token']`
- `d_model`
- `n_heads`
- `n_layers`
- `mlp_ratio`
- `driver`
- `optimizer`
- `learning_rate`
- `diag_shift`
- `momentum`
- `n_iter`
- `n_samples`
- 其他 runtime / sampler 控制参数

它们分别决定的是：

- 变分波函数 ansatz 长什么样
- 优化器怎样更新参数
- Monte Carlo 怎样采样和跑多久

因此，如果你的问题是“系统 Hamiltonian 与哪些参数相关”，核心还是第 6 节里那几项物理参数。

## 10. 一句话总结

这个 benchmark 对应的是一个 1D 的 full Kondo-Heisenberg chain：

- `t` 控制电子最近邻 hopping
- `J_K` 控制每个 site 上电子自旋与局域自旋的 Kondo 耦合
- `J1`、`J2` 控制局域自旋链上的最近邻和次近邻交换
- `delta_z` 控制局域自旋交换项里的 XXZ 各向异性
- `Lx` 和 `pbc` 决定键结构
- `n_fermions`、`joint_two_sz` 决定 sector / 初始化，但不改变 Hamiltonian 的公式
