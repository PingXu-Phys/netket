# Kondo-Heisenberg Transformer 接口说明

更新日期: 2026-04-14  
适用代码:
- `netket/nqs_state/transformer_joint_nnbf.py`
- `netket/my_run/run_kondoheisenbergchain_transformers.py`
- `netket/Hamiltonian/KondoHeisenbergChain.py`

## 1. 这份文档回答什么问题

这份文档只回答三件事：

1. `KondoHeisenbergChain` 的 `hi` 是怎么变成 Transformer token 的。
2. 费米子与局域自旋这两个子空间，在采样时是如何分别处理、再合并成联合样本的。
3. `run_kondoheisenbergchain_transformers.py` 里几个关键接口应该怎么调用。

## 2. Hilbert 空间的真实结构

`KondoHeisenbergChain.py` 里真正的 Hilbert 是

```python
joint_hi = fermion_hi * local_spin_hi
```

其中：

- `fermion_hi = SpinOrbitalFermions(Lx, s=1/2, ...)`
- `local_spin_hi = Spin(s=1/2, N=Lx, ...)`
- `joint_hi` 是 `TensorHilbert`

也就是说，费米子与局域自旋在 NetKet 里并没有混成一个单独的 Hilbert 类型，而是标准的张量积结构。

对 `Kondo` 模型尤其要注意：

- 当 `J_K != 0` 时，不能同时固定 `n_fermions_per_spin`。
- 这是因为 Kondo 自旋翻转会改变电子的 `N_up/N_dn`，只能固定总电子数 `n_fermions`。
- 这也是为什么 `transformer_joint_nnbf.py` 里必须支持 generalized determinant，而不是只支持分块 `up/dn` determinant。

## 3. 原始样本的排列顺序

当前接口假定费米子样本顺序与 `KondoHeisenbergChain.py` 一致：

```text
[fermion down block | fermion up block | local-spin block]
```

对于长度 `Lx = N` 的链：

- 前 `N` 个位置: `n_dn[0:N]`
- 接着 `N` 个位置: `n_up[0:N]`
- 最后 `N` 个位置: `S^z_loc[0:N]`

所以总长度是 `3N`。

例如 `Lx=4` 时，一个联合样本的结构是

```text
[n_dn(4) | n_up(4) | S_loc(4)]
```

## 4. 从 hi 到 token 的路径

### 4.1 公共第一步: 解析布局

`resolve_spin_fermion_layout(system_or_hilbert)` 会把联合 Hilbert 解析成一个轻量布局对象：

- `n_sites`
- `fermion_size`
- `spin_block_sizes`
- `n_fermions`
- 如果可用，再带上 `n_dn` 和 `n_up`

这样做的目的有两个：

- 避免把完整 `Hamiltonian/system` 重对象塞进 Flax module
- 把 token 化和 determinant 后端需要的静态尺寸集中起来

### 4.2 方案 A: `LogSiteTokenTransformerNNBF`

这是“四态 site token”方案。

每个 site 上，费米子局域态由

```text
(n_up[i], n_dn[i])
```

压成一个四态 token：

- `0`: 空占
- `1`: 只有 up
- `2`: 只有 dn
- `3`: 双占

然后：

1. 对这个四态 token 做 lookup embedding。
2. 加上 `site_embedding[i]`。
3. 如果有局域自旋链，再把每个 site 的自旋 token embedding 加进去。
4. 送入长度为 `N` 的 site-level Transformer。

这个方案的优点：

- token 序列长度短，只是 `N`
- 双占是原生局域态，不需要模型自己从两个二值 channel 重建
- 一般比 `spin-channel` 更省显存

### 4.3 方案 B: `LogSpinChannelTransformerNNBF`

这是“分 channel”方案。

每个 site 对应多个 channel：

- `dn` channel
- `up` channel
- `local spin` channel
- 如果以后有更多 site-aligned 自旋链，还可以继续加 channel

每个 token 的表示是：

```text
value_embedding + site_embedding + channel_embedding
```

然后：

1. 把 `(site, channel)` 展平成一条更长的序列。
2. 送入 Transformer。
3. 再把各 channel 的输出重新聚合回 site feature。

这个方案的优点：

- 保留了“不同自由度分 channel”这一点，接口上更显式
- 更容易扩展到多个 site-aligned 通道

代价是：

- 序列更长
- attention 更贵
- 显存通常比 `site-token` 更大

## 5. Transformer 输出后如何接 determinant

这里当前实现是一个 **adaptive determinant backend**，不是固定只有一种 determinant。

### 5.1 如果 Hilbert 固定了 `n_dn` 和 `n_up`

就走分块 determinant：

```text
det(Phi_dn) * det(Phi_up)
```

这对应标准的自旋分块 Slater/backflow 结构。

当前实现里，up/down 两个 backflow head 会分别输出：

- `DeltaPhi_dn in R^{N x N_dn}`
- `DeltaPhi_up in R^{N x N_up}`

因此即使 `N_dn != N_up`，block determinant 也仍然成立；这里不要求费米子 `S^z_f = 0`，只要求 Hilbert 本身固定了这两个 sector 的粒子数。

### 5.2 如果 Hilbert 只固定总电子数 `n_fermions`

就走 generalized determinant：

```text
Phi_all in R^{(2N) x N_f}
```

再从所有被占据的 spin-orbital 行里抽取子矩阵，做一个总 determinant。

这一步对 `KondoHeisenbergChain` 是必须的，因为：

- `J_K != 0` 时电子自旋可以翻转
- 因此不能把 ansatz 硬限制在固定的 `N_dn/N_up` 子空间里
- 在这种 full Kondo 情况下，更自然的是固定总电子数，再视需要固定联合总 `S^z`

## 6. 采样时两个空间为什么“看起来是分开的”

这是 `TensorHilbert + TensorRule` 的直接结果。

### 6.1 factorized 采样的调用链

如果在 `my_run` 里选择：

```text
--joint-sampler factorized
```

那么联合采样器结构是：

```python
fermion_rule = ...
spin_rule = ...
rule = nk.sampler.rules.TensorRule(joint_hi, (fermion_rule, spin_rule))
sampler = nk.sampler.MetropolisSampler(joint_hi, rule, ...)
```

`TensorRule.transition(...)` 的逻辑非常直接：

1. 把当前联合样本 `sigma` 按子空间切开。
2. 费米子块 `sigma_ferm` 交给 `fermion_rule.transition(...)`。
3. 自旋块 `sigma_spin` 交给 `spin_rule.transition(...)`。
4. 得到新的 `sigma_ferm'` 和 `sigma_spin'`。
5. 再把它们按原顺序拼回一个新的联合样本 `sigma'`。
6. 如果各个子规则返回了 proposal 修正项 `log_prob_corr`，就把它们加起来。
7. 最终的 Metropolis 接受率，仍然是对完整的联合波函数 `psi(sigma') / psi(sigma)` 来算。

所以“分开采样”的准确含义不是：

- 两个空间各自做一套独立的 Metropolis，最后再手工平均

而是：

- proposal 在子空间上分块生成
- acceptance 在联合波函数上统一判断

## 7. factorized 模式里，费米子 proposal 有两种

### 7.1 标准 fermion hop

如果选择：

```text
--fermion-sampler standard
```

费米子 proposal 是：

```python
nk.sampler.rules.FermionHopRule(...)
```

它做的是常规的“在允许的 cluster 上，把一个粒子从占据 mode hop 到空 mode”。

### 7.2 带 occupation proposal 的 fermion hop

如果选择：

```text
--fermion-sampler with-proposal
```

费米子 proposal 是：

```python
FermionHopRule_with_proposal(...)
```

它会用一组目标 occupations 去偏置 hop proposal 权重。当前接口支持两种输入：

- `--proposal-occupation-value 0.5`
  直接给每个 orbital 一个统一 occupation，长度自动按 `n_sites` 扩展
- `--proposal-occupations-file path`
  从文件读 occupation 数组，长度可以是 `n_sites` 或 `2*n_sites`

额外还可以控制：

- `--proposal-noise-strength`
- `--proposal-mixing`

也就是说，`with-proposal` 改变的是“费米子块如何 proposal”，不是“整个联合接受率如何算”。

## 8. factorized 模式的限制

这里要说清楚一个物理上的限制。

如果系统是完整 Kondo 链，并且：

- `J_K != 0`
- 只固定总电子数 `n_fermions`

那么真正的 Kondo 自旋翻转是一个“费米子自旋 + 局域自旋”耦合变化。

但 factorized 模式里：

- 费米子 proposal 在费米子子空间里单独做
- 自旋 proposal 在自旋子空间里单独做

因此它不会在“一个 proposal 步”里直接生成 Kondo 的耦合翻转。

这并不意味着它完全不能用，但要明白：

- 它更像一个显式分块、便于调试和做 proposal 对比的 sampler
- 如果你想严格沿 Hamiltonian 的连通结构 proposal，更稳妥的是用 `HamiltonianRule`

## 9. Hamiltonian 采样模式

如果在 `my_run` 里选择：

```text
--joint-sampler hamiltonian
```

那么 sampler 直接是：

```python
nk.sampler.MetropolisHamiltonian(joint_hi, hamiltonian=H, ...)
```

这时：

- proposal 直接从完整 Hamiltonian 的非对角连接里来
- 会自然包含 Kondo 的联合自旋翻转
- 也会自然保留 Hamiltonian 的对称性与连通结构

这也是为什么对于完整 Kondo 问题，这通常是更物理、更稳的默认选项。

注意：

- `--fermion-sampler standard|with-proposal` 只在 `--joint-sampler factorized` 时有效
- 在 `hamiltonian` 模式下，这个开关会被禁用

## 10. `my_run` 脚本里现在有哪些关键接口

### 10.1 构造系统

```python
system = KondoHeisenbergChainSpinFermion(...)
```

它给出：

- `system.hamiltonian`
- `system.joint_hilbert`
- `system.fermion_hilbert`
- `system.local_spin_hilbert`
- `system.geometry`

### 10.2 构造布局

```python
layout = resolve_spin_fermion_layout(system)
```

### 10.3 构造模型

```python
model = LogSiteTokenTransformerNNBF(layout=layout, ...)
```

或者

```python
model = LogSpinChannelTransformerNNBF(layout=layout, ...)
```

### 10.4 构造采样器

如果是 factorized:

```python
sampler, sampler_info = build_sampler(system, layout, args)
```

内部会走：

- `build_factorized_sampler(...)`
- 选择 `standard` 或 `with-proposal` 的 fermion rule
- 再和 spin rule 一起交给 `TensorRule`

如果是 hamiltonian:

- 同样还是 `build_sampler(...)`
- 但内部直接走 `MetropolisHamiltonian`

### 10.5 构造 variational state

```python
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=...,
    n_discard_per_chain=...,
    seed=...,
)
```

### 10.6 joint sector seeding

当：

- `J_K != 0`
- 只固定总电子数
- 并且使用 `--joint-sampler hamiltonian`

脚本会自动调用：

```python
seed_joint_sz_sector(vstate, two_sz=args.joint_two_sz)
```

用来把初态放进给定 joint `2S^z` 扇区。

## 11. 推荐的调用方式

### 11.1 更物理、默认推荐

```powershell
python my_run/run_kondoheisenbergchain_transformers.py \
  --Lx 50 \
  --models site-token spin-channel \
  --joint-sampler hamiltonian
```

适用场景：

- 完整 Kondo 模型
- 希望 proposal 直接尊重 Hamiltonian 连通结构
- 希望自然包含 Kondo 自旋翻转

### 11.2 需要显式比较 fermion proposal 时

标准 hop:

```powershell
python my_run/run_kondoheisenbergchain_transformers.py \
  --Lx 50 \
  --models site-token \
  --joint-sampler factorized \
  --fermion-sampler standard
```

带 occupation proposal:

```powershell
python my_run/run_kondoheisenbergchain_transformers.py \
  --Lx 50 \
  --models site-token \
  --joint-sampler factorized \
  --fermion-sampler with-proposal \
  --proposal-occupation-value 0.5
```

或：

```powershell
python my_run/run_kondoheisenbergchain_transformers.py \
  --Lx 50 \
  --models site-token \
  --joint-sampler factorized \
  --fermion-sampler with-proposal \
  --proposal-occupations-file path_to_occ.txt
```

## 12. 最后一句话总结

- 模型接口上，费米子和自旋是“分开表征、联合判断”。
- determinant 后端是“固定分自旋时 blocked / 仅固定总数时 generalized”。
- factorized 采样上，proposal 是“分开生成、统一接受”。
- 完整 Kondo 物理上，如果你要 proposal 也体现费米子-自旋耦合翻转，优先用 `--joint-sampler hamiltonian`。