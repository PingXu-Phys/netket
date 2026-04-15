﻿﻿# Kondo-Heisenberg 当前最佳方案：Hamiltonian With Proposal + Move-Type Balance

## 0. 采用的方案

当前正式采用的方案是：
- 这条路线脱胎于“方案 A / HamiltonianRule 基线”
- 仍然以完整 Hamiltonian 的非对角连通性作为 proposal 支持集
- 默认只对费米子部分引入 occupation-biased proposal
- 对 `ss / ff / sf` 三类 move 额外加入一个单参数的软平衡层

第一阶段新增内容严格限制为：
- 新增一个库文件：`netket/sampler/rules/hamiltonian_with_proposal.py`
- 在 `netket/sampler/metropolis.py` 中新增一个工厂函数：`MetropolisHamiltonianWithProposal(...)`

这条方案能够实现：
- 保持 Hamiltonian 连通性，不漏掉真实允许的 move
- 用自然轨道 occupation 提升费米子相关 move 的 proposal 质量
- 用一个参数控制 `ss / ff / sf` 三类 move 的提议概率质量，使它们保持在相近量级
- 兼容一般的一体费米子过程 `c_a^\dagger c_b`，包括同自旋 hopping、同轨道自旋翻转，以及基变换后可能出现的跨轨道自旋翻转

这份文档与以下两份文档保持联动：
- [run_kondoheisenbergchain_hamiltonian_with_proposal.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal.md:1>)
- [run_kondoheisenbergchain_sampler_manual.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_sampler_manual.md:1>)

## 1. 设计原则

当前方案严格遵守下面三条原则：

1. 运行效率优先
   只在 `HamiltonianRule` 已经给出的 connected states 上做一次向量化分类、加权和归一化；不再手写独立的 hop / spin / Kondo 子枚举器。

2. 接口兼容且简洁
   用户接口只比现有 `MetropolisHamiltonian(...)` 多出极少数 proposal 参数；其余 `n_chains`、`sweep_size`、`reset_chains` 等参数全部沿用现有 sampler 工厂的传参方式。

3. 默认行为要稳健可退化
   当 `occupations=None` 且 `balance_beta=0` 时，新方案必须严格退化回原始 `HamiltonianRule` 的均匀 connected-state 采样。

## 2. 用户接口

### 2.1 新工厂函数

建议工厂函数签名写成：

```python
def MetropolisHamiltonianWithProposal(
    hilbert,
    hamiltonian,
    *,
    occupations=None,
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    **kwargs,
) -> MetropolisSampler:
    ...
```

含义：
- `hilbert`：联合 Hilbert 空间
- `hamiltonian`：用于给出真实连通性的算符
- `occupations`：费米子轨道 occupation 输入
- `balance_beta`：控制 `ss / ff / sf` 三类 move 平衡程度的单参数
- `noise_strength`、`mixing`：完全复用 `fermion_2nd_proposal.py` 的噪声方案
- `**kwargs`：直接传递给 `MetropolisSampler`

### 2.2 新 rule 类

建议新 rule 名称为：

```python
HamiltonianRuleWithProposal
```

第一阶段建议优先做 JAX 版本，并保留和现有 `HamiltonianRule` 接近的组织方式：

```python
HamiltonianRuleWithProposalJax
HamiltonianRuleWithProposal(...)
```

其中：
- `HamiltonianRuleWithProposalJax` 负责真实实现
- `HamiltonianRuleWithProposal(...)` 作为薄包装入口
- 若传入的不是 JAX-compatible operator，第一阶段直接抛出清晰错误即可

## 3. 用户输入的 occupation 约定

### 3.1 occupation 只作用在费米子模式上

当前最佳方案的默认版只对费米子模式引入 occupation bias。

也就是说：
- 不默认对 local spin 引入 pseudo-occupation
- 不默认去拟合电子-自旋联合分布
- 自旋部分默认保持中性

这与当前目标一致：
- 优先提升自然轨道基底下费米子相关 move 的 proposal 质量
- 同时尽量保持 proposal 风格接近原始 `HamiltonianRule`

### 3.2 输入长度为 `L`

当用户传入长度为 `L` 的数组时：

```text
occupations = [occ_1, occ_2, ..., occ_L]
```

解释为：
- `up` 和 `dn` 共用同一组轨道 occupation
- 实现层按 `fermion_2nd_proposal.py` 的逻辑用 `np.tile` 扩展到两个自旋子扇区

在当前 Kondo 样本布局下，这会扩展成：

```text
[occ_1, ..., occ_L, occ_1, ..., occ_L]
```

也就是：
- 前 `L` 个给 `dn` block
- 后 `L` 个给 `up` block

### 3.3 输入长度为 `2L`

当用户传入长度为 `2L` 的数组时，必须明确采用和当前费米子 block 一致的顺序：

```text
occupations = [occ_dn(1), ..., occ_dn(L), occ_up(1), ..., occ_up(L)]
```

也就是：
- 先 `dn` block
- 后 `up` block

这条顺序约定必须与当前联合样本的扁平布局保持一致：

```text
sigma = [fermion_dn_block | fermion_up_block | local_spin_block]
```

### 3.4 `occupations=None` 的默认语义

如果用户不传 `occupations`，则默认含义是：
- 所有费米子相关 proposal 权重都视为中性
- 用户层可以把它理解成“所有费米子轨道的提议权重统一为 1”

但实现层必须注意：
- 不能把 `occupations=None` 机械实现为把每个 `v_m` 真设成 1
- 因为若直接代入 `v_target (1-v_source)`，会导致权重退化到 0

因此正确实现是：

```math
occupations = None \quad \Longrightarrow \quad B_f(\sigma\to\eta) \equiv 1
```

也就是走一个单独的 neutral branch，而不是构造一个全 1 的 occupation 数组。

## 4. move 类型分类

对当前状态 `\sigma` 和某个 connected state `\eta`，只需比较它们在两个 block 上是否发生变化：

- fermion block 是否变化
- local spin block 是否变化

由此把每个 connected move 分为三类：

```math
C_{ff}(\sigma),\qquad C_{ss}(\sigma),\qquad C_{sf}(\sigma)
```

定义如下：
- `ff`：只有 fermion block 改变，spin block 不变
- `ss`：只有 spin block 改变，fermion block 不变
- `sf`：fermion 与 spin block 同时改变

在当前 Kondo-Heisenberg 模型里，它们分别对应：
- `ff`：纯费米子 hopping 或基变换后一体费米子过程
- `ss`：纯局域自旋交换
- `sf`：Kondo 联合翻转，以及更一般的 spin-fermion 联合过程

## 5. 支持集与基础公式

### 5.1 支持集

proposal 的支持集严格取 Hamiltonian 的非对角连通性：

```math
\mathcal C_H^{\mathrm{off}}(\sigma)=\{\eta\neq\sigma: H_{\sigma,\eta}\neq 0\}
```

因此：
- 不再自己手写 hop cluster 支持集
- 不再自己手写 Kondo move 支持集
- 一切以 `operator.get_conn_padded(...)` 给出的真实 connected states 为准

### 5.2 `with_proposal` 的噪声机制

若用户提供了 `occupations`，则 proposal 中使用的不是静态 `v_m`，而是沿用 `fermion_2nd_proposal.py` 的安全噪声版本：

```math
\bar v_m = (1-\lambda) v_m + \lambda/2
```

```math
\tilde v_m \sim \mathrm{Beta}(\nu\bar v_m,\ \nu(1-\bar v_m))
```

其中：
- `\lambda = mixing`
- `\nu = noise_strength`

这一机制的作用是：
- 避免 `v_m` 过于靠近 `0/1`
- 保持 proposal 稳健和遍历性
- 避免过拟合一组固定 occupations

实现时应与 `fermion_2nd_proposal.py` 一样：
- 每个 chain、每一步采样一组新的 `\tilde v`
- forward / backward correction 使用同一组本步采样到的 `\tilde v`

## 6. 默认的费米子 proposal 权重

对任意 connected move `\sigma \to \eta`，只根据费米子 block 的 occupation 变化定义未归一化 bias：

```math
B_f(\sigma\to\eta)
=
\prod_{m:0\to1}\tilde v_m
\prod_{m:1\to0}(1-\tilde v_m)
```

如果 `occupations=None`，则定义：

```math
B_f(\sigma\to\eta)=1
```

这个公式统一兼容所有一体费米子非对角过程，因为它只关心：
- 哪些模式从 `0\to1`
- 哪些模式从 `1\to0`

因此天然兼容：
- 同自旋 hopping `c_{j,\sigma}^\dagger c_{i,\sigma}`
- 同轨道自旋翻转 `c_{i,\uparrow}^\dagger c_{i,\downarrow}`
- 基变换后跨轨道自旋翻转 `c_{j,\downarrow}^\dagger c_{i,\uparrow}`
- 更一般的一体过程 `c_a^\dagger c_b`

## 7. `ss / ff / sf` 的软平衡方案

### 7.1 为什么要加这一层

原始 `HamiltonianRule` 是在所有 connected states 上均匀抽样，因此某一类 move 若连接态特别多，就会天然占主导。这样会出现：
- proposal 可能大部分时间都在 spin 中变
- 或者大部分时间都在 fermion 中变
- `sf` 联合 move 可能长期提得太少

因此需要一层额外的 move-type 平衡机制。

### 7.2 分组和组内和

对三类 move，定义每一组的总权重：

```math
S_t(\sigma)=\sum_{\eta\in C_t(\sigma)} B_f(\sigma\to\eta),
\qquad t\in\{ss,ff,sf\}
```

### 7.3 单参数软平衡

定义组权重：

```math
G_t(\sigma)=\mathbf 1_{S_t>0}\,(S_t(\sigma)+\varepsilon)^{1-\beta}
```

其中：
- `\beta = balance_beta \in [0,1]`
- `\varepsilon` 是仅用于数值安全的小常数
- `\mathbf 1_{S_t>0}` 保证空组不被错误分配概率质量

然后按两阶段采样：

1. 先选 move 类型

```math
P(t\mid \sigma)=\frac{G_t(\sigma)}{\sum_u G_u(\sigma)}
```

2. 再在该类型内部选具体 connected state

```math
P(\eta\mid \sigma,t)=\frac{B_f(\sigma\to\eta)}{S_t(\sigma)}
```

因此总 proposal 为：

```math
g(\sigma\to\eta)=
\frac{G_{t(\eta)}(\sigma)}{\sum_u G_u(\sigma)}
\cdot
\frac{B_f(\sigma\to\eta)}{S_{t(\eta)}(\sigma)}
```

### 7.4 这个参数的物理含义

- `balance_beta = 0`
  不做人为类型平衡，只保留 connected-state 层面的加权。

- `balance_beta = 1`
  对非空的 `ss / ff / sf` 三组做最强平衡，也就是三组先被近似等概率选中。

- `0 < balance_beta < 1`
  在“尊重原始 connected-set 结构”和“让三组 move 比例更接近”之间做软插值。

### 7.5 推荐默认值

当前推荐：

```text
balance_beta = 0.5
```

理由：
- 能明显抑制某一类 move 因连接态数过多而独占 proposal 的问题
- 又不会像 `balance_beta = 1` 那样过于强硬地拉平三类 move
- 更符合“运行效率优先”的原则

若 warmup 后仍发现 `ss` 明显压过 `ff / sf`，可再提高到 `0.7` 或 `0.8`。

## 8. 重要退化极限

这一节是实现正确性的核心检查。

### 8.1 退化回原始 `HamiltonianRule`

若：

```math
occupations=None,
\qquad balance_beta=0
```

则有：

```math
B_f \equiv 1,
\qquad S_t = |C_t|,
\qquad G_t = |C_t|
```

于是：

```math
g(\sigma\to\eta)=\frac{1}{|\mathcal C_H^{\mathrm{off}}(\sigma)|}
```

这正好严格退化为原始 `HamiltonianRule`。

### 8.2 只有 occupation bias，没有类型平衡

若：

```math
occupations\neq None,
\qquad balance_beta=0
```

则：

```math
g(\sigma\to\eta)
\propto
B_f(\sigma\to\eta)
```

这正是“只用费米子 occupation 对 Hamiltonian 连通态做重排”的版本。

### 8.3 只有类型平衡，没有 occupation bias

若：

```math
occupations=None,
\qquad balance_beta=1
```

则：
- 先在非空 `ss / ff / sf` 三组之间近似等概率选一组
- 再在该组内均匀选 connected state

这给出一个只做 move-type 平衡、不做 occupation bias 的版本。

## 9. 需要新增的代码结构

### 9.1 新文件

新增：

```text
netket/sampler/rules/hamiltonian_with_proposal.py
```

建议内部结构如下：

```python
_build_mode_occupations_or_none(...)
_infer_kondo_layout(...)
_sample_noisy_occupations(...)
_classify_move_types(...)
_compute_fermion_log_bias(...)
_compute_group_scores(...)
_log_proposal_selected(...)

@struct.dataclass
class HamiltonianRuleWithProposalJax(...):
    ...

def HamiltonianRuleWithProposal(...):
    ...
```

### 9.2 helper 的职责

- `_build_mode_occupations_or_none`
  校验 `occupations` 输入，支持 `None`、长度 `L`、长度 `2L` 三种情况。

- `_infer_kondo_layout`
  从联合 Hilbert 中推断：
  - 费米子 block 长度 `2L`
  - local spin block 长度 `L`
  - 三段 slice 的边界

- `_sample_noisy_occupations`
  复用 `with_proposal` 的 mixing + Beta noise。

- `_classify_move_types`
  从 `x` 与 `xp` 的差异中给出 `mask_ss / mask_ff / mask_sf`。

- `_compute_fermion_log_bias`
  对每个 candidate 计算
  `sum log(tilde_v)` 与 `sum log(1-tilde_v)` 形式的费米子 bias。

- `_compute_group_scores`
  根据 `S_t` 和 `balance_beta` 给出 `G_t`。

- `_log_proposal_selected`
  对已选中的 candidate 直接返回 `log q(selected)`，供 forward/backward correction 使用。

## 10. `transition(...)` 的实现流程

对 JAX rule 的 `transition(...)`，建议实现顺序如下：

1. 用 `operator.get_conn_padded(x)` 取出 `xp, mels`
2. 用 `abs(mels) > 0` 过滤掉 padding 和零矩阵元
3. 若 `occupations is not None`，先采样本步用到的 noisy occupations `tilde_v`
4. 比较 `xp` 与 `x`，得到每个 candidate 的 fermion / spin block 变化掩码
5. 生成 `mask_ss / mask_ff / mask_sf`
6. 计算每个 candidate 的 `B_f`
7. 聚合得到 `S_ss / S_ff / S_sf`
8. 根据 `balance_beta` 得到组权重 `G_ss / G_ff / G_sf`
9. 先采样组，再在组内采样具体 candidate
10. 得到 `x_proposed`
11. 用同一组 `tilde_v` 在 `x_proposed` 上重新计算 backward 所需的 `log q(x_proposed -> x)`
12. 返回：

```math
\log\mathrm{corr}=\log q(x'\to x)-\log q(x\to x')
```

## 11. backward correction 的实现原则

为了保证细致平衡，必须注意：
- forward 和 backward correction 必须使用同一组本步采样的 `tilde v`
- backward 端也必须重新做一次 `get_conn_padded(x_proposed)` 后的分组、组内归一化和总归一化
- 不能只修正选中的单个 move，而不修正对应的组归一化与总归一化

也就是说，选中的 move 的前向 log-proposal 应为：

```math
\log q(\sigma\to\eta)
=
\log G_{t(\eta)}
- \log\sum_u G_u
+ \log B_f(\sigma\to\eta)
- \log S_{t(\eta)}
```

反向也完全同理。

## 12. 从用户角度的调用示例

### 12.1 区分 `up / dn` occupation

```python
sa = MetropolisHamiltonianWithProposal(
    hi,
    hamiltonian=ha,
    occupations=np.concatenate([occ_dn, occ_up]),
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    n_chains_per_rank=16,
)
```

这里要求：

```text
occupations = [occ_dn(1), ..., occ_dn(L), occ_up(1), ..., occ_up(L)]
```

### 12.2 `up / dn` 共用 occupation

```python
sa = MetropolisHamiltonianWithProposal(
    hi,
    hamiltonian=ha,
    occupations=occ_orbital,
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    n_chains_per_rank=16,
)
```

其中：

```text
occ_orbital = [occ_1, ..., occ_L]
```

会自动扩展到两个自旋子扇区。

### 12.3 不传 occupation

```python
sa = MetropolisHamiltonianWithProposal(
    hi,
    hamiltonian=ha,
    occupations=None,
    balance_beta=0.5,
    n_chains_per_rank=16,
)
```

这时：
- 费米子 proposal 不再有额外 occupation bias
- 但仍保留 `ss / ff / sf` 的软平衡层

若希望完全回到原始 `HamiltonianRule`：

```python
sa = MetropolisHamiltonianWithProposal(
    hi,
    hamiltonian=ha,
    occupations=None,
    balance_beta=0.0,
    n_chains_per_rank=16,
)
```

## 13. 第一阶段明确不要做的事

为了保证实现最小、稳健，第一阶段明确不做：
- 不默认加入 local spin pseudo-occupation
- 不默认加入 `|H_{\sigma,\eta}|^\alpha` 因子
- 不拆成 `MultipleRules` 的多 rule 框架
- 不改 runner
- 不改 `__init__.py`
- 不额外引入第二个库文件

## 14. 必须注意的实现细节

1. `occupations=None` 只能走 neutral branch，不能替换成全 1 occupation 数组。
2. 空组不能因为 `epsilon` 而被错误分到概率质量，必须有 `1_{S_t>0}` 这一层。
3. move 分类必须根据 `xp - x` 的实际状态差异来做，而不是根据算符类型标签来猜。
4. 费米子 bias 必须按模式变化统一处理，不能把“同自旋 hop”硬编码成唯一情况。
5. backward correction 必须重算 proposed state 的组内与总归一化。
6. 第一阶段实现必须优先保持 JAX-friendly，避免在 `transition(...)` 中写 Python 层循环。
7. 样本切片顺序必须和当前 Kondo 代码一致：

```text
[fermion_dn_block | fermion_up_block | local_spin_block]
```

## 15. 代码生成时的最短实现清单

如果后续按本文档直接生成代码，最小实现清单就是：

1. 新增 `netket/sampler/rules/hamiltonian_with_proposal.py`
2. 在其中实现 `HamiltonianRuleWithProposalJax`
3. 在其中实现 `HamiltonianRuleWithProposal(...)` 包装入口
4. 在 `netket/sampler/metropolis.py` 中新增 `MetropolisHamiltonianWithProposal(...)`
5. 不改其他库文件

这就是当前收敛出的最佳第一阶段实现路线。

## 16. Repeated Connected Entries

This point must be stated explicitly.

### 16.1 The proposal actually implemented by the code

The current implementation does not deduplicate Hamiltonian-connected target states before assigning proposal mass.
Instead, it enumerates the valid connected entries returned by `get_conn_padded(...)` and assigns probability mass on those entries.

For a current state `sigma`, let the valid connected entries be `E(sigma)`, and let each entry `e` point to a target state `T_sigma(e)=eta`.
The actual proposal implemented by the code is

```math
q_{code}(eta\mid sigma)
=
\sum_{e:T_sigma(e)=eta}
\frac{W_e(sigma)}{\sum_{e'\in E(sigma)} W_{e'}(sigma)}.
```

So:
- the proposal is first defined on connected entries;
- if the same target state appears through multiple entries, its total proposal probability is the sum of those entry probabilities.

### 16.2 What this changes

If the intended ideal proposal is defined on unique target states as

```math
q_{ideal}(eta\mid sigma)
\propto W(sigma\to eta),
```

then in the presence of repeated connected entries the implemented `q_code` is generally not equal to `q_{ideal}`.
It carries the multiplicity of the target state inside `get_conn_padded(...)`.

Therefore:
- it changes the shape of the proposal itself;
- it also changes the stationary distribution of a proposal-only chain with no MH correction;
- one should not interpret the proposal-only chain as an exact realization of the ideal unique-target proposal.

### 16.3 Why this does not change the final Metropolis target distribution

For Metropolis-Hastings, the key point is not whether the proposal equals `q_{ideal}`.
The key point is whether the acceptance correction uses the same effective proposal that the code actually samples from.

The current implementation already computes forward and backward probabilities using the effective `q_{code}`:
- it builds log-probabilities on connected entries;
- it then uses `_log_state_probability(...)` to sum repeated entries pointing to the same target by `logsumexp`;
- it finally returns `log_q_bwd - log_q_fwd`.

Therefore the acceptance rule is

```math
A(sigma\to eta)
=
\min\left(
1,
\frac{|\Psi(eta)|^2\,q_{code}(sigma\mid eta)}{|\Psi(sigma)|^2\,q_{code}(eta\mid sigma)}
\right).
```

As long as both forward and backward corrections use the same `q_{code}`, detailed balance still holds.
Therefore the final Metropolis target distribution remains

```math
pi(sigma)=|\Psi(sigma)|^2.
```

Conclusion:
- repeated connected entries change the proposal shape;
- they do not bias the final Metropolis target away from `|Psi|^2`;
- only if one wants an exact unique-target proposal does one need deduplication or an explicit multiplicity correction before sampling.

## 17. Complexity And Duplicate Aggregation Switch

This section records the current design choice around duplicate connected entries.

### 17.1 Complexity levels

It is useful to separate three cost regimes.

1. Baseline `HamiltonianRule`
   - enumerate Hamiltonian-connected entries
   - draw one entry uniformly
   - use forward/backward connected counts only

2. Neutral fast path
   - `occupations=None`
   - `balance_beta=0.0`
   - `aggregate_duplicate_entries=False`

   This path reuses the original `HamiltonianRule` implementation and therefore has the same cost class as the baseline sampler.

3. General proposal path
   - compute proposal scores for all connected candidates
   - sample from the full candidate distribution
   - recompute the backward proposal on the proposed state

   If `B` is the number of chains, `M` is the typical number of connected entries, `D` is the full state size, and `F` is the fermion-block size, then the extra proposal work is roughly of order

```math
O(B M (D + F))
```

on top of the connectivity backend itself.

### 17.2 Do we need explicit duplicate aggregation

Not always.

If repeated connected entries have symmetric forward/backward multiplicities, which is the typical situation for standard Hermitian Hamiltonians, then the original entry-level treatment is usually acceptable.

However, if one wants the Metropolis correction to use the fully aggregated state-level proposal

```math
q(target\mid source),
```

then repeated entries should be explicitly summed.

### 17.3 The new switch

The implementation now exposes

```python
aggregate_duplicate_entries=False
```

Interpretation:
- `False`:
  - closer to the original `HamiltonianRule` treatment of connected entries
  - cheaper
  - the neutral limit can reuse the original fast path exactly
- `True`:
  - explicitly aggregates repeated connected entries leading to the same target state
  - uses state-level proposal probabilities in the MH correction
  - is more conservative, but more expensive

### 17.4 Recommended default

The current recommended default is

```python
aggregate_duplicate_entries=False
```

Reason:
- it preserves the closest behavior to the original Hamiltonian sampler;
- it keeps the neutral limit exactly on the original fast path;
- for standard Hermitian Hamiltonians this is usually sufficient.

If a later benchmark or operator-specific study shows that duplicate-entry asymmetry matters, the switch can be turned on without changing the rest of the sampler interface.

## 18. Local-Difference Complexity And Optimization Directions

The earlier estimate

```math
O(B M (D + F))
```

is a safe upper bound for the current implementation, not a statement that the full state length must always be scanned in an optimal implementation.

### 18.1 Why the current bound is conservative

For Hamiltonian-connected proposals in this Kondo-Heisenberg setting, a connected move usually changes only a small local subset of the configuration:
- a fermion hop changes only a small number of fermion modes;
- a fermion spin flip changes only one source mode and one target mode;
- a local spin exchange changes only a small number of local-spin entries;
- a Kondo-like joint move changes only one local fermion pattern and one local spin.

Therefore, from an algorithm-design point of view, the true amount of information needed to score a move is often local. One does not need to rescan the whole state vector if the implementation can work directly with the changed coordinates.

### 18.2 What can be optimized

There are three obvious optimization directions.

1. Local move classification
   Instead of checking candidate differences by comparing the whole state vector, one can classify a move from the small set of changed indices only. In other words, `ss / ff / sf` can be inferred from the local delta pattern.

2. Local fermion-bias evaluation
   The current fermion bias conceptually depends only on the modes whose occupation changed:

```math
B_f(x\to x')
=
\prod_{m:0\to1}\tilde v_m
\prod_{m:1\to0}(1-\tilde v_m).
```

This means the cost should really scale with the number of changed fermion modes, not with the full fermion block length `F`, if the operator can expose the local change set efficiently.

3. Local duplicate handling
   Duplicate-target detection also does not fundamentally require comparing the full state vector in an optimal implementation. If two connected entries differ only on a small local support, then a local canonical representation of that support can be used for duplicate aggregation or representative-entry selection.

### 18.3 What this means for complexity

A more physically faithful way to think about the ideal optimized cost is:
- `B`: number of chains
- `M`: number of connected entries per chain
- `k_move`: typical number of changed degrees of freedom in one connected move
- `k_f`: typical number of changed fermion modes in one connected move

Then the move-dependent work should scale more like

```math
O(B M (k_move + k_f))
```

with `k_move` and `k_f` small, instead of the current conservative upper bound in terms of the full state sizes `D` and `F`.

### 18.4 Why the current implementation still uses the conservative form

The present implementation is written against the generic output of `get_conn_padded(...)`, which provides connected states as full target configurations. With only that interface, the simplest robust implementation is to compare full states. This is why the current code still behaves like the conservative upper bound.

So the current complexity is best understood as:
- a robust generic implementation cost;
- not the fundamental lower bound of the proposal idea itself.

### 18.5 Most promising future optimization path

The best optimization path is not to change the proposal formula, but to enrich the local information available from the connectivity backend.

The most promising improvements are:
- expose changed-index information together with each connected entry;
- evaluate `ss / ff / sf` from local deltas only;
- evaluate fermion proposal factors from changed fermion modes only;
- perform duplicate aggregation using local move signatures rather than full-state equality checks.

If these local descriptors become available, the proposal can stay mathematically the same while its implementation cost drops substantially.
