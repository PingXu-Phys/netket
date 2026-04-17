# HamiltonianRuleJax 的 n_conn 问题分析（2026-04-16）

## 1. 目的

这份文档分析的是 NetKet 原版 JAX `HamiltonianRule` 的一个更底层问题，不是 `with_proposal` 自己引入的问题。

目标是回答下面几件事：

1. 原版 `HamiltonianRuleJax` 想实现的 proposal 是什么。
2. 代码里实际实现成了什么。
3. 哪个文件、哪几行有问题。
4. 这个问题为什么不是“仅仅有重复条目”，而是会真正改变 `q(x'|x)`。
5. 这个问题进入 MH 接受率以后，为什么不能保证最终分布仍然正确。
6. 在一个最小 Kondo 体系上，这个问题具体是怎样发生的。

---

## 2. 先说结论

结论很直接：

- 问题核心在 `netket/operator/_discrete_operator_jax.py` 的默认 `n_conn` 实现。
- 它没有数 `mels != 0` 的个数，而是在数输入状态 `x` 里非零分量的个数。
- `HamiltonianRuleJax.transition` 把这个错误的 `n_conn` 当成“可提议 connected entries 的总数”来使用。
- 因此它真正执行的 proposal kernel，不等于它在 MH 修正里假设的那个 kernel。
- 一旦 proposal kernel 和 MH correction 不匹配，就不能保证 detailed balance，也不能保证最终平衡分布正确。

这不是“重复条目可以在 state-level 合并，所以没关系”那一类问题。

这里的问题更严重：

- 有些独立目标态会被完全漏掉；
- 有些状态会有一部分概率质量落到非法零状态；
- 某些 `x -> y` 是可提议的，但 `y -> x` 在代码实际 proposal 中却根本不可提议；
- 但代码仍然用 `log n_conn(x) - log n_conn(y)` 做修正。

---

## 3. 原版公式想实现什么

`HamiltonianRuleJax` 的文档写的是：

```math
T(x \to x') = \frac{1}{\mathcal N(x)} \theta(|H_{x,x'}|)
```

意思是：

- 先列出所有 `mels != 0` 的 connected entries；
- 然后在这些非零 entry 里均匀随机选一个。

如果记：

- `M(x)` = `get_conn_padded(x)` 返回的非零 entry 个数；
- `p_0 < p_1 < ... < p_{M(x)-1}` = 这些非零 entry 在 padded 列表中的位置；

那么理想的 entry-level proposal 应该是：

```math
q_{\text{ideal,entry}}(p_j \mid x) = \frac{1}{M(x)}, \qquad j=0,\dots,M(x)-1.
```

如果同一个目标态 `y` 由多个 entry 指向，那么 state-level proposal 是：

```math
q_{\text{ideal,state}}(y \mid x)
= \frac{\#\{j : x_{p_j}' = y\}}{M(x)}.
```

这时重复条目本身不是 bug；它只是意味着“按 entry 均匀”与“按 state 均匀”不同。

---

## 4. 相关代码在哪里

### 4.1 `n_conn` 的实现

文件：`netket/operator/_discrete_operator_jax.py`

问题位置：`n_conn()` 的下面几行。

```python
_, mels = self.get_conn_padded(x)
nonzeros = jnp.abs(x) > 0
_n_conn = nonzeros.sum(axis=-1)
```

这里的问题非常明确：

- 取到了 `mels`，但根本没用；
- 应该数的是 `jnp.abs(mels) > 0`；
- 实际数的是 `jnp.abs(x) > 0`。

所以它返回的不是“connected 非零 entry 数量”，而是“输入状态里非零分量数量”。

### 4.2 `HamiltonianRuleJax.transition` 怎样使用它

文件：`netket/sampler/rules/hamiltonian.py`

核心流程是：

```python
xp, mels = self.operator.get_conn_padded(x)
n_conn = self.operator.n_conn(x)
rand_i = jax.random.randint(key, shape=(x.shape[0],), minval=0, maxval=n_conn)

nonzeros = jnp.abs(mels) > 0
nonzero_i_plus1 = (jnp.cumsum(nonzeros, axis=-1)) * nonzeros
rand_i_mask = nonzero_i_plus1 == jnp.expand_dims(rand_i + 1, -1)
x_proposed = (xp * jnp.expand_dims(rand_i_mask, -1)).sum(axis=1, dtype=xp.dtype)

n_conn_proposed = self.operator.n_conn(x_proposed)
log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
```

这个流程的数学含义是：

1. `get_conn_padded(x)` 返回完整 `xp, mels`；
2. 用 `mels != 0` 标出真实非零 entry；
3. 但随机索引 `rand_i` 的取值范围不是由“非零 entry 数”决定，而是由 `n_conn(x)` 决定；
4. 最后又继续用 `n_conn(x)` 和 `n_conn(x')` 去构造 MH correction。

### 4.3 Metropolis 接受率在哪里使用 correction

文件：`netket/sampler/metropolis.py`

```python
σp, log_prob_correction = self.rule.transition(...)
do_accept = uniform < jnp.exp(
    proposal_log_prob - state.log_prob + log_prob_correction
)
```

这说明：

- `transition()` 返回的 `log_prob_correction` 必须等于
  `log q(x|x') - log q(x'|x)`；
- 否则 MH 接受率就不是针对“实际 proposal kernel”的正确接受率。

---

## 5. 当前代码实际实现的 proposal 是什么

记：

- `M(x)` = 实际非零 entry 个数；
- `c(x)` = 代码使用的 `n_conn(x)`；
- `p_0 < p_1 < ... < p_{M(x)-1}` = 非零 entry 的 padded 位置；

当前代码实际做的是：

1. 先从整数集合 `{0,1,...,c(x)-1}` 中均匀抽一个 `rand_i`；
2. 再寻找“第 `rand_i+1` 个非零 entry”；
3. 如果存在，就选中那个 entry；
4. 如果不存在，就 `rand_i_mask` 全为 `False`，于是 `x_proposed` 变成全零向量。

因此，当前代码真正实现的 entry-level proposal 是：

```math
q_{\text{code,entry}}(p_j \mid x) =
\begin{cases}
1/c(x), & j < \min(c(x), M(x)), \\
0, & j \ge c(x).
\end{cases}
```

并且当 `c(x) > M(x)` 时，还会有额外的非法质量：

```math
q_{\text{code}}(0\text{-state} \mid x) = \frac{c(x)-M(x)}{c(x)}.
```

所以只有在一个特殊条件下，当前代码才等价于理想公式：

```math
c(x) = M(x).
```

只要这个等式不成立，proposal 就已经变了。

---

## 6. 为什么这不是“重复条目问题”

如果只有重复条目，但 `c(x)` 真的等于非零 entry 数量，那么：

- 每个非零 entry 仍然是均匀被抽到的；
- state-level 概率只是“重复次数 / 总 entry 数”；
- MH correction 仍然可以是正确的。

那只是“按 entry 均匀”还是“按 state 均匀”的选择问题。

这里不是这个问题。

这里的问题是：

- `c(x)` 不是 entry 数量；
- 于是 entry-level proposal 本身就变了；
- 一些 entry 永远不会被选中；
- 一些不存在的 entry 还会吞掉概率质量。

所以这不是“重复态合并前后概率表达方式不同”，而是“代码实际 proposal kernel 和注释里的 proposal kernel 已经不是同一个东西”。

---

## 7. 为什么 MH 也救不回来

MH 只有在接受率里使用的是**实际 proposal kernel 的精确比值**时，才保证目标分布。

也就是说，需要：

```math
\log\text{corr}(x \to y) = \log q(y \to x) - \log q(x \to y).
```

但当前代码使用的是：

```math
\log\text{corr}_{\text{code}}(x \to y)
= \log c(x) - \log c(y).
```

这隐含假设：

```math
q_{\text{assumed}}(y \mid x) = \frac{1}{c(x)}
```

对所有可提议 entry 都成立。

而实际 proposal 是上一节给出的 `q_code`。两者一般不相等。

因此只要存在某对状态 `(x,y)` 满足：

```math
\frac{q_{\text{code}}(x \mid y)}{q_{\text{code}}(y \mid x)}
\neq
\frac{c(x)}{c(y)},
```

当前的 MH correction 就不是正确的 proposal ratio，detailed balance 就不能保证。

---

## 8. 数据流：错误是怎样传进去的

当前实现的数据流可以写成：

```text
输入状态 x
  -> operator.get_conn_padded(x)
      -> xp, mels
  -> operator.n_conn(x)
      -> c(x)   # 这里数错了，数的是 x!=0 的个数
  -> rand_i ~ Uniform{0, ..., c(x)-1}
  -> 根据 mels!=0 的累计编号，尝试选第 rand_i 个非零 entry
      -> 成功: 得到某个 x_proposed
      -> 失败: 得到全零非法状态
  -> operator.n_conn(x_proposed)
      -> c(x_proposed)   # 同样可能是错的
  -> log_prob_corr = log c(x) - log c(x_proposed)
  -> Metropolis acceptance
```

所以错误不是局限在一个辅助计数函数里，而是一路传播到：

- forward proposal；
- backward correction；
- 最终接受率。

---

## 9. 具体例子 A：漏掉独立目标态，导致单向 proposal

### 9.1 体系设置

使用仓库里的测试辅助函数 `_build_small_kondo_system()`：

- `Lx = 2`
- `t = 1.0`
- `J_K = 1.0`
- `J1 = 1.0`
- `J2 = 0.0`
- `delta_z = 1.0`
- `n_fermions = 2`

这个体系的规模是：

- fermion size = 4
- local spin size = 2
- joint size = 6
- 总基态数 = 24

下面选两个具体状态：

```text
x = [0, 0, 1, 1, -1, 1]
y = [1, 0, 0, 1, 1, 1]
```

### 9.2 从 x 出发，真实 connected entries 是什么

`get_conn_padded(x)` 的非零 entry 一共有 5 个：

| 非零 entry 序号 | target state | mel |
| --- | --- | --- |
| 4 | `[0, 0, 1, 1, -1, 1]` | `-0.25` |
| 5 | `[0, 0, 1, 1, 1, -1]` | `0.5` |
| 6 | `[0, 0, 1, 1, -1, 1]` | `-0.25` |
| 7 | `[1, 0, 0, 1, 1, 1]` | `0.5` |
| 9 | `[0, 0, 1, 1, -1, 1]` | `0.25` |

也就是说，state-level 上它们对应 3 个目标态：

- `x` 本身，重复了 3 次；
- `[0, 0, 1, 1, 1, -1]`，1 次；
- `y = [1, 0, 0, 1, 1, 1]`，1 次。

### 9.3 代码里如何 sample

对这个 `x`，代码得到：

```text
reported_n_conn = c(x) = 4
actual_nonzero_entry_count = M(x) = 5
```

于是 `rand_i` 只会在 `{0,1,2,3}` 上均匀抽取。

这意味着它只可能选到前 4 个非零 entry，也就是：

- entry 4
- entry 5
- entry 6
- entry 7

entry 9 永远不会被选到。

因此，代码实际的 state-level proposal 分布是：

```text
q_code(x | x) = 0.5
q_code([0,0,1,1,1,-1] | x) = 0.25
q_code(y | x) = 0.25
```

这里还看不出支持集缺失，因为 entry 9 只是又一次指向 `x` 本身。

### 9.4 从 y 出发，问题就暴露出来了

对 `y = [1, 0, 0, 1, 1, 1]`，真实非零 entry 一共有 6 个：

| 非零 entry 序号 | target state | mel |
| --- | --- | --- |
| 0 | `[1, 0, 1, 0, 1, 1]` | `-1.0` |
| 1 | `[0, 1, 0, 1, 1, 1]` | `-1.0` |
| 4 | `[1, 0, 0, 1, 1, 1]` | `0.25` |
| 6 | `[1, 0, 0, 1, 1, 1]` | `-0.25` |
| 8 | `[0, 0, 1, 1, -1, 1]` | `0.5` |
| 9 | `[1, 0, 0, 1, 1, 1]` | `0.25` |

注意：

- 这里 entry 8 对应的目标态正是 `x`；
- 也就是说，真实连通关系里 `y -> x` 是存在的。

但代码里：

```text
reported_n_conn = c(y) = 4
actual_nonzero_entry_count = M(y) = 6
```

所以它只会从前 4 个非零 entry 里抽，也就是：

- entry 0
- entry 1
- entry 4
- entry 6

entry 8 和 entry 9 永远不会被选到。

于是代码实际的 state-level proposal 分布变成：

```text
q_code([1,0,1,0,1,1] | y) = 0.25
q_code([0,1,0,1,1,1] | y) = 0.25
q_code(y | y) = 0.5
q_code(x | y) = 0
```

这就是关键反例：

- `q_code(y | x) = 0.25`
- `q_code(x | y) = 0`

也就是说，在代码实际 proposal 中：

- `x -> y` 能提议；
- `y -> x` 根本提议不到。

### 9.5 代码使用的 correction 又是什么

因为：

```text
c(x) = 4
c(y) = 4
```

所以代码会使用：

```math
\log\text{corr}_{\text{code}}(x \to y) = \log 4 - \log 4 = 0.
```

也就是它假设：

```math
\frac{q(x \mid y)}{q(y \mid x)} = 1.
```

但实际 proposal 比值是：

```math
\frac{q_{\text{code}}(x \mid y)}{q_{\text{code}}(y \mid x)}
= \frac{0}{0.25} = 0.
```

两者完全不一致。

### 9.6 这会造成什么结果

如果模型的 `proposal_log_prob - state.log_prob = 0`，例如常数模型对应的均匀目标分布，那么：

- 代码会以概率 `1` 接受 `x -> y`；
- 但反向 `y -> x` 根本提议不到。

于是：

- 从 `x` 到 `y` 有净概率流；
- 从 `y` 到 `x` 没有回流。

这显然不满足对称目标分布下的 detailed balance。

所以这里不能说“虽然 proposal 有问题，但 MH 会修正回来”。

不是的。MH correction 本身就已经和实际 proposal 不匹配了。

---

## 10. 具体例子 B：`reported_n > actual_entries` 时会产生非法零状态

再看另一个状态：

```text
x0 = [0, 0, 1, 1, 1, 1]
```

对这个状态：

```text
reported_n_conn = c(x0) = 4
actual_nonzero_entry_count = M(x0) = 3
```

真实非零 entry 只有 3 个：

| 非零 entry 序号 | target state | mel |
| --- | --- | --- |
| 4 | `[0, 0, 1, 1, 1, 1]` | `0.25` |
| 6 | `[0, 0, 1, 1, 1, 1]` | `0.25` |
| 9 | `[0, 0, 1, 1, 1, 1]` | `0.25` |

也就是说，第 4 个非零 entry 根本不存在。

但代码会在 `{0,1,2,3}` 上均匀抽 `rand_i`。

- 当 `rand_i = 0,1,2` 时，还能匹配到前 3 个非零 entry；
- 当 `rand_i = 3` 时，代码在 `nonzero_i_plus1 == 4` 的位置找不到任何匹配；
- 这会使 `rand_i_mask` 全为 `False`；
- 然后
  `x_proposed = (xp * mask).sum(...)`
  就会得到全零向量。

因此，这个状态下代码的非法零状态概率是：

```math
q_{\text{code}}(0\text{-state} \mid x0) = \frac{4-3}{4} = 0.25.
```

也就是说，有 `25%` 的 proposal 质量会掉到一个根本不是合法物理配置的零状态上。

`hamiltonian.py` 下面其实已经有注释提到这个风险：

```python
# if there are no connected elements x_proposed might
# contain illegal states (i.e. zeros)
# and log_prob_corr will be nan
```

这说明作者自己也知道“会冒出非法零状态”这一类边界问题，但当前实现没有真正处理。

---

## 11. 在整个小体系上，这不是偶发问题

我对这个 `Lx=2` 的小 Kondo 测试系统全部 24 个状态做了扫描，结果是：

- `reported_n_conn < actual_nonzero_entries` 的状态：14 个
- `reported_n_conn > actual_nonzero_entries` 的状态：6 个
- `reported_n_conn == actual_nonzero_entries` 的状态：4 个
- 代码实际 state-level support 与真实 support 不一致的状态：10 个
- 存在非法零状态风险的状态：6 个

所以这不是某一个角落状态才会触发的问题，而是这个小体系里就已经非常常见。

---

## 12. 为什么这件事和 `with_proposal` 的检查有关

前面验证里，`with_proposal` 在 `beta=0, occupations=None` 下与原版 `HamiltonianRule` 一致，这个结论仍然成立。

但它的含义必须说清楚：

- 这只能证明“当前 `with_proposal` 的中性路径复现了原版 JAX `HamiltonianRule` 的行为”；
- 不能证明“原版 JAX `HamiltonianRule` 本身就已经实现了正确的 Hamiltonian proposal 分布”。

也就是说：

- “和原版一致”是一个兼容性结论；
- 不是一个物理正确性结论。

---

## 13. 最终判断

最终判断可以压缩成四句话：

1. `HamiltonianRuleJax` 的问题核心不是重复条目，而是默认 `n_conn` 计数对象错了。  
2. 这个错误会改变代码实际 proposal kernel `q_code(x'|x)`。  
3. `transition()` 返回的 `log_prob_corr` 又继续把这个错误的 `n_conn` 当成真实 proposal 比值来用。  
4. 因此在当前 JAX 路径上，原版 `HamiltonianRule` 不能自动保证最终平衡分布仍然正确。  

如果后续要修这个问题，最小正确方向是：

- 先让 `n_conn` 真正数 `mels != 0`；
- 然后重新核对 `HamiltonianRuleJax.transition` 的实际 entry-level proposal 与 correction 是否严格一致；
- 再讨论 duplicate entries 是否要在 state-level 聚合。
