# 小体系 Correctness Benchmark 方案（2026-04-16）

## 1. 目标

这个 benchmark 的目标不是看“训练出来的能量是否接近 ED”，而是直接检查采样 kernel 本身是否正确。

这里的“正确”分成三层：

1. `proposal` 是否等于代码声称的那个 proposal。
2. `log_prob_corr` 是否等于实际 proposal 的反向/正向比值。
3. `proposal + MH acceptance` 组合后的完整转移核，是否以目标分布 `|psi(x)|^2` 为平衡分布。

只要第 2 条不成立，第 3 条一般就不能保证成立。

---

## 2. 为什么不能只看能量

只看能量有两个问题：

1. 能量只是单一标量，对分布错误不总是敏感。
2. 即使采样分布有偏，能量偏差也可能因为误差小、抵消、或 observable 不敏感而仍然只有 `1e-3`。

因此，这个 benchmark 要直接检查：

- `q(y|x)`
- `log q(x|y) - log q(y|x)`
- detailed balance residual
- stationary distribution 偏差

能量只作为最后一项辅助指标，不作为 correctness 主判据。

---

## 3. 固定测试体系

建议先用仓库里已经存在的小体系：

- builder: `_build_small_kondo_system()`
- 文件: [test_hamiltonian_with_proposal.py](/D:/Seafile/PHD/NQS/NetKet/netket/test/sampler/test_hamiltonian_with_proposal.py#L50)

它对应：

- `Lx = 2`
- `t = 1.0`
- `J_K = 1.0`
- `J1 = 1.0`
- `J2 = 0.0`
- `delta_z = 1.0`
- `n_fermions = 2`

规模是：

- fermion size = 4
- local spin size = 2
- joint size = 6
- 总基态数 = 24

这个体系足够小，可以把整个转移矩阵精确枚举出来。

---

## 4. 要比较的对象

建议至少比较下面 4 个 kernel。

### 4.1 Kernel A: 当前原版 JAX `HamiltonianRule`

- 文件: [hamiltonian.py](/D:/Seafile/PHD/NQS/NetKet/netket/netket/sampler/rules/hamiltonian.py#L167)
- `n_conn` 来源: [\_discrete_operator_jax.py](/D:/Seafile/PHD/NQS/NetKet/netket/netket/operator/_discrete_operator_jax.py#L199)

这是当前需要被审计的对象。

### 4.2 Kernel B: 当前 `with_proposal` 的中性路径

参数固定为：

- `occupations=None`
- `balance_beta=0.0`
- `aggregate_duplicate_entries=False`

这一路径当前的目标不是“物理上更正确”，而是验证它是否复现了 Kernel A 的行为。

### 4.3 Kernel C: 修正后的 entry-level reference kernel

这个 reference 不需要改库文件，可以在 benchmark 脚本里自己定义。

定义方式：

- `M(x) = count_nonzero(mels)`
- forward proposal 在所有非零 entries 上均匀抽样
- correction 用
  `log M(x) - log M(y)`

这是“按非零 entry 均匀”的最小 reference。

### 4.4 Kernel D: 可选的 state-aggregated reference kernel

如果你还想单独研究“重复条目是否应该按 state 聚合”，可以再定义一个 state-level reference：

- 先把相同 target state 的 entries 合并
- 按合并后的 state-level 质量定义 proposal

这不是修原版 bug 的必要步骤，但能帮助分清：

- “重复条目定义差异”
- “`n_conn` 计数错误”

---

## 5. benchmark 的核心定义

### 5.1 代码实际 proposal `q_code(y|x)`

对某个状态 `x`，记：

- `xp, mels = get_conn_padded(x)`
- `I(x) = {p_0 < p_1 < ... < p_{M(x)-1}}` 为 `mels != 0` 的位置
- `c(x) = operator.n_conn(x)` 为当前代码使用的 `n_conn`

那么当前 `HamiltonianRuleJax` 实际实现的 entry-level proposal 是：

```math
q_{code,entry}(p_j | x) =
\begin{cases}
1 / c(x), & j < \min(c(x), M(x)) \\
0, & j \ge c(x)
\end{cases}
```

当 `c(x) > M(x)` 时，还会有：

```math
q_{code}(0\text{-state} | x) = (c(x)-M(x)) / c(x)
```

state-level proposal 则由这些 entry 概率按 target state 聚合得到。

### 5.2 代码实际 correction

当前代码返回：

```math
log\_corr_{code}(x \to y) = \log c(x) - \log c(y)
```

### 5.3 benchmark 要检查的关键一致性条件

对任意一对可达状态 `(x,y)`，如果代码是正确的，就应有：

```math
log\_corr_{code}(x \to y)
= \log q_{code}(x|y) - \log q_{code}(y|x)
```

如果这个等式不成立，当前 MH correction 就不是针对“代码实际 proposal”的正确修正。

---

## 6. 建议的 5 组检查

## 6.1 检查一：support mismatch

目标：检查当前代码 proposal 的支持集是否和真实 connected support 一致。

对每个状态 `x`：

1. 列出全部 `mels != 0` 对应的 target states。
2. 构造代码实际可提议到的 state-level support。
3. 比较两者是否相同。

输出指标：

- `n_support_mismatch_states`
- 每个 mismatch 状态的：
  - `x`
  - `reported_n_conn`
  - `actual_nonzero_entries`
  - `actual_target_states`
  - `code_accessible_target_states`

判据：

- 如果有任何一个状态 support mismatch，这个 kernel 就已经不是理想 Hamiltonian proposal。

## 6.2 检查二：zero-state mass

目标：检查 `reported_n_conn > actual_nonzero_entries` 时，是否存在非法零状态质量。

对每个状态 `x` 计算：

```math
p_{zero}(x) = \max(c(x)-M(x), 0) / c(x)
```

输出指标：

- `n_zero_state_risk_states`
- `max_zero_state_probability`
- 具体例子列表

判据：

- 只要 `p_zero(x) > 0`，当前 direct path 就存在非法 proposal 质量。

## 6.3 检查三：correction residual

目标：检查 `log_prob_corr` 是否真的等于代码实际 proposal 的反向/正向比值。

对每个有 `q_code(y|x) > 0` 的 pair `(x,y)`，定义：

```math
\Delta_{corr}(x,y)
=
\bigl[\log q_{code}(x|y) - \log q_{code}(y|x)\bigr]
-
\bigl[\log c(x) - \log c(y)\bigr]
```

输出指标：

- `max_abs_delta_corr`
- `n_delta_corr_nonzero_pairs`
- 若 `q_code(x|y)=0` 而 `q_code(y|x)>0`，单独列成 one-way edge 统计

判据：

- 正确 kernel 应满足 `Delta_corr = 0` 到数值精度。

## 6.4 检查四：detailed balance residual

选一个目标分布 `pi(x)`，直接构造完整 MH 转移核：

```math
P(x \to y) = q_{code}(y|x) A(x \to y), \qquad y \neq x
```

其中：

```math
A(x \to y) = \min\left(1,
\frac{\pi(y)}{\pi(x)}
\exp(log\_corr_{code}(x \to y))
\right)
```

然后检查：

```math
R(x,y) = \pi(x)P(x \to y) - \pi(y)P(y \to x)
```

输出指标：

- `max_abs_db_residual`
- `sum_abs_db_residual`
- 最大残差对应的 `(x,y)`

判据：

- 正确 MH kernel 应满足 `R(x,y)=0` 到数值精度。

## 6.5 检查五：stationary distribution bias

把完整转移矩阵 `P` 构造出来以后，直接解平衡分布 `mu`：

```math
mu^T P = mu^T
```

然后比较 `mu` 与目标分布 `pi`。

输出指标：

- total variation distance
- `L1` distance
- `KL(mu || pi)` 或 `KL(pi || mu)`

判据：

- 正确 kernel 应该得到 `mu = pi` 到数值精度。

---

## 7. 建议使用的目标分布

建议至少做两个版本。

### 7.1 Case A: 常数模型，对应均匀分布

令所有状态的模型 log-amplitude 相同，则：

```math
pi(x) = 1 / |\mathcal H|
```

这个 case 最干净，因为：

- `proposal_log_prob - state.log_prob = 0`
- 接受率只剩 proposal correction
- 能最直接暴露“correction 与 proposal 不匹配”的问题

这是第一优先级。

### 7.2 Case B: 精确可控的非均匀正分布

再定义一个严格正的分布，例如：

```math
pi(x_i) \propto e^{-w_i}
```

其中 `w_i` 是对每个 basis state 预先固定的一组实数。

实现上不需要真的训练模型，可以在 benchmark 脚本中直接用一个 lookup table。

这个 case 用来检查：

- 问题是否只在均匀分布下可见；
- 还是对一般非均匀目标分布也会留下明确 detailed balance 残差。

---

## 8. 最小反例必须固定打印出来

benchmark 报告里必须固定包含两个反例。

## 8.1 反例 A：漏掉独立目标态

在小体系里，取：

```text
x = [0, 0, 1, 1, -1, 1]
y = [1, 0, 0, 1, 1, 1]
```

对 `x`：

- `reported_n_conn = 4`
- `actual_nonzero_entry_count = 5`

非零 entries 为：

| entry index | target state | mel |
| --- | --- | --- |
| 4 | `[0, 0, 1, 1, -1, 1]` | `-0.25` |
| 5 | `[0, 0, 1, 1, 1, -1]` | `0.5` |
| 6 | `[0, 0, 1, 1, -1, 1]` | `-0.25` |
| 7 | `[1, 0, 0, 1, 1, 1]` | `0.5` |
| 9 | `[0, 0, 1, 1, -1, 1]` | `0.25` |

代码实际 direct path 只会从前 4 个非零 entries 里抽，因此：

```text
q_code(x | x) = 0.5
q_code([0,0,1,1,1,-1] | x) = 0.25
q_code(y | x) = 0.25
```

对 `y`：

- `reported_n_conn = 4`
- `actual_nonzero_entry_count = 6`

其中有一个真实非零 entry 指向 `x`，但它在非零 entry 列表中的位置超出了前 4 个，所以代码实际 proposal 满足：

```text
q_code(x | y) = 0
```

于是得到：

```text
q_code(y | x) = 0.25
q_code(x | y) = 0
log_corr_code(x -> y) = log 4 - log 4 = 0
```

这就是最关键的反例：

- 实际 proposal ratio 不是 1；
- 代码 correction 却把它当成 1。

## 8.2 反例 B：非法零状态

取：

```text
x0 = [0, 0, 1, 1, 1, 1]
```

对这个状态：

- `reported_n_conn = 4`
- `actual_nonzero_entry_count = 3`

也就是说第 4 个非零 entry 根本不存在。

于是代码 direct path 有：

```math
p_zero(x0) = (4 - 3) / 4 = 0.25
```

即有 `25%` 的概率质量会落到非法零状态。

这个例子必须在 benchmark 输出里单独列出来。

---

## 9. 推荐输出格式

建议 benchmark 运行后输出两个文件。

### 9.1 机器可读 JSON

例如：

- `my_run/benchmark_hamiltonianrule_jax_correctness_small_system_20260416.json`

包含：

- 系统参数
- basis state 列表
- 每个状态的 `reported_n_conn`
- 每个状态的 `actual_nonzero_entries`
- 每个状态的 actual/code support
- `Delta_corr` 统计
- detailed balance residual 统计
- stationary distribution 统计
- 反例 A/B 的完整展开

### 9.2 人读 Markdown 报告

例如：

- `my_run/benchmark_hamiltonianrule_jax_correctness_small_system_20260416.md`

按下面顺序组织：

1. 体系设置
2. 核心结论
3. support mismatch 汇总
4. zero-state risk 汇总
5. correction residual 汇总
6. detailed balance residual 汇总
7. stationary distribution 偏差
8. 反例 A
9. 反例 B
10. 对 `with_proposal beta=0 noocc` 的对照说明

---

## 10. 推荐实现方式

为了不改动库文件，benchmark 应写成独立脚本，放在 `my_run` 或单独的 benchmark 文件里。

推荐新增：

- `my_run/benchmark_hamiltonianrule_jax_correctness_small_system.py`

脚本内部只做三件事：

1. 枚举小体系全部状态。
2. 对每个状态重建当前代码实际的 `q_code(y|x)`。
3. 构造完整 MH kernel 并计算 residual。

关键点是：

- 不要调用 Monte Carlo 采样去“估计”这些量；
- 直接用穷举把它们精确算出来。

这样结果是可重复、可定位、可复核的。

---

## 11. benchmark 的主判据

这个 benchmark 最终应该给出一个简单结论。

### 11.1 如果是正确 kernel，应满足

- `n_support_mismatch_states = 0`
- `n_zero_state_risk_states = 0`
- `max_abs_delta_corr` 在浮点误差内为 0
- `max_abs_db_residual` 在浮点误差内为 0
- `TV(mu, pi)` 在浮点误差内为 0

### 11.2 如果当前 `HamiltonianRuleJax` 有问题，应表现为

- 存在 support mismatch
- 或存在 zero-state risk
- 或存在 `Delta_corr != 0`
- 或存在明显 detailed balance residual
- 或 `mu != pi`

只要其中任意一项成立，就不能再用“能量看起来接近 ED”来替代 correctness 判断。

---

## 12. 与当前 `with_proposal` 检查的关系

这个 benchmark 还能顺手回答一个重要问题：

- 当前 `with_proposal(beta=0, occupations=None, aggregate_duplicate_entries=False)`
- 到底是在“复现原版行为”，还是“复现正确的 Hamiltonian proposal”

预期结论应该分开写：

1. 它可以复现当前原版 JAX `HamiltonianRule` 的行为。
2. 但如果原版 JAX `HamiltonianRule` 本身就有 correctness 问题，那么“复现原版行为”并不等于“复现正确 proposal”。

---

## 13. 最值得先做的版本

如果只做一个最小版本，我建议顺序是：

1. 固定小体系 `_build_small_kondo_system()`
2. 只做 Case A: 常数模型 / 均匀分布
3. 先比较 Kernel A 与 Kernel C
4. 直接输出：
   - support mismatch
   - zero-state risk
   - `Delta_corr`
   - detailed balance residual
5. 再把 Kernel B 加进去，确认 `with_proposal` 中性路径与 Kernel A 一致

这个版本已经足够回答最关键的问题：

- 当前原版 JAX `HamiltonianRule` 的 kernel 到底是不是正确的 MH kernel。
