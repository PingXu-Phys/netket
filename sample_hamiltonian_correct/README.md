# sample_hamiltonian_correct

## 1. 这个目录放什么

这个目录专门集中 `HamiltonianRuleJax` / `sample Hamiltonian` 相关的 correctness 材料，避免继续散落在 `my_run` 里。

当前包含：

- `run_kondoheisenbergchain_hamiltonianrule_jax_n_conn_issue_analysis_20260416.md`
  详细分析：哪里错了、为什么错了、公式与代码如何对应、两个具体反例。
- `run_kondoheisenbergchain_hamiltonianrule_jax_correctness_benchmark_plan_20260416.md`
  小体系 correctness benchmark 方案。
- `benchmark_hamiltonianrule_jax_correctness_small_system.py`
  当前实际运行的 benchmark 脚本。
- `benchmark_hamiltonianrule_jax_correctness_small_system_20260416.json`
  benchmark 的机器可读结果。
- `benchmark_hamiltonianrule_jax_correctness_small_system_20260416.md`
  benchmark 的简要人读报告。

---

## 1.1 与实现优化草案的联动

下面这份实现草案现在已经同步加入 correctness 边界摘要，并把 detailed balance、对角自提议、重复条目 / state-level aggregation 的约束写入实现规划中：

- [run_kondoheisenbergchain_hamiltonian_with_proposal_optimization_draft_final_20260416.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_optimization_draft_final_20260416.md>)

这个联动的目的不是把 correctness 讨论搬回 `my_run`，而是让实现文档在引用 baseline、neutral path、duplicate handling 与 backward correction 时，明确以本目录的结论为边界条件。

`my_run` 中与本目录重复的三份 correctness Markdown 应以本目录版本为准，并从 `my_run` 移除：
- `benchmark_hamiltonianrule_jax_correctness_small_system_20260416.md`
- `run_kondoheisenbergchain_hamiltonianrule_jax_correctness_benchmark_plan_20260416.md`
- `run_kondoheisenbergchain_hamiltonianrule_jax_n_conn_issue_analysis_20260416.md`

## 2. 现在确认的问题在哪里

核心问题在两个文件。

### 2.1 `netket/operator/_discrete_operator_jax.py`

默认 `n_conn` 的实现拿到了 `mels`，但没有数 `mels != 0`，而是在数 `x != 0`。

当前逻辑等价于：

```python
_, mels = self.get_conn_padded(x)
nonzeros = jnp.abs(x) > 0
_n_conn = nonzeros.sum(axis=-1)
```

这里的 `_n_conn` 不是“非零 connected entries 的数量”，而是“输入构型里非零分量的数量”。

这一步是根问题。

### 2.2 `netket/sampler/rules/hamiltonian.py`

`HamiltonianRuleJax.transition()` 把上面的 `n_conn(x)` 当成 proposal 的真实 entry 数来使用：

1. `xp, mels = operator.get_conn_padded(x)`
2. `n_conn = operator.n_conn(x)`
3. 在 `[0, n_conn)` 上均匀抽 `rand_i`
4. 用 `mels != 0` 的累计编号去选第 `rand_i` 个非零 entry
5. 再用 `log n_conn(x) - log n_conn(x')` 作为 MH correction

所以一旦 `n_conn` 不是真实非零 entry 数，错误会同时进入：

- forward proposal
- backward correction
- 最终 MH acceptance

---

## 3. 为什么这个问题不是“只有重复条目”

如果只是重复条目，那么只要 proposal 与 correction 自洽，MH 仍然可以是正确的。

这里的问题更严重：`n_conn` 数错了对象。

记：

- `M(x)` = `get_conn_padded(x)` 中 `mels != 0` 的真实个数
- `c(x)` = 当前代码返回的 `n_conn(x)`

理想的 entry-level proposal 应该是：

```math
q_{ideal,entry}(p_j|x) = 1 / M(x)
```

理想 correction 应该是：

```math
log_corr_{ideal}(x -> y) = log M(x) - log M(y)
```

但当前代码实际实现的是：

```math
q_{code,entry}(p_j|x) =
\begin{cases}
1 / c(x), & j < \min(c(x), M(x)) \\
0, & j \ge c(x)
\end{cases}
```

这意味着：

- 如果 `c(x) < M(x)`，后面的真实非零 entry 永远采不到；
- 如果 `c(x) > M(x)`，会有一部分概率质量落到非法零状态；
- correction 却仍然继续使用 `log c(x) - log c(y)`。

因此当前代码不是“重复条目表达方式不同”，而是 proposal kernel 自身变了。

---

## 4. 为什么 MH 也救不回来

MH 只有在接受率里使用的 correction 等于实际 proposal 的反向/正向比值时，才保证目标分布。

要求的是：

```math
log_corr(x -> y) = log q(x|y) - log q(y|x)
```

当前代码使用的是：

```math
log_corr_{code}(x -> y) = log c(x) - log c(y)
```

但真实 proposal 是 `q_code`，不是按 `c(x)` 定义的理想 proposal。

只要存在某对状态 `(x,y)` 使得：

```math
log q_{code}(x|y) - log q_{code}(y|x) != log c(x) - log c(y)
```

当前 MH correction 就不是正确的 proposal ratio。

对于 `q_code(y|x) > 0` 但 `q_code(x|y) = 0` 的 one-way edge，这个问题最明显。

---

## 5. benchmark 已经给出的证据

小体系 benchmark 使用的是：

- `Lx = 2`
- `t = 1.0`
- `J_K = 1.0`
- `J1 = 1.0`
- `J2 = 0.0`
- `delta_z = 1.0`
- `n_fermions = 2`
- 总基态数 = `24`

当前实际结果是：

- `n_states = 24`
- `n_support_mismatch_states = 10`
- `n_zero_state_risk_states = 6`
- `max_zero_state_probability = 0.25`
- `n_one_way_edges = 4`
- `max_abs_db_residual_uniform_pi = 0.010416666666666666`
- `with_proposal(beta=0, occupations=None, aggregate_duplicate_entries=False)` 中性路径与当前原版 direct path 一致

这些结果说明：

1. 原版 JAX `HamiltonianRule` 在这个小体系上不是严格正确的 MH kernel。
2. 当前 `with_proposal` 中性路径是“复现了原版行为”，而不是“已经自动修正了原版问题”。

---

## 6. 两个最关键的反例

### 6.1 反例 A：one-way edge

取：

```text
x = [0, 0, 1, 1, -1, 1]
y = [1, 0, 0, 1, 1, 1]
```

benchmark 给出的当前代码实际 proposal 是：

```text
q_code(y|x) = 0.25
q_code(x|y) = 0
log_prob_corr_code(x->y) = log 4 - log 4 = 0
```

也就是说：

- `x -> y` 可提议；
- `y -> x` 在当前 direct path 中根本提议不到；
- 但 correction 仍然把这对状态当成 proposal ratio 为 `1`。

这已经足够破坏 detailed balance。

### 6.2 反例 B：非法零状态

取：

```text
x0 = [0, 0, 1, 1, 1, 1]
```

benchmark 给出：

```text
reported_n_conn = 4
actual_nonzero_entries = 3
p_zero = 0.25
```

这意味着当前 direct path 有 `25%` 的 proposal 质量会落到非法零状态。

---

## 7. 现在最合理的修改方案

我建议分成两层。

### 7.1 最小修复：先修 `n_conn`

优先修改：

- `netket/operator/_discrete_operator_jax.py`

把默认 `n_conn` 改成真正数 `mels != 0`：

```python
_, mels = self.get_conn_padded(x)
nonzeros = jnp.abs(mels) > 0
_n_conn = nonzeros.sum(axis=-1)
```

这是最小修复。

它的意义是：

- 让默认 `DiscreteJaxOperator.n_conn()` 回到和 `get_conn_padded()` 一致的语义；
- 直接修掉“漏 entry”与“非法零状态”的根源。

### 7.2 更稳妥的修复：让 `HamiltonianRuleJax` 自己按 `mels` 计数

建议同时修改：

- `netket/sampler/rules/hamiltonian.py`

在 `transition()` 里不要盲信 `operator.n_conn(x)`，而是：

- forward 直接用已经拿到的 `mels` 计算 `n_conn`
- backward 对 `x_proposed` 调一次 `get_conn_padded(x_proposed)`，再从 `mels_proposed` 计算 `n_conn_proposed`

也就是把当前文件里那段已经写出来的注释版 `_n_conn(mels)` 恢复成正式实现。

这样做的好处是：

- `transition()` 的 proposal 与 correction 由同一份 `mels` 驱动；
- sampler 自己不再依赖外部 `n_conn()` 是否与 `get_conn_padded()` 完全一致；
- 逻辑上更封闭，也更容易 audit。

我现在更推荐把这一步也做上，而不是只改 `_discrete_operator_jax.py`。

---

## 8. 修完原版后，还要同步哪里

### 8.1 `netket/sampler/rules/hamiltonian_with_proposal.py`

当前 `with_proposal` 里有一个 neutral-limit fast path：

- `occupations is None`
- `balance_beta == 0`
- `aggregate_duplicate_entries == False`

这条路径现在是刻意复现原版 `HamiltonianRule` 的 direct path。

因此，如果原版 `HamiltonianRule` 修正了，这里也要同步修。

否则会出现：

- 原版 baseline 已修正；
- `with_proposal(beta=0, noocc)` 还在复现旧错误路径。

所以原版修完后，这个文件也要跟着改。

### 8.2 测试与 benchmark

建议保留并继续使用这个目录里的 benchmark 脚本做回归验证。

另外建议新增或补充测试：

- `n_conn(x)` 与 `count_nonzero(get_conn_padded(x)[1])` 一致
- small-system 上不再出现 zero-state risk
- small-system 上不再出现 one-way edge
- `with_proposal(beta=0, noocc)` 与修正后的 baseline 一致

---

## 9. 推荐的实际改动文件

如果现在开始修，我建议优先顺序是：

1. `netket/operator/_discrete_operator_jax.py`
   修默认 `n_conn`
2. `netket/sampler/rules/hamiltonian.py`
   让 `transition()` 直接按 `mels` 计数，确保 sampler 侧自洽
3. `netket/sampler/rules/hamiltonian_with_proposal.py`
   同步 neutral-limit fast path
4. `test/sampler/...`
   加 small-system 回归测试

---

## 10. 当前总判断

当前阶段最准确的判断是：

- `with_proposal` 中性路径已经成功复现了原版行为；
- 但小体系 benchmark 证明，原版 JAX `HamiltonianRule` 本身存在 correctness 问题；
- 因此下一步如果继续推进 proposal 方案，最好先把原版 Hamiltonian direct path 修正掉；
- 否则后续所有“与原版一致”的验证，都只是兼容性验证，不是 correctness 验证。
