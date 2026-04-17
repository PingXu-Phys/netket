# Hamiltonian With Proposal Optimization Final Draft 2026-04-16

## 0. Final Tier Naming

为了避免和前一版 draft 里的命名混淆，当前最终版采用下面三层定义：

- Tier 0
  sampler 内部统一做 dense diff 提取与复用，但不强制压缩成正式 local descriptor。

- Tier 0.5 / Tier 1-sampler
  sampler 从 `x` 与 `xp` 提取固定形状 local descriptor；后续 move classification、duplicate aggregation、proposal 计算都只消费 descriptor。

- Tier 1 / Tier 1-operator
  operator 额外直接提供 local descriptor；sampler 不再从 dense `xp` 反推出 descriptor。

这三层都保持用户侧 `MetropolisHamiltonianWithProposal(...)` 接口不变。

## 1. Current Bottleneck In One Paragraph

当前 primitive operator 内核并不完全 naive。例如 fermion backend 内部已经会做：
- bit packing；
- acted-on 位点的局部翻转；
- Jordan-Wigner sign 的局部 mask 计算。

但是，当前 operator 的公共输出仍然是完整 target states `xp`。随后 sampler 又会对 `xp` 进行重复整态扫描：
- 判断 `ss / ff / sf`；
- 计算 fermion `0->1 / 1->0`；
- 做 duplicate handling；
- backward 再来一遍。

因此：
- Tier 0 / Tier 0.5 主要优化 sampler 对 dense `xp` 的后处理；
- Tier 1 才真正让 local information 跨过 `operator -> sampler` 边界。

## 2. Final Goal Of This Draft

本最终版只回答两个问题：

1. 三层 tier 各自需要修改哪些接口。
2. 当前最合理的实施顺序是什么。

结论先写在前面：
- 第一阶段建议先做 Tier 0.5，也就是只动 sampler，但把 proposal 逻辑彻底改成 descriptor-driven。
- Tier 1-operator 作为下一阶段预留能力，不建议现在直接大改主 operator 接口。

## 2.1 Correctness Boundary And Cross-Links

这份优化草案现在需要和 `sample_hamiltonian_correct` 目录下的 correctness 文档配套阅读；后文讨论的是 proposal / descriptor 的实现路线，不等于默认当前 JAX direct path 已经是正确的 MH baseline。

联动文档：
- [sample_hamiltonian_correct/README.md](</D:/Seafile/PHD/NQS/NetKet/netket/sample_hamiltonian_correct/README.md>)
- [run_kondoheisenbergchain_hamiltonianrule_jax_n_conn_issue_analysis_20260416.md](</D:/Seafile/PHD/NQS/NetKet/netket/sample_hamiltonian_correct/run_kondoheisenbergchain_hamiltonianrule_jax_n_conn_issue_analysis_20260416.md>)
- [run_kondoheisenbergchain_hamiltonianrule_jax_correctness_benchmark_plan_20260416.md](</D:/Seafile/PHD/NQS/NetKet/netket/sample_hamiltonian_correct/run_kondoheisenbergchain_hamiltonianrule_jax_correctness_benchmark_plan_20260416.md>)
- [benchmark_hamiltonianrule_jax_correctness_small_system_20260416.md](</D:/Seafile/PHD/NQS/NetKet/netket/sample_hamiltonian_correct/benchmark_hamiltonianrule_jax_correctness_small_system_20260416.md>)

当前应固定的 correctness 边界如下。

1. Detailed balance 的当前主问题是 `n_conn` 计数对象错了，不是“重复条目天然有害”。当前 JAX `n_conn` 数的是 `jnp.abs(x) > 0`，而不是 `jnp.abs(mels) > 0`。`HamiltonianRuleJax.transition(...)` 随后又用这个错误的 `n_conn(x)` 决定 `rand_i` 的抽样范围，并用 `log n_conn(x) - log n_conn(x')` 作为 MH correction。因此代码实际采样到的 `q_code(x'|x)` 与 correction 假设的 proposal ratio 不一致，详细平衡不能保证。

2. 对角项 `x -> x` 需要被当成 proposal 支持集的一部分，而不是单独排除。只要某个 connected entry 满足 `xp[j] = x` 且 `mels[j] != 0`，它就会给 `q(x|x)` 贡献概率质量。在当前 Kondo-Heisenberg 里，这类自回项主要来自 `S^z_i S^z_j` 与 `s^z_i S^z_i` 的对角矩阵元。因此任何后续优化都必须明确保留 self-proposal 的 forward support 与 backward correction；不能把“只处理状态改变的 move”误当成完整 Hamiltonian proposal。

3. 重复条目与对角重复条目本身不是 correctness bug；问题在于 proposal 定义和 correction 是否自洽。`SumDiscreteJaxOperator` 当前是直接拼接 child `ys, mels`，不会自动把相同 target state 合并。于是同一个 target state，包括 `x` 本身，都可能在 candidate list 里重复出现。如果 forward sampling 与 backward correction 采用同一个 entry-level proposal，重复条目只改变 proposal 的定义，不自动破坏 detailed balance。如果想转成 aggregated state-level proposal，就必须在采样和 correction 两边同时做 duplicate aggregation，不能只改一侧。

4. 这对 `HamiltonianRuleWithProposal` 的实现约束也应固定下来。`occupations=None, balance_beta=0, aggregate_duplicate_entries=False` 的 neutral path 目前只是“复现原版 direct path 行为”的兼容分支，不能再被当成 correctness 参考基线。后续 Tier 0 / Tier 0.5 / Tier 1 优化要以“给定某个明确 proposal 定义后，forward 与 backward 使用同一有效 `q`”为约束，而不是以“保持当前错误的 `n_conn` 行为”作为目标。

因此，下面的 tier 规划都应理解为：
- correctness 问题已在 `sample_hamiltonian_correct` 目录中单独追踪；
- 当前文档负责说明在这些边界下，如何推进 `HamiltonianRuleWithProposal` 的性能与接口演进；
- 若未来需要 bug-compatible neutral fallback，也应明确标成 regression / compatibility path，而不是正确性基线。

## 3. Interfaces To Modify By Tier

## 3.1 Tier 0

### 3.1.1 Public API changes

无。

不改：
- `netket/sampler/metropolis.py` 中 `MetropolisHamiltonianWithProposal(...)` 的用户签名；
- `aggregate_duplicate_entries` 的语义；
- operator 侧任何公开方法签名。

### 3.1.2 Files to modify

建议修改：
- `netket/sampler/rules/hamiltonian_with_proposal.py`

建议新增：
- `netket/sampler/rules/_hamiltonian_with_proposal_dense_delta.py`

Tier 0 的核心原则是：
- 尽量新增 helper 文件；
- 尽量少改原有 rule 文件中的公共结构；
- 只把 dense diff 提取逻辑集中起来复用。

### 3.1.3 Internal interfaces to add

建议在新 helper 文件中新增：

```python
def build_dense_delta_masks(x, xp, valid, fermion_size):
    ...
```

输出应至少包含：
- `fermion_changed`
- `spin_changed`
- `fermion_add_mask`
- `fermion_remove_mask`
- `same_state_mask` 或其可复用形式

建议新增：

```python
def classify_move_types_from_dense_delta(delta, valid):
    ...
```

```python
def compute_candidate_log_bias_from_dense_delta(delta, occ, eps):
    ...
```

```python
def build_same_state_mask(candidates, target, valid):
    ...
```

### 3.1.4 Existing functions to reroute

在 `hamiltonian_with_proposal.py` 中，下面这些函数不一定要改签名，但内部应改成复用统一 delta helper：

- `_classify_move_types(...)`
- `_compute_candidate_log_bias(...)`
- `_compute_chain_log_probs(...)`
- `_compute_chain_log_probs_neutral(...)`
- `_log_state_probability(...)`
- `_log_representative_entry_probability(...)`
- `HamiltonianRuleWithProposalJax.transition(...)`

### 3.1.5 Tier 0 effect

Tier 0 不会消除第一次 dense 比较。

它做到的是：
- dense `xp` 只做一次主要差分提取；
- 后续 proposal 逻辑不再各扫一遍整态。

这一步主要降低常数、减少重复 tensor 流量。

## 3.2 Tier 0.5 / Tier 1-sampler

### 3.2.1 Public API changes

仍然无。

不改：
- 用户工厂函数；
- operator 公共接口；
- Hamiltonian 构造方式。

### 3.2.2 Files to modify

建议修改：
- `netket/sampler/rules/hamiltonian_with_proposal.py`

建议新增：
- `netket/sampler/rules/_hamiltonian_with_proposal_local_descriptor.py`

如果希望与 Tier 0 兼容，也可以保留：
- `netket/sampler/rules/_hamiltonian_with_proposal_dense_delta.py`

推荐结构是：
- 一个 dense-diff helper；
- 一个 local-descriptor helper；
- 主 rule 文件只负责 orchestration。

### 3.2.3 New internal descriptor interface

建议新增一个固定形状 pytree / dataclass：

```python
@struct.dataclass
class LocalMoveDescriptor:
    move_type: jax.Array
    changed_count: jax.Array
    changed_indices: jax.Array
    fermion_add_count: jax.Array
    fermion_add_indices: jax.Array
    fermion_remove_count: jax.Array
    fermion_remove_indices: jax.Array
    spin_change_count: jax.Array
    spin_change_indices: jax.Array
    duplicate_signature: jax.Array
```

其中：
- `move_type` 建议编码为 `-1/0/1/2` 或 `0/1/2/3`；
- 所有数组必须固定形状；
- 所有 count 都是显式字段；
- `duplicate_signature` 直接服务于 duplicate aggregation。

### 3.2.4 New helper interfaces

建议新增：

```python
def extract_local_descriptor_from_dense_targets(
    x,
    xp,
    valid,
    fermion_size,
    spin_size,
):
    ...
```

```python
def compute_log_bias_from_descriptor(descriptor, occ, eps):
    ...
```

```python
def compute_move_masks_from_descriptor(descriptor):
    ...
```

```python
def compute_log_probabilities_from_descriptor(
    descriptor,
    occ,
    balance_beta,
    eps,
):
    ...
```

```python
def log_state_probability_from_signature(
    signatures,
    log_candidate_probs,
    target_signature,
    valid,
):
    ...
```

```python
def log_representative_probability_from_signature(
    signatures,
    log_candidate_probs,
    target_signature,
    valid,
):
    ...
```

### 3.2.5 Existing sampler interfaces to replace or wrap

`hamiltonian_with_proposal.py` 中下面这些函数应逐步转成 descriptor-driven：

- `_classify_move_types(...)`
- `_compute_candidate_log_bias(...)`
- `_compute_chain_log_probs(...)`
- `_compute_chain_log_probs_neutral(...)`
- `_log_state_probability(...)`
- `_log_representative_entry_probability(...)`

更具体地说：
- `_classify_move_types(...)` 应变成 descriptor 的轻包装，或直接删除；
- `_compute_candidate_log_bias(...)` 应改成基于 `fermion_add_indices` / `fermion_remove_indices`；
- duplicate handling 不再主依赖整态相等，而主依赖 `duplicate_signature`。

### 3.2.6 Optional static widths for the current Kondo use case

由于当前 rule 本身已经是针对 Kondo mixed Hilbert 写的，Tier 0.5 可以直接采用当前问题的固定小宽度：

- `KF = 2`
  单个一体 fermion process 最多改变两个 fermion modes。

- `KS = 2`
  local spin exchange 最多改变两个 local-spin sites。

- `K = 4`
  对当前 Kondo 组合过程，这是一个安全且仍然紧凑的 descriptor 上界。

如果后续发现某类 move 超过这个上界，再单独调大，而不要一开始退化成全尺寸 descriptor。

### 3.2.7 Tier 0.5 effect

Tier 0.5 的本质是：
- dense `xp` 仍然存在；
- 但 proposal 的主要计算已经不再以 dense state scan 为中心，而是以 local descriptor 为中心。

这一步已经足够支持：
- move type 分类；
- duplicate 缩并；
- proposal 权重计算；
- backward correction 的同构重算。

这是当前最推荐的第一阶段实现目标。

## 3.3 Tier 1 / Tier 1-operator

### 3.3.1 Public API changes

用户侧 API 仍然建议保持不变。

operator 主接口也不建议直接改成新的强制签名。
当前更推荐的是增加一个 optional capability，而不是修改 `get_conn_padded(...)` 的返回值。

### 3.3.2 Core operator interface to add

建议在：
- `netket/operator/_discrete_operator_jax.py`

中新增一个可选接口：

```python
def get_conn_local_descriptor(self, x, xp, mels):
    return None
```

或者：

```python
def get_conn_local_descriptor(self, x, xp, mels):
    raise NotImplementedError
```

由 sampler 侧用 capability detection 决定是否调用。

重点是：
- 不改 `get_conn_padded(...)`；
- 不强制所有 operator 立即实现；
- 允许没有该能力的 operator 自动 fallback 到 Tier 0.5 路径。

### 3.3.3 Sampler interfaces to modify

需要修改：
- `netket/sampler/rules/hamiltonian_with_proposal.py`

建议新增调用逻辑：

```python
def get_or_build_local_descriptor(operator, x, xp, mels, valid, fermion_size, spin_size):
    ...
```

逻辑为：
- 如果 operator 提供 `get_conn_local_descriptor(...)`，优先使用；
- 否则回退到 sampler-side `extract_local_descriptor_from_dense_targets(...)`。

### 3.3.4 Operator classes that would need implementations

如果要让当前 Kondo 路线真正吃到 Tier 1-operator 的收益，至少要覆盖下面这组 operator 层：

- `netket/operator/_fermion2nd/jax.py`
  对 `FermionOperator2ndJax` 提供 add/remove indices。

- `netket/operator/_local_operator/jax.py`
  对局域自旋算符提供 changed indices 或 local descriptor。

- `netket/operator/_embed/discrete_jax_operator.py`
  把子空间 local descriptor 提升到 joint Hilbert 坐标。

- `netket/operator/_prod/discrete_jax_operator.py`
  合并两个子算符的 descriptor。

- `netket/operator/_sum/discrete_jax_operator.py`
  把多个 child descriptors 连接成最终的 batch-candidate descriptor。

### 3.3.5 Minimum implementation rule for Tier 1

Tier 1 不要求一开始就让所有 operator family 都支持 local descriptor。

当前最小可行集合应是：
- `FermionOperator2ndJax`
- 当前 Kondo 路线实际用到的局域 spin JAX operator
- `EmbedDiscreteJaxOperator`
- `ProductDiscreteJaxOperator`
- `SumDiscreteJaxOperator`

只要这条链能跑通，sampler 就已经能在 Kondo workload 上测试 operator-supplied descriptor 路径。

### 3.3.6 Tier 1 effect

Tier 1 的结构性收益在于：
- sampler 不再需要从 dense `xp` 自己恢复 local descriptor；
- dense `xp` 仍可能保留用于兼容和 fallback；
- proposal 计算真正开始使用 operator 原生 local information。

这一步才是严格意义上的“让 local information 穿过 operator-to-sampler 边界”。

## 4. Interfaces That Should Stay Unchanged For Now

下面这些接口当前不建议改：

- `MetropolisHamiltonianWithProposal(...)` 用户签名
- `HamiltonianRuleWithProposal(...)` 用户签名
- `aggregate_duplicate_entries` 的用户语义
- `DiscreteJaxOperator.get_conn_padded(...)` 的返回值签名
- Hamiltonian 构造文件 `Hamiltonian/KondoHeisenbergChain.py` 的外部调用方式

原因：
- 这些接口一旦改动，影响面会比当前优化目标大得多；
- 当前三层 tier 都可以在保持它们不动的前提下推进。

## 5. Recommended Final Implementation Order

在进入下面的实施顺序前，需要先固定一个前提：sample_hamiltonian_correct 目录中整理的 correctness 问题已被单独跟踪。以下顺序讨论的是性能与接口演进，不等于认可当前 JAX direct path 已经是正确的基线。


最终建议顺序如下：

1. 先做 Tier 0.5。
   - 只动 sampler；
   - 引入 fixed-shape local descriptor；
   - proposal 全部基于 descriptor；
   - duplicate handling 从整态相等转向 descriptor signature。

2. 保留 Tier 0 的 dense-reuse helper。
   - 它是 Tier 0.5 的自然前置层；
   - 也是 fallback 的基础。

3. 在 Tier 0.5 benchmark 成立后，再做 Tier 1-operator。
   - 先加 optional `get_conn_local_descriptor(...)`；
   - 再按 Kondo 所需的最小 operator 链逐步实现。

## 6. Final Recommendation

当前最合理的判断是：

- 现在不应该直接大改 operator 主接口；
- 现在最值得做的是 Tier 0.5，也就是 sampler-side local descriptor path；
- Tier 1-operator 应该作为下一步能力扩展，而不是第一步。

一句话总结：

先把 proposal 从 dense-state-driven 改成 descriptor-driven；等这个版本 benchmark 稳定以后，再决定是否值得把 descriptor 正式下推到 operator 侧。
