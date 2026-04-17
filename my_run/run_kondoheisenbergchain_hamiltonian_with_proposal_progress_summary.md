# Kondo-Heisenberg `with_proposal` 当前进展中文总结

## 1. 当前进展处于哪一阶段

目前这条路线已经不再是概念讨论阶段，而是进入了“第一版实现已经落地，正确性基线已经建立，下一步进入复杂度优化设计”的阶段。

可以把当前状态概括为三句话：
- 数学方案已经收敛，正式采用的是 `Hamiltonian support + fermion occupation bias + ss/ff/sf move-type soft balance`。
- 代码已经落地到 NetKet sampler 中，并且保留了 neutral limit 下与原版 `HamiltonianRule` 完全一致的快速路径。
- 当前最重要的问题已经从“方案是否成立”转向“如何把实现复杂度从全状态扫描压缩到局部差分处理”。

## 2. 已经完成的实现内容

当前仓库中已经完成了以下内容：

1. 新增了 `HamiltonianRuleWithProposal` 与 `HamiltonianRuleWithProposalJax`。
2. 新增了工厂函数 `MetropolisHamiltonianWithProposal(...)`。
3. 已经支持：
   - `occupations=None` 的 neutral proposal；
   - 长度为 `L` 或 `2L` 的 fermion occupation 输入；
   - `balance_beta` 控制 `ss / ff / sf` 三类 move 的软平衡；
   - `noise_strength` 与 `mixing` 控制 noisy occupations；
   - `aggregate_duplicate_entries` 控制是否显式聚合重复 connected entries。
4. 已经写了基线一致性测试，验证在
   - `occupations=None`；
   - `balance_beta=0.0`
   的情况下，新 rule 和原版 `HamiltonianRule` 逐步转移结果一致，新 sampler 和原版 `MetropolisHamiltonian` 的采样轨迹也一致。

这意味着第一阶段最关键的正确性要求已经满足：
- 当不启用额外 bias 时，新方案不会悄悄改变原始 Hamiltonian sampler 的行为。

## 3. 当前正式采用的 proposal 结构

当前正式采用的 proposal 结构可以概括为：

### 3.1 支持集不变

proposal 的支持集严格取 Hamiltonian 的非对角 connected states，也就是：
- 不手写额外 hop 枚举器；
- 不手写额外 Kondo move 枚举器；
- 一切以 `operator.get_conn_padded(...)` 给出的真实连通性为准。

这保证了：
- proposal 不会漏掉 Hamiltonian 真正允许的 move；
- 新方案在支持集层面与原版 `HamiltonianRule` 保持一致。

### 3.2 bias 只默认作用在 fermion block

当前版本不对 local spin 额外构造 pseudo-occupation，也不拟合联合 spin-fermion 分布。
默认 bias 只作用在 fermion occupancy change 上：

```math
B_f(x\to x')
=
\prod_{m:0\to1}\tilde v_m
\prod_{m:1\to0}(1-\tilde v_m).
```

这样可以统一覆盖：
- 同自旋 hopping；
- 同轨道自旋翻转；
- 基变换后的一般一体费米子过程 `c_a^\dagger c_b`。

### 3.3 三类 move 的软平衡已经确定

对于每个 connected candidate，当前实现按实际状态差分分成：
- `ss`：只改 spin block；
- `ff`：只改 fermion block；
- `sf`：fermion 与 spin block 同时变化。

然后先在三组之间按 `balance_beta` 决定组概率，再在组内按 candidate bias 归一化采样。
这一步的作用是：
- 避免某一类 move 因为连接态数量太多而长期垄断 proposal；
- 在“保留 Hamiltonian 原始 connected-set 结构”和“人为平衡三类 move”之间做软插值。

### 3.4 重复 connected entries 已经被正视并显式参数化

文档和代码都已经接受一个事实：
- `get_conn_padded(...)` 可能返回多个不同的 connected entry，但这些 entry 最终指向同一个 target state。

当前实现因此提供了：

```python
aggregate_duplicate_entries=False
```

含义是：
- `False`：更接近原版 `HamiltonianRule` 的 entry-level 处理，成本更低；
- `True`：显式把重复 entry 对应到同一 target 的 proposal mass 做求和，得到更保守的 state-level 修正。

当前推荐默认值仍然是：
- `aggregate_duplicate_entries=False`。

## 4. 当前复杂度问题到底是什么

### 4.1 当前实现的保守复杂度上界

文档已经明确：当前实现的一个安全上界是

```math
O(B M (D + F)),
```

其中：
- `B`：chain 数；
- `M`：每个状态的 connected entries 数；
- `D`：完整状态向量长度；
- `F`：fermion block 长度。

这个上界来自 sampler 侧的额外工作，而不是 proposal 数学公式本身。

### 4.2 额外成本具体来自哪里

当前 generic 实现是围绕完整 target states 写的，因此它在每个 candidate 上都可能做以下事情：
- 比较完整 fermion block，判断 fermion 是否变化；
- 比较完整 spin block，判断 spin 是否变化；
- 扫描完整 fermion block，构造 `0->1` 和 `1->0` 的 occupation bias；
- 在需要 duplicate aggregation 时，再做整态相等性比较；
- 在 backward correction 上，对 proposed state 再完整重算一遍 proposal 分布。

所以当前最重的额外负担并不是 Hamiltonian connectivity 本身，而是 sampler 在 connectivity 结果之上做的“全状态后处理”。

### 4.3 为什么说这个上界是保守的

在 Kondo-Heisenberg 的目标问题里，一个合法 move 往往只改动很少几个局部自由度。例如：
- fermion hop 往往只改两个 fermion modes；
- fermion spin flip 也只改一个 source 和一个 target；
- local spin move 只改少数自旋位点；
- Kondo-like joint move 只改局域 fermion pattern 与局域 spin。

也就是说，从物理结构上讲，proposal 的信息需求本来就是局部的；当前之所以看起来像全状态复杂度，是因为 `get_conn_padded(...)` 只把“完整 target state”暴露给 sampler，而没有把“本次 move 改了哪些局部坐标”一起暴露出来。

## 5. 理想优化后的复杂度应该怎么理解

当前文档已经给出了更合理的目标复杂度视角：

```math
O(B M (k_move + k_f)),
```

其中：
- `k_move`：单个 move 改变的局部自由度数；
- `k_f`：单个 move 改变的 fermion 模数。

在目标模型里，这两个量都应该远小于完整状态长度。

因此，真正的优化目标不是“继续微调 proposal 公式”，而是：
- 让 sampler 直接拿到局部 delta 信息；
- 把 move 分类、fermion bias、duplicate handling 都改成只依赖局部改变量；
- 尽量避免对 `xp` 与 `x` 做整块扫描。

## 6. 原版 `HamiltonianRule` 是否也有同类复杂度问题

要区分两层：

### 6.1 底层 connectivity 成本

原版 `HamiltonianRule` 和当前 `with_proposal` 都依赖 Hamiltonian backend 给出的 connected states，因此它们共享底层 connectivity 枚举成本。
这一层不是 `with_proposal` 特有的。

### 6.2 sampler 侧的额外后处理成本

但当前 `with_proposal` 比原版多出来的复杂度问题，确实主要是新方案自己的 sampler-side 逻辑造成的。

原版 `HamiltonianRule` 做的事情比较简单：
- 枚举 connected entries；
- 在这些 entry 中均匀抽一个；
- 用 `n_conn(x)` 和 `n_conn(x')` 做 MH 修正。

而当前 `with_proposal` 在 general path 下还要额外做：
- move-type 分类；
- fermion bias 计算；
- move-type 组间平衡；
- duplicate handling；
- backward proposal 重算。

因此可以说：
- 原版也受限于 connectivity backend；
- 但当前最值得优化的那部分附加复杂度，主要是 `with_proposal` 当前 generic 实现独有的。

## 7. 当前已经达成的工程共识

目前从文档和代码看，已经形成了几个比较稳定的共识：

1. proposal 的数学结构不用推翻。
   当前公式已经足够统一，也兼容一般一体费米子过程。

2. neutral fast path 必须保留。
   当
   - `occupations=None`；
   - `balance_beta=0.0`；
   - `aggregate_duplicate_entries=False`
   时，应当直接复用原版 `HamiltonianRule` 的快速路径。

3. 默认 duplicate policy 先保持保守但便宜。
   默认仍然建议 `aggregate_duplicate_entries=False`，除非后续 benchmark 明确证明某些 operator 上必须做 state-level aggregation。

4. 下一步优化重点不在 sampler 数学层，而在 operator -> sampler 的信息流。
   核心问题不是“proposal 该不该换”，而是“sampler 今天拿到的信息太粗糙”。

## 8. 当前最合理的优化方向

### 8.1 最值得做的是 optional local-delta metadata

目前最合理的方向不是直接改主 API，而是给 operator 增加一个可选的、JAX-friendly 的 metadata helper，例如：

```python
def get_conn_delta_metadata(self, x, xp, mels):
    return {
        "move_type": ...,
        "fermion_add_indices": ...,
        "fermion_remove_indices": ...,
        "spin_change_indices": ...,
        "duplicate_signature": ...,
    }
```

有了这个 helper，sampler 就可以：
- 不再扫描整个状态来区分 `ss / ff / sf`；
- 不再扫描整个 fermion block 来构造 bias；
- 不再用整态相等性比较来做 duplicate handling；
- 在 metadata 不存在时仍然安全 fallback 到当前 generic 实现。

### 8.2 为什么这比直接改主接口更平衡

因为它同时满足三点：
- 不破坏现有 `get_conn_padded(...)` 兼容性；
- 能逐步给不同 operator 增量接入；
- 仍然允许 JAX 端走固定形状、可 `jit` / `vmap` 的数组路径。

### 8.3 什么方向暂时不该优先

下面这些方向目前不应排在最前面：
- 重新设计 proposal 公式；
- 把 duplicate aggregation 默认改成 `True`；
- 先攻 backward-path reuse；
- 用 Python 对象或可变长 metadata 承载 delta 信息。

原因是它们要么收益不直接，要么破坏 JAX-first 原则，要么正确性验证成本太高。

## 9. 推荐的下一步实施顺序

当前最稳妥的顺序应该是：

1. 保持现有公式和接口不动。
2. 先补 benchmark，把当前 general path 的主要耗时拆出来。
3. 增加 optional `get_conn_delta_metadata(...)`。
4. 先用 metadata 优化 move classification。
5. 再用 metadata 优化 fermion-bias 计算。
6. 之后再做 local duplicate signature。
7. 最后才考虑 backward-path reuse 或 operator-native delta API。

## 10. 一句话结论

如果只用一句话概括当前状态，那就是：

当前 `with_proposal` 已经完成了“正确方案 + 可运行实现 + 基线一致性验证”这一步，接下来真正的主线任务已经明确为“在不改 proposal 数学结构的前提下，把 generic 的全状态后处理改造成局部 delta 驱动的 JAX-friendly 实现”。

## 11. 进度更新（2026-04-16）

### 11.1 当前推进到哪一步

截至 2026-04-16，实际代码实现已经推进到：
- `Tier 0.5 / Tier 1-sampler` 已完成第一版落地；
- `Tier 1 / Tier 1-operator` 还没有开始改动。

也就是说：
- 当前仍然由 operator 输出完整的 `xp, mels`；
- sampler 会在拿到 `xp` 之后，立即压缩出 fixed-shape local descriptor；
- proposal 的主要后处理已经从 dense-state-driven 改成 descriptor-driven。

### 11.2 这次实际完成了什么

这次完成的核心工作是：

1. 新增了 sampler 内部 helper：
   - `netket/sampler/rules/_hamiltonian_with_proposal_local_descriptor.py`

2. 在其中实现了第一版 `LocalMoveDescriptor` 及相关内部接口：
   - `extract_local_descriptor_from_dense_targets(...)`
   - `build_transition_signature(...)`
   - `compute_move_masks_from_descriptor(...)`
   - `compute_log_bias_from_descriptor(...)`
   - `log_state_probability_from_signature(...)`
   - `log_representative_probability_from_signature(...)`

3. 修改了：
   - `netket/sampler/rules/hamiltonian_with_proposal.py`

   使 weighted proposal path 改为：
   - 先由 `x` 与 `xp` 构造 local descriptor；
   - 再基于 descriptor 做 move classification；
   - 基于 descriptor 的 add/remove indices 计算 fermion bias；
   - 基于 `duplicate_signature` 做 duplicate handling；
   - backward correction 也走同一套 descriptor 逻辑。

4. 修改了测试文件：
   - `test/sampler/test_hamiltonian_with_proposal.py`

   补充了：
   - handcrafted descriptor extraction test；
   - signature-based duplicate aggregation test；
   - descriptor path 的运行测试。

5. 新增了实现说明：
   - `my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_tier05_implementation_notes_20260416.md`

### 11.3 这一步在 tier 语义上意味着什么

这一步对应的是：
- 不是纯 `Tier 0`；
- 也还不是 `Tier 1-operator`；
- 而是中间的 `Tier 0.5 / Tier 1-sampler`。

其含义是：
- 第一次 dense 比较仍然存在；
- 但 proposal 后续主逻辑已经不再围绕完整 `xp` 展开；
- 而是围绕一个固定形状的 local descriptor 展开。

因此当前阶段已经实现了：
- 从“反复扫整态”转向“先压缩，再基于 descriptor 计算”。

### 11.4 还没有做什么

截至这次更新，下面这些事情还没有做：

1. 还没有修改 operator 公共接口。
2. 还没有增加 `get_conn_local_descriptor(...)` 之类的 operator-side capability。
3. 还没有把 local descriptor 下推到：
   - `FermionOperator2ndJax`
   - `LocalOperatorJax`
   - `EmbedDiscreteJaxOperator`
   - `ProductDiscreteJaxOperator`
   - `SumDiscreteJaxOperator`
4. 还没有开始 `Tier 1 / Tier 1-operator` 的实现。

### 11.5 当前状态的简短判断

当前状态可以概括为：
- proposal 数学结构不变；
- sampler-side 的主要后处理已经局部化；
- 下一步如果要继续向下优化，才需要判断是否值得把 descriptor 正式推到 operator 一侧。

因此目前更准确的进度判断是：
- `Tier 0.5` 已落地；
- `Tier 1` 暂缓，尚未开工。
