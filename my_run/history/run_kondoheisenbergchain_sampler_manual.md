# Kondo-Heisenberg 混合体系采样改造操作手册

## 当前采用方案

当前正式采用的方案只有一条：
- 采用 `Hamiltonian With Proposal + Move-Type Balance`
- 主文档是 [run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md:1>)
- 当前实现目标是：新增 `netket/sampler/rules/hamiltonian_with_proposal.py`，并在 `netket/sampler/metropolis.py` 中新增 `MetropolisHamiltonianWithProposal(...)`

这份 manual 下面保留的 `joint_kondo_flip.py` / `MetropolisKondoMixedProposal(...)` 内容，只作为备选和历史保留，不是当前采用方案。
## 0. 前置边界与风格要求

这份手册从这里开始，统一采用下面这组硬约束。后面的所有实现建议都必须服从这组约束。

### 0.1 代码改动边界

第一阶段我建议严格限制为两处：
- 只新增一个库文件：当前优先路线为 `netket/sampler/rules/hamiltonian_with_proposal.py`；旧的 `joint_kondo_flip.py` 方案保留为备选
- 只修改一个已有库文件：`netket/sampler/metropolis.py`，并且只在里面新增一个对应的工厂函数

这意味着第一阶段明确不做：
- 不改 `netket/sampler/rules/__init__.py`
- 不改 `run_kondoheisenbergchain_transformers.py`
- 不改已有 rule 的旧逻辑
- 不额外再新增第二个库文件

如果后续需要把这个新 sampler 暴露给特定 runner，再作为第二阶段单独处理；但第一阶段的库内改造边界就到这里为止。

### 0.2 风格要求

新实现必须尽量复用并模仿现有库文件的风格，原因有两个：
- 第一，可读性更好，后续更容易维护和 debug
- 第二，更容易继承现有实现已经验证过的效率、稳健性和 JAX 支持方式

因此第一阶段的代码风格要求是：
- 优先模仿 `netket/sampler/rules/exchange.py`
- 对有 proposal bias 的写法优先模仿 `netket/sampler/rules/fermion_2nd_proposal.py`
- 在 `netket/sampler/metropolis.py` 里的工厂函数风格优先模仿 `MetropolisFermionHop` 与 `MetropolisFermionHopWithProposal`
- 优先使用已有 rule 组合，不重复发明新框架
- 优先保持 JAX-friendly 实现，不写会破坏 jit / vmap 习惯的 Python 逻辑
- 保持联动：新增的 rule 文件、`metropolis.py` 中的工厂函数以及本文档里的接口说明必须同步更新；只要 rule 的构造参数、默认行为或调用约定发生变化，相关 factory 和文档说明也必须一起改

### 0.3 目标不是“最花”，而是“最稳”

当前采用方案的第一阶段目标是：
- 保持完整 Hamiltonian 连通性作为 proposal 支持集
- 只对费米子部分加入 occupation-biased proposal
- 用单参数 `balance_beta` 软平衡 `ss / ff / sf` 三类 move
- 同时尽量不动已有库的其他部分
- 把新功能做成一个可以像 `with-proposal` 一样被外部调用的库入口

### 0.4 当前优先路线更新

结合 [run_kondoheisenbergchain_hamiltonian_with_proposal.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal.md:1>) 的进一步收敛，当前更优先的单文件库实现路线已调整为：
- 新增 `netket/sampler/rules/hamiltonian_with_proposal.py`
- 在 `netket/sampler/metropolis.py` 中新增 `MetropolisHamiltonianWithProposal(...)`

本 manual 中围绕 `joint_kondo_flip.py` 与 `MetropolisKondoMixedProposal(...)` 的内容保留为备选分支混合方案；如果第一阶段目标是“最自然地复用 Hamiltonian 连通性”，则以新文档中的路线为优先。并且当前优先路线必须在 rule 级别兼容一般的一体 fermion 非对角过程，包括同轨道自旋翻转 `c_{i,\uparrow}^\dagger c_{i,\downarrow}`，以及基变换后可能出现的跨轨道自旋变换 `c_{j,\downarrow}^\dagger c_{i,\uparrow}`。

## 1. 文档目的

这份手册的目的，是把前一份方案文档收敛成一套严格受限的、可直接进入代码库实施的方案。

这里不再追求“大而全”的采样器重构，而是只回答下面几个问题：
- 在“只新增一个库文件 + 在 `metropolis.py` 增加一个工厂函数”的约束下，最合理的实现是什么
- 这个新实现应当如何最大化复用现有库文件
- 新 rule 和新工厂函数的接口应该怎么设计
- 哪些文件第一阶段明确不要碰

## 2. 为什么采用这条边界

之所以把边界限制得这么死，是因为这条路线最符合你现在的目标。

### 2.1 它最接近 `with-proposal` 的加入方式

你当前已经有一个很好的先例：
- `fermion_2nd_proposal.py` 作为新增 rule 文件存在
- `metropolis.py` 中增加一个对应工厂函数
- 旧的 `FermionHopRule` 和其他通用框架不用动

这正是最稳健的集成方式。

### 2.2 它能最大限度隔离新功能

如果第一阶段就去改：
- `TensorRule`
- `MultipleRules`
- `ExchangeRule`
- runner
- `__init__.py`

那么出问题时很难判断：
- 是新思路本身有问题
- 还是接线面太大导致的副作用

而现在这条边界把风险压到了最小。

### 2.3 它更符合库代码的维护方式

真正值得进入库层的第一阶段能力，不应该是“为一个特定 runner 定制一堆分支逻辑”，而应该是：
- 一个明确的新 rule
- 一个明确的新 sampler factory

这样外部脚本、runner、notebook 都可以用同一个入口调用它。

## 3. 备选方案（历史保留）

这一节及其后续围绕 `joint_kondo_flip.py` 的内容，只保留为备选路线，便于回溯思路；真正采用的实现方案请以上面的“当前采用方案”和最佳方案文档为准。

在这组约束下，我推荐的最终方案是：

1. 新增 `netket/sampler/rules/joint_kondo_flip.py`
2. 在 `netket/sampler/metropolis.py` 中新增 `MetropolisKondoMixedProposal(...)`
3. 工厂函数内部复用已有规则，把总 sampler 拼出来
4. 第一阶段不改任何 runner

一句话理解：
- 新增一个真正缺失的联合 rule
- 再在 `metropolis.py` 里用现有积木把它装成一个可直接调用的采样器入口

## 4. 这条方案里真正新增的能力是什么

第一阶段真正新增的，不是一个巨大的“全能 joint sampler”，而是下面这件更小、更清楚的事：

- 一个只负责 onsite Kondo 联合翻转的 `JointKondoFlipRule`

其他能力全部来自已有模块复用：
- fermion hop：`FermionHopRule` 或 `FermionHopRule_with_proposal`
- spin exchange：`ExchangeRule`
- 子空间冻结：`FixedRule`
- 子空间拼装：`TensorRule`
- 多分支混合：`MultipleRules`
- 总 sampler 构造：`MetropolisSampler`

因此第一阶段的新东西只有一块，其他都是已有积木。

## 5. 第一阶段明确不要碰的文件

为了严格遵守边界，第一阶段明确不要动：
- `netket/sampler/rules/__init__.py`
- `netket/sampler/rules/exchange.py`
- `netket/sampler/rules/fermion_2nd.py`
- `netket/sampler/rules/fermion_2nd_proposal.py`
- `netket/sampler/rules/tensor.py`
- `netket/sampler/rules/multiple.py`
- `run_kondoheisenbergchain_transformers.py`
- `Hamiltonian/KondoHeisenbergChain.py`

原因很直接：
- 第一阶段不是框架重构
- 第一阶段不是 runner 接线阶段
- 第一阶段只做一个新库能力的最小闭环

## 6. 新增 rule 文件：`joint_kondo_flip.py`

### 6.1 这个文件的职责

它只负责一种 move：
- onsite 电子自旋翻转
- 与此同时同 site 局域自旋反向翻转

也就是只覆盖 Hamiltonian 中真正缺失的那类联合 Kondo move。

它不负责：
- fermion hopping
- 自旋 exchange
- branch mixing
- sampler 构造

### 6.2 推荐类名

推荐直接叫：

```python
JointKondoFlipRule
```

原因：
- 名字足够直白
- 与现有 `ExchangeRule`、`HamiltonianRule`、`FermionHopRule` 风格一致
- 一眼能看出职责边界

### 6.3 推荐接口

第一阶段建议接口尽量小：

```python
class JointKondoFlipRule(MetropolisRule):
    def __init__(self, hilbert):
        ...

    def transition(self, sampler, machine, parameters, state, key, sigma):
        ...
```

不要第一版就让构造器接收太多参数，例如：
- site weights
- spin-dependent bias
- external occupation profile
- graph
- custom branch probabilities

这些都不是这个 rule 该负责的事情。

### 6.4 构造器里应该做什么

`__init__` 中建议只做静态结构校验与缓存：
- 检查 `hilbert` 是否为 `TensorHilbert`
- 检查子空间是否正好是 `(SpinOrbitalFermions, Spin)`
- 推断 `n_sites = fermion_hi.n_orbitals`
- 记录三个 block 的切片边界

建议这个 rule 尽量自描述，不依赖 runner 额外传 `n_sites`。

原因：
- 更像库级 API
- 更稳健
- 更接近已有工厂函数只传 `hilbert` 的风格

### 6.5 样本布局假设

这个 rule 必须严格遵守当前 joint sample 的扁平布局：

```text
sigma = [fermion_dn_block | fermion_up_block | local_spin_block]
```

也就是：
- `sigma[..., 0:L]` 是 `n_dn`
- `sigma[..., L:2L]` 是 `n_up`
- `sigma[..., 2L:3L]` 是 local spin

这个布局必须和当前 Hamiltonian / sector helper 保持一致，不能自作主张改。

### 6.6 合法 Kondo flip 的局部条件

对 site `i`，只允许下面两类局部构型。

#### 分支 A：`s_plus * S_minus`

合法条件：
- `n_dn[i] = 1`
- `n_up[i] = 0`
- `S_loc[i] = +1`

更新后：
- `n_dn[i] -> 0`
- `n_up[i] -> 1`
- `S_loc[i] -> -1`

#### 分支 B：`s_minus * S_plus`

合法条件：
- `n_dn[i] = 0`
- `n_up[i] = 1`
- `S_loc[i] = -1`

更新后：
- `n_dn[i] -> 1`
- `n_up[i] -> 0`
- `S_loc[i] -> +1`

### 6.7 不合法 site

下列 site 都不应进入 valid site 列表：
- 空占据
- 双占据
- 单占据但局域自旋方向不匹配

原因：
- 这些局部态上 Kondo 非对角翻转项为零
- proposal 不应凭空创造 Hamiltonian 中不存在的联合 move

### 6.8 第一版 proposal 规则

第一版建议最简单：

1. 找出所有 valid Kondo sites
2. 从 valid sites 中均匀随机选一个
3. 执行对应联合翻转
4. 在新状态上重新计算 valid site 数
5. 返回 `(sigma_p, log_prob_corr)`

为什么用这个最简单版本：
- 它最像已有 `ExchangeRule` / `HamiltonianRule` 的风格
- 正确性最容易验证
- 第一版不要在 Kondo 分支上再堆 bias

### 6.9 `log_prob_corr` 的写法

第一版推荐：

```text
q(sigma' | sigma) = 1 / N_valid(sigma)
q(sigma | sigma') = 1 / N_valid(sigma')
```

因此：

```text
log_prob_corr = log(N_valid(sigma)) - log(N_valid(sigma'))
```

如果当前 `N_valid = 0`：
- 返回原样 `sigma`
- 返回 `log_prob_corr = 0`

不要抛异常。

### 6.10 必须保守的量

这个新 rule 必须保守：
- 总电子数
- 联合 `2S^z`

并且不应人为保守：
- 单独的 `N_up`
- 单独的 `N_dn`

这正是它相对 fermion-only hop 的结构性价值所在。

## 7. 在 `metropolis.py` 中新增工厂函数

### 7.1 为什么工厂函数要放在 `metropolis.py`

因为这和现有库风格最一致。

已有模式就是：
- rule 在 `netket/sampler/rules/*.py`
- sampler factory 在 `netket/sampler/metropolis.py`

例如：
- `FermionHopRule` <-> `MetropolisFermionHop`
- `FermionHopRule_with_proposal` <-> `MetropolisFermionHopWithProposal`

所以这次也应继续遵守这个模式。

### 7.2 推荐工厂函数名

推荐：

```python
MetropolisKondoMixedProposal
```

原因：
- 名字和现有 `MetropolisFermionHopWithProposal` 风格接近
- 它表达的是一个“联合 mixed proposal sampler”
- 不把它误写成只做 Kondo flip 的工厂

### 7.3 工厂函数的职责

这个工厂函数要做的事情是：
- 校验 joint hilbert 结构
- 构造 fermion 分支
- 构造 spin 分支
- 构造 Kondo 分支
- 用 `MultipleRules` 把三条分支组装起来
- 返回 `MetropolisSampler`

也就是说：
- `JointKondoFlipRule` 只负责一个 move
- `MetropolisKondoMixedProposal` 才负责把完整 mixed sampler 拼出来

### 7.4 推荐函数签名

在“只新增一个工厂函数”的约束下，我建议签名设计成：

```python
def MetropolisKondoMixedProposal(
    hilbert,
    *,
    graph=None,
    fermion_clusters=None,
    spin_clusters=None,
    d_max=1,
    occupations=None,
    noise_strength=100.0,
    mixing=0.05,
    branch_probabilities=(0.45, 0.20, 0.35),
    dtype=np.int8,
    **kwargs,
):
    ...
```

### 7.5 为什么这样设计

这个签名背后的考虑是：
- `hilbert` 作为主入口，保持和现有工厂一致
- `occupations=None` 时自动退化为标准 fermion hop
- `occupations` 非空时自动启用 `FermionHopRule_with_proposal`
- 这样就不需要在第一阶段额外再新增第二个工厂函数

这其实是对“只允许一个工厂函数”的约束做出的折中。

如果没有这个约束，更像库风格的做法其实会是：
- 一个 `MetropolisKondoMixed`
- 一个 `MetropolisKondoMixedWithProposal`

但既然你现在要求只加一个工厂函数，那就把是否使用 occupation bias 收进同一个 factory 的参数逻辑里。

### 7.6 工厂函数内部的组装逻辑

推荐内部结构：

1. 如果 `occupations is None`：
   - 用 `FermionHopRule`
2. 如果 `occupations is not None`：
   - 用 `FermionHopRule_with_proposal`
3. spin 分支统一用 `ExchangeRule`
4. Kondo 分支用 `JointKondoFlipRule`
5. 用 `TensorRule + FixedRule` 做 hop / spin 两个子空间分支
6. 用 `MultipleRules` 混合三条分支
7. 返回 `MetropolisSampler`

### 7.7 推荐的内部骨架

推荐工厂函数内部按下面的骨架组织：

```python
hop_branch = TensorRule(hilbert, (fermion_rule, FixedRule()))
spin_branch = TensorRule(hilbert, (FixedRule(), spin_rule))
kondo_branch = JointKondoFlipRule(hilbert)

rule = MultipleRules(
    [hop_branch, spin_branch, kondo_branch],
    branch_probabilities,
)

return MetropolisSampler(hilbert, rule, dtype=dtype, **kwargs)
```

### 7.8 关于 `graph`、`fermion_clusters`、`spin_clusters`

为了最小改动，我建议第一阶段支持：
- 传入一个统一 `graph`
- 或者分别传 `fermion_clusters` / `spin_clusters`

推荐优先逻辑：
- fermion 分支优先读 `fermion_clusters`，否则读 `graph`
- spin 分支优先读 `spin_clusters`，否则读 `graph`

这样既保持灵活性，也不需要为不同分支再新增第二个 factory。

## 8. 新实现必须如何模仿现有库风格

这是你这次特别强调的要求，我把它单独展开。

### 8.1 文件头与导入风格

推荐直接模仿：
- `exchange.py`
- `fermion_2nd_proposal.py`
- `metropolis.py`

包括：
- 版权和 license 头注释
- import 排列方式
- docstring 风格
- `__repr__` 的书写方式

原因：
- 这样后续读代码时不会有“这像外挂脚本，不像库代码”的感觉

### 8.2 JAX 实现风格

在 `transition(...)` 里推荐严格遵守已有 rule 的风格：
- 用 `jax.random.split`
- 用 `jnp.asarray`
- 用 `jax.vmap` 做 chain 维度映射
- 用 `jnp.where` / `.at[...]` 更新状态
- 不在核心 transition 路径里写 Python 层 for 循环去遍历 chains
- 不使用 Python list 收集每条链的状态
- 不引入 callback

这部分最应该模仿：
- `ExchangeRule.transition(...)`
- `FermionHopRule_with_proposal.transition(...)`

### 8.3 静态信息与构造器风格

推荐在 `JointKondoFlipRule` 中只保存少量静态信息，例如：
- `n_sites`
- block 边界

如果需要标记非 pytree 配置，优先模仿 `fermion_2nd_proposal.py` 的静态字段风格。

### 8.4 数值稳健性

第一版必须保证：
- 当没有合法 Kondo site 时，不报错
- `log_prob_corr` 不产生 `nan`
- 返回状态 dtype 与输入样本 dtype 保持一致
- branch 概率和 sampler dtype 处理风格模仿已有 factory

### 8.5 效率观

第一版不要为了“看起来高级”而写复杂优化。

优先顺序应当是：
1. 正确
2. 稳健
3. 结构清楚
4. 再考虑局部优化

原因：
- 这个 rule 本身局部性很强
- 第一版主要风险来自逻辑正确性，而不是算得不够快

## 9. 为什么第一阶段不改 runner

在当前边界下，我建议第一阶段完全不改 `run_kondoheisenbergchain_transformers.py`。

原因：
- 你要求库代码改动面尽量小
- runner 接线本质上是第二层消费逻辑
- 只要 `metropolis.py` 里已有新的 factory，runner 以后随时可以调用
- 现在先把库层接口稳定下来更重要

也就是说，第一阶段的交付目标不是：
- “马上让现有 runner 暴露一个新 CLI 选项”

而是：
- “先把库里新增一个像 `MetropolisFermionHopWithProposal` 那样的可调用能力”

## 10. 在当前边界下，第一阶段如何验证

既然第一阶段不新增第二个库文件，也不改 runner，那么验证方式也要跟着收敛。

### 10.1 推荐验证方式

第一阶段建议用：
- 临时 notebook
- 临时脚本
- 交互式小系统 smoke test

去验证：
- factory 能否成功构造 sampler
- `JointKondoFlipRule` 是否保持总电子数
- 是否保持联合 `2S^z`
- 是否不会因为无合法 site 而崩溃

### 10.2 为什么第一阶段不把测试也写进库里

因为你当前给出的边界是：
- 只新增一个库文件
- 只改 `metropolis.py`

如果严格遵守这个边界，那么第一阶段就不再新增 `test/...` 文件。

这不是说测试不重要，而是说：
- 在这组边界下，正式测试文件属于第二阶段
- 第一阶段先把库接口最小闭环做出来

## 11. 第二阶段才做什么

等第一阶段接口稳定后，第二阶段再考虑：
- 给 runner 增加 `mixed-proposal` 入口
- 增加正式测试文件
- 公开 branch weight CLI
- 做 conditional occupation bias

也就是说，第一阶段只做一个最小、稳健、像库功能的新增点；后续的消费层和增强层全部延后。

## 12. 第一阶段的具体操作清单

如果按这份手册动手，顺序建议严格如下。

### 第 1 步

新增：
- `netket/sampler/rules/joint_kondo_flip.py`

并且只实现：
- `JointKondoFlipRule`
- 最小构造器
- 最小 `transition(...)`
- `__repr__`

### 第 2 步

修改：
- `netket/sampler/metropolis.py`

只新增：
- `MetropolisKondoMixedProposal(...)`

不要顺手改：
- 其他 factory
- 其他 docstring
- 不相关格式

### 第 3 步

用临时脚本验证：
- `occupations=None` 时是否能正确退化到标准 fermion hop 分支
- `occupations` 非空时是否能正确切换到 `FermionHopRule_with_proposal`
- `JointKondoFlipRule` 是否真的参与 proposal
- 总电子数和联合 `2S^z` 是否保持

### 第 4 步

如果第一阶段一切稳定，再单独讨论：
- 要不要改 runner
- 要不要补正式测试文件

## 13. 一句话结论

在你现在给出的约束下，最合理、最稳健、也最像现有库风格的实现方案是：
- 只新增 `netket/sampler/rules/joint_kondo_flip.py`
- 只在 `netket/sampler/metropolis.py` 里新增一个 `MetropolisKondoMixedProposal(...)` 工厂函数
- 新工厂函数内部完全复用现有 `FermionHopRule` / `FermionHopRule_with_proposal`、`ExchangeRule`、`FixedRule`、`TensorRule`、`MultipleRules`
- 第一阶段不改 runner，不改 `__init__.py`，不改其他旧 rule

这就是最接近你所说“像 `with-proposal` 一样加入新功能”的落地方式。






