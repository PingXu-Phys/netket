# KondoHeisenbergChain 采样方案结论笔记

## 1. 这份笔记的目的

这份文件只记录一个问题：当前 `run_kondoheisenbergchain_transformers.py` 里的联合自旋-费米子采样到底是如何工作的，以及后续如果要继续修改 sampler，哪些方案是物理上兼容的，哪些只是数值上的辅助 proposal。

它不是完整使用手册。完整参数和运行接口见 `run_kondoheisenbergchain_transformers_guide.md`。这份笔记更偏向“改 sampler 前先统一认识”。

## 2. 联合 Hilbert 的结构

Kondo-Heisenberg 链的联合 Hilbert 不是单一费米子空间，而是

`joint_hi = fermion_hi * local_spin_hi`

其中：
- `fermion_hi` 是 `SpinOrbitalFermions`
- `local_spin_hi` 是局域自旋链
- 一个联合样本可以写成 `sigma = [sigma_f | sigma_s]`

也就是说，采样器面对的是联合分布

`p(sigma) = p(sigma_f, sigma_s) \propto |psi(sigma_f, sigma_s)|^2`

不是单独的费米子边际分布，也不是单独的自旋分布。

这件事决定了一个核心原则：

- proposal 可以分块设计
- acceptance 必须对完整联合态统一判断

## 3. 当前 run 里有两条采样路径

### 3.1 `joint-sampler = hamiltonian`

这条路径直接使用完整 Hamiltonian 的连通性来生成 proposal。

代码入口：
- `run/run_kondoheisenbergchain_transformers.py`
- `build_hamiltonian_sampler(...)`

它的物理含义是：proposal 来自完整 Hamiltonian 的非零非对角矩阵元，因此会自然包含：
- 电子 hopping
- 局域自旋交换或翻转
- Kondo 耦合项导致的“电子自旋翻转 + 局域自旋翻转”联合更新

因此，对于 full Kondo 模型，这条路径最物理，也最忠实于 Hamiltonian 本身的连通结构。

### 3.2 `joint-sampler = factorized`

这条路径先分别为两个子空间构造 proposal，再把它们组合起来。

代码入口：
- `run/run_kondoheisenbergchain_transformers.py`
- `build_factorized_sampler(...)`

它的逻辑是：
- 费米子块用 `fermion_rule`
- 自旋块用 `spin_rule`
- 再由 `TensorRule(joint_hi, (fermion_rule, spin_rule))` 组合成联合 proposal

这条路径的优点：
- 结构清晰
- 便于单独替换费米子 proposal
- 便于比较 `standard` 和 `with-proposal`

但它不是完整 Kondo 动力学的精确镜像。最关键的限制是：

- 它不会在一次 proposal 中显式构造“电子自旋翻转 + 局域自旋翻转”的耦合 Kondo move
- 它只是分别更新费米子块和自旋块，然后把结果拼起来

所以：
- 如果目标是 full Kondo 的物理一致性，优先用 `hamiltonian`
- 如果目标是研究 proposal 机制本身，特别是研究费米子 hop 的数值效率，才用 `factorized`

## 4. `TensorRule` 到底做了什么

`TensorRule` 不是新的物理更新规则，它只是 proposal 组合器。

给定
- 当前联合样本 `sigma = [sigma_f | sigma_s]`
- 费米子子 rule
- 自旋子 rule

它做的事情可以概括为：

1. 按子空间把 `sigma` 切开
2. 对 `sigma_f` 调用费米子 rule，得到 `sigma_f'`
3. 对 `sigma_s` 调用自旋 rule，得到 `sigma_s'`
4. 把它们拼回 `sigma' = [sigma_f' | sigma_s']`
5. 把各子 rule 返回的 `log_prob_corr` 相加

因此，`factorized` 的真实含义是：

- proposal 是分块生成的
- acceptance 不是分块做的，而是对完整 `sigma'` 统一计算

这点非常重要。`TensorRule` 没有把联合问题拆成两个彼此独立的 MCMC；它只是把 proposal 的生成过程模块化了。

## 5. factorized 路径下，费米子 proposal 有两种

### 5.1 `standard`

使用标准 `FermionHopRule`。

这类 proposal 的特点是：
- 在合法的占据模式之间做 hop
- 不带额外 occupation bias
- 更像均匀或结构上较简单的提案

### 5.2 `with-proposal`

使用自定义的 `FermionHopRule_with_proposal`，实现位于：
- `netket/sampler/rules/fermion_2nd_proposal.py`

这条 rule 的目标是解决一种纯费米子问题：当占据数结构高度集中时，标准 hop 往往会频繁提出质量较差的 proposal，导致接受率偏低。

它的大致做法是：
- 给每个 mode 一个目标 occupation
- 对 hop `i -> j` 赋予偏置权重
- 再通过随机噪声和数值混合避免 proposal 过于极端或退化
- 最后返回对应的 forward/backward proposal 修正项

在纯费米子模型里，这种设计是合理的，也可能明显改善 proposal 质量。

## 6. 为什么 `with-proposal` 在联合分布里会出现“不兼容感”

这里要区分三层含义。

### 6.1 数学上是否兼容

如果这个 rule 只作用在 `fermion_hi` 上，并且通过 `TensorRule` 与自旋 rule 组合，同时正确返回 `log_prob_corr`，那么从 Metropolis-Hastings 的角度看，它仍然是自洽的。

也就是说：
- 不是 MH 数学错了
- 不是 acceptance 公式失效了

### 6.2 物理上是否兼容

真正的问题在这里。

Kondo 模型的关键非对角过程之一是耦合翻转：
- 一个电子的自旋发生翻转
- 与此同时，一个局域自旋也发生相反翻转

这是联合空间里的单个物理 move。

而 `FermionHopRule_with_proposal`：
- 只看费米子块
- 只提出费米子 hop
- 不会同时改局域自旋

所以在 `factorized` 路径下，它无法单独表达 full Kondo 中最重要的那类联合更新。

### 6.3 效率上是否兼容

这也是问题来源之一。

`with-proposal` 的 bias 只建立在费米子 occupation 信息上，但联合目标分布实际是

`p(sigma_f, sigma_s)`

如果系统处在强自旋-费米子纠缠区域，真正更有效的 proposal 往往应该依赖某种条件结构，例如：

`q(sigma_f' | sigma_f, sigma_s)`

而不是只依赖 `sigma_f` 或外部给定的 occupation 轮廓。

换句话说：
- 它改善的是费米子边际 proposal
- 但 full Kondo 的难点经常来自联合相关性

因此它可能“局部有效”，但未必是联合分布里的主瓶颈解法。

## 7. 一个更具体的不兼容点：`N_up/N_dn` 守恒

如果费米子 rule 仍然按自旋分块 hop，或者等价地保持每个自旋 subsector 的粒子数不变，那么它实际上默认了：

- `N_up` 守恒
- `N_dn` 守恒

但对 full Kondo 而言，这通常不是正确的对称性。更准确的是：
- 总电子数守恒
- 联合 `S^z` 扇区守恒
- 单独的 `N_up` 和 `N_dn` 未必守恒

因此，如果 factorized 里的费米子 proposal 天然保守 `N_up/N_dn`，而自旋 proposal 只动局域自旋，那么它和 full Kondo 的真实连通结构之间就会再多出一层不匹配。

补充说明：这里讨论的是 factorized sampler 的 proposal 连通性，而不是 ansatz 本身的 determinant backend。当前 `transformer_joint_nnbf.py` 已经支持在 blocked determinant 路径下处理 `N_dn != N_up`，因此非零费米子 `S^z_f` 不再是 backflow determinant 这一层的实现瓶颈。对 `J_K != 0` 的 full Kondo，真正仍需谨慎的是 proposal 是否能够生成与 Kondo 自旋翻转相匹配的联合电子-局域自旋连通。

这也是为什么当前实现里：
- full Kondo 的默认推荐 sampler 是 `hamiltonian`
- `factorized + with-proposal` 更适合当作数值实验工具，而不是主采样器

## 8. 当前可以怎么理解这两条路径的职责

### 8.1 `hamiltonian` 的职责

适合：
- full Kondo
- 需要忠实保留 Hamiltonian 连通结构
- 需要自然包含 Kondo 耦合翻转

不适合：
- 专门隔离研究费米子 hop proposal 的数值差异

### 8.2 `factorized` 的职责

适合：
- 把费米子和自旋 proposal 分开调试
- 比较 `standard` 与 `with-proposal`
- 做 sampler 机制研究

限制：
- 它不是 full Kondo 联合更新的最自然表达
- 它不能自动把耦合翻转当成单个 proposal move 来处理

## 9. 如果后面要继续改 sampler，建议怎么改

### 9.1 最保守的建议

如果目标是正式的 full Kondo 计算：
- 继续优先使用 `joint-sampler = hamiltonian`
- 把 `factorized` 留作对照或实验模式

### 9.2 如果想保留 `with-proposal`

可以保留，但建议明确把它定位成：
- 费米子子空间 proposal 的加速器
- 不是 full Kondo 联合 proposal 的完整替代

### 9.3 更理想的下一步

如果后续真的要提高联合分布里的采样效率，更值得做的是一个新的 joint rule，例如：

`JointSpinFermionProposalRule`

它应当直接作用在 `joint_hi` 上，并显式包含多类 move：
- 同自旋电子 hop
- Kondo 耦合翻转 move
- 局域自旋 exchange 或 flip

其中：
- 电子 hop 这一类 move 可以继续复用 `with-proposal` 的 occupation bias 思想
- Kondo flip 这一类 move 则需要单独设计联合 proposal 权重

这会比单独强化费米子 hop 更符合 full Kondo 的真实结构。

## 10. 一个可直接使用的判断表

### 10.1 如果你的目标是物理一致性

用：
- `joint-sampler = hamiltonian`

原因：
- proposal 直接来源于完整 Hamiltonian
- 自动覆盖 Kondo 耦合翻转
- 不会把联合更新强行拆成两个子问题

### 10.2 如果你的目标是研究费米子 proposal

用：
- `joint-sampler = factorized`
- `fermion-sampler = standard` 或 `with-proposal`

原因：
- 这时你的研究对象就是费米子子 proposal 本身
- `TensorRule` 正好提供了干净的对照框架

但需要明确写在实验记录里：
- 这不是 full Kondo 最物理的采样方式
- 这是一个“模块化 proposal 实验平台”

## 11. 对后续修改 run 的直接影响

如果后面你准备继续改 `run` 目录下的 sampler 方案，建议按下面顺序考虑：

1. 先决定目标是“物理一致性”还是“proposal 数值优化”
2. 如果是前者，就围绕 `hamiltonian` 路径做优化
3. 如果是后者，就围绕 `factorized + TensorRule` 做对照实验
4. 只有当你准备引入真正的联合 move 时，才值得新写 `JointSpinFermionProposalRule`

一句话总结：

- `with-proposal` 在纯费米子空间里是合理的
- 放到联合 Kondo 分布里，它不是错误，而是不完整
- 真正缺失的不是 MH 修正，而是“联合自旋-费米子 move”的 proposal 机制
## 12. 我的思路

如果后面真的要把这套 sampler 往前推进，我自己的思路不是一开始就重写所有 proposal，而是分三层做。

### 12.1 第一层：先把“物理正确的基线”固定住

我的判断是，full Kondo 的主基线仍然应该是：
- `joint-sampler = hamiltonian`

原因很直接：
- 它天然包含 Kondo 耦合翻转
- 它不需要额外猜测哪些联合 move 是重要的
- 它最适合作为所有后续 proposal 方案的参照组

也就是说，后面无论你再怎么改 `factorized` 或新 joint rule，都应该先能回答一个问题：

- 它相对 `hamiltonian` 基线到底改善了什么
- 它改善的是接受率、混合速度，还是 wall-clock 效率

如果没有这条基线，后面的优化很容易变成只优化 proposal 的表面指标。

### 12.2 第二层：把 `factorized` 明确定位成“实验平台”

我不建议把 `factorized` 当成 full Kondo 的最终正式采样器，但我认为它非常适合做 proposal 研究。

原因是它有两个好处：
- 各子空间职责清楚，容易定位接受率下降到底来自费米子还是自旋部分
- 费米子 rule 可以独立替换，非常适合测试 `standard`、`with-proposal`、以及以后新的 bias 形式

所以我的倾向是：
- `factorized` 保留
- 但文档和脚本注释里始终明确，它主要是 sampler ablation / proposal benchmark 平台
- 不把它和 full Kondo 的物理保真方案混为一谈

### 12.3 第三层：真正值得新增的是 joint proposal，而不是继续单独强化 fermion hop

我对现在问题的判断是：
- `with-proposal` 已经把“纯费米子 occupation 很集中时，怎么改进 hop”这件事做到了一个合理方向
- full Kondo 里更大的缺口，不在费米子 hop 本身，而在缺少联合自旋-费米子 move

所以如果后续值得投入开发时间，我更倾向于新增一个真正的 joint rule，而不是继续在现有 fermion-only rule 上堆更多参数。

这个 joint rule 的最小版本，我会只放三类 move：
- 电子同自旋 hop
- Kondo flip
- 局域自旋 exchange 或 flip

并且用混合概率在三类 move 之间抽样。

这里面：
- 电子 hop 分支可以继续复用 `with-proposal` 的 occupation bias
- Kondo flip 分支则应该直接看当前 site 上的电子与局域自旋局部构型
- 自旋分支只负责补充局域自旋空间的探索

这样设计的好处是，旧的费米子 proposal 思想不会浪费，但它会被放进一个更正确的联合采样框架里。

### 12.4 我对实现顺序的建议

如果你后面真要改代码，我建议按这个顺序来：

1. 先继续保留 `hamiltonian` 路线，作为 full Kondo 的默认正式方案。
2. 在 `factorized` 路线里继续做 `standard` 和 `with-proposal` 的对照，把接受率、self-correlation time、单位时间有效样本数都记下来。
3. 如果确认瓶颈主要来自联合翻转缺失，而不是单独 fermion hop 太差，再开始写 `JointSpinFermionProposalRule`。
4. joint rule 的第一版先做最小闭环，不要一上来就把所有 bias 都做复杂。
5. 只有当 joint rule 的收益明确后，再考虑把 `with-proposal` 的 occupation 信息扩展成依赖局域自旋背景的 conditional proposal。

### 12.5 我最看重的评价指标

如果后面你拿这份笔记继续改 sampler，我建议不要只盯着 acceptance ratio。对这种联合系统，更有意义的是同时看：
- acceptance ratio
- 每步 proposal 的计算代价
- integrated autocorrelation time
- 单位 wall-clock 时间的有效样本数
- 最终能量估计的方差

因为有些 proposal 会让 acceptance 看起来更高，但每一步更贵，或者链混合得并没有更快。

### 12.6 一句话总结我的判断

我的核心判断是：
- ansatz 侧的 blocked determinant 现在已经可以处理 `N_dn != N_up`，因此非零费米子 `S^z_f` 本身不再构成 backflow 接口障碍
- `hamiltonian` 是 full Kondo 的物理基线
- `factorized` 是 proposal 研究平台
- `with-proposal` 是有价值的 fermion 子 proposal
- 真正的下一步，不是继续把 fermion hop 调得更花，而是把联合自旋-费米子 move 正式引入 proposal 设计