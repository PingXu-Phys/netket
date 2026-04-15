# Kondo-Heisenberg 混合体系采样改造方案

## 1. 核心结论

- NetKet 本身可以正确处理 `SpinOrbitalFermions * Spin` 这样的混合 Hilbert 空间；费米子反对易的符号问题是在算符层处理的，不是当前 sampler 出错的根源。
- 当前真正的问题不在 Metropolis-Hastings 数学是否成立，而在 proposal 的连通性是否和 full Kondo 的真实非对角过程相匹配。
- 只为 fermion hop 设计的 occupation-biased proposal 放到混合体系中有天然缺陷：它只能改善 fermion 子空间的边际 hop 质量，不能补出缺失的“电子自旋翻转 + 局域自旋翻转”联合 move。
- 因此，后续改造的优先级应当是：
  1. 先保证正确联合扇区内的遍历性；
  2. 再考虑利用费米子 occupation 信息提高 hop 分支的数值效率；
  3. 并且尽量复用现有代码，避免一开始就重写全部 sampler。
- 如果目标是 full Kondo 的正式计算，当前默认物理基线仍应继续是 `joint-sampler = hamiltonian`。
- 如果目标是在尽量小改动下，把当前 factorized 路径推进到“更合理、更稳健的联合 proposal 平台”，我最推荐的方案不是继续强化 fermion-only hop，而是引入最小的联合分支：`JointKondoFlipRule`，再用 `MultipleRules` 组织多类 move。

## 2. 现状判断

### 2.1 NetKet 是否能安全处理这种混合 Hilbert

结论：能处理，问题不在这里。

原因：
- 联合 Hilbert 本来就是显式写成 `joint_hi = fermion_hi * local_spin_hi`。
- sampler 的接受率是对完整联合态 `sigma = (sigma_f, sigma_s)` 统一计算的，并没有把联合分布错误拆开。
- 费米子的反对易符号由 `FermionOperator2nd` 的算符实现负责，在 `get_conn` / Jordan-Wigner parity 这一层处理，而不是靠 sampler rule 手工补偿。
- 因此，只要 proposal 本身和 `log_prob_corr` 写对，MH 数学上是自洽的。

所以当前需要担心的不是：
- “混合 Hilbert 会不会让 fermion 反对易处理错”；
- “TensorRule 会不会把联合采样拆坏”。

当前真正需要担心的是：
- proposal 能不能覆盖 full Kondo 里真正重要的联合更新；
- proposal 是否无意中保守了不该保守的量；
- proposal 是否在正确守恒扇区内保持足够的遍历性。

### 2.2 当前 factorized 路径的根本缺陷

当前 factorized 路径的本质是：

`TensorRule(joint_hi, (fermion_rule, spin_rule))`

这意味着：
- fermion 子空间按自己的规则提议；
- local-spin 子空间按自己的规则提议；
- 最后把两个 proposal 拼成一个联合 proposal。

这个结构在“做 proposal 机制实验”时是清楚的，但在 full Kondo 场景里有一个根本缺口：

full Kondo 的关键非对角过程不是“fermion move + spin move 的简单并列”，而是一个真正的联合 move：
- 电子自旋翻转；
- 同 site 的局域自旋反向翻转；
- 二者必须在一次 proposal 中一起出现。

而当前 factorized 结构做不到这一点。

### 2.3 为什么只改 fermion hop 不够

`FermionHopRule_with_proposal` 的改进方向本身没有问题。

它的价值在于：
- 在 pure fermion 场景下，根据 occupation 信息改善 hop proposal 的质量；
- 在 mixed 场景下，也仍然可以作为 fermion 子 proposal 的加速器使用。

但它解决不了下列问题：
- 它不看局域自旋背景；
- 它不能产生 Kondo 耦合翻转；
- 它天然更像是在优化 `q(sigma_f' | sigma_f)`，而不是优化真正需要的联合 proposal。

所以我的判断是：
- 这个 rule 不是“错误的”；
- 但它在 mixed Kondo 分布里天然是“不完整的”。

## 3. 设计原则

后续 sampler 改造我建议遵守下面五条原则。

### 3.1 原则一：遍历性优先于 proposal 精修

先回答“能不能在正确联合扇区里走遍重要连通分量”，再去回答“hop 的接受率还能不能更高”。

原因：
- 如果 proposal 本身缺少关键联合 move，occupation bias 调得再漂亮，也只是在错误或不完整的连通结构里局部优化。
- 对 full Kondo，缺失 Kondo flip 比 hop 权重不够聪明更严重。

### 3.2 原则二：保守正确的守恒量，不保守多余的量

对 `J_K != 0` 的 full Kondo，应当保守的是：
- 总电子数；
- 联合 `2S^z` 扇区。

不应被硬编码额外保守的是：
- 单独的 `N_up`；
- 单独的 `N_dn`。

原因：
- full Kondo 的电子自旋和局域自旋之间可以交换磁量子数；
- 如果 proposal 把 `N_up/N_dn` 锁死，就会把真实连通图人为切碎。

### 3.3 原则三：优先做最小联合补丁，而不是立刻重写大一统 rule

最优先补的是“缺失的联合 Kondo move”，而不是先把所有 hop/spin/conditional bias 一起写成一个超大 rule。

原因：
- 当前最明显的结构性缺口是 joint Kondo flip；
- 先补这个缺口，收益最明确；
- 这样也最利于验证每一步改动到底带来了什么改进。

### 3.4 原则四：保留 `hamiltonian` 作为物理基线

无论后续做多复杂的 proposal 优化，`hamiltonian` 路线都应保留为基线。

原因：
- 它天然忠实于完整 Hamiltonian 的连通结构；
- 它自动包含 Kondo flip；
- 它是判断新 proposal 是否真的改进了“有效采样效率”而不是只改进了表面 acceptance 的参照组。

### 3.5 原则五：尽量复用现有成熟模块

推荐优先复用：
- `FermionHopRule` / `FermionHopRule_with_proposal`；
- `TensorRule`；
- `FixedRule`；
- `ExchangeRule`；
- `MultipleRules`。

原因：
- 这些模块已经在 NetKet 结构里存在；
- 可以把新增代码局限在一个很小的 joint rule 上；
- 更利于后续调试和回退。

## 4. 方案比较

下面把我认为值得认真考虑的几条路线按“改动量、稳健性、物理一致性、是否利于复用现有 occupation proposal”来比较。

### 4.1 方案 A：继续以 `hamiltonian` 路径作为正式主线

思路：
- 正式 full Kondo 计算继续默认使用 `joint-sampler = hamiltonian`。
- `factorized` 只保留为 proposal 研究和 ablation 平台。

优点：
- 物理上最稳健。
- 不需要重新设计联合 move。
- 自动继承完整 Hamiltonian 的连通性。
- 自动包含 Kondo 耦合翻转。
- 实现改动最小，风险最低。

缺点：
- 不能直接利用你现在专门为 fermion hop 设计的 occupation-biased proposal。
- 不方便单独研究“hop proposal 是否真的改善了数值效率”。
- 如果未来瓶颈真在 proposal 质量本身，这条路线给出的调参空间较小。

适用场景：
- full Kondo 正式算例；
- 优先保证物理一致性，而不是优先研究 sampler 机制。

我的评价：
- 这是必须保留的基线。
- 但它不是“研究新 proposal”的最方便平台。

### 4.2 方案 B：保持当前 factorized 结构，只继续强化 fermion hop proposal

思路：
- 保持 `TensorRule(joint_hi, (fermion_rule, spin_rule))` 的大框架不变；
- 继续围绕 `FermionHopRule_with_proposal` 做 occupation bias、conditional bias、参数调节等优化。

优点：
- 代码改动最小。
- 直接复用你现有的 `fermion_2nd_proposal.py`。
- 对 pure fermion 子问题的研究非常方便。

缺点：
- 不能补出真正缺失的联合 Kondo move。
- 只改 fermion hop，无法解决 full Kondo 的核心连通性缺口。
- 如果 fermion rule 保守 `N_up/N_dn`，就会把 full Kondo 的联合扇区切碎。
- 如果 spin 分支用 `LocalRule`，还可能离开正确的联合 `S^z` 扇区。

为什么我不推荐把它当成正式方案：
- 因为它是在一个结构上不完整的 proposal 框架里继续精修局部细节；
- 这对 full Kondo 主问题来说优先级不对。

我的评价：
- 可以继续保留；
- 但应明确定位为“子 proposal 实验平台”，不应继续把它当作 full Kondo 主路线的主要候选。

### 4.3 方案 C：推荐方案，做一个“分支混合式联合 proposal”

思路：
- 不再用一个 `TensorRule(fermion_rule, spin_rule)` 同时更新两个子空间；
- 而是改成在几类 move 分支之间抽样，每一步只执行一种 move 类型；
- 这些分支由 `MultipleRules` 组织。

最小版本我建议只放三类分支：
- hop 分支：只做 fermion hop；
- spin 分支：只做 local-spin exchange；
- Kondo 分支：做 onsite 的联合 spin-fermion flip。

一个最小骨架可以写成：

```python
hop_branch = TensorRule(joint_hi, (fermion_rule, FixedRule()))
spin_branch = TensorRule(joint_hi, (FixedRule(), spin_exchange_rule))
kondo_branch = JointKondoFlipRule(joint_hi)

rule = MultipleRules(
    [hop_branch, spin_branch, kondo_branch],
    probabilities=[1/3, 1/3, 1/3],
)
```

其中：
- `fermion_rule` 可以直接复用 `FermionHopRule_with_proposal`；
- `spin_exchange_rule` 推荐优先用 `ExchangeRule`，而不是 `LocalRule`；
- 只需要新增一个 `JointKondoFlipRule`。

优点：
- 这是我认为“最小改动”和“补足关键缺口”之间最平衡的方案。
- 只新增一个真正新的联合 rule，其他都复用现有模块。
- fermion occupation proposal 不会浪费，而是自然变成 hop 分支的一部分。
- 结构非常清楚，后续 benchmark 容易定位收益到底来自哪条分支。
- 比直接重写一个大一统 joint rule 更稳健、更容易调试。

缺点：
- 仍然需要新写一个 `JointKondoFlipRule`。
- 需要决定各分支的混合概率。
- 相对 `hamiltonian` 路径，它仍然是“人为设计的联合 proposal”，不是完整 Hamiltonian 连通性的逐项镜像。

为什么这是我最推荐的路线：
- 当前最关键的结构缺口就是缺少 joint Kondo move；
- 这个方案正好只补这个缺口，不会把系统一次性改得过大；
- 同时还能保留你现有的 occupation-biased hop 思路。

### 4.4 方案 D：直接写一个单体化的 `JointSpinFermionProposalRule`

思路：
- 写一个新的 joint rule，内部自己管理 hop / spin / Kondo flip 三类 move；
- 所有 proposal 权重、条件信息、`log_prob_corr` 都在一个 rule 里统一完成。

优点：
- 最灵活。
- 最容易以后扩展成真正依赖局域自旋背景的 conditional proposal。
- 长期看，可能是表达能力最强的联合 sampler 方案。

缺点：
- 改动量最大。
- 最难调试。
- 最难验证 `log_prob_corr` 是否完全正确。
- 第一版就写这种 rule，风险显著高于方案 C。

我的评价：
- 这是可以做的长期目标；
- 但不适合作为第一步。

### 4.5 方案 E：在现有 fermion proposal 上直接引入依赖自旋背景的 conditional occupation bias

思路：
- 不只给每个 fermion mode 一个静态 `occupation`；
- 而是让 proposal occupation 或 hop 权重显式依赖当前局域自旋背景。

优点：
- 如果系统在强自旋-费米子纠缠区域，理论上可能更有效；
- 比静态 occupation profile 更贴近联合分布的真实条件结构。

缺点：
- 没有 joint Kondo move 时，这个改进仍然不解决核心连通性问题；
- 参数更多、调试更难、收益更不透明；
- 第一阶段就做，容易把问题复杂化。

我的评价：
- 这是方案 C 或 D 稳定之后才值得考虑的第二阶段增强；
- 不是现在最优先的事情。

## 5. 为什么我最终推荐方案 C

我推荐方案 C，不是因为它最“花”，而是因为它最符合你给出的三个要求：
- 先保证遍历性；
- 改动尽量小；
- 尽量稳健。

### 5.1 它先补的是结构性缺口，而不是局部性能细节

当前最大问题不是 hop 分布还不够聪明，而是 full Kondo 中缺少真正的 joint move。

方案 C 直接对准这个缺口：
- hop 分支负责电子运动；
- spin 分支负责局域自旋空间探索；
- Kondo 分支负责电子自旋与局域自旋之间的联合翻转。

### 5.2 它能最大限度复用现有代码

它不是推倒重来，而是：
- 保留 `FermionHopRule_with_proposal`；
- 保留 `TensorRule`；
- 保留 `ExchangeRule`；
- 保留 `MultipleRules`；
- 只新增一个最小的 `JointKondoFlipRule`。

这意味着：
- 你的现有工作不会浪费；
- 新增代码面最小；
- 发生问题时也容易定位。

### 5.3 它比“一步到位的大一统 joint rule”更稳健

第一版 sampler 改造最怕的不是“功能还不够全”，而是“结构一下子太大，结果什么都难验证”。

方案 C 把系统拆成清晰分支后：
- 每条分支的职责明确；
- 每条分支都能单独 benchmark；
- 是否需要进一步升级到真正单体化 joint rule，也可以在后面再决定。

### 5.4 它给 occupation-biased fermion proposal 一个正确的位置

我的判断不是“with-proposal 没价值”，而是：
- 它的正确角色是 hop 分支的局部加速器；
- 不是 full Kondo 联合 sampler 的全部。

方案 C 正好把它放回到最合适的位置上。

## 6. 推荐实施顺序

我建议按下面顺序推进，而不是一口气重写全部 sampler。

### 6.1 第一步：固定 baseline

保留并继续使用：
- `joint-sampler = hamiltonian`

用途：
- full Kondo 正式计算基线；
- 所有新 proposal 的参照组。

### 6.2 第二步：明确降级当前 factorized 路径的定位

把当前：
- `factorized + standard`
- `factorized + with-proposal`

明确标记为：
- proposal 研究平台；
- ablation 平台；
- 不作为 `J_K != 0` 的正式主 sampler。

### 6.3 第三步：新增 `JointKondoFlipRule`

第一版只做最小 onsite joint move。

建议规则：
- 只在同一个 site 上工作；
- 只对“单占据电子 + 一个局域自旋”这种 Kondo 项真正可作用的局部构型提议翻转；
- 采用最简单的均匀选 site 方案；
- `log_prob_corr` 先按“前后可行动 site 数”的比值来做。

这样做的原因：
- 先保证 correctness；
- 不一开始就在 Kondo flip 分支上叠加复杂 bias。

### 6.4 第四步：把 factorized 的总 rule 改成 `MultipleRules`

推荐第一版结构：
- hop 分支：`TensorRule((fermion_rule, FixedRule()))`
- spin 分支：`TensorRule((FixedRule(), ExchangeRule()))`
- Kondo 分支：`JointKondoFlipRule`

为什么 spin 分支优先推荐 `ExchangeRule` 而不是 `LocalRule`：
- `ExchangeRule` 更适合保守局域自旋总磁化；
- fermion 不变时，它不会把链带出当前联合 `S^z` 扇区；
- 对 full Kondo 的“稳健默认方案”更安全。

### 6.5 第五步：做 benchmark，而不是只看 acceptance

我最建议看的指标是：
- acceptance ratio；
- integrated autocorrelation time；
- ESS / wall-clock；
- 最终能量估计方差；
- proposal 单步代价。

原因：
- 有些 proposal 看起来 acceptance 更高，但每步更贵；
- 真正重要的是单位时间内得到多少有效样本。

### 6.6 第六步：只有在确认值得之后，再做 conditional occupation proposal

如果方案 C 的 joint sampler 已经稳定，后面再考虑：
- occupation 不再只是静态输入；
- 而是升级为依赖局域自旋背景的 conditional bias。

这一步应当放在后面，原因是：
- 它是效率增强，不是结构补洞；
- 没有先补 joint Kondo move，做这一步意义有限。

## 7. 各方案的最终判断

### 7.1 如果目标是 full Kondo 的正式主方案

推荐：
- 继续用方案 A 作为默认正式路线。

原因：
- 最稳健；
- 最物理；
- 已经包含 joint Kondo flip。

### 7.2 如果目标是“在尽量小改动下，做一个更合理的联合 proposal 平台”

推荐：
- 采用方案 C。

原因：
- 它先解决缺失 joint move 这个根问题；
- 同时最大化复用你已经写好的 fermion proposal；
- 风险显著小于直接上方案 D。

### 7.3 如果目标只是继续研究 fermion hop 的 occupation bias 思路

可以：
- 保留方案 B，继续做对照实验。

但必须明确：
- 这不是 full Kondo 的正式主路线；
- 这只是 fermion 子 proposal 的实验平台。

## 8. 一句话总结我的建议

我的最终建议是：

- 不要再把“继续强化 fermion-only hop”当成 full Kondo sampler 的主要前进方向；
- 正式主线继续保留 `hamiltonian`；
- 真正值得做的最小改动，是在联合 Hilbert 上补一个 `JointKondoFlipRule`，然后用 `MultipleRules` 把 hop / spin / Kondo 三类 move 组织起来；
- 这样既能保证遍历性的基础，又能把你现有的 occupation-biased hop proposal 放到一个更合理的位置上继续发挥作用。
