# Kondo-Heisenberg 中基于 Hamiltonian 连通性的 Proposal 方案

## 0. 与 manual 的联动说明

这份文档是对 [run_kondoheisenbergchain_sampler_manual.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_sampler_manual.md:1>) 的进一步收敛与细化。

这里给出的判断是：
- 如果严格遵守“只新增一个库文件，并只在 `metropolis.py` 增加一个工厂函数”这一边界；
- 同时又希望 proposal 尽量自然、尽量完整地反映 Kondo-Heisenberg 的真实连通性；

那么相比此前的“`JointKondoFlipRule + MultipleRules` 分支混合方案”，更自然的第一阶段实现其实是：

- 新增 `netket/sampler/rules/hamiltonian_with_proposal.py`
- 在 `netket/sampler/metropolis.py` 中新增 `MetropolisHamiltonianWithProposal(...)`

也就是说，这份文档给出的路线，会把 manual 中原先围绕 `joint_kondo_flip.py` 的那条实现思路降级为备选方案；当前更优先的单文件库实现路线，改为“基于 Hamiltonian 连通性的 occupation-biased proposal”。

## 1. 为什么这条路线更自然

相比最简单的 `FermionHopRule`，Kondo-Heisenberg 模型的真实非对角连通性包含三类过程：

1. 同自旋电子 hopping
2. Kondo 耦合引起的“电子自旋翻转 + 局域自旋翻转”联合 move
3. 局域自旋之间的 Heisenberg exchange

而最简单的 fermion hop proposal 只覆盖了第一类，并且通常只在同自旋子空间里移动电子。

因此，如果我们继续沿着“只在 fermion hop 上做 proposal bias”这条路往前走，会天然遇到两个问题：
- proposal 的支持集本身就不完整；
- 即便 occupation bias 做得很好，也仍然只是在优化 fermion 边际子问题，而不是优化 full Kondo 的联合 proposal。

从这个角度看，更自然的第一阶段方案不是再把 fermion-only rule 调得更复杂，而是：

- 直接沿用完整 Hamiltonian 的连通性；
- 然后只在这些 Hamiltonian-allowed 的 connected states 之间，用 occupation / pseudo-occupation 去构造非均匀提议分布。

这条路线的优点是：
- proposal 的支持集天然正确；
- 同时还能把你已经在 `fermion_2nd_proposal.py` 中使用的 occupation bias 思路保留下来；
- 而且改动边界正好还能压缩到“一个新 rule 文件 + 一个新 factory”。

## 2. 建议的库级实现名称

为了尽量模仿现有代码风格，我建议第一阶段使用下面这一组名称：

- 新 rule 文件：`netket/sampler/rules/hamiltonian_with_proposal.py`
- 新 rule 类：`HamiltonianRuleWithProposal`
- 新工厂函数：`MetropolisHamiltonianWithProposal(...)`

这样命名的原因是：
- 它直接对应现有 `hamiltonian.py` / `HamiltonianRule` / `MetropolisHamiltonian`
- 同时又和 `fermion_2nd_proposal.py` / `MetropolisFermionHopWithProposal` 的命名风格保持一致
- 用户一眼就能看出它是“基于 HamiltonianRule 的 with-proposal 版本”

## 3. 核心输入：两类 occupation profile

这个方案里，proposal bias 需要的输入不再只有一类 occupation，而是两类。

### 3.1 费米子 occupation profile

定义：
- `v_{i,\sigma} \in (0,1)` 表示第 `i` 个费米子轨道、电子自旋 `\sigma` 的目标 occupation

推荐支持两种输入方式：

#### 方式 A：长度为 `L`

输入：

```text
v = [v_1, v_2, ..., v_L]
```

解释：
- 自动扩展为
  - `v_{i,\uparrow} = v_i`
  - `v_{i,\downarrow} = v_i`

这对应“up 与 dn occupation 完全相同”的情形。

#### 方式 B：长度为 `2L`

输入：

```text
v = [v_{1,\downarrow}, ..., v_{L,\downarrow}, v_{1,\uparrow}, ..., v_{L,\uparrow}]
```

解释：
- up 和 dn occupation 分开指定
- 顺序上建议直接和当前 sample layout 的 fermion block 保持一致

这对应“up 与 dn occupation 不同，需要区分对待”的情形。

这正是你提出的第一个重要改进点，我认为非常合理，而且应当直接写进新 rule 的输入约定里。

### 3.2 局域自旋 pseudo-occupation profile

对局域自旋 site `i`，我们不再把它当成一个 Bernoulli occupation mode，而是把它看成一个二态变量：
- `s_i = \uparrow`
- `s_i = \downarrow`

因此更自然的输入是一个 categorical profile：

```math
u_{i,\uparrow} > 0, \quad \nu_{i,\downarrow} > 0, \quad \nu_{i,\uparrow} + \nu_{i,\downarrow} = 1
```

默认推荐：

```math
u_{i,\uparrow} = \nu_{i,\downarrow} = 1/2
```

也就是：
- 每个局域自旋 site 默认不偏向上也不偏向下；
- 如果以后需要，也可以引入 site-dependent 的 `\nu_i`。

## 4. 为什么局域自旋不能简单套用纯 fermion 的公式

这是整个方案里最关键的技术点之一。

在 `fermion_2nd_proposal.py` 中，纯 fermion hop 的未归一化权重写成：

```math
W_{i \to j} = v_j (1-v_i)
```

这个公式成立的原因是：
- fermion mode 的局域变量是 `0/1` occupation
- 我们使用的是 Bernoulli 型目标分布

但局域自旋 site 不是一个 Bernoulli 变量，而是一个**单占据二态变量**。如果我们把局域自旋硬写成两维 one-hot occupation，并机械地照搬：

```math
W_{\alpha \to \beta} \stackrel{?}{=} \nu_{\beta}(1-\nu_{\alpha})
```

在 `\nu_{\uparrow} + \nu_{\downarrow} = 1` 的约束下，会得到：

```math
W_{\uparrow \to \downarrow} = \nu_{\downarrow}^2
```

反向则是：

```math
W_{\downarrow \to \uparrow} = \nu_{\uparrow}^2
```

它们的比值为：

```math
\frac{W_{\uparrow \to \downarrow}}{W_{\downarrow \to \uparrow}} = \left(\frac{\nu_{\downarrow}}{\nu_{\uparrow}}\right)^2
```

这会把正确的细致平衡比值平方化，因此是不对的。

所以，对局域自旋的 proposal prior，正确做法不是继续使用 Bernoulli-hop 公式，而是使用 categorical 目标概率。

也就是说，当局域自旋从 `\alpha` 变到 `\beta` 时，正确的 target-side proposal weight 应当直接取：

```math
W_{\alpha \to \beta}^{\mathrm{spin}} = \nu_{\beta}
```

这会给出正确的比值：

```math
\frac{W_{\alpha \to \beta}^{\mathrm{spin}}}{W_{\beta \to \alpha}^{\mathrm{spin}}} = \frac{\nu_{\beta}}{\nu_{\alpha}}
```

这正是 categorical prior 下应有的 detailed-balance ratio。

## 5. Proposal 的辅助目标分布

为了统一推导所有 move 的权重，我们定义一个“proposal prior”，它不是最终采样目标，而是用来指导提议分布偏置的辅助分布：

```math
\pi_{\mathrm{prop}}(\sigma)
\propto
\Bigg[
\prod_{i,\sigma_e}
 v_{i,\sigma_e}^{n_{i,\sigma_e}}
 (1-v_{i,\sigma_e})^{1-n_{i,\sigma_e}}
\Bigg]
\Bigg[
\prod_i \nu_{i,s_i}
\Bigg]
```

其中：
- `n_{i,\sigma_e} \in \{0,1\}` 是 conduction electron 的 occupation
- `s_i \in \{\uparrow,\downarrow\}` 是 local spin 的状态

注意：
- 这个 `\pi_{\mathrm{prop}}` 只是 proposal prior，不是最终 VMC 目标分布
- 最终接受率仍然由波函数模方和 MH 修正共同决定

## 6. 基于 Hamiltonian 连通性的总 proposal

对当前状态 `\sigma`，用 Hamiltonian 的连通性定义 proposal 的支持集：

```math
\mathcal{C}_H(\sigma) = \{\sigma' : H_{\sigma,\sigma'} \neq 0\}
```

这一步完全模仿现有 `HamiltonianRule`：
- 不是我们自己手写 hop clusters
- 不是我们自己额外枚举 Kondo flip 或 Heisenberg exchange
- 直接以 Hamiltonian 的 `get_conn_padded` / `get_conn_flattened` 为准

然后只在这个支持集上做 bias。

## 7. 通用未归一化权重公式

设 `\sigma \to \sigma'` 是一个 Hamiltonian-allowed move。

定义：
- `\mathcal{F}_{0\to1}`：所有从空变满的 fermion modes
- `\mathcal{F}_{1\to0}`：所有从满变空的 fermion modes
- `\mathcal{S}`：所有发生局域自旋状态变化的 spin sites

则推荐的通用未归一化权重为：

```math
W(\sigma \to \sigma')
=
\prod_{m \in \mathcal{F}_{0\to1}} v_m
\prod_{m \in \mathcal{F}_{1\to0}} (1-v_m)
\prod_{(i:s_i\to s_i') \in \mathcal{S}} \nu_{i,s_i'}
```

这个公式的意义非常直接：
- 对 fermion mode，沿用 `with-proposal` 的 Bernoulli occupation 逻辑
- 对 local spin site，使用 categorical target-probability 逻辑
- 所有变化的自由度只看“目标态在 proposal prior 下多自然”

## 8. 三类具体 move 的权重

这个通用公式自动覆盖 Kondo-Heisenberg 中最重要的三类非对角过程。

### 8.1 同自旋 fermion hopping

若 move 为：
- 电子从 `(i,\sigma)` hop 到 `(j,\sigma)`
- 即 `n_{i,\sigma}: 1 \to 0`
- `n_{j,\sigma}: 0 \to 1`

则：

```math
W_{\mathrm{hop}} = v_{j,\sigma}(1-v_{i,\sigma})
```

这和现有 `fermion_2nd_proposal.py` 完全一致。

### 8.2 Kondo flip：`s_i^+ S_i^-`

若当前局部构型为：
- conduction electron: `\downarrow \to \uparrow`
- local spin: `\uparrow \to \downarrow`

则：

```math
W_{\mathrm{Kondo}}^{(\downarrow\to\uparrow)}
=
 v_{i,\uparrow}(1-v_{i,\downarrow})\, \nu_{i,\downarrow}
```

### 8.3 Kondo flip：`s_i^- S_i^+`

若当前局部构型为：
- conduction electron: `\uparrow \to \downarrow`
- local spin: `\downarrow \to \uparrow`

则：

```math
W_{\mathrm{Kondo}}^{(\uparrow\to\downarrow)}
=
 v_{i,\downarrow}(1-v_{i,\uparrow})\, \nu_{i,\uparrow}
```

### 8.4 Heisenberg exchange：`S_i^+ S_j^-`

若当前 local spin 构型为：
- `s_i: \downarrow \to \uparrow`
- `s_j: \uparrow \to \downarrow`

则：

```math
W_{\mathrm{spin-ex}} = \nu_{i,\uparrow}\,\nu_{j,\downarrow}
```

反向 `S_i^- S_j^+` 则是：

```math
W_{\mathrm{spin-ex}}' = \nu_{i,\downarrow}\,\nu_{j,\uparrow}
```

### 8.5 默认 `\nu = 1/2` 时的含义

如果我们默认：

```math
\nu_{i,\uparrow} = \nu_{i,\downarrow} = 1/2
```

则：
- 所有纯 local-spin exchange move 的自旋因子都只是常数；
- 所有 Kondo flip move 的 local-spin 因子也只是 `1/2` 常数；
- proposal bias 的主体就主要来自 conduction electron occupation profile

这和你的直觉是一致的：
- 第一阶段最重要的 bias 信息仍然在 fermion occupation 上；
- local spin 先用最中性的 `1/2,1/2` 即可。

## 9. 归一化 proposal 概率与 MH 修正

在当前状态 `\sigma` 下，对所有 Hamiltonian-connected neighbors 做归一化：

```math
Z(\sigma) = \sum_{\eta \in \mathcal{C}_H(\sigma)} W(\sigma \to \eta)
```

因此 proposal 概率为：

```math
g(\sigma \to \sigma') = \frac{W(\sigma \to \sigma')}{Z(\sigma)}
```

反向 proposal 则为：

```math
g(\sigma' \to \sigma) = \frac{W(\sigma' \to \sigma)}{Z(\sigma')}
```

最终 MH 接受率仍为：

```math
A(\sigma \to \sigma')
=
\min\left(1,
\frac{|\Psi(\sigma')|^2}{|\Psi(\sigma)|^2}
\frac{g(\sigma'\to\sigma)}{g(\sigma\to\sigma')}
\right)
```

因此需要返回的 log correction 为：

```math
\log\mathrm{corr}
=
\log g(\sigma'\to\sigma)
-
\log g(\sigma\to\sigma')
```

也就是：

```math
\log\mathrm{corr}
=
\log W(\sigma'\to\sigma)
-
\log Z(\sigma')
-
\log W(\sigma\to\sigma')
+
\log Z(\sigma)
```

这和现有 `fermion_2nd_proposal.py` 的核心结构是一致的，只是：
- 支持集从 fermion clusters 扩展成 Hamiltonian 连通集；
- weight 从纯 fermion hop 权重扩展成 joint spin-fermion weight。

## 10. 与 `fermion_2nd_proposal.py` 的关系

这条新路线不是推翻 `fermion_2nd_proposal.py`，而是把它的逻辑推广到 full Kondo 的 Hamiltonian 连通性上。

可以把两者的关系理解成：

- 旧 rule：支持集 = fermion hop clusters；权重 = `v_target (1-v_source)`
- 新 rule：支持集 = `Hamiltonian.get_conn`；权重 = “fermion Bernoulli 因子 × local-spin categorical 因子”

因此在代码风格上，最值得复用的是：
- `_safe_log(...)`
- occupation 输入校验与展开
- mixing / noise 的数值稳定处理思路
- `log_prob_corr` 的写法

## 11. 关于 noise 与 mixing 的建议

为了保持和 `with-proposal` 风格一致，我建议新方案仍然保留：
- `noise_strength`
- `mixing`

但第一阶段应当区分两类 profile 的处理方式。

### 11.1 conduction fermion profile

继续模仿现有 `fermion_2nd_proposal.py`：

```math
\bar v = (1-\lambda)v + \lambda/2
```

然后逐 mode 做 Beta 采样：

```math
\tilde v_m \sim \mathrm{Beta}(\nu \bar v_m, \nu(1-\bar v_m))
```

### 11.2 local spin profile

如果第一阶段默认 local spin 先取 `1/2,1/2`，那么实际上可以先不对它加噪声，直接用常数 profile。

原因：
- 这样实现最稳；
- 也最符合“第一阶段先让 fermion occupation bias 发挥主作用”的思路。

如果以后确实需要，也可以对 `\nu_{i,\uparrow}` 做 Beta 采样：

```math
\tilde \nu_{i,\uparrow} \sim \mathrm{Beta}(\nu_s \bar \nu_{i,\uparrow}, \nu_s(1-\bar \nu_{i,\uparrow}))
```

并定义：

```math
\tilde \nu_{i,\downarrow} = 1 - \tilde \nu_{i,\uparrow}
```

但我不建议第一版就做这一步。

## 12. 实现路径：为什么这比 `JointKondoFlipRule + MultipleRules` 更适合当前边界

如果目标是：
- 只新增一个库文件
- 只在 `metropolis.py` 加一个工厂函数
- 尽量模仿现有 `HamiltonianRule` 和 `MetropolisHamiltonian` 风格

那么这条路线比 `JointKondoFlipRule + MultipleRules` 更合适。

原因有四个：

### 12.1 它天然覆盖完整支持集

不用再额外分三条 branch 去手拼：
- fermion hop
- Kondo flip
- spin exchange

因为它们本来就都在 Hamiltonian 的 `get_conn` 里。

### 12.2 它更接近现有 `HamiltonianRule`

新 rule 可以直接模仿 `HamiltonianRuleJax` 的基本骨架：
- 用 `operator.get_conn_padded(x)` 取所有 connected states
- 只是在“均匀选一个 neighbor”这一步，改成“按 proposal weights 选一个 neighbor”

### 12.3 它更符合“只改一个库文件 + 一个 factory”的边界

不需要：
- 额外引入 `FixedRule`
- 额外引入 `MultipleRules`
- 额外在 runner 里拼装分支

整个新增逻辑都可以压在：
- `hamiltonian_with_proposal.py`
- `MetropolisHamiltonianWithProposal(...)`

里面完成。

### 12.4 它和你当前的物理直觉完全一致

你的直觉是：
- 真实支持集应来自 Hamiltonian 连通性；
- occupation bias 只是用来重新分配这些 Hamiltonian-connected proposals 的概率；
- 对 local spin site，也可以引入最中性的 pseudo-occupation，比如 `1/2,1/2`

我认为这个直觉是对的，而且比“先把 joint move 单独拆出来”更自然。

## 13. 建议的新 rule 实现框架

下面是我建议的新 rule 在库里的结构，不是逐字代码，而是实现骨架。

### 13.1 新文件

```text
netket/sampler/rules/hamiltonian_with_proposal.py
```

### 13.2 新类

```python
class HamiltonianRuleWithProposal(HamiltonianRuleBase):
    operator: DiscreteJaxOperator = struct.field(pytree_node=True)
    occupations: jax.Array
    local_spin_profile: jax.Array
    noise_strength: float = struct.field(pytree_node=False)
    mixing: float = struct.field(pytree_node=False)
```

第一阶段建议只支持 JAX operator。

原因：
- 当前最需要的就是 JAX-friendly 路线；
- 也更贴近你强调的稳健性和效率要求；
- 先不做 numba 版，避免把单文件边界搞复杂。

### 13.3 transition 的高层流程

对每个 batch state `x`：

1. `xp, mels = operator.get_conn_padded(x)`
2. 计算 nonzero mask
3. 对每个 neighbor `xp[k]`，比较 `xp[k] - x` 的变化模式
4. 按第 7 节的通用公式算 `W(x -> xp[k])`
5. 归一化得到 `g(x -> xp[k])`
6. 抽样一个 proposal neighbor
7. 对 proposal state `x'` 再算一次 `Z(x')`
8. 返回 `x'` 和 `log_prob_corr`

### 13.4 第一版不要做的优化

第一版不建议做：
- 用局部更新去手工增量维护 `Z(x')`
- 对不同 move type 写完全不同的采样子内核
- 同时支持 JAX 和 Numba 两套实现

原因：
- 第一版最重要的是 correctness
- `get_conn_padded` 已经提供了一个清楚的统一接口
- 当前连通度本来就是 `O(L)` 级别，不是指数爆炸的全连接图

## 14. 建议的新工厂函数

在 `metropolis.py` 中新增：

```python
def MetropolisHamiltonianWithProposal(
    hilbert,
    *,
    hamiltonian,
    occupations,
    local_spin_profile=None,
    noise_strength=100.0,
    mixing=0.05,
    dtype=np.int8,
    **kwargs,
):
    ...
```

### 14.1 参数建议

- `occupations`
  - 长度 `L` 或 `2L`
  - 长度 `L` 表示 up/dn 共用一个 fermion profile
  - 长度 `2L` 表示区分 up/dn

- `local_spin_profile`
  - 默认 `None`，此时自动用 `1/2, 1/2`
  - 若提供长度 `L`，表示 `\nu_{i,\uparrow}`，并自动令 `\nu_{i,\downarrow}=1-\nu_{i,\uparrow}`
  - 若以后确实有必要，也可以再扩展成支持显式 `2L` 输入

## 15. 我对第一版的最终建议

如果现在就要把方案真正落进代码库，我的推荐已经从之前的 `joint_kondo_flip.py` 方案收敛为下面这条：

- 支持集：直接用 Hamiltonian 连通性
- bias：用 fermion occupation + local spin categorical profile 共同定义
- local spin 默认 profile：`1/2,1/2`
- conduction fermion profile：支持共享 up/dn 或区分 up/dn
- 新增文件：`netket/sampler/rules/hamiltonian_with_proposal.py`
- 新增 factory：`MetropolisHamiltonianWithProposal(...)`

这一版最重要的优点是：
- 物理上最自然
- 与现有 `HamiltonianRule` 风格最接近
- 真正满足“只新增一个库文件 + 只在 `metropolis.py` 增加一个工厂函数”的约束
- 同时把你在 `with-proposal` 里已经验证过的 occupation-bias 思路完整继承下来

## 16. 一句话总结

我现在的核心判断是：

- 在 Kondo-Heisenberg 中，最自然的 proposal 不是继续把 fermion-only hop 做得更复杂；
- 而是保持 `HamiltonianRule` 的真实连通性支持集不变，再用 fermion occupation 与 local spin pseudo-occupation 去重排这些 connected states 的提议概率；
- 其中 fermion 用 Bernoulli-style `v_target (1-v_source)`，local spin 用 categorical-style `\nu_{target}`，两者组合成统一的 Hamiltonian-connected proposal weight。

## 17. 公式自检：这个 proposal 定义本身是否有问题

先给结论：这套定义在数学上是自洽的，但要作为一个稳健实现，还必须同时满足几条技术约束。

### 17.1 正确的支持集应当是非对角 Hamiltonian 连通集

推荐显式定义：

```math
\mathcal{C}_H^{\mathrm{off}}(\sigma)=\{\eta\neq\sigma:\ H_{\sigma,\eta}\neq 0\}
```

也就是说：
- proposal 的支持集只来自 Hamiltonian 的真实非对角连通性；
- 不能把对角元对应的 `\sigma' = \sigma` 也混进来；
- 否则 self-loop 会吞掉太多 proposal 概率，使链变钝。

### 17.2 proposal prior 与权重比值

定义辅助 prior：

```math
\pi_{\mathrm{prop}}(\sigma)
\propto
\Bigg[\prod_m v_m^{n_m}(1-v_m)^{1-n_m}\Bigg]
\Bigg[\prod_i \nu_{i,s_i}\Bigg]
```

对应的基础未归一化权重记作：

```math
W_0(\sigma\to\sigma')
=
\prod_{m:0\to1} v_m
\prod_{m:1\to0} (1-v_m)
\prod_{i:s_i\to s_i'} \nu_{i,s_i'}
```

则有：

```math
\frac{W_0(\sigma\to\sigma')}{W_0(\sigma'\to\sigma)}
=
\frac{\pi_{\mathrm{prop}}(\sigma')}{\pi_{\mathrm{prop}}(\sigma)}
```

因此如果再定义：

```math
g(\sigma\to\sigma') = \frac{W_0(\sigma\to\sigma')}{Z(\sigma)},
\qquad
Z(\sigma)=\sum_{\eta\in\mathcal C_H^{\mathrm{off}}(\sigma)}W_0(\sigma\to\eta),
```

那么 `log_prob_corr` 仍然可以写成：

```math
\log\mathrm{corr}
=
\log W_0(\sigma'\to\sigma)
-
\log Z(\sigma')
-
\log W_0(\sigma\to\sigma')
+
\log Z(\sigma)
```

这一点与现有 `fermion_2nd_proposal.py` 的逻辑是一致的，因此定义本身没有原则性错误。

### 17.3 local spin 不能继续套用 fermion 的 Bernoulli 公式

对 fermion hop，现有公式是：

```math
W_{i\to j}^{\mathrm{fermion}} = v_j(1-v_i)
```

这成立的前提是：
- 一个 fermion mode 是 Bernoulli 型 `0/1` occupation 变量。

但 local spin site 不是 Bernoulli occupation，而是单占据二态变量，因此它不应继续使用：

```math
\nu_{\beta}(1-\nu_{\alpha})
```

而应直接使用 categorical 目标概率：

```math
W_{\alpha\to\beta}^{\mathrm{spin}} = \nu_{\beta}
```

否则 forward/backward 比值会被错误平方化。

### 17.4 为了保持遍历性，所有 profile 都不能真的取到 0 或 1

要保证 proposal 不会把某些 Hamiltonian-allowed move 永久权重压成 0，必须保证：
- `v_m` 不取到严格的 0 或 1；
- `\nu_{i,\uparrow}`、`\nu_{i,\downarrow}` 也不取到严格的 0 或 1。

因此第一版实现仍应保留和 `with-proposal` 类似的数值稳定策略：
- clipping
- mixing
- 对 fermion profile 的 Beta noise

对 local spin profile，如果第一版默认 `1/2,1/2`，则可以先不加噪声，但仍应避免未来允许用户输入严格的 0 或 1。

## 18. 兼容性说明：必须兼容一般的费米子自旋翻转与基变换

这是当前方案里非常重要的一点。

### 18.1 不能只把 fermion 部分理解成“同自旋 hop”

如果 proposal 真正建立在 Hamiltonian 连通性上，那么 fermion 非对角过程不应被事先限制为：
- 只允许 `c_{j,\sigma}^\dagger c_{i,\sigma}` 这样的同自旋 hopping。

它必须兼容更一般的一体 fermion 变化，只要这种变化确实出现在 Hamiltonian 的连通性里。

### 18.2 通用 fermion 改变量应按“哪些 mode 变了 occupancy”来定义

也就是说，权重公式不应依赖“这个 move 被我们叫作 hop 还是 flip”，而应只依赖：
- 哪些 fermion mode 从 `0 \to 1`
- 哪些 fermion mode 从 `1 \to 0`

因此对任意一体 fermion move，都统一写成：

```math
W_{\mathrm{fermion-part}}
=
\prod_{m:0\to1} v_m
\prod_{m:1\to0} (1-v_m)
```

这条写法天然兼容下面这些过程。

### 18.3 同轨道自旋翻转

例如：

```math
c_{i,\uparrow}^\dagger c_{i,\downarrow}
```

它对应：
- `(i,\downarrow): 1\to0`
- `(i,\uparrow): 0\to1`

因此 fermion 部分的 proposal weight 自然就是：

```math
v_{i,\uparrow}(1-v_{i,\downarrow})
```

### 18.4 跨轨道自旋翻转

如果你对 fermion 做了基变换，Hamiltonian 里完全可能出现：

```math
c_{j,\downarrow}^\dagger c_{i,\uparrow}
```

这种“一个轨道的 up 变成另一个轨道的 dn”的一体过程。

这时 fermion proposal weight 也仍然天然统一为：

```math
v_{j,\downarrow}(1-v_{i,\uparrow})
```

所以只要我们是：
- 先通过 `Hamiltonian.get_conn` 得到真实 connected states；
- 再通过比较 `\sigma' - \sigma` 识别哪些 fermion modes 改变了 occupation；

那么这类基变换后的跨轨道自旋变换就不会有兼容性问题。

### 18.5 Kondo 联合翻转也只是这个统一公式的一个特例

例如 onsite Kondo flip：
- fermion `\downarrow \to \uparrow`
- local spin `\uparrow \to \downarrow`

则总 proposal weight 自然就是：

```math
W_{\mathrm{Kondo}}
=
 v_{i,\uparrow}(1-v_{i,\downarrow})\,\nu_{i,\downarrow}
```

如果 fermion 部分将来变成更一般的 basis-rotated spin-changing move，而 local spin 也同步改变，那么公式仍然完全兼容，因为它本来就是按“所有改变的自由度”逐个乘起来定义的。

### 18.6 对当前模型，费米子部分本质上需要兼容所有一体 `c_a^\dagger c_b`

因此我对实现兼容性的要求可以压缩成一句话：

- 新 rule 不能把 fermion 非对角过程硬编码成“同自旋 hopping”或“只有一种 Kondo flip 结构”；
- 它必须把所有一体 fermion 变化统一理解成：从某个已占据 mode `b` 移除电子，并在某个空 mode `a` 中加入电子；
- 无论 `a,b` 是否同轨道、是否同自旋、是否来自基变换后混合的表示，公式都应统一写成 `v_a (1-v_b)`。

这才是真正保证兼容性没有问题的关键。

## 19. 这个 proposal 能否成为一个好的提议分布

我的判断是：
- 它可以成为一个合理而自然的第一版 proposal；
- 但要把它做成“更好的 proposal”，我建议再加入一个对称的 Hamiltonian 边权因子。

### 19.1 为什么它已经比 fermion-only proposal 更自然

因为它直接满足：
- 支持集正确：只在 Hamiltonian-allowed states 中提议；
- Kondo flip 不再缺失；
- Heisenberg exchange 也自然包含；
- fermion occupation bias 仍然能继承下来。

因此它比 `fermion-only with-proposal` 更接近 full Kondo 的真实结构。

### 19.2 但只用 prior 因子，还不一定是最好的权重

如果只用：

```math
W_0(\sigma\to\sigma')
```

那么 proposal 会反映：
- 各 fermion mode 的 occupation prior
- 各 local spin 目标朝向的 pseudo-occupation

但它还没有利用一个信息：
- 不同 Hamiltonian-connected moves 的 matrix element 强弱可能本身就很不一样。

### 19.3 推荐的增强版权重

因此我更推荐把真正实现的未归一化权重写成：

```math
W(\sigma\to\sigma')
=
|H_{\sigma,\sigma'}|^{\alpha}
\prod_{m:0\to1} v_m
\prod_{m:1\to0} (1-v_m)
\prod_{i:s_i\to s_i'} \nu_{i,s_i'}
```

其中：
- `\alpha \ge 0`
- 第一版最自然可取 `\alpha = 1`
- 如果希望更保守，也可以先从 `\alpha = 0` 开始退化到纯 prior 版本

### 19.4 为什么这个增强版仍然是自洽的

因为对 Hermitian Hamiltonian，通常有：

```math
|H_{\sigma,\sigma'}| = |H_{\sigma',\sigma}|
```

因此这个对称边权因子在 forward/backward 比值中会相互抵消，不会破坏 proposal prior 所对应的 detailed-balance 比值。

也就是说：
- 它只是在同一个支持集上，进一步把 proposal 权重按物理连通强度重排；
- 但不会破坏 MH 修正的数学自洽性。

### 19.5 我对“好不好”的最终判断

如果“好”的标准是：
- 支持集物理上自然；
- 能兼容 Kondo flip 与一般 fermion spin-changing move；
- 能直接利用 natural-orbital occupation 信息；
- 在只加一个 rule 文件和一个 factory 的边界下尽量稳健；

那么这套 proposal 是好的。

如果“好”的标准进一步要求：
- 显著提升 acceptance
- 显著降低 autocorrelation time
- 显著提升 ESS / wall-clock

那么它是否足够好，还取决于：
- 输入 occupation prior 是否已经足够接近真实一体边际；
- 系统瓶颈是不是主要来自一体 occupation，而不是更强的联合相关；
- 是否引入了上面的 `|H_{\sigma,\sigma'}|^{\alpha}` 对称边权因子。

### 19.6 我对第一版公式的最终推荐

如果现在就要收敛成一个第一版可实现公式，我建议直接用下面这一版：

```math
W(\sigma\to\sigma')
=
|H_{\sigma,\sigma'}|^{\alpha}
\prod_{m:0\to1} v_m
\prod_{m:1\to0} (1-v_m)
\prod_{i:s_i\to s_i'} \nu_{i,s_i'}
```

其中：
- `\alpha = 1` 可作为自然默认值；
- `v_m` 支持长度 `L` 或 `2L`；
- 长度 `L` 时表示 up/down 共用一个 fermion profile；
- 长度 `2L` 时表示 up/down 分开；
- `\nu_{i,\uparrow},\nu_{i,\downarrow}` 第一版默认 `1/2,1/2`；
- 所有输入都应通过 mixing / clipping 保证不会把某些 Hamiltonian-allowed move 权重压成严格 0。

## 20. 当前收敛出的最佳方案

结合最新讨论，当前收敛出的最佳方案已经进一步固定为：
- 仍以完整 Hamiltonian 的非对角连通性作为 proposal 支持集
- 默认只对费米子部分引入 occupation bias
- 对 `ss / ff / sf` 三类 move 再加一个单参数软平衡层 `balance_beta`

最关键的收敛结果有三条：
- `occupations=None` 时，正确实现不是把每个 occupation 设成 1，而是走 neutral branch：`B_f \equiv 1`
- `occupations=None, balance_beta=0` 时，新规则必须严格退化回原始 `HamiltonianRule`
- `balance_beta` 用来调节三类 move 的 proposal 概率质量，使它们保持在相近量级，而不是让某一类因为连接态数过多而长期主导采样

当前完整的“最佳方案 + 接口 + 实现流程 + 注意事项”已经整理到：
- [run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md:1>)

## 21. 与最佳方案文档的联动要求

从现在开始，下面三处内容必须保持联动：
- 本文档中的公式定义与默认行为
- 新增 rule 文件 `netket/sampler/rules/hamiltonian_with_proposal.py` 的真实实现
- 新文档 [run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md](</D:/Seafile/PHD/NQS/NetKet/netket/my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_best_scheme.md:1>) 中给出的接口与调用说明

只要以下任一项发生变化：
- `occupations` 的输入顺序
- `occupations=None` 的默认语义
- `balance_beta` 的定义与默认值
- `MetropolisHamiltonianWithProposal(...)` 的函数签名

上述三处都必须同步更新。
