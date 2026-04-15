# Rende et al. (2026) Transformer Embedding Strategy Notes

## 文献信息
- 论文：Riccardo Rende, Alexander Nikolaenko, Luciano Loris Viteritti, Subir Sachdev, Ya-Hui Zhang, "Transformer Neural-Network Quantum States for lattice models of spins and fermions: Application to the Ancilla Layer Model"
- 版本：arXiv:2603.02316v1
- 日期：2026-03-04
- 本笔记聚焦：文中 Transformer 的 tokenization / embedding / 结构编码策略，而不是整篇论文的物理结果。

## 一句话总结
这篇文章采用的是“整站点 composite-state tokenization + 可训练 lookup embedding + attention 中注入空间结构”的策略。更具体地说，每个格点的完整局域构型先离散成一个 token，再映射成 embedding；晶格结构信息主要不是通过标准 positional embedding 加到输入上，而是通过 factored attention 和 spatial bias 注入到注意力机制中。

## 1. 局域状态如何变成 token
- ALM 每个格点的局域构型写成 `s_i = (n_{i\uparrow}, n_{i\downarrow}, S^z_{1i}, S^z_{2i})`。
- 其中 `n_{i\sigma} \in {0,1}`，两个 ancilla spin 的 `z` 分量分别取 `\pm 1/2`。
- 因而每个格点的局域 Hilbert 空间维度是 `V = 24`。
- 论文对每一种完整局域构型分配一个唯一整数标签 `t_i \in {0, 1, ..., V-1}`。
- 整个 many-body 构型因此表示成长度为 `N` 的 token 序列 `t = (t_1, ..., t_N)`。

## 2. 这篇文章的 tokenization 粒度
- 真正落地的实现是“每个格点一个 token，对应整个格点的完整局域构型”。
- 也就是说，费米占据和两个局域自旋不是分别喂给 Transformer 的三个或四个 token，而是先合并成一个 composite local state，再映射成一个离散 token。
- 这一点很重要，因为它决定了 embedding 的对象不是“单个自由度”，而是“整站点 local basis state”。
- 引言里有一句较宽泛的表述，说是把局域自由度 tokenization 后送入 Transformer；但 Section II 和 Fig. 4 的正式定义明确表明，实际实现应以整站点 token 为准。

## 3. Embedding 本体是什么
- 每个 token `t_i` 先通过一个可训练的 embedding lookup table 映射成向量 `x_i \in \mathbb{R}^d`。
- embedding 矩阵形状是 `V x d`。
- 这里 `d` 是 embedding dimension，也是网络超参数。
- 在 Appendix V.A 里，主实验使用的是 `d = 72`，同时 `n_l = 4` 层、`h = 12` 个 attention heads。
- 在这个初始阶段，`x_i` 只依赖该格点自己的局域构型，即 `x_i = x_i(s_i)`，还没有混入其他格点的信息。

## 4. 这种 embedding 的物理含义
- 由于一个 token 对应完整局域构型，所以一个 embedding 向量实际上编码了“该格点上费米占据 + 两个 ancilla spin”的联合状态。
- 也就是说，onsite 的异质自由度不是靠多个分支分别编码，而是在 token vocabulary 层面直接统一处理。
- 这样做的直接好处是：局域复合 Hilbert 空间可以被当成离散词表处理，Transformer 输入形式与 NLP 中的离散 token 序列非常接近。

## 5. 位置编码 / 晶格结构并不是标准 positional embedding
- 文中没有强调使用标准的 sinusoidal positional embedding，也没有把位置向量直接加到 `x_i` 上的明确描述。
- 他们引入晶格结构的主要位置在 attention，而不是 embedding lookup 本身。
- 文章采用的是 factored attention：注意力权重 `\alpha_{ij}` 只依赖格点索引 `i, j`，并由一个可训练的 `N x N` 矩阵参数化。
- 在此基础上，又加入了 spatial bias，使得注意力强度随格点间距离增大而衰减。
- 因此，这篇文章的“结构编码”更准确地说是“attention kernel 中的 site-index / distance bias”，而不是常见的“输入端 positional embedding”。

## 6. Embedding 之后发生了什么
- Transformer 把输入序列 `(x_1, ..., x_N)` 映射成上下文相关的输出 `(y_1, ..., y_N)`。
- 这些 `y_i` 不再只是局域标签，而是已经吸收了整条链上其他格点信息的 context-aware representations。
- 后续他们不是直接把 `y_i` 用来输出波函数标量，而是用它们去参数化 backflow 单粒子轨道：`\Phi_{i\sigma\alpha} = \sum_\beta y_{i\beta} W_{i\sigma\alpha\beta}`。
- 再把 `\Phi_{i\sigma\alpha}` reshape 成 `\Phi_{r\alpha}`，其中 `r = (i, \sigma)`，最后根据占据的费米轨道选行并取 Slater determinant。
- 所以 embedding 与 Transformer 输出是整个 backflow-determinant 构造的前端表示层，而不是独立存在的特征工程模块。

## 7. 这套 embedding 策略的关键特征
- 统一编码：spin 和 fermion 自由度通过单个 composite token 联合编码。
- 站点粒度：序列长度是 `N`，而不是“每站点多个 token”导致的更长序列。
- 词表驱动：局域复合 Hilbert 空间的所有可能 basis state 构成离散 vocabulary。
- 结构分工：onsite 结构主要由 token identity 承载，非局域晶格结构主要由 attention + spatial bias 承载。
- 无显式因子化：即便在 `J_K = 0` 的解耦极限，他们也没有显式把 spin sector 和 fermion sector 分成两个独立波函数分支；而是让统一表示自己学出需要的近似因子化结构。

## 8. 如果以后要和别的 embedding 策略比较，建议优先比这些维度
- token 粒度：full-site token、per-DOF token、patch token，还是连续物理特征直接投影。
- 词表大小标度：这里是 `V x d`，当每站点自由度增加时，`V` 可能指数增大。
- 结构信息注入位置：是在输入端加 positional embedding，还是像本文一样放进 attention bias。
- 自由度融合方式：是先分开编码再融合，还是像本文一样从一开始就联合编码。
- 下游接口：embedding 之后是直接回归波函数振幅，还是像本文一样先生成 backflow orbitals 再走 Slater determinant。
- attention 形式：本文使用 factored attention，权重不显式依赖 token 内容，而主要依赖格点索引和空间偏置。

## 9. 这篇文章策略的优点
- 对小到中等规模的复合局域 Hilbert 空间非常直接。
- 不需要手工拆分和重组多个 onsite 自由度。
- 便于把“局域异质性”和“非局域相关”分层处理。
- 对 multi-orbital / 更复杂 composite local space 的推广路径是清楚的，只要 `V` 仍然可控。

## 10. 这篇文章策略的潜在局限
- 如果每站点自由度继续增加，`V` 会很快膨胀，embedding table 的规模也会一起增长。
- 整站点 tokenization 会隐藏子自由度之间更细的 compositional structure；如果后续方法特别依赖“occupation token”和“spin token”的显式交互，这种做法可能不如 per-DOF token 透明。
- factored attention 降低了计算成本，但注意力权重不再显式依赖 token 内容；表达力更多依赖深层堆叠和值通道。

## 11. 可直接复用的抽象模板
```text
local composite state s_i
  -> discrete site token t_i in {0, ..., V-1}
  -> trainable embedding x_i = E[t_i], E in R^{V x d}
  -> Transformer with factored attention + spatial bias
  -> context-aware features y_i
  -> backflow orbitals Phi
  -> Slater determinant amplitude
```

## 12. 便于以后快速回忆的结论
- 这篇文章不是“每个自由度一个 token”的细粒度 tokenization。
- 它是“每个格点完整局域构型一个 token”的 coarse-grained tokenization。
- embedding 是一个标准可训练 lookup table。
- 空间/位置结构主要通过 attention bias 注入，而不是通过标准 positional embedding 注入。
- Transformer 输出不是直接读出波函数，而是先去生成 backflow orbitals，再进入 determinant 结构。

## 13. NetKet compatibility check for `KondoHeisenbergChain.py`
- I do not see an obvious structural bug in `KondoHeisenbergChain.py`.
- The returned Hilbert is `joint_hi = fermion_hi * local_spin_hi`, so the flattened configuration is ordered as `fermion block` followed by `local-spin block`.
- A small probe confirms that for `Lx = N`, the fermion block has length `2N`, the local-spin block has length `N`, the fermion entries are binary `0/1`, and the local-spin entries are encoded as `-1/+1`.
- The Kondo coupling and XXZ prefactors are also consistent with the convention `S^z = sigma^z / 2` used in the code.
- Therefore, the main compatibility work belongs in the variational ansatz, not in the Hamiltonian.

## 14. NetKet-compatible Scheme A: four-state site token + adaptive determinant backend
### 14.1 Core idea
Keep NetKet's external fermionic interface unchanged, but internally compress the conduction-electron occupancies on each site into a four-state token:
- `0`: empty
- `1`: up only
- `2`: down only
- `3`: double occupation

For a Kondo-Heisenberg-type model with one local spin chain, the per-site Transformer input can be built as
`h_i = E_f[site4_i] + E_spin[S_i] + E_site[i]`.
If there are multiple local-spin chains, add one embedding per spin chain:
`h_i = E_f[site4_i] + sum_m E_spin_m[S_{m,i}] + E_site[i]`.

### 14.2 Determinant side
The current repository uses an adaptive determinant backend rather than a single hard-coded choice:
- if the Hilbert fixes both fermionic spin subsectors, use a blocked determinant;
- if the Hilbert fixes only the total fermion number, use a generalized determinant over the full spin-orbital space.

In the blocked path, the backflow outputs should be sized independently:
- `Delta Phi_dn(i, alpha_dn)` with `alpha_dn = 1, ..., N_dn`
- `Delta Phi_up(i, alpha_up)` with `alpha_up = 1, ..., N_up`

This means spin-imbalanced fixed-sector states with `N_up != N_dn` are still valid. The important requirement is not `S^z_f = 0`, but matching the determinant backend to the Hilbert constraint.

In the generalized path, the model produces one orbital matrix over `r = (i, sigma)` and selects occupied rows directly from the full spin-orbital occupancy pattern.

### 14.3 Why this is the most stable interface
- It preserves the existing `SpinOrbitalFermions` occupation convention used by NetKet observables and samplers.
- It does not require changing the Hamiltonian or the Hilbert definition.
- It reduces the Transformer sequence length from `2N` to `N`, which is noticeably cheaper for attention.
- Double occupation is represented explicitly as a native local state instead of being inferred from two separate binary tokens.
- It is the cleanest bridge between the paper's site-level tokenization idea and the repository's adaptive determinant backend.

### 14.4 Limitation
- In full Kondo problems with spin flips, the physically natural choice is still the generalized determinant path.
- The blocked path is best viewed as a constrained-sector specialization, not as a replacement for the paper's unified spin-orbital determinant.

## 15. NetKet-compatible Scheme B: split `up/dn` orbitals + shared site embedding + channel embedding
### 15.1 Why shared positional embedding alone is not enough
If `up` and `down` are kept as separate binary tokens, using only a shared site positional embedding leaves an ambiguity:
- the model knows two tokens belong to the same site,
- but it does not know which token is `up` and which token is `down`.

Therefore, the minimal correction is to add a separate `channel` or `spin-type` embedding.

### 15.2 Recommended input form
For the conduction electrons:
`h_{i,sigma} = E_occ[n_{i,sigma}] + E_site[i] + E_channel[sigma]`
with `sigma in {up, down}`.

For a Kondo-Heisenberg-type model with local spins, extend the same idea to all channels:
`h_{i,c} = E_value_c[x_{i,c}] + E_site[i] + E_channel[c]`
where the channels are, for example,
- `c = 0`: fermion down
- `c = 1`: fermion up
- `c = 2`: local spin chain 1
- `c = 3`: local spin chain 2
- etc.

In this view, the local spin chain is not special; it is just another site-aligned channel with its own value embedding and channel identity.

### 15.3 Output side
After the Transformer, reshape the channel sequence back to per-site form and pool or mix the channels to obtain a site-resolved feature `y_i`.
Then:
- if the Hilbert fixes `N_dn` and `N_up`, emit blocked backflow corrections with independent `dn/up` widths;
- if the Hilbert fixes only `N_f`, emit one generalized determinant over all spin orbitals.

### 15.4 Pros and cons
Pros:
- minimal conceptual change if the current implementation already works with flattened spin-orbital tokens;
- explicit `site` and `channel` identities make the representation much cleaner than a single flat positional embedding;
- naturally extends to multiple site-aligned spin channels.

Cons:
- sequence length is larger than in Scheme A;
- double occupation is not a native token and must still be represented through the joint state of two channels;
- without an explicit channel marker, the model is under-specified.

## 16. Recommendation
If the goal is a long-term interface that is both efficient and physically clear, Scheme A should be the default.

More specifically:
- for `J_K != 0` and only fixed total fermion number, prefer the generalized determinant path;
- for fixed spin subsectors with `N_dn` and `N_up` separately constrained, a blocked determinant is still fine, but the `dn/up` backflow heads must have independent widths;
- in either scheme, channel identity should always be explicit rather than left for the network to infer from site position alone.