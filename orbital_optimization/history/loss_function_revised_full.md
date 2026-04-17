# 基于自然轨道 / 1DM 信息的损失函数设计（完整版）

> 文档整理说明：
> 旧版 `loss_function.md` 的核心内容已经被本文件与 `loss_function_revised_minimal.md` 覆盖。
> 本文件保留为完整理论主文档，旧版不再单独维护。
>
> 阅读建议：
> 如果你现在只是想先跑通实现，先看 `loss_function_revised_minimal.md`。
> 如果你想比较原版 / soft / barrier 三个代码版本，先看 `optimize_no_variant_comparison.md`。

## 0. 文档目的

本文档是在原始 `loss_function.md` 的基础上，对损失函数设计做一次更系统的整理。目标不是简单罗列更多项，而是把以下几个问题讲清楚：

1. 原文件中每一项到底在约束什么；
2. 哪些项应该视为**主目标**，哪些项更适合作为**正则 / 约束**；
3. 结构代价函数 \(\mathcal S\) 的确切含义是什么，它能表达什么、不能表达什么；
4. “轨道对”为什么是一个核心中间层对象；
5. 为什么推荐将“保持自然轨道占据结构”视为约束或 barrier，而把“简化变换后哈密顿量”视为主优化目标；
6. 在利用 1DM 时，应当如何看待“从自然轨道出发”与“从格点轨道出发”两条路线；
7. 给出一个**完整版**推荐 loss，以及一个更接近实现的伪代码框架。

这份文档比最小版更完整，保留了更多讨论细节、更多可选项，以及与原文件的逐项比较。

---

## 1. 原始文件中的总体结构

原文件采用

\[
\mathcal L(V)=\lambda_\gamma C_\gamma+\lambda_t C_t+\lambda_{\mathrm{int}} C_{\mathrm{int}}+\lambda_{\mathrm{loc}} C_{\mathrm{loc}}
\]

其中：

- \(U_0\)：参考态 1-RDM 的自然轨道矩阵；
- \(V\)：在自然轨道空间内继续优化的块对角旋转矩阵；
- \(U=U_0V\)：最终优化轨道；
- 块间混合由自动分块规则直接禁止，因此原实现中除了 loss 之外，已经包含一层更强的**硬约束**。

这一定义的优点是：结构清晰、计算上可行、并且和当前代码实现高度一致。

但从优化视角看，原形式也有几个问题：

1. 参数较多，初期调参负担较重；
2. \(C_t\) 与 \(C_{\mathrm{int}}\) 虽然都在描述“哈密顿量复杂度”，但分成两项后需要单独权衡；
3. \(\mathcal S\) 描述的是**形状**（集中度与短程性），不直接约束**总强度**；
4. \(C_{\mathrm{loc}}\) 若直接取平均 IPR 作为正号 loss，需要检查符号方向是否正确；
5. \(C_{\mathrm{int}}\) 中若使用旋转后的占据数 \(\tilde n_a\) 构造 \(\Xi_{ab}\)，则可能和“保持 occupation 结构”的目标发生竞争。

---

## 2. 统一记号与基本对象

### 2.1 1-RDM 与自然轨道

设参考态的 1-RDM 为

\[
\gamma = U_0\,\mathrm{diag}(n_1,\dots,n_M)\,U_0^\dagger.
\]

其中 \(n_a\) 是自然轨道占据数，按从大到小排序。

自然轨道的核心信息有两类：

1. **轨道形状**：即本征矢 \(U_0\)；
2. **占据谱结构**：即本征值 \(n_a\)。

在优化里，这两类信息不应混为一谈。

- 占据谱结构主要告诉我们：哪些轨道近满、哪些近空、哪些 active；
- 轨道形状则未必必须被原封不动保留。

这是本文后面建议“把 1DM 视为子空间与权重信息来源，而不一定把全局 NO 视为唯一优化起点”的原因。

---

### 2.2 最终轨道与优化变量

设

\[
U=U_0V,
\]

其中 \(V\) 是在 NO 空间内的正交/酉旋转。若使用硬分块，则 \(V\) 仅允许在给定子块内旋转。

这样做的含义是：

- 先利用 1DM 找到一个“谱结构清楚”的坐标系；
- 再在该坐标系内寻找更适合哈密顿量表示的轨道。

---

### 2.3 轨道对：整个框架的核心中间层对象

很多复杂度指标天然不是单轨道量，而是**两轨道关系**的量。于是我们引入轨道对 \((a,b)\)。

轨道 \(a\) 在格点表象中的波函数记作

\[
\phi_a(i)=U_{ia}.
\]

则：

- 单体耦合由 \(t'_{ab}\) 给出；
- active/frozen 混合由 \(|V_{ab}|^2\) 或切空间元素 \(|K_{ab}|^2\) 描述；
- interaction exposure 可由 \(\Lambda_{ab}\) 或 \(W_{ab}\) 描述。

因此，优化的核心并不只是“单个轨道是否局域”，而是：

> 复杂度到底分布在哪些轨道对通道上。

图论上可以这样理解：

- 轨道是节点；
- 轨道对是边；
- 边权 \(W_{ab}\) 表示该 pair-channel 的重要性；
- 结构代价函数 \(\mathcal S\) 则在优化“这张边权图的形状”。

---

## 3. 原文件中各项的详细含义

### 3.1 \(C_\gamma\)：占据结构保持项

原定义：

\[
C_\gamma=
\frac{\|\bigl(V^\dagger\,\mathrm{diag}(\mathbf n)\,V\bigr)_{\mathrm{off}}\|_F^2}
{\sum_i n_i^2}.
\]

#### 物理意义

初始 NO 基底下，1-RDM 本来就是对角的；若继续旋转，旋转后的 1-RDM 为

\[
\gamma_{\mathrm{rot}}=V^\dagger\,\mathrm{diag}(\mathbf n)\,V.
\]

因此 \(C_\gamma\) 实际上是在度量：

> 旋转后的 1-RDM 偏离“对角形态”的程度。

若 \(V=I\)，则 \(C_\gamma=0\)。任何将 occupation 差异明显的轨道混起来的旋转，都会让 \(C_\gamma\) 增大。

#### 小旋转展开

若 \(V\approx e^K\)，其中 \(K\) 为反厄米矩阵，则近似有

\[
(\gamma_{\mathrm{rot}})_{ab}\approx (n_a-n_b)K_{ab},
\]

从而

\[
C_\gamma\propto \sum_{a<b}(n_a-n_b)^2|K_{ab}|^2.
\]

这说明：

- 占据数差小的轨道混合，惩罚较弱；
- 占据数差大的轨道混合，惩罚较强。

#### 在优化中的角色

\(C_\gamma\) 更适合作为**正则 / soft constraint**，而不是主优化目标。它告诉优化器：

- 可以离开 NO 基底；
- 但不要以严重破坏 occupation 结构为代价。

#### 与原文件比较

这一项的定义是合理的，应保留。主要建议是：

- 不要让 \(\lambda_\gamma\) 过大，否则优化会过早黏回 NO 基底；
- 若已有硬分块，则 \(C_\gamma\) 应主要承担**块内 regularization** 的角色，而不是承担全部“防混合”任务。

---

### 3.2 \(C_t\)：单体部分复杂度项

原定义：

\[
C_t=\mathcal S\!\left(|t'_{ab}|^2\right),
\qquad
 t'=V^\dagger t_{\mathrm{NO}}V,
\qquad
 t_{\mathrm{NO}}=U_0^\dagger t U_0.
\]

#### 物理意义

这项描述的是：

> 在优化后的轨道表示中，单体 hopping 的耦合图是否更集中、是否更短程。

换言之，\(C_t\) 在问：

- 重要的 hopping 是否只落在少数 pair 上；
- 这些 pair 是否主要是“近”的。

#### 在优化中的作用

最小化 \(C_t\) 会推动 \(V\) 使得单体哈密顿量更接近一种“结构化稀疏”形式：

- 非零元素更集中；
- 长程耦合减少；
- 更利于后续表示与截断。

#### 主要局限

\(C_t\) 只通过 \(\mathcal S\) 约束“形状”，不直接约束“总系数预算”。

例如：

- 三个近邻项各为 100；
- 三个近邻项各为 0.01；

若分布形状一致，则 \(\mathcal S\) 会近似给出相同结果。

#### 改进建议

更完整的版本建议写成

\[
C_t^\star=
\eta_t\,\widetilde{\|t'_{\mathrm{off}}\|}_{1,w}
+(1-\eta_t)\,\mathcal S\!\left(|t'_{ab}|^2\right),
\]

其中：

- 第一项为 weighted smooth-\(L_1\) 范数，控制总预算；
- 第二项保留原来的结构偏好。

---

### 3.3 \(C_{\mathrm{int}}\)：相互作用复杂度项

原定义：

\[
\Lambda_{ab}=
\sum_{ij}S_{ij}|U_{ia}|^2|U_{jb}|^2,
\]

\[
\Xi_{ab}=n_a(1-n_b)+n_b(1-n_a),
\]

\[
W_{ab}=\Lambda_{ab}\,\Xi_{ab},
\qquad
C_{\mathrm{int}}=\mathcal S(W_{ab}).
\]

#### \(\Lambda_{ab}\) 的含义

\(\Lambda_{ab}\) 衡量轨道对 \((a,b)\) 在相互作用图上的几何曝光度。它反映的是：

- 轨道 \(a\) 的概率云在格点 \(i\) 上的权重；
- 轨道 \(b\) 的概率云在格点 \(j\) 上的权重；
- 以及格点间相互作用 \(S_{ij}\) 的强度。

因此 \(\Lambda_{ab}\) 是一个 pair-level 的几何量：

> 轨道对 \((a,b)\) 在相互作用网络中到底有多“暴露”。

#### \(\Xi_{ab}\) 的含义

\(\Xi_{ab}\) 是一个占据活性因子。它表明轨道对 \((a,b)\) 在粒子-空穴散射意义上是否活跃。

- 一个近满、一个近空：\(\Xi_{ab}\) 大；
- 两个都近满或都近空：\(\Xi_{ab}\) 小；
- 两个都在 active 区域：\(\Xi_{ab}\) 也可能较大。

#### \(W_{ab}\) 的含义

\[
W_{ab}=\Lambda_{ab}\Xi_{ab}
\]

同时结合了：

- 轨道对的相互作用几何暴露度；
- 轨道对的占据活性。

所以 \(W_{ab}\) 可以理解为：

> 这条 interaction pair-channel 作为复杂度来源到底有多重要。

#### 主要局限与关键修正

原文件已经指出：若 \(\Xi_{ab}\) 使用旋转后的占据数 \(\tilde n_a\)，则优化器可能通过“涂抹 occupation”来间接降低 \(C_{\mathrm{int}}\)。这会和“保持 occupation 结构”的目标形成竞争。

因此更保守、更稳的写法是固定为初始 NO 占据：

\[
\Xi_{ab}^{\mathrm{NO}}=
 n_a^{\mathrm{NO}}(1-n_b^{\mathrm{NO}})
+n_b^{\mathrm{NO}}(1-n_a^{\mathrm{NO}}).
\]

此时只让 \(\Lambda_{ab}\) 随 \(V\) 变化，而不让 \(\Xi_{ab}\) 也跟着变化。

#### 更完整的推荐形式

\[
C_{\mathrm{int}}^\star=
\eta_{\mathrm{int}}\,\widetilde{\|W\|}_{1,w}
+(1-\eta_{\mathrm{int}})\,\mathcal S(W),
\qquad
W_{ab}=\Lambda_{ab}\Xi_{ab}^{\mathrm{NO}}.
\]

这样兼顾：

- 总 interaction budget；
- interaction 通道的结构集中度与短程性。

---

### 3.4 \(C_{\mathrm{loc}}\)：轨道局域性项

原文件写作

\[
C_{\mathrm{loc}}=\langle \mathrm{IPR}_a\rangle_a.
\]

其中常见定义为

\[
\mathrm{IPR}_a=\sum_i|U_{ia}|^4.
\]

#### 问题所在

按通常定义：

- 越局域，IPR 越大；
- 越离域，IPR 越小。

因此若总 loss 是被最小化的，而 \(C_{\mathrm{loc}}\) 又以正号加入：

\[
\mathcal L\supset +\lambda_{\mathrm{loc}}\langle\mathrm{IPR}\rangle,
\]

那么优化器实际上会偏向**减小 IPR**，即使轨道更离域。这和“惩罚轨道过于离域”的文字意图相反。

#### 更稳的替代形式

建议改成 spread 型局域性指标：

\[
C_{\mathrm{loc}}^{\mathrm{spread}}=
\frac1M\sum_a
\Bigl(\langle r^2\rangle_a-|\langle r\rangle_a|^2\Bigr),
\]

其中

\[
\langle r\rangle_a=\sum_i|U_{ia}|^2r_i.
\]

这样最小化 \(C_{\mathrm{loc}}^{\mathrm{spread}}\) 才确实是在鼓励局域。

若必须使用 IPR，建议改为：

\[
C_{\mathrm{loc}}^{\mathrm{IPR}}=-\frac1M\sum_a\mathrm{IPR}_a
\]

或

\[
C_{\mathrm{loc}}^{\mathrm{invIPR}}=
\frac1M\sum_a\frac{1}{\mathrm{IPR}_a+\varepsilon}.
\]

#### 建议

\(C_{\mathrm{loc}}\) 应保留为一个**小权重正则项**，不要取太大。

---

## 4. 额外可选项：\(C_{\mathrm{mix}}\) 与 \(C_{\mathrm{AF}}\)

原文件还给出了两个很有价值的补充项，它们不是默认项，但物理针对性很强。

### 4.1 \(C_{\mathrm{mix}}\)：按 occupation gap 抑制混合

\[
C_{\mathrm{mix}}=
\frac{\sum_{a<b}|n_a-n_b|^p|V_{ab}|^2}
{\sum_{a<b}|n_a-n_b|^p+\varepsilon}.
\]

它直接惩罚 occupation 差大的轨道对混合。

#### 评价

- 优点：定义直接，工程上清楚；
- 局限：依赖 \(V_{ab}\) 本身，更适合作为辅助项，不宜取代 \(C_\gamma\)。

---

### 4.2 \(C_{\mathrm{AF}}\)：直接抑制 active / frozen 混合

先定义 active 指标

\[
A_a=4n_a(1-n_a).
\]

则 near-full / near-empty 轨道有 \(A_a\approx 0\)，最 active 的轨道有 \(A_a\approx 1\)。于是可定义

\[
C_{\mathrm{AF}}=
\frac{\sum_{a<b}(A_a-A_b)^2|V_{ab}|^2}
{\sum_{a<b}(A_a-A_b)^2+\varepsilon}.
\]

#### 评价

这一项和 \(C_\gamma\) 不同，它并不泛泛保护整个 NO 结构，而是专门针对：

> active 空间不要和 frozen 空间互相污染。

若问题的真正重点是 active/frozen 分离，则 \(C_{\mathrm{AF}}\) 往往比单纯调大 \(\lambda_\gamma\) 更有针对性。

---

## 5. 结构代价函数 \(\mathcal S(W)\) 的详细解释

这是整套框架中一个很关键但也最容易被误解的对象。

### 5.1 定义

对非负权重矩阵 \(W_{ab}\)，定义

\[
P_{\mathrm{eff}}=
\frac{\left(\sum_{a<b}W_{ab}\right)^2}
{\sum_{a<b}W_{ab}^2},
\]

\[
\mathrm{participation}=
\frac{P_{\mathrm{eff}}-1}{P_{\max}-1},
\]

\[
\mathrm{decay}=
\frac{\sum_{a<b}W_{ab}(d_{ab}/d_{\max})^p}
{\sum_{a<b}W_{ab}},
\]

以及

\[
\mathcal S(W)=\frac12\bigl(\mathrm{participation}+\mathrm{decay}\bigr).
\]

### 5.2 participation 的意义

若将所有轨道对记为一个离散集合 \(\mu\)，并定义归一化权重

\[
p_\mu = \frac{w_\mu}{\sum_\nu w_\nu},
\]

则

\[
P_{\mathrm{eff}}=\frac{1}{\sum_\mu p_\mu^2}.
\]

这正是一个“有效参与 pair 数目”。

- 若只有一个 pair 很重要，则 \(P_{\mathrm{eff}}\approx 1\)；
- 若所有 pair 都差不多重要，则 \(P_{\mathrm{eff}}\) 大。

因此 participation 惩罚的是：

> 权重分布得太广，而不是某个单独元素太大。

### 5.3 decay 的意义

\[
\mathrm{decay}=\sum_\mu p_\mu\tilde d_\mu^p,
\qquad \tilde d_\mu\in[0,1].
\]

它是一个按权重加权的平均距离。因此：

- 权重主要落在短程 pair 上，则 decay 小；
- 权重重心落在长程 pair 上，则 decay 大。

### 5.4 \(\mathcal S\) 的优点

\(\mathcal S\) 对整体缩放不敏感：

\[
\mathcal S(\alpha W)=\mathcal S(W),\qquad \alpha>0.
\]

所以它度量的是“形状”，不是“规模”。这对描述“复杂度是否集中、是否短程”很有价值。

### 5.5 \(\mathcal S\) 的局限

正因为它只看形状，它不能取代范数项。若两组 \(W\) 只有整体尺度不同，但分布形状一样，则 \(\mathcal S\) 给出的值近似相同。

因此：

- \(\mathcal S\) 适合做**结构项**；
- 若想控制总系数预算，仍应额外加入 weighted \(L_1\) / smooth-\(L_1\) 项。

### 5.6 关于距离 \(d_{ab}\)

\(d_{ab}\) 不应简单取轨道编号差。更合理的是：

1. 轨道中心距离

\[
\bar r_a=\sum_i |U_{ia}|^2r_i,
\qquad
 d_{ab}=|\bar r_a-\bar r_b|;
\]

2. 双分布平均距离

\[
d_{ab}^2=
\sum_{ij}|U_{ia}|^2|U_{jb}|^2d_{ij}^2.
\]

第二种定义更稳，也与 \(\Lambda_{ab}\) 的构造方式更一致。

---

## 6. “约束”和“惩罚项”有什么不同

这个问题在方法设计上非常关键。

### 6.1 三种写法

#### 硬约束

\[
\min_V C_H(V)
\quad\text{s.t.}\quad C_\gamma(V)\le \delta_\gamma.
\]

#### 普通软惩罚

\[
\min_V C_H(V)+\lambda_\gamma C_\gamma(V).
\]

#### barrier / thresholded penalty

\[
\min_V C_H(V)+\mu_\gamma\,\phi\bigl(C_\gamma(V)-\delta_\gamma\bigr),
\]

例如

\[
\phi(x)=\mathrm{softplus}(x/\tau)^2.
\]

### 6.2 本质区别

- **硬约束**：定义可行域，越界解不允许；
- **软惩罚**：允许越界，只要收益足够大；
- **barrier**：在阈值以内几乎不罚，超过阈值后迅速变重。

### 6.3 为什么当前实现已经有一层真正约束

当前自动分块规则已经直接禁止某些 block 间混合，因此这不是“代价高”，而是“根本不在搜索空间里”。

因此原实现实际上已经是：

- 一层硬约束：block 间禁止混合；
- 一层软约束：块内用 \(C_\gamma\) 抑制偏离 NO；
- 主目标：简化 \(H\)。

### 6.4 推荐的理解方式

若某种混合在物理上根本不应发生，就应当用硬约束；若边界本身是模糊的、模型依赖的，则更适合用软约束或 barrier。

---

## 7. 从 1DM 出发：NO-first 还是 site/local-first？

### 7.1 路线 A：从自然轨道出发

先对 1DM 对角化，得到 \(U_0\) 与 \(n_a\)，然后在 NO 空间中继续优化 \(V\)。

#### 优点

- occupation 结构最清楚；
- active/frozen 划分直接；
- 适合把多体波函数表示得更紧凑。

#### 缺点

- 原本局域的哈密顿量在 NO 基底里往往更长程；
- 可能使哈密顿量复杂度优化离“局域 basin”较远。

### 7.2 路线 B：从格点 / 局域轨道出发，但用 1DM 提供结构信息

这一思路并不是直接采用 NO 的轨道形状，而是只使用 1DM 的谱信息：

- 哪些轨道近满、近空、active；
- 哪些 occupation gap 大；
- 哪些子空间不宜混合。

然后在 site / local basis 上优化，以保留原始哈密顿量的局域性。

### 7.3 更推荐的混合路线

对于当前问题，更推荐：

1. 用 1DM 提取 active/frozen 结构与 occupation 权重；
2. 用这些信息定义 block / projector / penalty；
3. 再在保留局域性的表示中，优化哈密顿量复杂度。

也就是说：

> 1DM 主要用于定义“哪些方向重要、哪些方向不应混”，而不一定要求最终轨道始终等于那组全局 NO。

---

## 8. 推荐的完整版 loss

为了把“保持 occupation 结构”和“简化哈密顿量”这两件事分清楚，推荐把完整版写成“主目标 + barrier + 小正则”的形式。

### 8.1 主目标：哈密顿量复杂度

定义

\[
C_H(V)=\alpha_t C_t^\star(V)+\alpha_{\mathrm{int}}C_{\mathrm{int}}^\star(V),
\]

其中

\[
C_t^\star=
\eta_t\,\widetilde{\|t'_{\mathrm{off}}\|}_{1,w}
+(1-\eta_t)\,\mathcal S\!\left(|t'_{ab}|^2\right),
\]

\[
C_{\mathrm{int}}^\star=
\eta_{\mathrm{int}}\,\widetilde{\|W\|}_{1,w}
+(1-\eta_{\mathrm{int}})\,\mathcal S(W),
\qquad
W_{ab}=\Lambda_{ab}\Xi_{ab}^{\mathrm{NO}}.
\]

weighted smooth-\(L_1\) 项可写为

\[
\widetilde{\|X\|}_{1,w}=
\frac{\sum_{a<b}w_{ab}\sqrt{|X_{ab}|^2+\varepsilon^2}}
{\sum_{a<b}w_{ab}\sqrt{|X^{(0)}_{ab}|^2+\varepsilon^2}+\varepsilon}.
\]

其中 \(w_{ab}\) 可以取距离权重，例如

\[
w_{ab}=1+\kappa(d_{ab}/d_0)^q.
\]

### 8.2 约束 / barrier：occupation 结构保持

定义

\[
C_\gamma(V)=
\frac{\|\bigl(V^\dagger\mathrm{diag}(\mathbf n)V\bigr)_{\mathrm{off}}\|_F^2}
{\sum_i n_i^2},
\]

以及可选的

\[
C_{\mathrm{AF}}(V)=
\frac{\sum_{a<b}(A_a-A_b)^2|V_{ab}|^2}
{\sum_{a<b}(A_a-A_b)^2+\varepsilon}.
\]

则总 loss 推荐写成

\[
\mathcal L_{\mathrm{full}}(V)=
C_H(V)
+\mu_\gamma\,\phi\bigl(C_\gamma(V)-\delta_\gamma\bigr)
+\mu_{\mathrm{AF}}\,\phi\bigl(C_{\mathrm{AF}}(V)-\delta_{\mathrm{AF}}\bigr)
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}(V).
\]

其中 \(\phi\) 可取

\[
\phi(x)=\mathrm{softplus}(x/\tau)^2.
\]

### 8.3 若不想用 barrier，可退化成纯加权和

\[
\mathcal L_{\mathrm{full,soft}}=
\alpha_t C_t^\star
+\alpha_{\mathrm{int}}C_{\mathrm{int}}^\star
+\lambda_\gamma C_\gamma
+\lambda_{\mathrm{AF}}C_{\mathrm{AF}}
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}.
\]

但从可解释性上讲，barrier 形式更适合表达“occupation 结构最多允许扭曲到什么程度”。

---

## 9. 为什么推荐这个完整版，而不是原版直接加权和

### 9.1 把“主目标”与“结构保护”分开

原文件中 \(C_\gamma\)、\(C_t\)、\(C_{\mathrm{int}}\)、\(C_{\mathrm{loc}}\) 以统一地位并列写在 loss 里。数值上当然可行，但概念上它们并不对等。

更准确的分工应为：

- \(C_t\)、\(C_{\mathrm{int}}\)：主目标，负责简化 \(H\)；
- \(C_\gamma\)、\(C_{\mathrm{AF}}\)：结构保护，负责 occupation 结构不被破坏；
- \(C_{\mathrm{loc}}\)：弱正则，负责轨道形状不要跑得太野。

### 9.2 让 \(\mathcal S\) 回到最合适的位置

\(\mathcal S\) 最适合表达的是：

- 复杂度是否集中在少数 pair 上；
- 是否偏短程。

它不是总范数，也不是 sign-problem surrogate。因此将它和显式范数项组合起来，远比单独依赖 \(\mathcal S\) 更稳。

### 9.3 让 interaction proxy 更自洽

固定 \(\Xi_{ab}^{\mathrm{NO}}\) 可以避免：

- \(C_{\mathrm{int}}\) 通过涂抹 occupation 获益；
- \(C_\gamma\) 又试图阻止 occupation 被涂抹；
- 两者互相竞争。

### 9.4 保留灵活性

完整版仍允许：

- 不同模型采用不同的 \(S_{ij}\)；
- 对 site / local / hybrid 初始化做不同选择；
- 视情况打开或关闭 \(C_{\mathrm{AF}}\)。

---

## 10. 与原文件的逐项比较总结

### 保留

1. \(C_\gamma\) 的定义与物理解释；
2. \(\Lambda_{ab}\Xi_{ab}\) 作为 interaction complexity 的 pair-level surrogate；
3. \(\mathcal S\) 作为结构项；
4. “硬分块 + 软正则”的总体思路；
5. \(C_{\mathrm{mix}}\)、\(C_{\mathrm{AF}}\) 作为补充候选项的思路。

### 修改

1. \(C_{\mathrm{int}}\) 中建议固定 \(\Xi_{ab}^{\mathrm{NO}}\)，而非使用旋转后占据数；
2. \(C_{\mathrm{loc}}\) 建议改成 spread 型，或更改 IPR 的符号；
3. \(C_t\) 与 \(C_{\mathrm{int}}\) 中加入显式 weighted smooth-\(L_1\) 项，而非只保留 \(\mathcal S\)。

### 提升

1. 将 loss 从“所有项并列加权”提升为“主目标 + 约束/barrier + 弱正则”的结构；
2. 明确把 1DM 的作用拆成“提供子空间信息”和“提供轨道形状起点”两层，而不再把两者混为一谈；
3. 明确把“轨道对”作为单体项、interaction proxy、active/frozen 混合的统一语言。

---

## 11. 推荐的实现路径

### 阶段 1：最小可行版

只使用

\[
\mathcal L_{\mathrm{core}}(V)=C_H(V)+\lambda_{\mathrm{occ}}C_{\mathrm{occ}}(V)+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}(V),
\]

其中 \(C_H\) 为合并后的哈密顿量复杂度项，\(C_{\mathrm{occ}}\) 默认取 \(C_\gamma\)。

这是适合最初探索的版本，参数最少。

### 阶段 2：加入结构项 \(\mathcal S\)

当基础版本稳定后，再把

- weighted \(L_1\) 与 \(\mathcal S\) 做混合；
- 或把 \(C_t\) 与 \(C_{\mathrm{int}}\) 重新拆分开来。

### 阶段 3：加入更细的 occupation 结构保护

若发现 active/frozen 污染仍明显，则加入 \(C_{\mathrm{AF}}\) 或将 \(C_\gamma\) 改写为 barrier 形式。

---

## 12. 一个更接近代码实现的伪代码框架

```python
# inputs:
# U0, n_occ, t, Sij, site_positions
# block structure already enforced in parameterization of V

U = U0 @ V

# ----- occupation protection -----
gamma_rot = V.conj().T @ diag(n_occ) @ V
C_gamma = fro_norm(offdiag(gamma_rot))**2 / (sum(n_occ**2) + eps)

A = 4.0 * n_occ * (1.0 - n_occ)
C_AF = weighted_pair_ratio((A[:,None]-A[None,:])**2, abs(V)**2)

# ----- single-particle complexity -----
t_no = U0.conj().T @ t @ U0
t_rot = V.conj().T @ t_no @ V
Wt = abs(t_rot)**2
Ct_struct = S_metric(Wt, d_ab)
Ct_norm   = smooth_L1_weighted(offdiag(t_rot), w_ab)
Ct_star   = eta_t * Ct_norm + (1.0 - eta_t) * Ct_struct

# ----- interaction complexity -----
Lambda = build_pair_exposure(U, Sij)     # sum_ij Sij |U_ia|^2 |U_jb|^2
Xi_NO  = n_occ[:,None] * (1.0 - n_occ[None,:]) \
       + n_occ[None,:] * (1.0 - n_occ[:,None])
Wint   = Lambda * Xi_NO
Cint_struct = S_metric(Wint, d_ab)
Cint_norm   = smooth_L1_weighted(Wint, w_ab)
Cint_star   = eta_int * Cint_norm + (1.0 - eta_int) * Cint_struct

# ----- Hamiltonian objective -----
C_H = alpha_t * Ct_star + alpha_int * Cint_star

# ----- locality regularizer -----
C_loc = orbital_spread(U, site_positions)

# ----- full loss -----
L = C_H \
  + mu_gamma * barrier(C_gamma - delta_gamma) \
  + mu_AF    * barrier(C_AF - delta_AF) \
  + lambda_loc * C_loc
```

---

## 13. 总结

若只用一句话概括本文档的核心建议，那就是：

> 保持自然轨道 occupation 结构这件事，更适合作为约束或 barrier；
> 简化变换后哈密顿量这件事，才应当是主优化目标；
> 而“轨道对”则是把这两类需求统一到同一语言中的关键中间层对象。

因此，一个更成熟的设计不应只是把更多项简单相加，而应当明确区分：

- 什么是主目标；
- 什么是允许偏离但不希望过度偏离的结构保护；
- 什么是仅用于数值稳定与形状控制的弱正则。

这也正是从原文件走向更稳定、更可解释、更便于后续扩展的方向。


---

## 14. 完整版、soft 版与 minimal 版之间的关系

这一节专门说明三者之间的包含关系与参数映射，避免后续实现时因为“同一个符号在不同版本里承担不同角色”而混淆。

### 14.1 三个层次的关系

推荐把三个版本理解为：

1. **完整版（barrier 版）**：最强调“主目标 + 容忍边界”的形式；
2. **完整版（soft 版）**：把约束近似改成普通加权和，便于实现与初期试验；
3. **minimal 版**：在 soft 版基础上再做压缩，只保留最核心的 2–3 个权重。

也就是说：

\[
\text{minimal} \subset \text{full-soft},
\qquad
\text{full-soft} \neq \text{full-barrier},
\]

更准确地说：

> **minimal 是完整版 soft 版的特例，而不是 barrier 版通过把若干参数简单设为 0 就自然得到的特例。**

---

### 14.2 完整版（barrier 版）

\[
\mathcal L_{\mathrm{full}}(V)=
C_H(V)
+\mu_\gamma\,\phi\bigl(C_\gamma(V)-\delta_\gamma\bigr)
+\mu_{\mathrm{AF}}\,\phi\bigl(C_{\mathrm{AF}}(V)-\delta_{\mathrm{AF}}\bigr)
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}(V).
\]

其中：

- \(C_H\)：主目标；
- \(C_\gamma, C_{\mathrm{AF}}\)：结构保护项；
- \(\delta_\gamma,\delta_{\mathrm{AF}}\)：允许的最大偏离阈值；
- \(\mu_\gamma,\mu_{\mathrm{AF}}\)：越界后的惩罚强度；
- \(\phi\)：平滑 barrier 函数。

---

### 14.3 完整版（soft 版）

若不想显式引入阈值 \(\delta\)，则可退化为普通加权和：

\[
\mathcal L_{\mathrm{full,soft}}=
\alpha_t C_t^\star
+\alpha_{\mathrm{int}}C_{\mathrm{int}}^\star
+\lambda_\gamma C_\gamma
+\lambda_{\mathrm{AF}}C_{\mathrm{AF}}
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}.
\]

这时：

- \(C_t^\star, C_{\mathrm{int}}^\star\) 都仍然是“norm + structure”的混合；
- \(C_\gamma, C_{\mathrm{AF}}\) 则不再是 barrier，而是普通正则项；
- 所有项都进入连续 trade-off。

---

### 14.4 minimal 版

minimal 版的目标是：在开始阶段尽量减少调参负担，同时保留最核心的物理逻辑。推荐写成

\[
\mathcal L_{\mathrm{core}}(V)
=
C_H^{\mathrm{min}}(V)
+\lambda_{\mathrm{occ}}\,C_{\mathrm{occ}}(V)
+\lambda_{\mathrm{loc}}\,C_{\mathrm{loc}}^{\mathrm{spread}}(V),
\]

其中

\[
C_H^{\mathrm{min}}(V)
=
\frac12\,\widehat C_t(V)
+\frac12\,\widehat C_{\mathrm{int}}(V),
\]

\[
\widehat C_t=\widetilde{\|t'_{\mathrm{off}}\|}_{1,w},
\qquad
\widehat C_{\mathrm{int}}=\widetilde{\|W\|}_{1,w},
\qquad
C_{\mathrm{occ}}=C_\gamma.
\]

也就是说，minimal 版只保留：

- 一个合并后的哈密顿量复杂度主目标；
- 一个 occupation 保持项；
- 一个小权重的 locality 正则项。

若想再进一步压缩，甚至可以暂时令 \(\lambda_{\mathrm{loc}}=0\)，变成两项版：

\[
\mathcal L_{\mathrm{core,2}}(V)
=
C_H^{\mathrm{min}}(V)+\lambda_{\mathrm{occ}}C_\gamma(V).
\]

---

### 14.5 从完整版 soft 版到 minimal：参数映射

若从

\[
\mathcal L_{\mathrm{full,soft}}=
\alpha_t C_t^\star
+\alpha_{\mathrm{int}}C_{\mathrm{int}}^\star
+\lambda_\gamma C_\gamma
+\lambda_{\mathrm{AF}}C_{\mathrm{AF}}
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}
\]

出发，则只要取

\[
\eta_t=1,\qquad
\eta_{\mathrm{int}}=1,
\]

使得

\[
C_t^\star=\widetilde{\|t'_{\mathrm{off}}\|}_{1,w}=\widehat C_t,
\qquad
C_{\mathrm{int}}^\star=\widetilde{\|W\|}_{1,w}=\widehat C_{\mathrm{int}},
\]

再令

\[
\alpha_t=\alpha_{\mathrm{int}}=\frac12,
\qquad
\lambda_{\mathrm{AF}}=0,
\qquad
\lambda_\gamma=\lambda_{\mathrm{occ}},
\]

即可得到

\[
\mathcal L_{\mathrm{full,soft}}
=
\frac12\widehat C_t
+\frac12\widehat C_{\mathrm{int}}
+\lambda_{\mathrm{occ}}C_\gamma
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}},
\]

这就是 minimal 版。

因此：

> **minimal 版就是完整版 soft 版在“关闭结构项 \(\mathcal S\)、关闭 \(C_{\mathrm{AF}}\)、并把 \(C_t\) 与 \(C_{\mathrm{int}}\) 平均合并”之后得到的特例。**

---

### 14.6 为什么 barrier 版不能仅靠“把某些参数设为 0”变成 minimal

这是因为 occupation 项的函数形式已经不同。

barrier 版里 occupation 结构是通过

\[
\mu_\gamma\,\phi(C_\gamma-\delta_\gamma)
\]

出现的，而 minimal 版里对应的是

\[
\lambda_{\mathrm{occ}}\,C_\gamma.
\]

两者的区别不只是参数多少，而是**作用机制不同**：

- barrier 版：阈值以内几乎不管，越界后迅速变硬；
- soft/minimal 版：始终线性参与 trade-off。

因此，即使把 barrier 版中的某些项关掉，例如令 \(\mu_{\mathrm{AF}}=0\)，得到的仍然是

\[
C_H+\mu_\gamma\phi(C_\gamma-\delta_\gamma)+\lambda_{\mathrm{loc}}C_{\mathrm{loc}},
\]

它一般并不等于

\[
C_H+\lambda_{\mathrm{occ}}C_\gamma+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}.
\]

除非额外强行把 barrier 本身退化成线性形式，这本质上已经是在改模型了，而不是“仅靠设零参数”。

---

## 15. soft 版与 barrier 版的区别

这一节单独总结两者的概念差异、参数含义差异以及数值优化行为差异。

### 15.1 一句话概括

- **soft 版**：在做不同目标之间的连续权衡（trade-off）；
- **barrier 版**：在做“主目标 + 容忍边界”的近似约束优化。

也可以更形象地说：

- soft 版像一根一直把你往回拉的弹簧；
- barrier 版像一堵墙：在安全区内几乎不干预，靠近边界时再猛烈推回去。

---

### 15.2 soft 版：所有项都可以交换

soft 版的典型形式是

\[
\mathcal L_{\mathrm{soft}}
=
C_H
+\lambda_\gamma C_\gamma
+\lambda_{\mathrm{AF}} C_{\mathrm{AF}}
+\lambda_{\mathrm{loc}} C_{\mathrm{loc}}.
\]

这意味着：

- 只要换来的 \(C_H\) 降低足够多，优化器就允许 \(C_\gamma\) 变差；
- \(\lambda_\gamma\) 的意义是“1 单位的 occupation 结构损失值多少钱”；
- 没有一个显式的“最大允许扭曲量”。

换句话说，soft 版是在求一个多目标折中点，而不是一个有明确可行域边界的问题。

---

### 15.3 barrier 版：先给阈值，再惩罚越界

barrier 版的典型形式是

\[
\mathcal L_{\mathrm{barrier}}
=
C_H
+\mu_\gamma\,\phi(C_\gamma-\delta_\gamma)
+\mu_{\mathrm{AF}}\,\phi(C_{\mathrm{AF}}-\delta_{\mathrm{AF}})
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}.
\]

这时：

- \(\delta_\gamma\) 表示“occupation 结构最多允许偏离到什么程度”；
- \(\mu_\gamma\) 表示“越界以后墙有多硬”；
- \(\phi\) 则决定墙是平滑过渡还是非常陡。

这种写法更接近于下面这个约束问题的可微近似：

\[
\min_V C_H(V)
\quad\text{s.t.}\quad
C_\gamma(V)\le\delta_\gamma,\quad
C_{\mathrm{AF}}(V)\le\delta_{\mathrm{AF}}.
\]

---

### 15.4 梯度行为的差异

#### soft 版

\[
\nabla \mathcal L_{\mathrm{soft}}
=
\nabla C_H
+\lambda_\gamma \nabla C_\gamma
+\lambda_{\mathrm{AF}}\nabla C_{\mathrm{AF}}
+\cdots
\]

只要 \(\lambda_\gamma\neq 0\)，occupation 项的梯度就始终在线。即使当前 \(C_\gamma\) 已经非常小，它仍然继续把你往更小拉。

因此 soft 版会持续偏好“更靠近 NO”。

#### barrier 版

\[
\nabla \mathcal L_{\mathrm{barrier}}
=
\nabla C_H
+\mu_\gamma\,\phi'(C_\gamma-\delta_\gamma)\,\nabla C_\gamma
+\cdots
\]

其中 \(\phi'(x)\) 的行为是：

- 当 \(x\ll 0\) 时，\(\phi'(x)\) 很小；
- 当 \(x\approx 0\) 时，\(\phi'(x)\) 开始变大；
- 当 \(x>0\) 时，\(\phi'(x)\) 快速变大。

所以 barrier 版具有明显的“安全区—边界—墙外”分区特征。

---

### 15.5 参数含义完全不同

#### soft 版中的参数

\[
\lambda_\gamma,\ \lambda_{\mathrm{AF}}
\]

是**权衡系数**。它们回答的是：

> 允许多大的 \(C_\gamma\) 或 \(C_{\mathrm{AF}}\) 增加，以换取多少 \(C_H\) 的下降。

但它们不直接告诉你“最大允许偏离多少”。

#### barrier 版中的参数

\[
\delta_\gamma,\ \delta_{\mathrm{AF}}
\]

是**容忍阈值**，直接对应：

> occupation 结构允许扭曲到哪里为止。

\[
\mu_\gamma,\ \mu_{\mathrm{AF}}
\]

则是越界后惩罚的强弱；  
\(\tau\) 则是 barrier 的平滑尺度。

所以 barrier 版比 soft 版更容易和“物理上可接受的最大扭曲量”建立对应。

---

### 15.6 数值上的优缺点

#### soft 版的优点

- 参数少；
- 调参直接；
- 对普通优化器更友好；
- 很适合最初阶段快速试探。

#### soft 版的缺点

- occupation 正则始终在线，容易过度把解往 NO 基底拉回；
- “允许偏离多少”是隐含的，不易解释；
- 若 \(\lambda_\gamma\) 太大，主目标几乎动不了；若太小，又可能偏离过头。

#### barrier 版的优点

- 更贴近“occupation 结构应视为约束而非主目标”的物理直觉；
- 可以显式表达“允许偏离多少”；
- 在安全区内可更专注于优化 \(C_H\)。

#### barrier 版的缺点

- 参数更多；
- \(\delta,\mu,\tau\) 之间存在耦合；
- barrier 太硬会导致训练不稳定，太软则失去约束意义。

---

### 15.7 在当前问题中的推荐用法

结合本问题的目标，推荐采用分阶段策略：

#### 第一阶段：先用 soft / minimal 版

\[
\mathcal L_{\mathrm{core}}
=
C_H^{\mathrm{min}}
+\lambda_{\mathrm{occ}}C_\gamma
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}.
\]

原因是：

- 参数最少；
- 更容易快速探索；
- 先确定 \(C_H\) 与 \(C_\gamma\) 的相对量级。

#### 第二阶段：若出现以下情况，再升级到 barrier 版

1. \(\lambda_{\mathrm{occ}}\) 非常难调，稍大稍小效果就截然不同；
2. 已经可以从经验上判断一个“可接受的最大扭曲量”；
3. 明显感觉 soft 版在安全区里也一直过度把解往 NO 拉回。

这时再切换到

\[
\mathcal L_{\mathrm{full}}
=
C_H
+\mu_\gamma\,\phi(C_\gamma-\delta_\gamma)
+\lambda_{\mathrm{loc}}C_{\mathrm{loc}}^{\mathrm{spread}}
\]

往往更自然。

---

## 16. 补充总结

若把本次新增讨论压缩成最简短的结论，可以记成下面三句话：

1. **完整版的主逻辑是：主目标优化 \(H\)，occupation 结构只做保护。**
2. **minimal 版是完整版 soft 版的特例：关闭 \(\mathcal S\)、关闭 \(C_{\mathrm{AF}}\)、合并 \(C_t\) 与 \(C_{\mathrm{int}}\)。**
3. **soft 与 barrier 的差别不在于“项多项少”，而在于“连续 trade-off”与“容忍边界”的优化哲学不同。**
