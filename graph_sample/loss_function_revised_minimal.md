# 损失函数说明（修订版：最小可行 loss）

> 这份修订版的目标不是把所有可能的项都堆进去，而是先抓住**最核心、最少参数、最容易调通**的版本。
>
> 核心思想：
>
> 1. **主目标只有一个**：让变换后的哈密顿量更简单。  
> 2. **占据结构保护只保留一个软正则项**：防止优化把自然轨道结构完全涂抹掉。  
> 3. **局域性项只做弱正则，甚至可以在第一轮直接关闭**。  
> 4. **已有的 occupation 分块仍然作为硬约束保留**，因此不需要再用太多软项重复约束同一件事。  
>
> 阅读建议：
> 这份文档是默认入口，适合直接对应 `optimize_no_soft.py`。
> 如果你想看更完整的理论动机与 barrier 形式，再回到 `loss_function_revised_full.md`。

---

## 0. 结论先行：我推荐的最小版本

如果你现在的实现已经有：

- 基于 1-RDM 占据数的 **hard blocks**；
- 只允许 block 内旋转；

那么开始阶段我最推荐的 loss 是：

$$
\boxed{
\mathcal L_{\rm core}(V)
=
C_H(V)
+
\lambda_{\rm occ}\, C_{\rm occ}(V)
+
\lambda_{\rm loc}\, C_{\rm loc}(V)
}
$$

其中：

- $C_H$：**主目标**，统一度量变换后哈密顿量的复杂度；
- $C_{\rm occ}$：**占据结构软约束**；
- $C_{\rm loc}$：**弱局域性正则项**，可选。

这样真正需要调的只有 **2 个正则参数**：

- $\lambda_{\rm occ}$
- $\lambda_{\rm loc}$

默认可以固定主目标前的系数为 1，不再单独引入 $\lambda_t,\lambda_{\rm int}$。

---

## 1. 为什么要从“四项 loss”压缩到“三项以内”

原文件的总形式是

$$
\mathcal L(V)=
\lambda_\gamma C_\gamma
+\lambda_t C_t
+\lambda_{\rm int} C_{\rm int}
+\lambda_{\rm loc} C_{\rm loc}.
$$

并且还有可选的 $C_{\rm mix}$、$C_{\rm AF}$ 等附加项。

这个形式的问题不是“不对”，而是**起步阶段太重**：

1. **顶层权重太多**：$\lambda_\gamma,\lambda_t,\lambda_{\rm int},\lambda_{\rm loc}$ 至少 4 个。  
2. **很多项其实在管同一件事的不同侧面**：  
   - $C_\gamma, C_{\rm mix}, C_{\rm AF}$ 都在管“不该混的轨道不要混”；  
   - $C_t, C_{\rm int}$ 都在管“哈密顿量变简单”。  
3. **过多权重会掩盖问题本身**：你最后很难判断，是 loss 设计错了，还是只是调参没调对。  

因此更好的策略是：

- 先把“同类项”合并；
- 只保留每一类目标里最核心的一项；
- 等最小版本工作稳定后，再逐步打开更精细的附加项。

---

## 2. 推荐的最小版本：每一项到底是什么

---

### 2.1 主目标：统一的哈密顿量复杂度项 $C_H$

我不建议在开始阶段保留两个独立的主目标权重 $\lambda_t,\lambda_{\rm int}$。  
更好的做法是把单体部分和相互作用部分合并为**一个统一的主目标**：

$$
\boxed{
C_H(V)=\frac{1}{2}\left(\widehat C_t(V)+\widehat C_{\rm int}(V)\right)
}
$$

这里上面的帽子表示：**每一项都先用初始值归一化**，这样单体项和相互作用项天然处在相近尺度上，不再需要两个单独的顶层权重。

---

#### 2.1.1 单体复杂度项 $\widehat C_t$

定义旋转后的单体项：

$$
t' = V^\dagger t_{\rm NO} V,
\qquad
t_{\rm NO}=U_0^\dagger t\, U_0.
$$

我建议开始阶段先不用 $\mathcal S(|t'|^2)$，而改用一个更直接的、带距离权重的 smooth-$L_1$ 复杂度：

$$
\boxed{
\widehat C_t(V)
=
\frac{
\sum_{a<b}\omega_{ab}\sqrt{|t'_{ab}|^2+\varepsilon^2}
}{
\sum_{a<b}\omega_{ab}\sqrt{|t^{(0)}_{ab}|^2+\varepsilon^2}
+\varepsilon
}
}
$$

其中：

- $t^{(0)}$ 是初始基底下的对应矩阵元；
- $\varepsilon$ 是一个很小的数，用来平滑绝对值；
- $\omega_{ab}$ 是距离权重，默认可取
  $$
  \omega_{ab}=1+\left(\frac{d_{ab}}{d_{\max}}\right)^2.
  $$

##### 这项的作用

这项同时做了两件事：

1. **压小单体项的总预算**：  
   如果很多 $t'_{ab}$ 很大，那么这项大；
2. **更讨厌远程项**：  
   如果大的矩阵元主要出现在远距离 pair 上，$\omega_{ab}$ 会让代价更高。

因此它比纯粹的 $\mathcal S(|t'|^2)$ 更适合做开始阶段的主目标，因为它直接回答：

> 变换后的 hopping 到底是不是更少、更短、更便宜了？

##### 为什么不直接用原文件里的 $C_t=\mathcal S(|t'_{ab}|^2)$

因为 $\mathcal S$ 更像“形状指标”，而不是“总强度指标”。  
若只用 $\mathcal S$，它会偏向让耦合分布更集中、更短程，但不直接压制总量。  
开始阶段更稳的做法是先用上面这个 weighted smooth-$L_1$，以后若需要，再把 $\mathcal S$ 加回来做二阶段精修。

---

#### 2.1.2 相互作用复杂度项 $\widehat C_{\rm int}$

定义完整变换：

$$
U=U_0V.
$$

然后定义轨道对曝光度

$$
\Lambda_{ab}
=
\sum_{ij} S_{ij}\,|U_{ia}|^2 |U_{jb}|^2.
$$

其中 $S_{ij}$ 是格点对之间的相互作用强度矩阵。

再定义占据因子，但这里我建议**固定使用初始自然轨道占据数**，而不是旋转后的占据：

$$
\boxed{
\Xi_{ab}^{\rm NO}
=
n_a^{\rm NO}(1-n_b^{\rm NO})
+
n_b^{\rm NO}(1-n_a^{\rm NO})
}
$$

然后定义 pair weight

$$
W_{ab}=\Lambda_{ab}\,\Xi_{ab}^{\rm NO}.
$$

最后定义

$$
\boxed{
\widehat C_{\rm int}(V)
=
\frac{
\sum_{a<b}\omega_{ab}\, W_{ab}
}{
\sum_{a<b}\omega_{ab}\, W_{ab}^{(0)}+\varepsilon
}
}
$$

##### 这项的作用

这项本质上在问：

> 在变换后的轨道基底里，真正“暴露在相互作用中、而且占据上活跃”的轨道对，是否被压缩到了更少、更短、更弱的通道上？

它是整个 loss 里最物理、最重要的一项，因为你最终真正想简化的，往往不是单体 hopping，而是相互作用的有效复杂度。

##### 为什么这里固定使用 $\Xi^{\rm NO}_{ab}$

因为如果你用旋转后的占据数 $\tilde n_a$ 去构造 $\Xi_{ab}$，优化器有可能通过“把 occupation 涂抹掉”来减小 $C_{\rm int}$。  
这会和“保持自然轨道占据结构”的目标直接打架。

所以更保守、更干净的做法是：

- 让 $V$ 只通过 $\Lambda_{ab}$ 改变 interaction complexity；
- 让 $\Xi_{ab}$ 只编码“参考态 1DM 告诉你的活性结构”。

---

### 2.2 占据结构软约束：$C_{\rm occ}$

开始阶段我建议只保留 **一个** 占据结构约束项，不要让 $C_\gamma, C_{\rm mix}, C_{\rm AF}$ 同时出现。

最推荐的默认选择是：

$$
\boxed{
C_{\rm occ}(V)=C_\gamma(V)
=
\frac{
\left\|
\bigl(V^\dagger \mathrm{diag}(\mathbf n)V\bigr)_{\rm off-diag}
\right\|_F^2
}{
\sum_a n_a^2
}
}
$$

##### 这项的作用

它在做的事情是：

- 若某两个轨道 occupation 很接近，则允许它们较自由地重整；
- 若 occupation 差很多，则不鼓励它们混合。

小旋转下：

$$
C_\gamma \propto \sum_{a<b}(n_a-n_b)^2|K_{ab}|^2,
\qquad V\simeq e^K.
$$

所以它天然就是“**occupation gap weighted mixing penalty**”。

##### 为什么推荐它，而不是 $C_{\rm mix}$ 或 $C_{\rm AF}$

- $C_\gamma$ 直接来自旋转后 1-RDM 的几何结构，定义最自然；
- 它与 NO 基底直接对应；
- 你当前实现已经有 hard blocks，因此它只需要承担块内 regularization 的角色，不必太重。

##### 什么时候把它换成 $C_{\rm AF}$

如果你后面发现：

- 你最在意的并不是一般意义上的 occupation structure；
- 而是特别在意 **active / frozen 边界不要混**；

那么可以把 $C_{\rm occ}$ 换成

$$
A_a=4n_a(1-n_a),
$$

$$
\boxed{
C_{\rm AF}
=
\frac{
\sum_{a<b}(A_a-A_b)^2 |V_{ab}|^2
}{
\sum_{a<b}(A_a-A_b)^2+\varepsilon
}
}
$$

但**开始阶段不建议 $C_\gamma$ 和 $C_{\rm AF}$ 同时开**。  
否则你又回到“有太多类似的项、太多类似的超参数”的老问题。

---

### 2.3 弱局域性正则：$C_{\rm loc}$

这里我建议保留，但默认小权重，甚至第一轮可以直接关闭。

更重要的是：我不建议继续用原文件里“平均 IPR 直接加入 loss”的写法，因为若做最小化，这个符号方向很可能是反的。

更安全的做法是用 real-space spread：

$$
\boxed{
C_{\rm loc}(V)
=
\frac{1}{N_{\rm orb}}
\sum_a
\left(
\langle r^2\rangle_a-|\langle r\rangle_a|^2
\right)
}
$$

其中

$$
\langle r\rangle_a=\sum_i |U_{ia}|^2 r_i,
\qquad
\langle r^2\rangle_a=\sum_i |U_{ia}|^2 r_i^2.
$$

##### 这项的作用

它只是在说：

> 别为了压低 $C_H$，把轨道搞得过于扩展、失去基本空间直觉。

##### 为什么它只是弱正则

因为你的主目标不是“让轨道尽量局域”，而是“让哈密顿量在这个基底里更简单”。  
有时适度离域的轨道反而会让相互作用或 hopping 更有结构。  
所以 $C_{\rm loc}$ 只应该作为一个轻微的守门员，而不是主优化目标。

---

## 3. 我为什么不把 $\mathcal S(W)$ 放进最小版本

原文件里非常重视结构代价函数 $\mathcal S(W)$，这本身没有问题。  
但我不建议一开始就把它放进最小版本的主 loss 中，理由如下。

---

### 3.1 $\mathcal S(W)$ 本质上是“形状指标”

原文件里定义：

$$
P_{\rm eff}
=
\frac{(\sum_{a<b}W_{ab})^2}{\sum_{a<b}W_{ab}^2},
$$

$$
\mathrm{participation}
=
\frac{P_{\rm eff}-1}{P_{\max}-1},
$$

$$
\mathrm{decay}
=
\frac{
\sum_{a<b}W_{ab}(d_{ab}/d_{\max})^p
}{
\sum_{a<b}W_{ab}
},
$$

$$
\mathcal S(W)
=
\frac12
\left(
\mathrm{participation}
+
\mathrm{decay}
\right).
$$

它回答的是：

- 重要权重是否集中在少数轨道对上；
- 重要权重是否主要出现在短距离 pair 上。

这个思想很好，但有一个关键点：

$$
\mathcal S(\alpha W)=\mathcal S(W),\qquad \alpha>0.
$$

也就是说，$\mathcal S$ 对整体尺度不敏感。  
如果所有重要边的“形状分布”不变，只是整体强度放大了，$\mathcal S$ 几乎看不到。

---

### 3.2 所以 $\mathcal S$ 适合作为“二阶段结构精修”，不适合作为起步阶段唯一主目标

开始阶段你更想知道的是：

> 这个基底到底有没有让 Hamiltonian complexity 真正下降？

因此更适合先用带距离权重的 smooth-$L_1$ / weighted-$L_1$。  
它直接压总预算，也间接压远程项。

当最小版本稳定后，你可以把 $\mathcal S$ 加回来，形成：

$$
\widehat C_t^{\rm upgrade}
=
\alpha\,\widehat C_t
+
(1-\alpha)\,\mathcal S(|t'|^2),
$$

$$
\widehat C_{\rm int}^{\rm upgrade}
=
\alpha\,\widehat C_{\rm int}
+
(1-\alpha)\,\mathcal S(W).
$$

其中 $\alpha$ 可以固定为 0.5 或 0.7，而不必当成新的自由参数。

---

## 4. 约束 vs 惩罚：为什么这里只保留一个软约束项就够了

你的当前实现已经有一层非常关键的**硬约束**：

- 先根据 1DM 占据数自动分块；
- 只允许块内旋转；
- block 间混合直接禁止。

这意味着很多“不要让 full 和 empty 混”“不要让 active 和 frozen 瞎混”的任务，其实已经由 **搜索空间本身** 保证了。

因此在最小版本里：

- **不需要再叠很多类似的软罚项**；
- 只保留一个 $C_{\rm occ}$ 就够了；
- 它只负责在“允许旋转的块内”继续防止过度偏离 NO 结构。

换句话说：

- **硬约束** 决定“哪些方向根本不许走”；  
- **软惩罚** 决定“在允许走的方向上，别走太猛”。  

你的当前设置已经有 hard blocks，所以 soft side 不需要太复杂。

---

## 5. 与原文件逐项比较

---

### 5.1 总体形式对比

#### 原文件

$$
\mathcal L
=
\lambda_\gamma C_\gamma
+\lambda_t C_t
+\lambda_{\rm int} C_{\rm int}
+\lambda_{\rm loc} C_{\rm loc}
$$

还可再加 $C_{\rm mix}, C_{\rm AF}$。

#### 修订后的最小版本

$$
\boxed{
\mathcal L_{\rm core}
=
C_H
+
\lambda_{\rm occ}C_{\rm occ}
+
\lambda_{\rm loc}C_{\rm loc}
}
$$

其中

$$
C_H=\frac12(\widehat C_t+\widehat C_{\rm int}).
$$

##### 核心差别

1. 把 $C_t$ 与 $C_{\rm int}$ 从两个独立主目标，合并成一个统一主目标 $C_H$；
2. 把“占据结构保护”统一压缩成一个 $C_{\rm occ}$；
3. 保留局域性项，但只做弱正则；
4. 顶层待调参数从 **4 个以上** 压缩到 **2 个**。

---

### 5.2 对 $C_t$ 的比较

#### 原文件

$$
C_t=\mathcal S(|t'_{ab}|^2).
$$

#### 修订版

$$
\widehat C_t=
\frac{
\sum_{a<b}\omega_{ab}\sqrt{|t'_{ab}|^2+\varepsilon^2}
}{
\sum_{a<b}\omega_{ab}\sqrt{|t^{(0)}_{ab}|^2+\varepsilon^2}+\varepsilon
}.
$$

##### 为什么这样改

- 原版更强调“形状”；
- 修订版更强调“真实复杂度预算”；
- 对开始阶段调参更稳、更直接。

---

### 5.3 对 $C_{\rm int}$ 的比较

#### 原文件

$$
C_{\rm int}=\mathcal S(W),
\qquad
W_{ab}=\Lambda_{ab}\Xi_{ab},
$$

其中 $\Xi_{ab}$ 在当前实现里是用旋转后占据数构造的。

#### 修订版

$$
\widehat C_{\rm int}
=
\frac{
\sum_{a<b}\omega_{ab}\Lambda_{ab}\Xi_{ab}^{\rm NO}
}{
\sum_{a<b}\omega_{ab}W_{ab}^{(0)}+\varepsilon
}.
$$

##### 为什么这样改

1. 把 interaction complexity 直接做成“加权预算项”，而不是只看结构；
2. 固定使用 $\Xi^{\rm NO}$，避免 $C_{\rm int}$ 反过来鼓励 occupation 被涂抹；
3. 便于和 $\widehat C_t$ 统一成同一类主目标。

---

### 5.4 对占据约束项的比较

#### 原文件

- 默认：$C_\gamma$
- 可选：$C_{\rm mix}$、$C_{\rm AF}$

#### 修订版

- 开始阶段：**只保留一个**
- 默认选：$C_\gamma$
- 若 active/frozen 边界是重点：改成 $C_{\rm AF}$

##### 为什么这样改

因为它们本质上都在惩罚“不该出现的轨道混合”，不值得一开始同时开。

---

### 5.5 对局域性项的比较

#### 原文件

$$
C_{\rm loc}=\langle \mathrm{IPR}_a\rangle_a.
$$

#### 修订版

$$
C_{\rm loc}
=
\frac1{N_{\rm orb}}
\sum_a
\left(
\langle r^2\rangle_a-|\langle r\rangle_a|^2
\right).
$$

##### 为什么这样改

- 平均 IPR 若直接作为正号 loss 去最小化，方向很可能反；
- spread 形式更清楚：最小化它就是在鼓励局域。

---

## 6. 最小版本里每一步公式到底在做什么

---

### 第一步：从 1DM 得到参考信息

从参考态测量 1-RDM：

$$
\gamma_{ij}=\langle c_i^\dagger c_j\rangle.
$$

对角化：

$$
\gamma
=
U_0\,\mathrm{diag}(n_1,\dots,n_M)\,U_0^\dagger.
$$

这一步给出两样东西：

1. **自然轨道方向** $U_0$；
2. **占据结构** $n_a$。

---

### 第二步：根据 occupation 做 hard blocks

这是当前实现里已经存在的部分：

- near-full
- active
- near-empty
- 以及同一 sector 内进一步按 gap 分块

因此真正优化的不是任意酉矩阵，而是块对角的 $V$。

---

### 第三步：写总变换

$$
U=U_0V.
$$

于是单体项和轨道形状都由 $V$ 决定。

---

### 第四步：主目标 $C_H$ 压缩 Hamiltonian complexity

- 通过 $\widehat C_t$ 压缩旋转后 hopping 的总预算与长程性；
- 通过 $\widehat C_{\rm int}$ 压缩 interaction exposure 的总预算与长程性。

这一步是整个优化的主要驱动力。

---

### 第五步：$C_{\rm occ}$ 防止 occupation 结构被涂抹

- 若 $V$ 试图把 occupation 差很大的轨道混起来，$C_\gamma$ 会迅速变大；
- 因此优化只能在“保留基本 occupation 分层”的前提下去简化 $H$。

---

### 第六步：$C_{\rm loc}$ 防止轨道跑得太野

- 它不负责决定最终结构；
- 它只是避免出现“虽然 $C_H$ 很小，但轨道非常离谱”的解。

---

## 7. 建议的默认参数

最小版本下，建议固定主目标前系数为 1，只调两个正则参数：

```python
CoreLossConfig(
    lambda_occ = 0.05,   # 或 0.1
    lambda_loc = 0.0,    # 第一轮可直接关掉；若开，可取 0.01~0.05
)
```

更具体一些：

- **第一轮最简测试**：  
  `lambda_occ = 0.05`, `lambda_loc = 0.0`
- **若发现 occupation 被涂抹明显**：  
  把 `lambda_occ` 提到 `0.1`
- **若发现轨道变得非常扩展**：  
  再把 `lambda_loc` 从 `0.0` 开到 `0.01~0.05`

不建议一开始就把 `lambda_occ` 开很大，否则优化会直接黏在 NO 基底附近动不了。

---

## 8. 两个可选升级路线，但不要一开始就开

---

### 升级路线 A：把 $\mathcal S$ 加回来做结构精修

当最小版本已经能稳定降低能量方差、简化 $H$ 以后，可以把 $\mathcal S$ 作为 second-stage refinement 加回来：

$$
\widehat C_t \to \alpha \widehat C_t + (1-\alpha)\mathcal S(|t'|^2),
$$

$$
\widehat C_{\rm int}\to \alpha \widehat C_{\rm int} + (1-\alpha)\mathcal S(W).
$$

---

### 升级路线 B：把 $C_\gamma$ 换成 $C_{\rm AF}$

若你后面更关心 active / frozen 边界，而不是一般意义上的 occupation structure，则把

$$
C_{\rm occ}=C_\gamma
\quad\to\quad
C_{\rm occ}=C_{\rm AF}.
$$

不要两者一起开。

---

## 9. 从 1DM 出发：自然轨道起步还是格点轨道起步？

这点也给一个简短结论，供未来扩展时参考。

### 当前这份 loss 所对应的默认路线

它默认是：

- 先从 1DM 得到 $U_0$；
- 再在 NO 基底内优化块对角旋转 $V$。

这条路线的优点是：

- occupation 结构最清楚；
- active / frozen 分层最自然；
- 很适合当前已有的 hard blocks 框架。

### 但更长期的更强路线是“1DM 提供结构信息，局域基底负责复杂度优化”

也就是说：

- 用 1DM 定义哪些子空间不能混；
- 但不必永远执着于精确的 NO 形状本身；
- 真正的主优化仍然应该落在 Hamiltonian complexity 上。

因此，当前最小版本仍然沿用 NO-first/hybrid 路线；  
但未来若要更强地优化局域性和 Hamiltonian 稀疏结构，可以考虑 site-first / projector-guided 的扩展版。

---

## 10. 一个可直接实现的最终版本（推荐）

综合上面的讨论，我建议当前先实现下面这个版本：

$$
\boxed{
\mathcal L_{\rm core}(V)
=
\frac12\left(\widehat C_t(V)+\widehat C_{\rm int}(V)\right)
+
\lambda_{\rm occ}\, C_\gamma(V)
+
\lambda_{\rm loc}\, C_{\rm loc}(V)
}
$$

其中

$$
\widehat C_t(V)
=
\frac{
\sum_{a<b}\omega_{ab}\sqrt{|t'_{ab}|^2+\varepsilon^2}
}{
\sum_{a<b}\omega_{ab}\sqrt{|t^{(0)}_{ab}|^2+\varepsilon^2}
+\varepsilon
},
$$

$$
\widehat C_{\rm int}(V)
=
\frac{
\sum_{a<b}\omega_{ab}\Lambda_{ab}\Xi_{ab}^{\rm NO}
}{
\sum_{a<b}\omega_{ab}W_{ab}^{(0)}+\varepsilon
},
$$

$$
C_\gamma(V)
=
\frac{
\left\|
\bigl(V^\dagger \mathrm{diag}(\mathbf n)V\bigr)_{\rm off-diag}
\right\|_F^2
}{
\sum_a n_a^2
},
$$

$$
C_{\rm loc}(V)
=
\frac1{N_{\rm orb}}
\sum_a
\left(
\langle r^2\rangle_a-|\langle r\rangle_a|^2
\right),
$$

$$
\omega_{ab}=1+\left(\frac{d_{ab}}{d_{\max}}\right)^2.
$$

---

## 11. 相关文献线索（支持思路，不是逐项照搬）

下面这些文献支持的是“为什么这样取最小版本是合理的”，而不是说本文 loss 直接照搬它们。

### (1) 自然轨道作为压缩波函数表示、改善变分优化的依据

- P. Besserve et al., **Natural orbitalized variational quantum eigensolving of extended impurity models within a slave-boson approach**, arXiv:2108.10780 / Phys. Rev. B 105, 115108 (2022).  
  关键词：NOization，利用 1-RDM 逐步转向近似自然轨道基底，使变分优化更快、更准确。

### (2) 直接优化 Hamiltonian 表示复杂度，而不只是做传统局域化

- E. Koridon et al., **Orbital transformations to reduce the 1-norm of the electronic structure Hamiltonian for quantum computing applications**, arXiv:2103.14753 / Phys. Rev. Research 3, 033127 (2021).  
  关键词：用 Hamiltonian 1-norm 直接做 orbital optimization cost function，说明“主目标直接落在 Hamiltonian complexity 上”是合理路线。

### (3) sign problem 更适合作为特定求解器下的 surrogate，而不是默认主目标

- D. Hangleiter et al., **Easing the Monte Carlo sign problem**, arXiv:1906.02309 / Quantum 4, 286 (2020).  
  关键词：average sign 难以直接拿来优化，而 non-stoquasticity / $\ell_1$-type surrogate 更可计算；也说明 sign proxy 和真正 average sign 不是一回事。

### (4) 自然轨道与 split localization 的取舍依赖问题规模和目标

- 近期 DMRG / active-space 相关工作显示：  
  小体系中 natural orbitals 往往更适合“能量局域”；  
  更大体系中 split-localized orbitals 常常更利于降低重叠、纠缠和 bond dimension。  
  这支持“当前先用 NO-first/hybrid，后续再考虑更强 locality-first 扩展”的路线。

---

## 12. 最后的实践建议

若现在你只想尽快把框架跑通，我建议严格按下面顺序来：

### 版本 1：最小可行
$$
\mathcal L=C_H+\lambda_{\rm occ}C_\gamma,
\qquad
\lambda_{\rm loc}=0.
$$

### 版本 2：若轨道明显跑太散
$$
\mathcal L=C_H+\lambda_{\rm occ}C_\gamma+\lambda_{\rm loc}C_{\rm loc}.
$$

### 版本 3：若 active/frozen 边界问题特别突出
把
$$
C_\gamma \to C_{\rm AF},
$$
而不是再额外加一个新项。

### 版本 4：当最小版本已经稳定
再考虑把 $\mathcal S$ 加回来做结构精修。

---

# 附：原文件与本修订版的关系

- 原文件保留了一个更全面的“候选项库”；  
- 本修订版则是把这些候选项压缩成一个最小可行版本；  
- 逻辑上并不冲突：  
  - 原文件适合后续扩展；  
  - 本修订版适合开始阶段落地。  

如果后面要继续升级，最自然的顺序是：

1. 先跑通本修订版；
2. 再把 $\mathcal S$ 加回来；
3. 最后再考虑 $C_{\rm AF}$ 或 solver-specific 的 sign surrogate。
