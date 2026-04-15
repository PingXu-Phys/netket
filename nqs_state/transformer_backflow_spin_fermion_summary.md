# Transformer 自旋+费米子 backflow 架构中如何处理 $N_\uparrow \neq N_\downarrow$

## 先说结论

这个问题要分成两层来看：

1. **论文本身怎么做**
   - Rende et al. (2026) 用的是统一的 spin-orbital generalized determinant。
   - 在论文的写法里，不需要先把电子拆成两个独立的 up/down determinant 块。
   - 因此某个具体构型里出现 $N_\uparrow \neq N_\downarrow$ 时，只是“被选中的 spin-orbital 行数不一样”，不需要额外判断步骤。

2. **当前仓库实现怎么做**
   - 当前 `transformer_joint_nnbf.py` 不是只实现论文那一种 backend，而是实现了 **adaptive determinant backend**。
   - 如果 Hilbert 只固定总电子数 `n_fermions`，就走 generalized determinant。
   - 如果 Hilbert 固定了 `n_fermions_per_spin=(N_dn, N_up)`，就走 blocked determinant。
   - 这两条路径现在都应该支持费米子自旋不平衡；也就是说，`N_up != N_dn` 不再默认意味着必须走 `S^z_f = 0` 的特殊情形。

## 1. 论文中的原始结构

论文里，每个格点的局域构型写作

$$
s_i = (n_{i\uparrow}, n_{i\downarrow}, S^z_{1i}, S^z_{2i}),
$$

Transformer 输出依赖整个 many-body 构型的 backflow 轨道

$$
\Phi_{i\sigma\alpha}(s),
$$

并把格点指标 $i$ 与自旋指标 $\sigma$ 合并成一个复合单粒子指标

$$
r = (i, \sigma).
$$

于是轨道矩阵写成

$$
\Phi_{r\alpha} \in \mathbb{R}^{2N \times N_e}.
$$

最后的 many-body 振幅是：从所有 `2N` 个 spin-orbital 行中，按照当前构型的占据情况选出被占据的那些行，再取一个总 determinant。

所以对论文本身来说：

- $N_\uparrow$ 和 $N_\downarrow$ 不是两个需要单独传给 determinant backend 的静态尺寸；
- 它们是由当前构型的 occupation pattern 动态决定的；
- $N_\uparrow \neq N_\downarrow$ 只是“选中的 up/down 行数不同”。

## 2. 当前仓库实现的两条 determinant 路径

当前仓库里的 `netket/nqs_state/transformer_joint_nnbf.py` 不是把所有情况都强行写成论文式 generalized determinant，而是根据 Hilbert 约束自动选择 backend。

### 2.1 固定总电子数时

如果 Hilbert 只固定总电子数 `n_fermions`，则使用 generalized determinant：

$$
\Phi_{\mathrm{all}} \in \mathbb{R}^{(2N) \times N_f}.
$$

这里：

- `N_f = n_fermions`；
- 行空间是完整的 spin-orbital 空间；
- 当前构型中到底有多少个 up 与 down，直接由 occupation block 给出；
- 因此同一个 `n_fermions` 扇区里，费米子 $S^z_f$ 可以变化，而不需要改网络结构。

这条路径对应完整 Kondo 物理最自然的情形：

- `J_K != 0`
- 电子自旋可翻转
- 不能固定 `N_up/N_dn`
- 应只固定总电子数，并在需要时固定联合总 `S^z`

### 2.2 固定每个自旋分量粒子数时

如果 Hilbert 固定了 `n_fermions_per_spin=(N_dn, N_up)`，则当前实现走 blocked determinant：

$$
\det \Phi_{dn} \cdot \det \Phi_{up}.
$$

这里的关键点是：**两个 backflow head 必须分别按 `N_dn` 和 `N_up` 来定宽度。**

也就是说，当前实现应当构造：

- $\Delta \Phi_{dn} \in \mathbb{R}^{N \times N_{dn}}$
- $\Delta \Phi_{up} \in \mathbb{R}^{N \times N_{up}}$

而不是默认把两个 head 都做成同样大小。

这件事对 `N_up != N_dn` 很关键，因为：

- 如果两个 head 被错误地做成同一列数，那么一旦费米子 `S^z_f != 0`，down/up 两个 determinant 的矩阵尺寸就会不匹配；
- 修正后的接口允许在固定 `N_dn/N_up` 的 block-det 路径里处理自旋不平衡扇区，而不是默认只支持半填充或 `S^z_f = 0`。

## 3. 所以，是否需要“先根据 $S^z$ 判断当前用了几个 up/down 电子”？

### 3.1 在论文的 generalized determinant 里

不需要。

原因是：

- 当前构型本身就已经包含所有 occupation number；
- `N_up = \sum_i n_{i\uparrow}`，`N_dn = \sum_i n_{i\downarrow}`；
- determinant backend 直接根据当前占据向量去选行；
- 没有额外的“先判断，再切换 block 大小”的步骤。

### 3.2 在当前仓库的 blocked determinant 里

也不需要对“每一个样本”动态判断 block 大小。

因为在 blocked determinant 路径中：

- `N_dn` 和 `N_up` 是 Hilbert 的静态约束；
- 它们在整个变分态里本来就是固定的；
- 网络初始化时就应当为 down/up 两个 sector 分别分配对应宽度的 backflow 输出。

所以这里需要的不是“根据样本运行时判断”，而是“根据 Hilbert 约束在模型构造时给两个 head 正确分配静态尺寸”。

## 4. 与总 $S^z$ 的关系

在完整 Kondo 链里，更自然的守恒量通常不是电子子系统单独的 $N_\uparrow$ 与 $N_\downarrow$，而是整体的联合量，例如总电子数和联合总 $S^z$。

因此：

- 对 `J_K != 0` 的 full Kondo 情况，优先使用 `n_fermions` + generalized determinant；
- 对 `J_K = 0` 或显式固定 `N_dn/N_up` 的问题，blocked determinant 仍然是合法而高效的实现；
- 非零费米子 $S^z_f$ 并不要求你把整个接口都改成 generalized determinant，它只要求 block-det 路径正确支持 `N_dn != N_up`。

## 5. 采样层面怎么理解

这和 determinant backend 是两件相关但不同的事：

- determinant backend 决定的是波函数振幅如何从给定样本中计算出来；
- sampler 决定的是 proposal 如何在 Hilbert 空间中走。

对完整 Kondo 问题：

- 如果希望 proposal 也反映费米子自旋与局域自旋的耦合翻转，优先使用 `MetropolisHamiltonian`；
- 如果只是想比较显式分块 proposal，`TensorRule` 的 factorized sampler 仍然可以用，但它不会在单个 proposal 步里直接生成 Kondo 的耦合翻转。

## 6. 一句话总结

- **论文本身**：`N_up != N_dn` 通过统一的 spin-orbital generalized determinant 自动处理。
- **当前仓库实现**：采用 adaptive determinant backend。
  - 固定总电子数时，用 generalized determinant。
  - 固定 `N_dn/N_up` 时，用 blocked determinant，并且 up/down 两个 backflow head 必须分别按 `N_dn` 与 `N_up` 定宽。
- 因而“费米子 $S^z_f$ 不一定为 0”本身不是问题；真正重要的是：**determinant backend 必须和 Hilbert 约束保持一致。**