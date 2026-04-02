# 损失函数说明

## 总形式

$$\mathcal{L}(V) = \lambda_\gamma\, C_\gamma + \lambda_t\, C_t + \lambda_\text{int}\, C_\text{int} + \lambda_\text{loc}\, C_\text{loc}$$

$V$ 是作用在自然轨道空间内的块对角旋转矩阵（正交/酉），优化变量。
完整变换为 $U = U_0 V$，其中 $U_0$ 是 1-RDM 的自然轨道本征向量。

---

## 各项定义

### $C_\gamma$：占据数正则项

$$C_\gamma = \frac{\| (V^\dagger \,\mathrm{diag}(\mathbf{n})\, V)_\text{off-diag} \|_F^2}{\sum_i n_i^2}$$

- 分子是旋转后 1-RDM 的非对角范数平方；分母是归一化因子。
- $V = I$ 时 $C_\gamma = 0$（自然轨道基底）；任何旋转都会让它增大。
- **物理含义**：约束新轨道保持自然轨道的块结构（近满/活性/近空的分离）。
- **本质是正则项**，不是主优化目标。建议 $\lambda_\gamma \ll \lambda_\text{int}$。

---

### $C_t$：单体哈密顿量稀疏性

$$C_t = \mathcal{S}\!\left(|t'_{ab}|^2\right), \qquad t' = V^\dagger t_\text{NO}\, V$$

其中 $t_\text{NO} = U_0^\dagger\, t\, U_0$ 是 hopping 矩阵在自然轨道基底下的形式，
$\mathcal{S}(\cdot)$ 是结构代价函数（见下）。

- **约束对象**：H 的**单体部分**（动能/hopping $t_{ij}$）。
- 度量旋转后 hopping 矩阵的稀疏性和空间衰减，越集中/衰减越快越好。
- 需要传入 `hopping_matrix`（或完整 `H`）；不传时此项为 0。

---

### $C_\text{int}$：相互作用稀疏性（通用）

$$\Lambda_{ab} = \sum_{ij} S_{ij}\, |U_{ia}|^2 |U_{jb}|^2$$

$$\Xi_{ab} = n_a(1 - n_b) + n_b(1 - n_a)$$

$$C_\text{int} = \mathcal{S}(W_{ab}),\quad W_{ab} = \Lambda_{ab} \cdot \Xi_{ab}$$

其中 $U = U_0 V$ 是完整变换（格点 → 优化轨道），$S_{ij}$ 是**格点对之间的相互作用强度矩阵**。

**$\Lambda_{ab}$ 的物理含义**：轨道对 $(a, b)$ 在相互作用中的"曝光度"——轨道 $a$ 在格点 $i$ 的权重 $|U_{ia}|^2$ 乘以轨道 $b$ 在格点 $j$ 的权重 $|U_{jb}|^2$，再对所有格点对 $(i,j)$ 以 $S_{ij}$ 加权求和。

**$\Xi_{ab}$ 的物理含义**：相空间因子，衡量轨道对 $(a, b)$ 在低能激发中的散射可行性。$n_a \approx 1, n_b \approx 0$ 或反之时 $\Xi_{ab} \approx 1$（最大）；$n_a \approx n_b \approx 0.5$ 时也较大。

**$W_{ab} = \Lambda_{ab} \cdot \Xi_{ab}$**：同时编码了相互作用强度与占据结构，是衡量轨道对散射难度的核心量。

#### 如何传入 $S_{ij}$

| 模型 | `interaction_matrix` 参数 |
|------|--------------------------|
| Kondo $J_i$ | 传 1D 数组 `J`，自动当作 `diag(J)` |
| Heisenberg $J_{ij}$ | 传 2D 对称矩阵 `J_ij` |
| Kondo-Heisenberg 混合 | `np.diag(J_kondo) + J_heis` |
| Hubbard $U$（on-site） | `U * np.eye(n)` |
| 通用 $V_{ijkl}$ | 先边缘化：$S_{ij} = \sum_{kl} |V_{ijkl}|$ |

不传 `interaction_matrix` 时此项为 0，等价于 $\lambda_\text{int} = 0$。

---

### $C_\text{loc}$：轨道局域性正则项

$$C_\text{loc} = \langle \text{IPR}_a \rangle_a$$

基于逆参与率（inverse participation ratio），惩罚轨道过于离域。
提供 `site_positions` 时额外加入实空间 spread 项。
纯正则项，通常权重取小值（$\lambda_\text{loc} \sim 0.1$）。

---

## 结构代价函数 $\mathcal{S}(W)$

对任意非负权重矩阵 $W_{ab}$，$\mathcal{S}$ 度量其集中程度：

$$\text{participation} = \frac{P_\text{eff} - 1}{P_\text{max} - 1}, \quad P_\text{eff} = \frac{(\sum_{a<b} W_{ab})^2}{\sum_{a<b} W_{ab}^2}$$

$$\text{decay} = \frac{\sum_{a<b} W_{ab}\,(d_{ab}/d_\text{max})^p}{\sum_{a<b} W_{ab}}$$

$$\mathcal{S} = \frac{1}{2}(\text{participation} + \text{decay}) \quad (\text{hybrid 模式})$$

- **participation**：越集中在少数轨道对上，值越小（好）。
- **decay**：远程元素越小（按距离幂次加权），值越小（好）。
- `structure_metric` 参数控制使用哪种或混合。
- `distance_power` 控制衰减的幂次 $p$。

---

## 与哈密顿量的对应关系

| 损失项 | 约束 H 的哪一部分 | 计算方式 |
|--------|-------------------|----------|
| $C_\gamma$ | 无（仅约束旋转幅度） | 直接对 1-RDM |
| $C_t$ | 单体部分 $t_{ij}$ | 直接旋转矩阵 |
| $C_\text{int}$ | 相互作用部分 | 代理量 $\Lambda_{ab}$（格点权重乘积） |
| $C_\text{loc}$ | 无（纯正则） | 轨道 IPR |

$C_\text{int}$ 的代理量避免了直接计算 $O(n^8)$ 的两体积分旋转，
对于格点型相互作用（Kondo、Heisenberg、Hubbard）是精确或高度近似的。

---

## 权重调参建议

```python
NOOptimizationConfig(
    lambda_occupancy  = 0.1,   # 小值：允许离开 NO 基底
    lambda_hopping    = 1.0,   # 中等：hopping 稀疏性
    lambda_interaction= 2.0,   # 大值：主要物理目标
    lambda_locality   = 0.1,   # 小值：防止轨道过度离域
)
```

若相互作用比 hopping 更难截断，可适当提高 `lambda_interaction`。
若轨道局域性不重要（如均匀系统），可令 `lambda_locality = 0`。
