# optimize_no / soft / barrier 对比与选择

> 阅读建议：
> 如果你现在不知道该选哪个代码版本，先看这份文档。
> 如果你选定了 `soft`，下一步看 `loss_function_revised_minimal.md`；
> 如果你选定了 `barrier`，再看 `loss_function_revised_full.md`。

这份文档只回答两个问题：

1. 原版 `optimize_no.py`、新的 `optimize_no_soft.py`、新的 `optimize_no_barrier.py` 到底有什么区别。
2. 实际使用时应该优先选哪一个。

---

## 1. 核心结论

- 三者共用同一套**硬分块**机制：先按自然轨道占据数分块，再只允许块内旋转，不同块之间不会混合。
- 三者的主要区别不在“块怎么分”，而在：
  - loss 的组织方式
  - interaction 项是否固定使用初始 NO occupation
  - 主目标更偏向“结构指标”还是“显式压预算”
  - occupation 结构保护是普通 soft penalty 还是 barrier

---

## 2. 区别表

| 项目 | 原版 `optimize_no.py` | `optimize_no_soft.py` | `optimize_no_barrier.py` |
|---|---|---|---|
| 总体定位 | 通用多目标版本 | 最小可行版 | 完整 barrier 版 |
| 主优化哲学 | 所有项并列加权 | 主目标最简化，先把框架跑稳 | 主目标优化 `H`，occupation 结构当作约束边界 |
| 总 loss | `lambda_occ*C_gamma + lambda_hopping*C_t + lambda_interaction*C_int + lambda_locality*C_loc` | `alpha_hopping*C_t_norm + alpha_interaction*C_int_norm + lambda_occupancy*C_gamma + lambda_locality*C_loc` | `C_H + mu_gamma*barrier(C_gamma-delta_gamma) + mu_active_frozen*barrier(C_af-delta_af) + lambda_locality*C_loc` |
| hopping 主项 | 结构代价 `structure_metric` | weighted smooth-L1 norm | norm 与 structure 混合 |
| interaction 主项 | 结构代价 `structure_metric` | weighted smooth-L1 norm | norm 与 structure 混合 |
| interaction 相空间 | 用旋转后的 `diag(gamma_rot)` 构造 | 固定用初始 NO occupation | 固定用初始 NO occupation |
| occupation 保护 | `lambda_occupancy * C_gamma` | `lambda_occupancy * C_gamma` | `mu_gamma * barrier(C_gamma - delta_gamma)` |
| active/frozen 保护 | 无单独项 | 无单独项 | 可选 `C_af` barrier |
| 局域性项 | IPR / participation + 可选 spread 混合 | 只用 spread | 只用 spread |
| 参数数量 | 中等 | 最少 | 最多 |
| 参数可解释性 | 一般 | 较清楚 | 最清楚 |
| 调参难度 | 中等 | 最低 | 最高 |
| 适合阶段 | 老框架兼容、对照实验 | 第一次跑、先找量级 | 已知可接受 occupation 扭曲上限后 |

---

## 3. 三者分别在做什么

### 原版 `optimize_no.py`

它更像一个“广义多目标框架”：

- 一方面用 `C_gamma` 约束不要偏离 NO 基底太远；
- 另一方面用 `C_t` 和 `C_int` 去让变换后的 Hamiltonian 更集中、更短程；
- 再加一个 `C_loc` 防止轨道太离域。

它的优点是结构完整、和最初设计一致。  
它的缺点是：

- 顶层权重都在同时 trade-off；
- `C_t` / `C_int` 更偏“结构形状”，不直接压总预算；
- `C_int` 会随着旋转后的 occupation 一起变，容易和 `C_gamma` 相互牵扯。

---

### `optimize_no_soft.py`

它是最小版，目的是先把优化做得**简洁、稳定、好调**。

它做的事情是：

- 用 weighted smooth-L1 直接压 hopping 和 interaction 的总复杂度预算；
- 用一个小的 `lambda_occupancy * C_gamma` 保住 occupation 结构；
- 局域性只保留最直接的 spread 项，而且默认可以先关掉。

它的优点是：

- 参数少；
- 行为直接；
- 适合先观察 `C_H` 和 `C_gamma` 的量级关系；
- interaction 相空间固定用初始 NO occupation，不会“边转边改规则”。

它的缺点是：

- occupation 保护始终在线；
- 没有显式“允许扭曲到什么程度”的阈值概念；
- 如果 `lambda_occupancy` 不合适，仍然会出现“拉回太强”或“保护不够”的问题。

---

### `optimize_no_barrier.py`

它是更符合文档逻辑的“完整版”：

- 主目标 `C_H` 只负责简化 Hamiltonian；
- `C_gamma` 不再作为普通 soft penalty，而是作为 barrier；
- 如有需要，还能加 `active/frozen` barrier；
- `C_t` / `C_int` 都可以在 norm 和 structure 之间做混合。

它的优点是：

- 概念最清楚；
- 更接近“occupation 结构是约束，不是主目标”；
- 更容易表达“允许偏离多少”。

它的缺点是：

- 参数更多；
- `delta`、`mu`、`tau` 会耦合；
- 比 soft 版更难一次调好。

---

## 4. 选择表

| 你的目标 / 当前情况 | 推荐版本 | 原因 |
|---|---|---|
| 第一次把框架跑起来 | `optimize_no_soft.py` | 参数最少，最容易看清主目标与 occupation 正则的量级关系 |
| 先快速看这个系统能不能从 NO 基底中获益 | `optimize_no_soft.py` | minimal 版最适合做首轮扫描 |
| 想和旧实现直接对照 | 原版 `optimize_no.py` | 保持最接近原设计与原有指标 |
| 很在意 pair-weight 是否集中、是否短程 | 原版 `optimize_no.py` 或 `barrier` | 这两版都显式保留 structure metric |
| 更在意显式压低 Hamiltonian complexity 总预算 | `optimize_no_soft.py` | 它的主目标直接是 weighted smooth-L1 norm |
| 已经知道 occupation 扭曲最多允许到什么程度 | `optimize_no_barrier.py` | 可以直接用 `delta_gamma` 表达容忍边界 |
| 觉得 soft penalty 一直把解往 NO 拉回，太难调 | `optimize_no_barrier.py` | barrier 在安全区内干预更小 |
| 需要专门控制 active/frozen 混合 | `optimize_no_barrier.py` | 有单独的 `active_frozen` barrier |
| 只想要最简洁可靠、函数层次尽量浅 | `optimize_no_soft.py` | 逻辑最直、嵌套最少 |

---

## 5. 实用选择建议

推荐按下面顺序使用：

### 方案 A：大多数情况

1. 先用 `optimize_no_soft.py`
2. 观察：
   - `final_loss_terms["hopping"]`
   - `final_loss_terms["interaction"]`
   - `final_loss_terms["occupancy"]`
3. 如果发现 occupation 保护很难调，再升级到 `optimize_no_barrier.py`

这是最稳的默认路线。

---

### 方案 B：你已经很明确 occupation 只能扭曲一点点

直接用 `optimize_no_barrier.py`。

适合这种情况：

- 你已经从经验上知道 `C_gamma` 大致应控制在什么量级；
- 或者你明确希望“阈值以内基本不管，超过以后强力拉回”。

---

### 方案 C：你想保留和原始实现最直接的一致性

继续用原版 `optimize_no.py`。

适合这种情况：

- 你在做旧结果复现；
- 你希望继续沿用原始的 `structure_metric` 视角；
- 你要和旧脚本、旧记录直接比。

---

## 6. 最后一句建议

如果没有特别强的先验，推荐默认顺序是：

1. `optimize_no_soft.py`
2. `optimize_no_barrier.py`
3. 原版 `optimize_no.py` 作为对照

也就是说：

- `soft` 版负责先把问题跑通；
- `barrier` 版负责在你已经知道边界后做更物理的约束优化；
- 原版主要保留作兼容、对照和结构指标参考。
