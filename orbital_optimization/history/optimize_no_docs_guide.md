# optimize_no 文档导航

这份文档只做一件事：说明当前 `orbital_optimization` 目录里与 `optimize_no` / loss 设计有关的 Markdown 文件应该怎么读。

---

## 当前保留的文档

### 1. `loss_function_revised_full.md`

用途：

- 完整理论主文档
- 解释原版 loss 的问题、完整版推荐形式、soft 与 barrier 的关系
- 适合在你想统一理解整套设计逻辑时阅读

什么时候看：

- 你想弄清楚为什么要把主目标和 occupation 保护分开
- 你想理解 `soft` 和 `barrier` 的理论区别
- 你想看更完整的公式、约束形式和设计动机

---

### 2. `loss_function_revised_minimal.md`

用途：

- 最小可行版说明
- 面向“先把东西跑起来”的最短路径
- 对应 `optimize_no_soft.py` 的主要设计思路

什么时候看：

- 你现在主要想先跑通优化
- 你不想一开始就调太多参数
- 你想快速看推荐默认参数和实践顺序

---

### 3. `optimize_no_variant_comparison.md`

用途：

- 比较原版 `optimize_no.py`
- 新 `optimize_no_soft.py`
- 新 `optimize_no_barrier.py`

三者的区别和选择建议。

什么时候看：

- 你不知道该用哪个实现版本
- 你想快速比较三者的 loss 结构和适用场景

---

### 4. `optimize_no_design.md`

用途：

- 原版 `optimize_no.py` 的接口说明
- 偏 API / 调用层文档，不是理论文档

什么时候看：

- 你要查函数入口、返回字段、调用方式
- 你在对接旧版 `optimize_no.py`

---

## 推荐阅读顺序

### 路线 A：第一次接触

1. 先看 `optimize_no_variant_comparison.md`
2. 再看 `loss_function_revised_minimal.md`
3. 真要深究理论时，再看 `loss_function_revised_full.md`

---

### 路线 B：直接上手跑代码

1. 看 `loss_function_revised_minimal.md`
2. 用 `optimize_no_soft.py` 和对应 template 先跑
3. 如果 soft 难调，再回看 `loss_function_revised_full.md` 里的 barrier 设计

---

### 路线 C：要查旧实现接口

1. 看 `optimize_no_design.md`
2. 再结合 `optimize_no_variant_comparison.md` 看新旧版本差异

---

## 一句话总结

- 理论总文档：`loss_function_revised_full.md`
- 最小落地文档：`loss_function_revised_minimal.md`
- 版本选择文档：`optimize_no_variant_comparison.md`
- 原版接口说明：`optimize_no_design.md`
