# orbital_optimization 目录清单

下面这份清单只做一件事：说明当前目录下每个主要文件大概在干什么，并顺手标出哪些地方像是旧版本遗留或路径未更新，方便后续清理。

| 文件 | 当前作用 | 当前状态 / 备注 |
| --- | --- | --- |
| `optimization_flow.md` | 最新的总流程文档，已经把“稠密优化 -> 连续阈值化 -> 稳定结构 -> 稀疏重优化”写清楚了 | 建议保留，应该视为当前主纲 |
| `loss_function_revised_minimal.md` | 最小可行 loss 的理论说明，主要对应 `soft` 版 | 建议保留，适合作为默认理论入口 |
| `loss_function_revised_full.md` | 完整版 loss 理论文档，包含 barrier 设计与更完整动机 | 建议保留，适合作为完整版参考 |
| `optimize_no.py` | 原始 monolithic 实现，里面既有工具函数，也有 original loss，也有优化循环 | 建议先保留作 baseline 和工具来源，但不建议继续往里加新 sparse flow 逻辑 |
| `optimize_no_core.py` | 已抽出的公共优化骨架：base context、block 参数化、Optax 循环、Hamiltonian 旋转、结果打包 | 建议重点保留，后续应继续成为主骨架 |
| `optimize_no_losses.py` | 已抽出的 loss 构造层：soft / barrier / original config 与 loss builder | 建议重点保留，是未来最适合继续扩展的 loss 层 |
| `optimize_no_soft.py` | soft 版外层包装，负责准备 context、调用 core + soft loss module | 建议保留，作为 dense stage 的简洁入口 |
| `optimize_no_barrier.py` | barrier 版外层包装，负责准备 context、调用 core + barrier loss module | 建议保留，作为 dense stage 的约束强化入口 |
| `optimize_no_template.py` | 原版 `optimize_no.py` 的调用模板 | 可保留作旧版参考；导入路径用的是 `orbital_optimization.optimize_no`，相对较新 |
| `optimize_no_soft_template.py` | soft 版调用模板 | 内容有用，但导入路径写成了 `netket.graph_sample.optimize_no_soft`，与当前目录不一致，属于待更新项 |
| `optimize_no_barrier_template.py` | barrier 版调用模板 | 内容有用，但导入路径写成了 `netket.graph_sample.optimize_no_barrier`，与当前目录不一致，属于待更新项 |
| `optimize_no_design.md` | 设计/API 文档，解释了 `core + losses + wrappers` 的架构思路 | 内容有参考价值，但文内很多链接和导入示例仍指向 `graph_sample`，属于半过时文档 |
| `optimize_no_docs_guide.md` | 文档导航页，告诉读者先看哪些 md | 可保留，但文案还在说“graph_sample 目录”，建议后续一起改路径 |
| `optimize_no_variant_comparison.md` | 比较 `optimize_no.py`、`soft`、`barrier` 三种实现 | 建议保留，适合快速选型 |

## 额外观察

### 1. 当前目录真正的主线

如果只看“以后继续开发应该围绕谁转”，当前最值得作为主线的是：

- `optimization_flow.md`
- `optimize_no_core.py`
- `optimize_no_losses.py`
- `optimize_no_soft.py`
- `optimize_no_barrier.py`

### 2. 当前目录最明显的旧痕迹

最明显的旧痕迹有两类：

- 一些 md 还在写 `graph_sample` 路径；
- `soft/barrier` 两个 template 的导入路径也还是 `graph_sample`。

这说明目录经历过一次位置迁移或结构重组，但文档和模板没有完全跟上。

### 3. 最适合新增的文件

如果接下来要把 `flow` 真正落成代码，最自然的新文件不是再来一个新的 `optimize_no_xxx.py`，而是这几类：

- `optimize_no_utils.py`
- `optimize_no_pruning.py`
- `optimize_no_sparse.py`
- `optimize_no_flow_runner.py`
- 可选：`optimize_no_netket_eval.py`

### 4. 我对清理顺序的建议

如果后面要清理，我建议顺序是：

1. 先不删任何实现文件；
2. 先把 stale 路径文档和模板标出来；
3. 等 `pruning / sparse / runner` 落地后，再决定 `optimize_no.py` 是否只留作 baseline；
4. 文档层最后统一改一次路径和入口说明。
