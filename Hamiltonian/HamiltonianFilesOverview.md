# Hamiltonian 目录文件说明

本文档整理了下面 12 个文件的职责、主要功能和相互关系：

- `Hubbard.py`
- `HubbardExample.py`
- `HubbardPlot.py`
- `KondoHeisenberg_constraint_notes.md`
- `KondoHeisenberg.py`
- `KondoHeisenbergChain.py`
- `KondoHeisenbergChainExample.py`
- `KondoHeisenbergChainPlot.py`
- `KondoHeisenbergChainSpinFermion.py`
- `KondoHeisenbergExample.py`
- `KondoHeisenbergPlot.py`
- `KondoHeisenbergSpinFermion.py`

## 总体结构

这些文件大致分成 4 类：

1. **主模型实现**
   - `Hubbard.py`
   - `KondoHeisenberg.py`
   - `KondoHeisenbergChain.py`
2. **最小示例**
   - `HubbardExample.py`
   - `KondoHeisenbergExample.py`
   - `KondoHeisenbergChainExample.py`
3. **可视化脚本**
   - `HubbardPlot.py`
   - `KondoHeisenbergPlot.py`
   - `KondoHeisenbergChainPlot.py`
4. **补充说明 / 包装接口**
   - `KondoHeisenberg_constraint_notes.md`
   - `KondoHeisenbergSpinFermion.py`
   - `KondoHeisenbergChainSpinFermion.py`

---

## 1. `Hubbard.py`

### 文件作用
这是 **2D Fermi-Hubbard 模型** 的主实现文件，负责在 NetKet 中构造哈密顿量和对应的自旋费米子 Hilbert 空间。

### 主要功能
- 定义二维矩形晶格 `Lx x Ly`。
- 支持最近邻跃迁 `t1`。
- 支持对角线次近邻跃迁 `t2`。
- 支持同一格点上的 Hubbard 相互作用 `U n_up n_dn`。
- 支持 `x/y` 两个方向分别设置开放边界或周期边界。
- 支持固定总电子数 `n_fermions`，或固定每个自旋分量的电子数 `n_fermions_per_spin`。
- 如果没有显式传入粒子数，默认取半填充。

### 关键接口
- `hubbard_index(x, y, Ly)`
  - 把二维坐标 `(x, y)` 映射成单个 site 编号。
- `hubbard_geometry(...)`
  - 生成几何信息和键列表。
  - 输出最近邻 `shell_1` 和对角次近邻 `shell_2`。
- `build_hubbard_hilbert(...)`
  - 构造 `nk.hilbert.SpinOrbitalFermions`。
- `Hubbard(...)`
  - 构造完整哈密顿量 `H` 和 Hilbert 空间 `hi`。
- `build_hubbard`
  - `Hubbard(...)` 的别名。

### 适用场景
- 作为 2D Hubbard 模型的核心建模文件直接导入。
- 给绘图脚本 `HubbardPlot.py` 提供统一的几何信息。
- 给示例脚本 `HubbardExample.py` 提供最小可运行接口。

---

## 2. `HubbardExample.py`

### 文件作用
这是 `Hubbard.py` 的**最小示例脚本**，主要用于快速验证模型是否能正常构造。

### 主要功能
- 调用 `Hubbard(...)` 构造一个 `4x4` 的模型。
- 示例参数包括：
  - `t1=1.0`
  - `t2=0.2`
  - `U=8.0`
  - `pbc_x=True`
  - `pbc_y=False`
  - `n_fermions_per_spin=(8, 8)`
- 打印：
  - `H` 的类型
  - `hi` 的类型
  - Hilbert 空间大小 `hi.size`

### 适用场景
- 快速 smoke test。
- 检查 `Hubbard.py` 的接口是否可直接调用。
- 给后续写训练脚本的人一个最小调用模板。

---

## 3. `HubbardPlot.py`

### 文件作用
这是 `Hubbard.py` 的**可视化脚本**，用来把二维 Hubbard 模型的几何结构和自旋层结构画出来。

### 主要功能
- 动态加载同目录下的 `Hubbard.py` 或 `Hubbard_staging.py`。
- 根据 `hubbard_geometry(...)` 的输出绘图。
- 左图画空间晶格结构：
  - `shell_1` 最近邻键
  - `shell_2` 对角次近邻键
- 右图画“自旋分层图”：
  - 上层是 `spin up`
  - 下层是 `spin down`
  - 用虚线表示 onsite `U`
- 支持命令行参数：`Lx`、`Ly`、`pbc_x`、`pbc_y`、`t1`、`t2`、`U`、`save`、`no-show` 等。
- 提供：
  - `plot_hubbard_layout(...)`
  - `quick_plot(...)`
  - `main()`

### 特点
- 这个文件不负责物理求解，只负责结构展示。
- 支持把图保存成文件，也支持交互式显示。
- 对周期边界键做了弧线显示，便于区分“绕回去”的连接。

---

## 4. `KondoHeisenberg_constraint_notes.md`

### 文件作用
这是一个**设计说明文档**，专门解释 `KondoHeisenberg.py` 和 `KondoHeisenbergChain.py` 在约束选择上的物理与实现问题。

### 主要内容
- 指出当 `J_K != 0` 时，full Kondo 项包含自旋翻转：
  - `s_plus S_minus`
  - `s_minus S_plus`
- 因而以下量不再单独守恒：
  - `N_up`
  - `N_dn`
  - 局域自旋部分的单独 `S^z`
- 所以在 `J_K != 0` 时，不能继续使用：
  - `n_fermions_per_spin`
  - `local_total_sz`
- 真正守恒的是电子和局域自旋的**联合总磁量子数**。
- 文档给出了两条修正路线：
  1. 保留 full Kondo 物理，改用总电子数 `n_fermions`，必要时再加联合总 `S^z` 约束。
  2. 如果必须固定 `N_up/N_dn`，则把 Kondo 项改成 Ising 型或去掉横向翻转项。

### 价值
- 它不是运行脚本，而是帮助理解为什么某些参数组合会报错。
- 对以后修改模型接口或约束设计很重要。

---

## 5. `KondoHeisenberg.py`

### 文件作用
这是 **广义 2D Kondo-Heisenberg ladder 模型** 的主实现文件，也是这一组文件里功能最完整的一个核心模块。

### 物理模型
每个空间格点 `(x, leg)` 上同时有两类自由度：
- 一个自旋费米子轨道（导电电子）
- 一个局域自旋 `S=1/2`

哈密顿量包含三部分：
- 电子跃迁项
- 局域自旋之间的 Heisenberg / XXZ 相互作用
- 同站点的 Kondo 耦合 `s_i · S_i`

### 主要功能
- 描述 `Lx x n_legs` 的梯子或多腿系统。
- 支持三类键壳层：
  - `shell_1` 最近邻
  - `shell_2` 对角线键
  - `shell_3` 轴向第三近邻
- 支持电子跃迁参数：`t1, t2, t3`
- 支持局域自旋耦合参数：`J1, J2, J3`
- 支持 Kondo 耦合 `J_K`
- 支持局域自旋 XXZ 各向异性参数 `delta_z`
- 支持 `x` 方向和腿方向的边界条件控制
- 构造张量积 Hilbert 空间：`fermion_hi * local_spin_hi`
- 提供联合总 `2S^z` 的计算与初始化辅助函数

### 关键接口
- `ladder_index(x, leg, n_legs)`
  - 把二维 ladder 坐标映射到单个 site 编号。
- `kondo_heisenberg_geometry(...)`
  - 生成几何结构和各壳层键列表。
- `build_kondo_heisenberg_hilbert(...)`
  - 构造张量积 Hilbert 空间。
- `joint_total_two_sz_operator(...)`
  - 返回联合总 `2S^z` 算符。
- `joint_total_two_sz_of_states(...)`
  - 对采样态直接计算联合总 `2S^z`。
- `joint_sector_reference_state(...)`
  - 生成某个联合总 `2S^z` 扇区中的参考构型。
- `seed_joint_sz_sector(vstate, two_sz=0)`
  - 在不改变 Hilbert 空间定义的前提下，把 MCState 采样器初始化到指定联合扇区。
- `KondoHeisenberg(...)`
  - 构造广义梯子模型哈密顿量。
- `KondoHeisenbergChain(...)`
  - 在本文件里提供的链模型便捷包装器。

### 重要实现特点
- 当 `J_K != 0` 时，代码显式禁止：
  - `n_fermions_per_spin`
  - `local_total_sz`
- 推荐做法是：
  - 用 `n_fermions` 固定总电子数
  - 用 `seed_joint_sz_sector(...)` 固定联合总磁量子数扇区
- 因为 full Kondo 项包含自旋翻转，所以这部分约束逻辑是物理一致性的关键。

### 额外说明
- 文件末尾还提供了 `build_kondo_heisenberg` 和 `build_kondo_heisenberg_chain` 两个别名。
- 对于真正专门的 1D 链接口，作者也明确建议优先看 `KondoHeisenbergChain.py`。

---

## 6. `KondoHeisenbergChain.py`

### 文件作用
这是 **专门的 1D Kondo-Heisenberg 链模型** 实现文件。虽然 `KondoHeisenberg.py` 里已经提供了链的包装器，但这个文件是单独为 1D 情况写的，更直接、更清晰。

### 物理模型
每个空间点只有一维链坐标 `x`，但仍有两类自由度：
- 导电电子链
- 局域自旋链

哈密顿量包括：
- 电子最近邻跃迁 `t`
- 同站点 Kondo 耦合 `J_K`
- 局域自旋最近邻 `J1`
- 局域自旋次近邻 `J2`

### 主要功能
- 提供 1D 链几何：
  - `nn` 最近邻键
  - `nnn` 次近邻键
- 构造 1D 链的张量积 Hilbert 空间。
- 提供与二维广义版一致的联合 `2S^z` 工具函数。
- 构造专用链哈密顿量 `KondoHeisenbergChain(...)`。
- 支持 `delta_z` 控制局域自旋 XXZ 各向异性。
- 支持开边界或周期边界 `pbc`。

### 关键接口
- `kondo_heisenberg_chain_geometry(...)`
- `build_kondo_heisenberg_chain_hilbert(...)`
- `joint_total_two_sz_operator(...)`
- `joint_total_two_sz_of_states(...)`
- `joint_sector_reference_state(...)`
- `seed_joint_sz_sector(...)`
- `KondoHeisenbergChain(...)`

### 特点
- 与广义多腿版本相比，这个文件去掉了 `shell_1/shell_2/shell_3` 的通用结构，直接用 `nn/nnn` 表达 1D 链。
- 更适合只研究一维系统时直接使用。
- 同样处理了 `J_K != 0` 时约束与联合总 `S^z` 扇区的问题。

---

## 7. `KondoHeisenbergChainExample.py`

### 文件作用
这是 `KondoHeisenbergChain.py` 的最小演示脚本。

### 主要功能
- 构造一个 `Lx=8` 的 1D Kondo-Heisenberg 链。
- 示例参数：
  - `t=1.0`
  - `J_K=1.0`
  - `J1=1.0`
  - `J2=0.3`
  - `pbc=False`
  - `n_fermions=8`
- 用 `joint_sector_reference_state(...)` 构造联合总 `2S^z = 0` 的参考态。
- 用 `joint_total_two_sz_of_states(...)` 验证该参考态所在扇区。
- 打印哈密顿量类型、Hilbert 类型、Hilbert 大小和参考态扇区值。

### 适用场景
- 快速验证链模型接口。
- 给训练/测试脚本提供最小参考。

---

## 8. `KondoHeisenbergChainPlot.py`

### 文件作用
这是 **1D Kondo-Heisenberg 链的绘图脚本**。

### 主要功能
- 动态加载 `KondoHeisenbergChain.py` 或 `KondoHeisenbergChain_staging.py`。
- 画出两条平行的线：
  - 上面是 conduction chain
  - 下面是 local-spin chain
- 用不同颜色区分：
  - 电子跃迁 `t`
  - 局域自旋最近邻 `J1`
  - 局域自旋次近邻 `J2`
  - 竖直虚线 `J_K`
- 支持命令行参数：`Lx`、`pbc`、`t`、`J_K`、`J1`、`J2`、`save`、`no-show` 等。
- 提供：
  - `plot_kondo_heisenberg_chain(...)`
  - `quick_plot(...)`
  - `main()`

### 特点
- 这个图非常适合讲清楚“1D Kondo-Heisenberg chain 并不是两条空间腿，而是同一组空间点上的两类自由度”。
- 周期边界绕回连接也通过弧线处理。

---

## 9. `KondoHeisenbergChainSpinFermion.py`

### 文件作用
这是 **1D 链模型的结构化 spin-fermion 包装接口**。

### 主要功能
- 在不改变底层物理模型的前提下，把 `KondoHeisenbergChain.py` 的返回结果包装成一个 dataclass：
  - `hamiltonian`
  - `joint_hilbert`
  - `fermion_hilbert`
  - `local_spin_hilbert`
  - `geometry`
- 提供与底层链模型一致的 Hilbert 构造入口：
  - `build_kondo_heisenberg_chain_spin_fermion_hilbert(...)`
  - 支持 `n_fermions`
  - 支持 `n_fermions_per_spin`
  - 支持 `local_total_sz`
- 提供主构造函数：
  - `KondoHeisenbergChainSpinFermion(...)`
- 同模块重新导出了链模型里的联合扇区辅助接口：
  - `joint_total_two_sz_operator(...)`
  - `joint_total_two_sz_of_states(...)`
  - `joint_sector_reference_state(...)`
  - `seed_joint_sz_sector(...)`
  - `kondo_heisenberg_chain_geometry(...)`

### 适用场景
- 适合变分波函数或神经网络 ansatz 想把“电子部分”和“局域自旋部分”分开处理的情况。
- 方便外部代码直接拿到分离后的两个子空间，而不是只拿到整体 TensorHilbert。
- 现在也适合在 `J_K != 0` 时继续使用总电子数 `n_fermions` 加联合总 `S^z` 扇区初始化的完整工作流。

### 当前接口特点
- 这个包装层现在和底层 `KondoHeisenbergChain.py` 更接近，既保留了“结构化返回值”的优点，也补齐了 full Kondo 情况下最常用的约束接口。
- 如果你希望从同一个模块同时完成：
  - 构造结构化系统对象
  - 访问 `fermion_hilbert` / `local_spin_hilbert`
  - 初始化联合总 `S^z` 扇区
  那么这个文件现在已经可以直接承担这套工作流。

---

## 10. `KondoHeisenbergExample.py`

### 文件作用
这是 `KondoHeisenberg.py` 的最小示例脚本，对应广义多腿梯子版本。

### 主要功能
- 构造一个 `Lx=4, n_legs=2` 的模型。
- 示例参数包括：
  - `t1=1.0, t2=0.2, t3=0.0`
  - `J_K=1.0`
  - `J1=1.0, J2=0.3, J3=0.0`
  - `pbc_x=True, pbc_y=False`
  - `n_fermions=8`
- 构造联合总 `2S^z = 0` 的参考态并验证其扇区值。
- 打印哈密顿量类型、Hilbert 类型、Hilbert 大小和参考态扇区信息。

### 作用定位
- 说明在 `J_K != 0` 的标准 full Kondo 情况下，应该如何使用 `n_fermions` 和联合扇区参考态。
- 是理解 `KondoHeisenberg.py` 推荐用法的最短入口。

---

## 11. `KondoHeisenbergPlot.py`

### 文件作用
这是 **广义 2D Kondo-Heisenberg ladder** 的可视化脚本。

### 主要功能
- 动态加载 `KondoHeisenberg.py` 或 `KondoHeisenberg_staging.py`。
- 左图绘制空间几何：
  - `shell_1`
  - `shell_2`
  - `shell_3`
- 右图绘制分层结构：
  - `c layer` 表示导电电子层
  - `S layer` 表示局域自旋层
  - 站点间虚线表示 onsite `J_K`
- 可以分别显示：
  - 电子跃迁 `t1/t2/t3`
  - 自旋交换 `J1/J2/J3`
  - Kondo 耦合 `J_K`
- 支持命令行参数、保存图片和交互显示。
- 提供：
  - `plot_kondo_heisenberg_layout(...)`
  - `quick_plot(...)`
  - `main()`

### 特点
- 右图用了 pseudo-3D 的错层方式，让 `J_K` 的连线不会被站点遮住。
- 很适合检查你定义的几何壳层是否和预期一致。

---

## 12. `KondoHeisenbergSpinFermion.py`

### 文件作用
这是 **广义多腿 Kondo-Heisenberg 模型的结构化 spin-fermion 接口**，相当于 `KondoHeisenberg.py` 的包装层。

### 主要功能
- 使用 dataclass `SpinFermionKondoHeisenbergSystem` 返回：
  - `hamiltonian`
  - `joint_hilbert`
  - `fermion_hilbert`
  - `local_spin_hilbert`
  - `geometry`
- 提供：
  - `build_kondo_heisenberg_spin_fermion_hilbert(...)`
  - `KondoHeisenbergSpinFermion(...)`
- 调用底层 `KondoHeisenberg(...)` 构造真正的模型，再把分离后的子空间信息一起打包返回。

### 适用场景
- 当外部模型、神经网络或分析代码要分别读取电子部分和局域自旋部分时，这个接口比直接拿 `(H, hi)` 更方便。
- 特别适合“分支网络”“双流输入”“显式 sector 处理”等变分结构。

### 当前接口特点
- 和链版本一样，这个包装层更偏向“显式分离两个子 Hilbert 空间”的使用体验。
- 它当前也没有把 `n_fermions` 作为独立参数重新暴露出来；如果要支持 `J_K != 0` 下更一般的总电子数设置，可以继续扩展。

---

## 文件之间的关系

### 主依赖关系
- `HubbardExample.py` 依赖 `Hubbard.py`
- `HubbardPlot.py` 依赖 `Hubbard.py`
- `KondoHeisenbergExample.py` 依赖 `KondoHeisenberg.py`
- `KondoHeisenbergPlot.py` 依赖 `KondoHeisenberg.py`
- `KondoHeisenbergChainExample.py` 依赖 `KondoHeisenbergChain.py`
- `KondoHeisenbergChainPlot.py` 依赖 `KondoHeisenbergChain.py`
- `KondoHeisenbergSpinFermion.py` 是对 `KondoHeisenberg.py` 的结构化包装
- `KondoHeisenbergChainSpinFermion.py` 是对 `KondoHeisenbergChain.py` 的结构化包装
- `KondoHeisenberg_constraint_notes.md` 是对 `KondoHeisenberg.py` / `KondoHeisenbergChain.py` 约束设计的补充说明

### 使用建议
- 如果你要直接建模 2D Hubbard：看 `Hubbard.py`
- 如果你要直接建模广义多腿 Kondo-Heisenberg：看 `KondoHeisenberg.py`
- 如果你只研究 1D 链：优先看 `KondoHeisenbergChain.py`
- 如果你只是想快速跑通接口：看 `*Example.py`
- 如果你想确认几何和键连接是否写对：看 `*Plot.py`
- 如果你要给 ansatz 显式区分“电子”和“局域自旋”两个 sector：看 `*SpinFermion.py`
- 如果你在 `J_K != 0` 的约束上遇到错误：先看 `KondoHeisenberg_constraint_notes.md`

## 一句话总结

这一组文件已经形成了比较完整的结构：
- `Hubbard.py` 负责 2D Hubbard 主模型；
- `KondoHeisenberg.py` / `KondoHeisenbergChain.py` 负责 Kondo-Heisenberg 主模型；
- `Example` 文件负责最小示例；
- `Plot` 文件负责结构可视化；
- `SpinFermion` 文件负责更适合复杂 ansatz 的结构化接口；
- `constraint_notes` 负责解释 full Kondo 情况下约束为什么必须重新设计。