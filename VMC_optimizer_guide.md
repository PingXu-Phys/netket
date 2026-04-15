# NetKet VMC 优化器选择笔记

核对日期：2026-04-15

配套代码例子：[`VMC_SR_optimizer_examples.md`](./VMC_SR_optimizer_examples.md)

基于三部分信息整理：

- 本地源码：`D:\Seafile\PHD\NQS\NetKet\netket`，版本 `3.21.1.dev16+ged038304e`
- 官方文档：NetKet `latest` / `stable` 文档页面
- 官方文档：JAX / Optax 当前文档页面

## 1. 先说结论

如果你现在在 NetKet 里做基态 VMC 优化，最新推荐路线是：

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
)
```

如果你明确在问“Adam 能不能用”：能用。尤其在 plain gradient descent、pretraining、custom loss、需要 schedule/clip/weight decay 的工程化流程里，`Adam` / `AdamW` 很常见；但如果你在用 `VMC_SR`，官方源码和文档仍然把 `optax.sgd` 当作更合理的默认值，因为这时 `Adam` 更像“套在 SR 方向外面的一层经验型更新器”，而不是最纯的 SR/NGD。

不要默认再把 `nk.driver.VMC(..., preconditioner=nk.optimizer.SR(...))` 当成主路线。老路线还在，但现在官方和本地文档都更偏向 `VMC_SR`。

## 2. 为什么说 `VMC_SR` 是现在的主路线

本地文档 `docs/api/drivers.md` 明确写着：对于 ground-state optimization，通常推荐 `netket.driver.VMC_SR` 而不是普通 `netket.driver.VMC`。

本地源码里也有同样信号：

- `netket/driver/vmc.py` 注释提示，如果要用 SR，考虑直接用 `netket.driver.VMC_SR`
- `netket/optimizer/__init__.py` 明确暴露了 `Sgd` / `Momentum` / `AdaGrad` / `Adam` / `RmsProp` 这些外层 optimizer wrapper
- `netket/optimizer/sr.py` 注释也提示，考虑直接使用 `netket.driver.VMC_SR`
- `netket/driver/__init__.py` 已经把 `VMC_SR` 公开导出到公共 API

官方最新文档也把 `netket.driver.VMC_SR` 放在公开 Drivers API 中。

## 3. 概念上要分两层

在 NetKet 里，“VMC 优化器怎么选”其实有两层：

### 3.1 外层参数更新器

这是 `optimizer=` 传进去的对象，例如：

- `nk.optimizer.Sgd`
- `nk.optimizer.Adam`
- `nk.optimizer.RmsProp`

但对 SR / NGD 来说，源码和官方文档都强调：

- 更合理的选择通常是 `optax.sgd`，也就是 `nk.optimizer.Sgd(...)`
- `Adam` 之类高级优化器虽然可能能跑，但在 SR/NGD 语义下通常“不太数学合理”
- `netket.optimizer` 这一层本质上只是对 `optax` 的薄封装；如果你要 `AdamW`、schedule、gradient clipping、weight decay、参数分组或更复杂链式组合，官方更建议直接用 `optax`

因此，默认建议：

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
```

如果你只是想用 NetKet 已封装好的几种常见外层更新器，那么当前公开可用的是：

- `nk.optimizer.Sgd`
- `nk.optimizer.Momentum`
- `nk.optimizer.AdaGrad`
- `nk.optimizer.Adam`
- `nk.optimizer.RmsProp`

而从 JAX 这一层看，`jax.example_libraries.optimizers` 只是 example-only；真正要做正式训练，应该优先看 `Optax`。

### 3.2 更新方向的几何结构

这部分决定你到底是在做：

- 普通梯度下降
- SR / Natural Gradient
- minSR / NTK 版本
- 带 SPRING 的动量版

现在推荐用 `VMC_SR` 来统一承载这些选择。

## 4. 现在具体怎么选

### 4.1 最普通的推荐起点

如果你只是想要一个现代、官方主推、语义清晰的 VMC 设置：

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
)
```

适合：大多数新代码。

### 4.2 外层 optimizer 到底怎么选：Sgd / Adam / AdaGrad / RmsProp / AdamW

如果你关心的是“这一步在数学上还是不是标准 SR/NGD”，那就选 `Sgd`：

- 这是 `VMC_SR` 源码和官方文档都在直接强调的默认路线
- 如果你把 `Adam` / `RmsProp` / `AdaGrad` 套在 `VMC_SR` 外层，往往也能跑，但这时更应把它理解成“工程化更新技巧”，而不是最纯的 SR/NGD

如果你关心的是工程上的收敛速度、warmup、gradient clipping、weight decay 或参数分组，那么直接用 `optax` 往往更好：

- `nk.optimizer.Adam`：最常见的自适应起点。更适合 plain VMC、pretraining、custom loss，或者你明确接受“这不是最纯的 SR”
- `nk.optimizer.RmsProp`：也是自适应方法；当梯度尺度变化比较大、你想要比 Adam 更简单的状态时可以试
- `nk.optimizer.AdaGrad`：更适合稀疏梯度或不同参数块梯度量级差别很大时；缺点是累计平方梯度会让后期步长越来越小
- `nk.optimizer.Momentum`：如果你只是想给普通 Euclidean 梯度下降加惯性可以用它；它和 `VMC_SR(..., momentum=...)` 不是一回事，后者对应的是 SPRING
- `optax.adamw`：NetKet 没有同名包装器，但官方明确建议需要高级功能时直接用 `optax`，所以要 weight decay、warmup/cosine decay、clip 等时，它通常是比 `nk.optimizer.Adam` 更合适的入口

例如，如果你只是想做 plain VMC 或其他工程化优化，可以直接写：

```python
op = nk.optimizer.Adam(learning_rate=1e-3)
gs = nk.driver.VMC(ham, op, variational_state=vs)
```

如果你要更现代一点的训练配方，通常直接写 `optax`：

```python
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=50,
    decay_steps=1000,
    end_value=1e-5,
)
op = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-4),
)
gs = nk.driver.VMC(ham, op, variational_state=vs)
```

你本地仓库里的 `graph_sample/iteration_occ_func_simple.py` 其实就是这种“`optax.chain(...) + adaptive optimizer + schedule`”的工程化思路，只不过它用的是 `optax.adam` 加旧式 `SR` preconditioner。

### 4.3 什么时候选标准 SR / QGT

如果你想明确走标准 SR，也就是直接处理 QGT：

```python
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
    use_ntk=False,
)
```

更适合：

- 参数量不算特别大
- 你明确想用传统 SR / QGT 解释
- 你想和旧文献、旧代码一一对应

### 4.4 什么时候选 minSR / NTK

如果参数很多，`N_parameters > N_samples`，本地源码会自动偏向 NTK/minSR 路线。

显式写法：

```python
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
    use_ntk=True,
    on_the_fly=True,
)
```

更适合：

- 参数规模很大
- 想用 kernel trick / minSR
- 想避免直接处理大尺寸 QGT

本地源码逻辑：

- 若 `use_ntk is None`，则自动按 `n_parameters > n_samples` 选择 NTK 还是 QGT
- 若 `use_ntk=True`，默认又会把 `on_the_fly=True`

### 4.5 什么时候选 SPRING

SPRING 不是 `optimizer/` 目录下面一个独立类，也不是一个单独的 `nk.optimizer.SPRING(...)`。

它在当前实现里是 `VMC_SR` 的一个选项，通过 `momentum` 打开：

```python
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
    momentum=0.8,
)
```

本地源码和文档都说明：

- `momentum` 对应 SPRING
- 经验上 `momentum = 0.8` 左右常常工作得不错

## 5. SPRING 到底在哪里

高层入口在：

- `netket/_src/driver/vmc_sr.py`

真正执行更新的底层在：

- 标准 SR 路径：`netket/_src/ngd/sr.py`
- minSR / NTK 路径：`netket/_src/ngd/srt.py`
- 公共分发逻辑：`netket/_src/ngd/sr_srt_common.py`

也就是说：

- 你在 API 层面使用的是 `nk.driver.VMC_SR(..., momentum=...)`
- 底层实现再根据 `use_ntk` 决定走 `sr.py` 还是 `srt.py`

## 6. 一个关键实现细节

这版源码里：

- `momentum` 同时会传到 `sr.py` 和 `srt.py`
- 所以 SPRING 风格的动量并不只限于 `use_ntk=True`
- 但 `proj_reg` 只在 `srt.py` 那条 minSR/NTK 路径上实现了
- 在标准 `sr.py` 中，如果传 `proj_reg`，会直接报错：`proj_reg not implemented for SR`

因此：

- 只想开动量：`VMC_SR(..., momentum=...)` 即可
- 想用 `proj_reg`：更应考虑 `use_ntk=True`

## 7. 求解器怎么选

这是另一个重要分叉：线性系统怎么解。

### 7.1 本地这份源码的倾向

在 `VMC_SR` 的源码说明里，写得比较明确：

- `cholesky` 更快，但更容易数值不稳定，某些情况下可能导致 NaN
- `pinv_smooth` 更稳，但更贵
- 迭代法如 `cg` 以前很常用，但作者现在明显更谨慎，不再默认推荐它作为首选
- 如果你不被线性求解器本身卡住，源码注释建议优先考虑更稳健的 `pinv_smooth`

### 7.2 一个需要注意的小差异

我核对时发现：

- 你本地 `3.21.1.dev16` 源码里，`VMC_SR` 默认参数还是 `linear_solver=cholesky`
- 但官方 `latest` API 页面已经显示成了 `cholesky_with_fallback`

这说明 upstream 的方向是在把默认值往更稳健的实现推进。

为了避免默认值随版本变化带来行为差异，实践上更稳妥的做法是显式写出求解器。

例如：

```python
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
    linear_solver=nk.optimizer.solver.pinv_smooth,
)
```

如果你更看重速度，也可以试：

```python
linear_solver=nk.optimizer.solver.cholesky
```

## 8. 费米子问题的一个额外建议

本地 `VMC_SR` 源码对费米子系统专门给了提醒：

- 如果是 `SpinOrbitalFermions`
- 而且你的波函数本质是实值带符号，而不是有真正复相位
- 那通常应该显式写 `mode="real"`

例如：

```python
gs = nk.driver.VMC_SR(
    hamiltonian,
    op,
    variational_state=vstate,
    diag_shift=1e-3,
    mode="real",
)
```

这通常会更便宜。

## 9. 老路线还要不要用

老路线仍然可用：

```python
sr = nk.optimizer.SR(diag_shift=1e-3)
gs = nk.driver.VMC(
    hamiltonian,
    op,
    variational_state=vstate,
    preconditioner=sr,
)
```

它仍然适合：

- 你在维护旧代码
- 你只想单独实验 preconditioner API
- 你明确想复用 `SR` 对象接口

但如果是新代码，尤其你还关心：

- `minSR`
- `SPRING`
- `use_ntk`
- `on_the_fly`
- 统一 SR / NGD 接口

那么 `VMC_SR` 更合适。

## 10. 我给出的实用选择表

### 情况 A：一般新项目

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(ham, op, variational_state=vs, diag_shift=1e-3)
```

### 情况 B：参数很多，想上 minSR

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    use_ntk=True,
    on_the_fly=True,
)
```

### 情况 C：想试 SPRING

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    momentum=0.8,
)
```

### 情况 D：数值稳定性优先

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    linear_solver=nk.optimizer.solver.pinv_smooth,
)
```

### 情况 E：我就是想用 Adam / AdamW 这类工程化 optimizer

```python
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=50,
    decay_steps=1000,
    end_value=1e-5,
)
op = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-4),
)
gs = nk.driver.VMC(ham, op, variational_state=vs)
```

更适合：

- plain VMC
- pretraining / custom loss / orbital optimization
- 你明确想要 warmup、decay、clip、weight decay
- 你接受这不是最纯的 SR/NGD 语义

如果你不需要 `AdamW`、schedule、clip，换成 `nk.optimizer.Adam(...)` 或 `optax.adam(...)` 也可以。

### 情况 F：只是兼容旧代码

```python
sr = nk.optimizer.SR(diag_shift=1e-3)
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC(ham, op, variational_state=vs, preconditioner=sr)
```

## 11. 本地源码定位

关键文件如下：

- `netket/driver/vmc.py`
- `netket/driver/__init__.py`
- `netket/optimizer/__init__.py`
- `netket/optimizer/sr.py`
- `netket/_src/driver/vmc_sr.py`
- `netket/_src/ngd/sr_srt_common.py`
- `netket/_src/ngd/sr.py`
- `netket/_src/ngd/srt.py`
- `netket/optimizer/solver/solvers.py`
- `graph_sample/iteration_occ_func_simple.py`
- `docs/api/drivers.md`

## 12. 官方参考

- Drivers API  
  <https://netket.readthedocs.io/en/latest/api/drivers.html>

- `netket.driver.VMC_SR`  
  <https://netket.readthedocs.io/en/latest/api/_generated/driver/netket.driver.VMC_SR.html>

- `netket.optimizer`  
  <https://netket.readthedocs.io/en/stable/api/optimizer.html>

- SR / QGT 指南  
  <https://netket.readthedocs.io/en/stable/user-guides/sr.html>

- JAX `jax.example_libraries.optimizers`  
  <https://docs.jax.dev/en/latest/jax.example_libraries.optimizers.html>

- Optax Getting Started  
  <https://optax.readthedocs.io/en/latest/getting_started.html>

- Optax Optimizers API  
  <https://optax.readthedocs.io/en/latest/api/optimizers.html>

## 13. 最后一句话版结论

如果你问“NetKet 现在 VMC 优化器该怎么选”：

- 新代码优先 `VMC_SR + Sgd`
- 参数很大时优先考虑 `use_ntk=True`
- 想用 SPRING 就在 `VMC_SR` 里加 `momentum`
- 想更稳就显式考虑 `linear_solver=nk.optimizer.solver.pinv_smooth`
- 想上 `Adam` / `AdamW` / schedule / clip 时，优先直接用 `optax`；这更像工程化外层更新，而不是最纯的 SR/NGD
- `VMC + SR` 还可以用，但更像旧接口或兼容接口


