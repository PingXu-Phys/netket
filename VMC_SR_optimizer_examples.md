# NetKet `VMC_SR` + optimizer 对照表与代码例子

核对日期：2026-04-15

背景与选型说明：[`VMC_optimizer_guide.md`](./VMC_optimizer_guide.md)

这份文件只回答一个问题：

- `netket.driver.VMC_SR` 应该怎样和不同 optimizer 组合使用？

## 1. 先说结论

如果你没有特别强的理由，默认从这条开始：

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

原因很简单：

- `VMC_SR` 内部负责 SR / NGD 方向
- 外层 `optimizer` 负责把这个方向变成参数更新
- 对 `VMC_SR` 来说，官方源码和文档都把 `optax.sgd` 视作更合理的默认值

## 2. `nk.optimizer.*` 和 `optax.*` 的关系

`nk.optimizer.Sgd`、`nk.optimizer.Adam`、`nk.optimizer.RmsProp` 等，本质上是对 `optax` 的薄封装。

所以这两种写法都可以：

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
```

```python
import optax
op = optax.sgd(learning_rate=1e-2)
```

如果你需要：

- `AdamW`
- warmup / cosine decay / schedule
- gradient clipping
- `optax.chain(...)`
- 更复杂的参数分组

那么通常应直接用 `optax`。

## 3. 推荐对照表

| 场景 | 推荐写法 | 说明 |
| --- | --- | --- |
| 默认新项目 | `VMC_SR + Sgd` | 最接近标准 SR / NGD 语义 |
| 想显式用 Optax SGD | `VMC_SR + optax.sgd` | 和上面本质相同，只是少一层 NetKet wrapper |
| 想试自适应更新 | `VMC_SR + Adam` | 能跑，但更像工程化外层更新，不是最纯的 SR |
| 想要 warmup / clip / weight decay | `VMC_SR + optax.chain(...adamw...)` | 现代训练配方，推荐直接用 `optax` |
| 参数很多 | `VMC_SR(..., use_ntk=True)` | 走 minSR / NTK 路线 |
| 想用 SPRING | `VMC_SR(..., momentum=0.8)` | 这是 SR 内部的 SPRING，不是外层 optimizer 的 momentum |
| 数值稳定优先 | `linear_solver=pinv_smooth` | 更稳，但通常更慢 |
| 费米子实波函数 | `mode="real"` | 常能更便宜 |

## 4. 最小模板

后面的所有例子都默认你已经有：

```python
import netket as nk
import optax

ham = hamiltonian
vs = vstate
```

如果你不想引入 `optax`，只看使用 `nk.optimizer.*` 的小节即可。

## 5. 代码例子

### 5.1 推荐默认：`VMC_SR + nk.optimizer.Sgd`

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

适合：

- 大多数新代码
- 想保持最标准的 SR / NGD 语义
- 不需要复杂 schedule / clip / weight decay

### 5.2 等价写法：`VMC_SR + optax.sgd`

```python
import netket as nk
import optax

op = optax.sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

适合：

- 你想显式站在 Optax 生态里写代码
- 你后面可能继续叠加别的 Optax transformation

### 5.3 `VMC_SR + Adam`

```python
import netket as nk

op = nk.optimizer.Adam(learning_rate=1e-3)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

或者：

```python
import netket as nk
import optax

op = optax.adam(learning_rate=1e-3)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

适合：

- 你明确想尝试自适应外层更新
- 你更在意工程上的调参便利

注意：

- 这不是最纯的 SR / NGD 默认路线
- 更准确地说，这是“先算 SR 方向，再用 Adam 更新参数”

### 5.4 `VMC_SR + AdamW + schedule + clip`

```python
import netket as nk
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

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

适合：

- 想用现代训练配方
- 想要 warmup、decay、clip、weight decay
- 需要比 `nk.optimizer.Adam(...)` 更灵活的组合

### 5.5 参数很多时：`use_ntk=True`

```python
import netket as nk

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

适合：

- 参数数目很大
- 想走 minSR / NTK 路线
- 想避免直接处理大尺寸 QGT

### 5.6 想用 SPRING：`momentum=0.8`

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    momentum=0.8,
)
```

注意这里的 `momentum`：

- 是 `VMC_SR` 自己的参数
- 对应的是 SPRING
- 不是外层 optimizer 的 momentum

### 5.7 更稳的线性求解器：`pinv_smooth`

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    linear_solver=nk.optimizer.solver.pinv_smooth,
)
```

适合：

- 更担心数值稳定性
- 遇到 `cholesky` 导致 NaN 或奇异矩阵问题

代价：

- 通常会更慢

### 5.8 费米子常见设置：`mode="real"`

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    mode="real",
)
```

适合：

- `SpinOrbitalFermions`
- 波函数本质上是实值带符号，而不是真正复相位

## 6. 两种“momentum”不要混淆

### 6.1 外层 optimizer 的 momentum

```python
import netket as nk

op = nk.optimizer.Momentum(learning_rate=1e-2, beta=0.9)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
)
```

这里的含义是：

- 先算 SR 方向
- 再用带 momentum 的外层优化器更新参数

### 6.2 `VMC_SR` 自己的 SPRING momentum

```python
import netket as nk

op = nk.optimizer.Sgd(learning_rate=1e-2)

gs = nk.driver.VMC_SR(
    ham,
    op,
    variational_state=vs,
    diag_shift=1e-3,
    momentum=0.8,
)
```

这里的含义是：

- 在 SR 本身内部启用 SPRING
- 这是另一个层次的“动量”

如果你不想把问题搞复杂，先只用：

- `optimizer = Sgd(...)`
- 如果确实需要，再单独打开 `momentum=0.8`

## 7. 我会怎么选

### 情况 A：默认起点

```python
op = nk.optimizer.Sgd(learning_rate=1e-2)
gs = nk.driver.VMC_SR(ham, op, variational_state=vs, diag_shift=1e-3)
```

### 情况 B：参数很多

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

### 情况 C：想更稳

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

### 情况 D：想尝试 Adam

```python
op = nk.optimizer.Adam(learning_rate=1e-3)
gs = nk.driver.VMC_SR(ham, op, variational_state=vs, diag_shift=1e-3)
```

### 情况 E：想要现代训练配方

```python
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
gs = nk.driver.VMC_SR(ham, op, variational_state=vs, diag_shift=1e-3)
```

### 情况 F：想试 SPRING

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

## 8. 一句话版总结

如果你问“`VMC_SR + 指定 optimizer` 到底怎么写”：

- 把 optimizer 作为 `VMC_SR` 的第二个参数传进去
- 默认首选 `Sgd`
- 要高级训练配方时直接用 `optax`
- `Adam` 能用，但它更像 SR 外层的工程化更新器
- `VMC_SR(..., momentum=...)` 是 SPRING，不等于外层 optimizer 的 momentum

