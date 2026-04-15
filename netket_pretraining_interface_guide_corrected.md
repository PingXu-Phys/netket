# NetKet 预训练 / Target-State Fitting 接口修正版 Guide

更新日期: 2026-04-10  
核对对象: 本地仓库 `D:\Seafile\PHD\NQS\NetKet\netket`  
当前分支: `PingXu-withproposal`

---

## 1. 这份文档的目的

这份 guide 用来取代此前那份按版本整理的预训练接口笔记中若干过于粗略、或者在本地源码层面并不完全准确的说法。

它的目标不是简单复述文档，而是把下面三件事说清楚:

1. 本地 NetKet 仓库里，和 “target-state fitting / infidelity pretraining” 直接相关的接口到底在哪里。
2. 按版本看，哪些能力是何时引入的，哪些说法需要修正。
3. 以后如果要调用、扩展、或者修改这套接口，应该先看哪些文件，哪些边界条件不能弄错。

这份文档以本地源码、`CHANGELOG.md`、本地 tag、测试文件为准。

---

## 2. 最重要的结论

### 2.1 当前本地仓库中的正式结论

在当前本地仓库中，和 “预训练 / target-state fitting” 对应的官方现成接口是:

```python
netket.experimental.observable.InfidelityOperator
netket.experimental.driver.Infidelity_SR
```

不是:

```python
netket.driver.Infidelity_SR
```

也就是说，在这个本地代码树里，`Infidelity_SR` 仍然停留在 `experimental` 命名空间。

### 2.2 `target_state` 的真实要求

`target_state` 不是“任意目标态对象”。

它必须是一个:

```python
netket.vqs.VariationalState
```

因此可直接作为 teacher/target 的对象通常是:

- `MCState`
- `FullSumState`
- 其他实现了 `VariationalState` 接口的对象

而下面这些都不能直接传给 `target_state=`:

- 裸的 `numpy.ndarray` 或 `jax.numpy.ndarray` 波函数向量
- 普通 Python callable
- 一批监督数据 `[(sigma_i, logphi_i)]`
- 外部 MPS/DMRG 对象本体

这些对象如果要成为 target，必须先被包装进 `VariationalState`。

### 2.3 旧说法里最需要修正的两点

#### 修正 1: “3.21 才支持 `FullSumState`” 这个说法不够准确

更精确的说法应该是:

- `3.19.0` 时，`InfidelityOperator`、`FullSumState` 上的 exact infidelity evaluator、以及 `target_state` 的若干包装分支已经能看到 `FullSumState`。
- 但 `3.19.0` 的 `Infidelity_SR` 还显式拒绝把 **被优化的 variational state** 设成 `FullSumState`。
- 到 `3.21` 对应的改动里，`Infidelity_SR` 才真正补上了对 **driver state 侧** 的 `FullSumState` 支持。

所以:

- 若说的是 “`FullSumState` 完全没有出现过”，那不对。
- 若说的是 “`Infidelity_SR` 从 3.21 开始支持把 `variational_state` 本身设成 `FullSumState`”，这才准确。

#### 修正 2: “3.22+ 用 `netket.driver.Infidelity_SR`” 这件事不能直接用于本地仓库

外部文档可能已经把 `Infidelity_SR` 写成稳定接口，但在当前本地仓库中:

- `netket.driver.__init__` 只导出了 `VMC_SR`
- `Infidelity_SR` 仍然只在 `netket.experimental.driver.__init__` 中导出

因此:

- 如果你的代码目标是 **当前这个本地仓库**
- 那就必须按 `experimental` 路径写

即:

```python
import netket.experimental as nkx

driver = nkx.driver.Infidelity_SR(...)
```

不要在这个仓库上直接写:

```python
nk.driver.Infidelity_SR(...)
```

---

## 3. 本地源码中相关接口的准确位置

下面这些文件是以后调用或修改时应优先查看的入口。

### 3.1 Driver

`Infidelity_SR` 的实现文件:

```text
netket/experimental/driver/infidelity_sr.py
```

它负责:

- 接受 `target_state`
- 接受 `variational_state`
- 处理 `operator=U`
- 调用 SR / minSR / NTK 相关逻辑
- 执行参数更新

### 3.2 Observable

`InfidelityOperator` 的实现文件:

```text
netket/experimental/observable/infidelity/infidelity_operator.py
```

它负责:

- 检查 `target_state` 是否是 `VariationalState`
- 在 `operator` 非空时，把 `U|Phi>` 包装回 `MCState` 或 `FullSumState`
- 保存 target 侧信息给 `expect` 和 `expect_and_grad` 用

### 3.3 Exact evaluator

`FullSumState` 的 exact infidelity 计算逻辑在:

```text
netket/experimental/observable/infidelity/exact.py
```

这部分很重要，因为它说明:

- `FullSumState` 早就不是完全“没接入 infidelity 体系”
- 只是最开始 `Infidelity_SR` driver 本身对 `FullSumState` 作为优化 state 的支持还不完整

### 3.4 MC evaluator

蒙特卡洛估计相关逻辑在:

```text
netket/experimental/observable/infidelity/expect.py
```

以后如果要看采样分布、local estimator、control variate 或 target/student 的联合采样结构，应该先看这里。

### 3.5 VariationalState 基类

```text
netket/vqs/base.py
```

这决定了 `target_state` 为什么必须是 `VariationalState`，以及一个对象要满足什么接口才能被 NetKet 当成 variational target 使用。

### 3.6 FullSumState

```text
netket/vqs/full_summ/state.py
```

这里要看的是:

- `FullSumState` 的构造方式
- `model` / `apply_fun` / `variables` 的入口
- `log_value`
- `to_array`

### 3.7 Exact vector 包装模型

```text
netket/models/full_space.py
```

这里定义了:

```python
netket.models.LogStateVector
```

它是把一个完整波函数向量包装成 NetKet 可用 target 的关键工具。

### 3.8 测试

测试文件尤其重要:

```text
test/observable/test_infidelity.py
test/driver/test_infidelity_vmc.py
```

这两份文件基本决定了:

- 仓库当前认为哪些用法是受支持的
- 哪些组合以前不支持、后来才支持
- 以后改接口时最应该补哪些测试

---

## 4. 本地 `CHANGELOG` 和 git 历史给出的版本事实

### 4.1 `3.19.x`: 第一次正式引入 infidelity driver / observable

`CHANGELOG.md` 已明确写出:

- 新增 `netket.experimental.observable.InfidelityOperator`
- 新增 `netket.experimental.driver.Infidelity_SR`

对应提交历史里，核心引入提交是:

```text
36edec20 Infidelity driver (#2076)
```

这点与此前总结一致。

### 4.2 3.19 阶段应如何描述才准确

准确说法应为:

- `3.19` 开始，NetKet 首次提供官方 built-in 的 infidelity fitting 接口。
- 这套接口一开始就在 `experimental` 命名空间。
- `InfidelityOperator` 和 `Infidelity_SR` 都已经出现。

但要加上一个关键补充:

- `3.19.0` 的 `Infidelity_SR` 还会拒绝 `variational_state=FullSumState`
- 也就是 driver 侧尚未完整支持 FullSumState 作为 student/state

在 `v3.19.0` 的实现里可以看到:

```python
if isinstance(variational_state, FullSumState):
    raise TypeError(
        "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
    )
```

所以如果你的表述是:

- “3.19 完全不支持 FullSumState”

这不对。

如果你的表述是:

- “3.19 的 `Infidelity_SR` 还不支持把 `FullSumState` 作为被优化 state”

这才对。

### 4.3 `3.20.x`: 仍在 `experimental`

从当前本地代码树和导出路径来看，`Infidelity_SR` 在这个阶段仍然在:

```python
netket.experimental.driver.Infidelity_SR
```

没有进入稳定导出。

在本地代码里，稳定 `driver` 命名空间已经有:

```python
netket.driver.VMC_SR
```

但并没有:

```python
netket.driver.Infidelity_SR
```

因此以后如果要写“3.20 起 `Infidelity_SR` 已稳定”，不能直接这样写，至少当前本地仓库并不支持这个说法。

### 4.4 `3.21.x`: `Infidelity_SR` 对 `FullSumState` 的 driver 支持补齐

`CHANGELOG.md` 明确写了:

- `netket.driver.VMC_SR`
- `netket.experimental.driver.Infidelity_SR`

现在也支持:

- `netket.vqs.FullSumState`

这对应 git 历史里的关键提交:

```text
b94fd14a Extend infidelity_sr to FullSumState (#2171)
```

这个改动的正确理解是:

- 不是 `FullSumState` 第一次出现在 infidelity 体系里
- 而是 `Infidelity_SR` driver 本身开始支持 `FullSumState` 作为被优化 state

测试文件 `test/driver/test_infidelity_vmc.py` 也反映了这一点:

- 测试了 `FullSumState` 作为 `variational_state`
- 测试了 target 是 `FullSumState`
- 也测试了 target 是 `MCState`

所以以后如果你要描述 `3.21`，推荐使用下面这种更精确的说法:

> `3.21` 补齐了 `Infidelity_SR` driver 对 `FullSumState` 的支持，使其不仅能作为 target/exact evaluator 的一部分，也能作为被优化的 variational state。

### 4.5 本地仓库对 `3.22+` 的可确认范围

本地 tag 里我只看到:

- `v3.19.0`
- `v3.19.2`
- `v3.20.0` 到 `v3.20.5`
- `v3.21.0`

没有本地 `v3.22.x` tag。

因此，对 `3.22+` 的说法在这份 guide 里必须谨慎:

- 可以说“外部文档可能如此表述”
- 但不能把它写成本地仓库已验证的事实

本地仓库里当前能确认的事实只有:

- `VMC_SR` 已稳定导出
- `Infidelity_SR` 仍只在 `experimental` 下导出

---

## 5. 当前本地仓库中的真实接口语义

### 5.1 `InfidelityOperator`

构造大致是:

```python
InfidelityOperator(
    target_state: VariationalState,
    *,
    operator: AbstractOperator = None,
    cv_coeff: float | None = -0.5,
    dtype=None,
)
```

它的关键点有:

1. `target_state` 必须是 `VariationalState`
2. `operator` 非空时，语义是拟合 `U|Phi>`
3. `cv_coeff` 是 Monte Carlo 方差缩减项
4. 若 `target_state` 是 `FullSumState`，`cv_coeff` 会被禁用为 `None`

这个类本身不是优化器，它更接近“infidelity observable / projector observable”的包装。

### 5.2 `Infidelity_SR`

当前本地实现的核心签名大致是:

```python
Infidelity_SR(
    target_state,
    optimizer,
    *,
    operator=None,
    diag_shift,
    proj_reg=None,
    momentum=None,
    linear_solver=...,
    variational_state=None,
    chunk_size_bwd=None,
    mode=None,
    use_ntk=None,
    on_the_fly=None,
)
```

它做的事是:

- 根据 student 和 target 计算 infidelity estimator
- 选择 SR/QGT 或 minSR/NTK 路线
- 施加 `diag_shift`
- 解线性系统
- 更新 `variational_state.parameters`

### 5.3 `operator` 的含义

这不是“观测算符”意义下的普通 `expect(O)`。

在这里:

```python
operator=U
```

表示 student 去拟合:

```math
U|\Phi\rangle
```

而不是原始的 `|\Phi\rangle`。

### 5.4 `optimizer` 的推荐含义

driver docstring 明确建议:

- 若要保持 SR / NGD 的数学语义
- 推荐使用 `optax.sgd(...)`

这点应保留在文档里。

不过当前一些本地测试仍使用:

```python
nk.optimizer.Sgd(...)
```

因此更稳妥的说法是:

- 从“当前实现能否跑”的角度，优化器接口是开放的
- 从“这一步在数学上是否仍是纯 SR/NGD”的角度，首选 `optax.sgd`

### 5.5 `target_state` 与 `variational_state` 不能混为一谈

这一点是旧总结里最容易被说糊的地方。

要分开看:

- `target_state` 能不能是 `FullSumState`
- `variational_state` 能不能是 `FullSumState`

这两件事在历史上不是同时打通的。

更准确的历史是:

- `target_state` 侧较早就已有 `FullSumState` 痕迹
- `variational_state` 侧在 `Infidelity_SR` 里是后面才真正支持完整的 `FullSumState`

以后写说明时，必须明确“你说的是 target 侧，还是 student/driver 侧”。

---

## 6. teacher state 到底怎么输入

这一部分是实际调用时最常见的误区来源。

### 6.1 teacher 是另一个 NQS

这是最自然的情况。

例如:

```python
vs_target = nk.vqs.MCState(
    sampler=sa,
    model=teacher_model,
    n_samples=n_samples,
)
```

然后:

```python
driver = nk.experimental.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

这是官方接口直接支持的路径。

### 6.2 teacher 是完整波函数向量

这时不能直接把向量塞给 `target_state`。

推荐做法是先包装成:

```python
nk.models.LogStateVector
```

再构造:

```python
nk.vqs.FullSumState
```

典型写法:

```python
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import optax

psi_target = ...

vs_target = nk.vqs.FullSumState(
    hilbert=hi,
    model=nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
    variables={"params": {"logstate": jnp.log(psi_target)}},
)

driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

注意:

- `LogStateVector` 只适用于可 indexable 的小 Hilbert 空间
- `psi_target` 的基底顺序必须与 NetKet 的状态枚举顺序一致

### 6.3 teacher 是一个返回 `logpsi(sigma)` 的函数

普通 callable 不能直接给 `target_state`。

但它可以通过 `apply_fun` / `init_fun` 被包装成 `MCState` 或 `FullSumState`。

例如思路上可以写成:

```python
vs_target = nk.vqs.FullSumState(
    hilbert=hi,
    apply_fun=my_logpsi_fun,
    variables=my_variables,
)
```

或:

```python
vs_target = nk.vqs.MCState(
    sampler=sa,
    apply_fun=my_logpsi_fun,
    variables=my_variables,
    n_samples=n_samples,
)
```

这说明:

- callable 不是不能用
- 而是不能以“裸 callable”的身份直接传给 `target_state`

### 6.4 teacher 是监督数据集 `[(sigma_i, logphi_i)]`

当前本地官方接口没有:

```python
target_state=dataset
```

这样的入口。

这是需要明确写死在 guide 里的事实。

也就是说:

- 官方现成的是 state-to-state infidelity fitting
- 不是 dataset-to-state supervised pretraining

所以如果 teacher 只有样本标签，而没有可评估的 `VariationalState` 形式，那么你应该走自定义监督训练，而不是强行套 `Infidelity_SR`。

---

## 7. 当前本地仓库下推荐的调用写法

既然当前本地仓库中 `Infidelity_SR` 仍在 experimental，那么以本地代码为目标时，推荐模板应写成下面这样。

### 7.1 student 是 `MCState`，teacher 是 `MCState`

```python
import netket as nk
import netket.experimental as nkx
import optax

vs_student = nk.vqs.MCState(
    sampler=sa,
    model=student_model,
    n_samples=2**12,
)

vs_target = nk.vqs.MCState(
    sampler=sa,
    model=teacher_model,
    n_samples=2**12,
)

driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=5e-2),
    diag_shift=1e-4,
)

driver.run(n_iter=200)
```

### 7.2 student 是 `MCState`，teacher 是 exact vector 包出来的 `FullSumState`

```python
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import optax

psi_target = ...

vs_target = nk.vqs.FullSumState(
    hilbert=hi,
    model=nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
    variables={"params": {"logstate": jnp.log(psi_target)}},
)

vs_student = nk.vqs.MCState(
    sampler=sa,
    model=student_model,
    n_samples=2**12,
)

driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=5e-2),
    diag_shift=1e-4,
)

driver.run(n_iter=200)
```

### 7.3 加上 `operator=U`

```python
driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    operator=U,
    optimizer=optax.sgd(learning_rate=5e-2),
    diag_shift=1e-4,
)
```

此时 student 拟合的是 `U|Phi>`。

---

## 8. 以后写说明时建议直接采用的版本矩阵

下面这个版本矩阵是对旧总结的修正版，建议以后统一引用这个版本。

### 8.1 NetKet < 3.19

- 没有本地 `CHANGELOG` 能确认的官方 built-in `Infidelity_SR`
- 若要做预训练，只能自己写 custom loss / custom loop / custom driver

### 8.2 NetKet 3.19.x

- 首次引入:
  - `netket.experimental.observable.InfidelityOperator`
  - `netket.experimental.driver.Infidelity_SR`
- 已经进入官方代码库，但仍处于 `experimental`
- `FullSumState` 已经出现在 infidelity 相关实现里
- 但 `Infidelity_SR` 还不支持 `variational_state=FullSumState`

### 8.3 NetKet 3.20.x

- `Infidelity_SR` 仍在 `experimental`
- 本地可确认的稳定 driver 主要是 `VMC_SR`
- 不应在本地仓库语境中把 `Infidelity_SR` 写成稳定导出接口

### 8.4 NetKet 3.21.x

- `Infidelity_SR` 对 `FullSumState` 的 driver 支持补齐
- 现在不仅 target 能是 `FullSumState`
- 被优化的 state 也能是 `FullSumState`

### 8.5 当前本地仓库

- `netket.experimental.driver.Infidelity_SR` 仍然是正确入口
- `netket.driver.Infidelity_SR` 在本地代码中没有稳定导出
- 因此面向本地仓库写代码时，路径必须按 `experimental` 写

---

## 9. 如果以后要修改这套接口，优先看哪些文件

以下顺序是推荐的维护顺序。

### 9.1 第一层: 直接入口

先看:

```text
netket/experimental/driver/infidelity_sr.py
netket/experimental/observable/infidelity/infidelity_operator.py
```

原因:

- 一个控制 driver 侧逻辑
- 一个控制 observable / target 包装逻辑

这两处最直接决定 public interface。

### 9.2 第二层: 具体计算路径

然后看:

```text
netket/experimental/observable/infidelity/expect.py
netket/experimental/observable/infidelity/exact.py
```

原因:

- `MCState` 路线和 `FullSumState` 路线是分开的
- 如果只改 driver，不改这里，很容易造成某一条路径行为不一致

### 9.3 第三层: variational state 基础设施

再看:

```text
netket/vqs/base.py
netket/vqs/full_summ/state.py
netket/vqs/mc/mc_state/state.py
netket/models/full_space.py
```

原因:

- 这里决定了 `VariationalState` 的最小接口
- 决定了 `FullSumState` 和 `MCState` 各自能提供什么
- 决定了 exact vector 如何被包装进 NetKet

### 9.4 第四层: 测试

最后一定要回看:

```text
test/observable/test_infidelity.py
test/driver/test_infidelity_vmc.py
```

原因:

- 这些文件是最直接的行为契约
- 改接口不更新测试，很容易造成版本说明与实际行为再次脱节

---

## 10. 如果以后要加入 dataset pretraining，最合理的扩展路线

目前官方现成接口没有 dataset warm start。

如果未来要在这个仓库里补这一块，推荐分两条路线理解。

### 10.1 路线 A: 外部 warm start

特点:

- 不改 NetKet 核心源码
- 直接拿 `vstate.log_value(batch_sigma)` 写损失
- 用 JAX + Optax 单独训练
- 训练结束后再回到 `VMC_SR` 或 `Infidelity_SR`

优点:

- 最快
- 风险最小
- 不会破坏现有 infidelity 体系

缺点:

- 不属于 NetKet driver 框架内部的一部分
- 以后维护和记录实验时会有两套训练入口

### 10.2 路线 B: 新增 experimental driver / observable

可能的方向:

- 新建一个 dataset-based observable
- 或新建一个 dataset pretrain driver

例如概念上可以有:

```text
netket.experimental.observable.DatasetLogMSE
netket.experimental.driver.DatasetPretrain
```

优点:

- 更系统
- 更适合未来提 PR 或长期维护
- 能统一日志、driver 风格和训练入口

缺点:

- 设计成本更高
- 需要同时处理 `MCState` / `FullSumState` / batching / loss 定义 / tests

### 10.3 如果只想尽快可用，先做 A 再做 B

推荐顺序是:

1. 先做外部 warm start，验证 loss 和训练流程
2. 确认最常用的 dataset 形式
3. 再决定是否将其提升为 NetKet 内部 driver/observable

---

## 11. 以后写代码或文档时最容易踩坑的点

### 11.1 不要把外部最新文档的路径直接照搬到本地仓库

如果当前工作目标是这个本地仓库，就必须优先以本地源码导出路径为准。

当前本地仓库里:

- `nk.driver.VMC_SR` 是稳定导出
- `nk.experimental.driver.Infidelity_SR` 才是 infidelity 的正确入口

### 11.2 必须区分 `target_state` 支持和 `variational_state` 支持

一句 “支持 `FullSumState`” 太模糊了。

写清楚到底是:

- 作为 target 支持
- 作为 exact evaluator 支持
- 还是作为被优化 state 支持

### 11.3 teacher 和 student 必须在同一 Hilbert 空间

至少要保证:

- 同一个 `hilbert`
- 同一约束
- 同一子空间
- 同一基底顺序

否则 infidelity 的数学对象都不在同一个态空间里。

### 11.4 exact vector 的基底顺序必须与 NetKet 一致

尤其是:

- 自旋排序
- 粒子数约束下的状态枚举顺序

如果这里错了，`LogStateVector` 包出来的 teacher 会在数学上对应错态。

### 11.5 `dataset` 与 `state` 不是一回事

这句话以后最好直接写进所有相关说明:

> NetKet 当前现成支持的是 `state-to-state infidelity fitting`，不是 `dataset-to-state supervised pretraining`。

---

## 12. 建议以后直接复用的简短结论

如果以后你只想在别的文档里引用一个简洁版结论，可以直接复用下面这段:

> 在当前本地 NetKet 仓库中，预训练 / target-state fitting 的官方接口仍然是 `netket.experimental.observable.InfidelityOperator` 和 `netket.experimental.driver.Infidelity_SR`。`target_state` 必须是 `VariationalState`，所以 teacher 通常应包装成 `MCState` 或 `FullSumState`，而不是裸向量、普通函数或 dataset。需要特别修正的是: `3.19` 已经引入 infidelity 接口并出现了 `FullSumState` 相关路径，但 `3.21` 才补齐 `Infidelity_SR` 对 `FullSumState` 作为被优化 state 的支持；同时，本地仓库中并没有可确认的稳定导出 `netket.driver.Infidelity_SR`，因此当前代码应继续使用 `experimental` 路径。

---

## 13. 本文档核对过的关键依据

这份修正版 guide 主要根据以下本地内容整理:

- `CHANGELOG.md`
- `netket_pretraining_interfaces_by_version.md`
- `netket/experimental/driver/infidelity_sr.py`
- `netket/experimental/observable/infidelity/infidelity_operator.py`
- `netket/experimental/observable/infidelity/exact.py`
- `test/observable/test_infidelity.py`
- `test/driver/test_infidelity_vmc.py`
- 本地 git tag:
  - `v3.19.0`
  - `v3.19.2`
  - `v3.20.0` 到 `v3.20.5`
  - `v3.21.0`

---

## 14. 最后一句话

这份本地仓库里的正确基准线是:

- 预训练官方接口已经有
- 但仍主要是 `experimental` 下的 state-to-state infidelity fitting
- 对 `FullSumState` 的支持历史需要区分 target 侧和 driver 侧
- dataset supervised pretraining 仍然需要自行扩展