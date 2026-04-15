# NetKet 不同版本中的“预训练 / target-state fitting”接口说明

> 主题：在不同版本的 NetKet 中，如何把一个目标态（teacher state）输入给大 NQS，并做预训练/拟合。
>
> 这里的“预训练”指的是：先给定一个目标态 \(|\Phi\rangle\)，再最小化 student \(|\Psi_\theta\rangle\) 与 target 的 infidelity
>
> \[
> I(\Psi_\theta,\Phi)
> = 1 - \frac{|\langle \Psi_\theta|\Phi\rangle|^2}{\langle \Psi_\theta|\Psi_\theta\rangle\langle \Phi|\Phi\rangle}.
> \]
>
> NetKet 官方现成支持的是 **target-state infidelity minimization**，而不是“直接输入一批 \((\sigma_i, \log\phi(\sigma_i))\) 标签做监督回归”的通用 dataset driver。

---

## 0. 先说结论

### 版本分界

- **NetKet < 3.19**
  - 我查到的官方 release notes / changelog 中，**还没有**内置的 infidelity pretraining driver。
  - 若你要做预训练，只能自己写 custom loss / custom driver。

- **NetKet 3.19.x ~ 3.20.x**
  - 首次引入：
    - `netket.experimental.observable.InfidelityOperator`
    - `netket.experimental.driver.Infidelity_SR`
  - 这是官方第一次提供“把 variational state 拟合到 target state”的现成接口。
  - 此时接口还在 `experimental` 命名空间。

- **NetKet 3.21.x**
  - 在上面的基础上，`Infidelity_SR` 增加了对 `netket.vqs.FullSumState` 的支持。
  - 因而小系统/可全空间求和的 teacher state 可以更自然地接进去。

- **NetKet 3.22+（文档中的稳定接口）**
  - `Infidelity_SR` 从 `netket.experimental.driver` **迁移并稳定化**到
    - `netket.driver.Infidelity_SR`
  - 旧路径会有 deprecation warning。
  - 注意：我查到的 GitHub release 页面显示 **3.22 在我核查时仍有 dev candidate**；所以如果你实际安装的是 3.21.x，仍应使用 `experimental` 路径。

### 核心接口语义

无论哪个版本，这个功能的本质都不是“传入任意 Python 对象作为目标态”，而是：

- `target_state` 必须是 **`VariationalState`**；
- 因此 teacher 需要包装成：
  - `MCState`，或
  - `FullSumState`（3.21+ 支持在 `Infidelity_SR` 中使用），或
  - 其他实现了 `VariationalState` 接口的对象。

所以：

- **可以直接作为 teacher 的：**
  - 另一个 NQS（另一个 `MCState` / `FullSumState`）
  - 小系统的完整波函数，先包成 `LogStateVector` 再构造 `MCState` / `FullSumState`

- **不能直接作为 teacher 的：**
  - 裸的 `numpy.ndarray` 波函数向量
  - 外部 MPS/DMRG 对象本体
  - 一批监督数据 `[(sigma_i, logphi_i)]`
  - 一个普通 Python callable（除非你再包一层 `apply_fun` 变成 `VariationalState`）

---

## 1. 官方功能的数学对象

官方 driver 优化的是 infidelity：

\[
I(\Psi,\Phi)=1-\frac{|\langle\Psi|\Phi\rangle|^2}{\langle\Psi|\Psi\rangle\langle\Phi|\Phi\rangle}.
\]

更底层地，它对应一个 projector observable：

\[
\hat I_{\mathrm{op}} = \frac{|\Phi\rangle\langle \Phi|}{\langle\Phi|\Phi\rangle},
\qquad
I=1-\frac{\langle\Psi|\hat I_{\mathrm{op}}|\Psi\rangle}{\langle\Psi|\Psi\rangle}.
\]

文档还给出了联合 Born 分布下的 Monte Carlo 估计器：

\[
\chi(x,y)=\frac{|\Psi(x)|^2|\Phi(y)|^2}{\langle\Psi|\Psi\rangle\langle\Phi|\Phi\rangle}.
\]

这也是为什么 NetKet 要求 `target_state` 是一个 **可评估 / 可采样 / 可求和** 的 `VariationalState`，而不是单纯一组标签。

---

## 2. 版本矩阵

## 2.1 NetKet < 3.19

### 是否有官方预训练接口？

**没有我能查到的官方 built-in infidelity pretraining driver。**

### 你能做什么？

只能自己写：

- 自定义监督损失：
  - amplitude/log-amplitude MSE
  - overlap-like loss
  - 自己构造 infidelity estimator
- 然后用 JAX/Optax 直接训练，或者自己写一个 driver。

### 这意味着什么？

如果你在旧代码环境里工作，又想“先用小 NQS / ED / DMRG 的态预训练大 NQS”，最稳妥的方式是：

1. 用 `MCState.log_value(samples)` 得到 student 的 \(\log\psi_\theta(\sigma)\)；
2. 自己定义监督损失；
3. 直接用 `jax.grad` + `optax` 更新参数。

这种做法和后来的官方 `Infidelity_SR` 是两条路线：

- 前者：dataset-based supervised warm start；
- 后者：state-to-state infidelity minimization。

---

## 2.2 NetKet 3.19.x ~ 3.20.x

### 新增的官方对象

从 3.19 起，官方引入：

```python
netket.experimental.observable.InfidelityOperator
netket.experimental.driver.Infidelity_SR
```

这是 NetKet 第一次正式提供“拟合目标态”的 built-in 接口。

### 主要导入方式

```python
import netket as nk
import netket.experimental as nkx
import optax
```

### 最小接口

```python
import netket as nk
import netket.experimental as nkx
import optax

# student
vs_student = nk.vqs.MCState(
    sampler=sa,
    model=student_model,
    n_samples=n_samples,
)

# teacher: 必须是 VariationalState
vs_target = ...  # MCState / 其他 VariationalState

driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
    operator=None,
)

driver.run(n_iter=100)
```

### 参数语义（3.19~3.20 的 experimental 版本）

常用参数：

- `target_state`：目标态，类型必须是 `VariationalState`
- `variational_state`：被训练的 student
- `optimizer`：文档明确建议用 `optax.sgd(...)`
- `diag_shift`：QGT/NTK 的对角正则
- `operator`：可选，如果要学的是 \(U|\Phi\rangle\)
- `use_ntk`：切换到 kernel trick / minSR
- `on_the_fly`：懒计算 Jacobian/QGT 的模式
- `mode`：real/complex/onthefly Jacobian 模式

### 这一版本最重要的限制

`target_state` **不是裸态**，而是 `VariationalState`。

因此下面这些**不能直接传**：

```python
psi_target = np.array([...])      # 不行，裸向量
teacher_dataset = [...]           # 不行，标签数据集
teacher_fn = lambda sigma: ...    # 不行，普通函数
```

需要先包装。

---

## 2.3 NetKet 3.21.x

### 相比 3.19~3.20 的主要变化

`Infidelity_SR` 新增对 `FullSumState` 的支持。

这很关键，因为对小系统或可完整枚举的对称扇区，你通常更希望 teacher 不走 Monte Carlo，而是直接做全空间求和。

### 典型 teacher 包装方式 1：另一个 NQS

```python
vs_target = nk.vqs.MCState(
    sampler=sa,
    model=teacher_model,
    n_samples=n_samples,
)
```

然后：

```python
driver = nk.experimental.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

### 典型 teacher 包装方式 2：exact wavefunction -> `LogStateVector` -> `FullSumState`

这是小系统最推荐的 teacher 写法：

```python
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import optax

psi_target = ...  # shape = (hilbert.n_states,)

vs_target = nk.vqs.FullSumState(
    hilbert=hi,
    model=nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
    variables={"params": {"logstate": jnp.log(psi_target)}},
)

vs_student = nk.vqs.MCState(
    sampler=sa,
    model=student_model,
    n_samples=n_samples,
)

driver = nkx.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

### 为什么这比把 target 也做成 `MCState` 更好？

因为 teacher 端不再引入额外采样噪声：

- `FullSumState`：teacher 精确求和；
- `MCState`：teacher 仍然要采样。

如果系统够小，`FullSumState` 通常是更干净的验证/预训练基线。

---

## 2.4 NetKet 3.22+（文档中的稳定接口）

### 路径变化

从文档来看，3.22 起：

```python
netket.experimental.driver.Infidelity_SR
```

被稳定化并迁移到：

```python
netket.driver.Infidelity_SR
```

旧路径会给 deprecation warning。

### 推荐导入方式

```python
import netket as nk
import optax
```

### 最小接口

```python
import netket as nk
import optax

vs_student = nk.vqs.MCState(
    sampler=sa,
    model=student_model,
    n_samples=n_samples,
)

vs_target = ...  # VariationalState

driver = nk.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
    operator=None,
)

driver.run(n_iter=100)
```

### 若 target 是 exact vector

```python
import jax.numpy as jnp
import netket as nk
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
    n_samples=n_samples,
)

driver = nk.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

### 一个容易混淆的点

在最新文档中：

- `Infidelity_SR` 已经稳定到 `netket.driver`
- 但更底层的 `InfidelityOperator` 仍然放在
  - `netket.experimental.observable.InfidelityOperator`

所以代码里可能会出现一种“driver 已稳定、operator 还在 experimental”的过渡态，这并不矛盾。

---

## 3. teacher state 到底怎么输入？

这部分是实际使用时最容易出错的。

## 3.1 teacher 是另一个 NQS

这是最直接的情况。

```python
vs_target = nk.vqs.MCState(
    sampler=sa,
    model=teacher_model,
    n_samples=n_samples,
)
```

然后把 `vs_target` 直接传给 `target_state`。

适用情形：

- 你先训练了一个小模型 / 小网络 teacher；
- 想把它作为更大模型的 warm start target；
- Hilbert 空间相同、基底顺序相同、约束相同。

---

## 3.2 teacher 是完整波函数向量（ED / 自己的 exact solver）

需要先包成 `LogStateVector`。

```python
psi_target = ...

vs_target = nk.vqs.FullSumState(
    hilbert=hi,
    model=nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
    variables={"params": {"logstate": jnp.log(psi_target)}},
)
```

也可以用 `MCState` 包，但如果系统够小，一般 `FullSumState` 更合理。

---

## 3.3 teacher 是“给定 sigma 返回 logpsi 的函数”

不能直接把普通函数塞进 `target_state`。

但你可以通过 `apply_fun` / `init_fun` 把它包装成一个 `MCState` 或 `FullSumState`。

即，概念上它当然可以成为 teacher，只是要先适配到 `VariationalState` 接口。

---

## 3.4 teacher 是标签数据集 `[(sigma_i, logphi_i)]`

**官方没有现成的 `target_state=dataset` 接口。**

这是你后面最可能要自己扩展的地方。

此时推荐路线不是 `Infidelity_SR(target_state=...)`，而是：

1. 直接调用 `vstate.log_value(batch_sigma)`；
2. 写自定义监督损失：
   - log-amplitude MSE
   - amplitude + phase loss
   - empirical overlap loss
3. 用 JAX + Optax 做 warm start；
4. 预训练结束后再切回官方 `VMC_SR` 或 `Infidelity_SR`。

---

## 4. `operator=U` 是什么？

NetKet 的 infidelity 接口不只支持拟合 \(|\Phi\rangle\)，还支持拟合：

\[
U|\Phi\rangle.
\]

所以你可以写：

```python
driver = nk.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    operator=U,
    optimizer=optax.sgd(learning_rate=1e-2),
    diag_shift=1e-4,
)
```

这时 student 拟合的是变换后的目标态，而不是原始 teacher 本身。

这对“先由一个简单态生成目标态，再让更复杂模型去模仿”很有用。

---

## 5. 关于优化器：为什么文档强调 `optax.sgd`

`Infidelity_SR` 本质上是 **infidelity minimization + stochastic reconfiguration / natural gradient descent**。

因此在 driver 内部，更新方向本身已经经过 SR/QGT 预条件。文档特别提醒：

- 若你想保持这个更新在数学上就是 SR/NGD，
- `optimizer` 应当用 `optax.sgd`。

用 Adam/Yogi 不是一定不能跑，但就不再是“纯粹的 SR 更新”了。

因此，官方推荐写法是：

```python
optimizer=optax.sgd(learning_rate=...)
```

而不是 Adam。

---

## 6. 迁移建议

## 6.1 如果你当前是 3.19 / 3.20 / 3.21

优先写成：

```python
import netket as nk
import netket.experimental as nkx

# 使用 nkx.driver.Infidelity_SR
```

### 若系统很小

优先让 target 用 `FullSumState + LogStateVector`。

### 若 target 本身是另一个 NQS

直接用 teacher 的 `MCState` / `FullSumState`。

---

## 6.2 如果你当前是 3.22+（或跟随 latest docs）

直接用：

```python
nk.driver.Infidelity_SR
```

旧的 experimental 路径只用于兼容旧代码。

---

## 6.3 如果你准备自己往 NetKet 里加“dataset 预训练”

最合理的设计路线是：

### 路线 A：先做外部 warm start

- 不改 NetKet 源码；
- 直接拿 `vstate.log_value()` + 自定义 JAX/Optax 训练；
- 训完后把参数塞回 `MCState`；
- 再转入 `nk.driver.VMC_SR` 或 `nk.driver.Infidelity_SR`。

### 路线 B：补一个自定义 observable / driver

做一个类似：

- `DatasetOverlapObservable`
- `DatasetLogMSEObservable`
- `DatasetPretrainDriver`

这样就能把 dataset-based pretraining 融进 NetKet 的 driver 框架。

如果你的目标是未来给 NetKet 提 PR，这条路线更系统。

---

## 7. 最小模板速查

## 7.1 3.19 ~ 3.21：experimental 路径

```python
import netket as nk
import netket.experimental as nkx
import optax
import jax.numpy as jnp

# target from exact vector
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

> 注：`FullSumState` 在 `Infidelity_SR` 里是 **3.21 起明确支持**。如果你在 3.19~3.20 且遇到兼容性问题，可先把 teacher 包成 `MCState`。

---

## 7.2 3.22+：稳定路径

```python
import netket as nk
import optax
import jax.numpy as jnp

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

driver = nk.driver.Infidelity_SR(
    target_state=vs_target,
    variational_state=vs_student,
    optimizer=optax.sgd(learning_rate=5e-2),
    diag_shift=1e-4,
)

driver.run(n_iter=200)
```

---

## 7.3 dataset warm start：官方没有现成 driver

```python
# 伪代码：你需要自己写
batch_sigma, batch_logphi = ...

logpsi = vs_student.log_value(batch_sigma)

d = logpsi - batch_logphi
c = jnp.mean(d)
loss = jnp.mean(jnp.abs(d - c)**2)

# jax.grad(loss) + optax update
```

这不是官方 `Infidelity_SR` 的直接输入方式，而是你要自己实现的“监督预训练”。

---

## 8. 使用时最容易踩坑的几个点

### 8.1 teacher 和 student 必须定义在同一个 Hilbert 空间/子空间

必须保证：

- 同一个 `hilbert`
- 同一组约束（如固定粒子数、固定总磁化等）
- 同一基底顺序
- 同一对称性扇区

否则 `target_state` 在数学上就不是同一个态空间里的对象。

### 8.2 外部波函数向量的基底顺序必须与 NetKet 一致

尤其是自旋系统，若你从外部代码导入完整波函数到 `LogStateVector`，一定要核对 basis ordering。

### 8.3 teacher 是小系统态，并不自动意味着可迁移到大系统

`Infidelity_SR` 的前提是 target 和 student 处在同一 Hilbert 空间。

因此：

- **同一 system size** 的 teacher -> student：直接支持；
- **小系统 -> 大系统**：官方没有“自动尺寸迁移”的接口；你需要自己定义参数迁移/embedding 规则。

### 8.4 文档版本与已发布包版本可能不一致

目前 latest docs 已把 `Infidelity_SR` 写在 `netket.driver` 下，并说明它在 3.22 稳定化；但如果你本地安装的仍是 3.21.x，就应继续使用 `netket.experimental.driver.Infidelity_SR`。

---

## 9. 我建议你的实际策略

如果你准备后面在 NetKet 中加入“预训练”功能，我建议你按这个顺序做：

1. **先用官方现成的 state-to-state 接口**
   - 适用于 teacher 本身能包装成 `VariationalState` 的情况；
   - 优先用 `Infidelity_SR`。

2. **再单独实现 dataset-based warm start**
   - 适用于 teacher 只提供标签数据 \((\sigma_i, \log\phi_i)\) 的情况；
   - 这需要自己写损失和训练 loop，或者补一个新 driver。

3. **最后再考虑把 dataset warm start 整合到 NetKet 的 driver/observable 体系**
   - 这一步才是适合做 PR 或长期维护的实现。

---

## 10. 参考来源（官方文档 / release notes）

以下是我这份笔记整理时核对的主要来源：

1. NetKet latest 文档：`netket.driver.Infidelity_SR`
   - https://netket.readthedocs.io/en/latest/api/_generated/driver/netket.driver.Infidelity_SR.html

2. NetKet stable 文档：`netket.experimental.driver.Infidelity_SR`
   - https://netket.readthedocs.io/en/stable/api/_generated/experimental/driver/netket.experimental.driver.Infidelity_SR.html

3. NetKet latest 文档：`netket.experimental.observable.InfidelityOperator`
   - https://netket.readthedocs.io/en/latest/api/_generated/experimental/observable/netket.experimental.observable.InfidelityOperator.html

4. NetKet fidelity tutorial
   - https://netket.readthedocs.io/en/latest/tutorials/fidelity.html

5. NetKet latest changelog
   - https://netket.readthedocs.io/en/latest/changelog.html

6. GitHub releases / changelog（用于核对 3.19、3.21、3.22 相关变更）
   - https://github.com/netket/netket/releases
   - https://github.com/netket/netket/blob/main/CHANGELOG.md

7. `MCState` 文档（`apply_fun` / `init_fun`）
   - https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html

8. `FullSumState` 文档
   - https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.FullSumState.html

9. `LogStateVector` 文档
   - https://netket.readthedocs.io/en/stable/api/_generated/models/netket.models.LogStateVector.html

---

## 11. 一句话总结

在 NetKet 里，“预训练”官方对应的是：

- **state-to-state infidelity minimization**：已经有 built-in 接口；
- **dataset-to-state supervised pretraining**：还没有现成公共 driver，需要你自己补。

而不同版本最关键的区别就是：

- **3.19 引入 experimental 接口**；
- **3.21 支持 `FullSumState`**；
- **3.22 把 `Infidelity_SR` 稳定到 `netket.driver`**。
