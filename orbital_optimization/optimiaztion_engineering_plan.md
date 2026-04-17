# orbital_optimization 工程落地方案

## 1. 先说结论

这套代码现在已经有一个比较清楚的雏形：

- `optimization_flow.md` 给出了最新的总流程；
- `optimize_no_core.py` 已经把“公共优化骨架”抽出来了；
- `optimize_no_losses.py` 已经把 soft / barrier / original 三类 loss builder 集中起来了；
- `optimize_no_soft.py` 和 `optimize_no_barrier.py` 已经在做“外层包装”。

所以接下来的重点，不是推倒重写，而是把 `flow` 里还没有真正落地的那两段补齐：

1. 连续阈值化 / 裁剪；
2. 固定支撑后的稀疏重优化。

按工程实现看，最合适的结构是：

- `loss` 单独放；
- `dense optimization` 单独放；
- `pruning / clipping` 单独放；
- `masked sparse re-optimization` 单独放；
- 再用一个 `runner` 文件串起来。

这和你要求的“loss 优化、裁剪分开，再来一个总文件迭代调用”是一致的。

---

## 2. 对当前目录代码的判断

### 2.1 现在已经做对的部分

当前目录里最值得保留的主线是：

- `optimization_flow.md`：算法主线最清楚；
- `optimize_no_core.py`：已经把 base context、block 参数化、Optax 优化循环、Hamiltonian 旋转抽出来；
- `optimize_no_losses.py`：已经把不同 loss 的配置、loss 构造、diagnostics 抽出来；
- `optimize_no_soft.py` / `optimize_no_barrier.py`：已经是比较合理的外层 API 包装。

这说明目录已经从“一个大文件”走到“骨架 + loss + wrapper”的状态了。

### 2.2 现在还缺的部分

还没真正落地的，是 `flow` 中这两步：

- 连续升阈值找支撑；
- 支撑固定后做稀疏幺正重优化。

也就是说，当前目录已经有“第一步：稠密优化”，但还没有真正代码化的：

- `Step 2`: pruning / threshold schedule
- `Step 3`: sparse masked refinement

### 2.3 当前代码的几个工程债务

1. `optimize_no_core.py` 仍然从 `optimize_no.py` 借不少工具函数。
   这意味着 core 还没有完全独立。

2. 一些模板和 md 文档还在引用 `graph_sample` 路径，而现在代码实际放在 `orbital_optimization`。
   这属于文档/导入路径漂移。

3. 当前优化器是 JAX + Optax 的显式梯度优化，不是 NetKet 的 `MCState/VMC` 驱动。
   这本身没有错，但要在工程上明确：
   `orbital optimization` 和 `NQS benchmark` 是两个层次，不该混进一个 loss loop 里。

---

## 3. NetKet 在这里应该怎么用

## 3.1 basis 优化阶段怎么用

当前这套 orbital optimization，本质上是一个确定性 JAX 目标：

- 输入是 `rdm / hopping / interaction / distance / positions`；
- 参数是 block 内的轨道旋转；
- 优化器是 `optax`；
- 输出是新的轨道基和可选的旋转后 Hamiltonian。

这一层里，NetKet 最合理的作用只有两个：

1. Hamiltonian 类型与旋转：
   当前代码已经使用 `FermionOperator2ndBase` 和 `rotate_fermion_hamiltonian`。
2. 后续 NQS benchmark 的驱动容器：
   即 `nk.vqs.MCState`、`nk.driver.VMC` 或 `nk.driver.VMC_SR`。

所以 basis 优化层不要强行绑进 `MCState`。

## 3.2 这一步需不需要 SR

结论：

- 对 orbital 参数优化本身，不建议上 SR。
- 对 NQS benchmark / NQS 训练阶段，建议保留 SR 选项。

原因很直接：

1. 当前 orbital loss 是确定性的，没有 Monte Carlo 噪声；
2. 参数量只是块内旋转，不像 NQS 参数那样高维；
3. 当前 `optimize_no_core.py` 已经是 `jax.jit + value_and_grad + optax` 的干净结构；
4. 把 SR 引入 orbital 参数层，会显著增加实现复杂度，但收益并不明显。

所以推荐策略是：

- `dense orbital optimization`：只用 Optax，不用 SR；
- `sparse masked refinement`：只用 Optax，不用 SR；
- `basis 上的 NQS benchmark`：如果你要比较某个基底对 NQS 是否友好，再用 NetKet 的 VMC / VMC_SR。

## 3.3 NetKet benchmark 层怎么用

结合你仓库本地文档 `VMC_optimizer_guide.md`，当前更推荐的 SR 路线是：

```python
import netket as nk

vstate = nk.vqs.MCState(sampler, model, n_samples=...)
opt = nk.optimizer.Sgd(learning_rate=1e-2)
vmc = nk.driver.VMC_SR(
    hamiltonian,
    opt,
    variational_state=vstate,
    diag_shift=1e-3,
)
```

如果为了兼容旧代码，也可以继续用：

```python
sr = nk.optimizer.SR(diag_shift=1e-3)
vmc = nk.driver.VMC(
    hamiltonian,
    opt,
    variational_state=vstate,
    preconditioner=sr,
)
```

但这个应该放在 `optimize_no_netket_eval.py` 一层，而不是 basis loss loop 里。

---

## 4. 推荐的文件拆分

我建议最终目录按下面思路收敛：

```text
orbital_optimization/
  optimize_no_utils.py
  optimize_no_core.py
  optimize_no_losses.py
  optimize_no_pruning.py
  optimize_no_sparse.py
  optimize_no_netket_eval.py
  optimize_no_flow_runner.py
  optimize_no_soft.py
  optimize_no_barrier.py
  optimize_no_template.py
  optimize_no_flow_template.py
```

### 4.1 `optimize_no_utils.py`

职责：

- 从当前 `optimize_no.py` 抽出真正公共的工具函数；
- 包括 matrix validation、distance / interaction resolve、Hamiltonian term extraction、Hamiltonian rotation 等。

目的：

- 让 `optimize_no_core.py` 和 `optimize_no_losses.py` 不再依赖 `optimize_no.py` 这个 monolithic baseline。

### 4.2 `optimize_no_core.py`

保留并继续承担：

- `prepare_base_context(...)`
- hard block construction
- block 参数初始化
- block-local unitary build
- 通用 Optax 优化循环
- 最终 Hamiltonian 旋转
- 统一结果打包

这一层只回答：

> 怎么优化一组连续参数。

### 4.3 `optimize_no_losses.py`

保留并继续承担：

- soft / barrier / original loss builder
- loss config dataclass
- JAX loss function builder
- metrics / diagnostics builder

这一层只回答：

> 优化什么。

### 4.4 `optimize_no_pruning.py`

这是接下来最该新增的文件。

职责：

- 阈值序列管理；
- 从当前旋转矩阵构造支撑；
- top-m 保底；
- 行列空支撑修补；
- 记录每一轮的支撑；
- 计算持久度矩阵；
- 基于 `loss_module.evaluate_metrics(...)` 做 accept/reject。

这一层不要自己实现物理 loss，只依赖回调。

建议接口：

```python
@dataclass(slots=True)
class NOPruningConfig:
    taus: tuple[float, ...]
    top_m: int = 1
    eta_H: float = 0.03
    eta_gamma: float = 0.08
    f_min: float = 0.7


def build_support_from_rotation(post_no_rotation: np.ndarray, tau: float, *, top_m: int = 1) -> np.ndarray: ...

def repair_support(mask: np.ndarray, rotation: np.ndarray, *, top_m: int = 1) -> np.ndarray: ...

def run_pruning_schedule(*, base_context: dict, loss_module: dict, dense_result: dict, config: NOPruningConfig) -> dict: ...

def build_stable_support(supports: list[np.ndarray], *, f_min: float) -> np.ndarray: ...
```

### 4.5 `optimize_no_sparse.py`

这是第二个必须新增的文件。

职责：

- 在固定支撑下做 masked sparse refinement；
- 初版不要追求最漂亮的几何参数化，先做“投影到支撑 + polar retraction + Optax”即可；
- 最后输出 `V_sparse` 与对应 metrics。

建议接口：

```python
@dataclass(slots=True)
class NOSparseConfig:
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    gradient_clip: float | None = 1.0
    n_steps: int = 100
    log_every: int = 25
    lambda_fit: float = 0.3
    tol_unitary: float = 1e-6
    tol_mask: float = 1e-8


def optimize_sparse_with_support(
    *,
    base_context: dict,
    loss_module: dict,
    support: np.ndarray,
    init_post_no_rotation: np.ndarray,
    config: NOSparseConfig,
) -> dict: ...
```

### 4.6 `optimize_no_netket_eval.py`

职责：

- 把 basis 优化和 NQS benchmark 分开；
- 输入 rotated Hamiltonian；
- 外部传入 `vstate_builder` 或 `driver_builder`；
- 内部只负责跑 NetKet VMC / VMC_SR。

建议接口：

```python
@dataclass(slots=True)
class NONetKetEvalConfig:
    n_iter: int = 200
    use_sr: bool = True
    use_vmc_sr: bool = True
    learning_rate: float = 1e-2
    diag_shift: float = 1e-3


def evaluate_basis_with_netket(
    *,
    hamiltonian,
    vstate_builder,
    config: NONetKetEvalConfig,
    driver_builder=None,
) -> dict: ...
```

### 4.7 `optimize_no_flow_runner.py`

这是总调度文件。

职责：

- 串起 `flow`：
  - prepare context
  - dense optimize
  - pruning schedule
  - sparse masked refinement
  - optional NetKet benchmark
- 对外给一个总入口。

建议接口：

```python
def optimize_sparse_no_basis_flow(
    *,
    H=None,
    rdm: np.ndarray,
    dense_kind: str = "soft",
    dense_config=None,
    pruning_config=None,
    sparse_config=None,
    netket_eval_config=None,
    hopping_matrix=None,
    interaction_matrix=None,
    distance_matrix=None,
    site_positions=None,
    active_indices=None,
    spin_symmetric: bool = True,
    vstate_builder=None,
): ...
```

输出建议统一为：

```python
{
  "base_context": ...,
  "dense_result": ...,
  "pruning_result": ...,
  "sparse_result": ...,
  "netket_eval": ...,
}
```

---

## 5. loss / 裁剪 / 总调度三层怎么衔接

## 5.1 loss 层不关心裁剪

`optimize_no_losses.py` 只负责：

- 给定 `post_no_rotation`，算 loss；
- 给定 `post_no_rotation`，算 diagnostics；
- 给 dense / sparse 阶段提供同一套物理指标。

它不应该知道：

- 当前阈值是多少；
- 支撑是不是被裁掉了；
- 当前是 dense 还是 sparse 阶段。

## 5.2 pruning 层不关心物理公式细节

`optimize_no_pruning.py` 只负责：

- 生成 mask；
- 管阈值 schedule；
- 用 metrics 做 accept/reject；
- 做持久度统计。

它不应该自己重写 `C_H`、`C_gamma`。

它只需要调用：

```python
metrics = loss_module["evaluate_metrics"](post_no_rotation)
```

## 5.3 sparse 层做“带约束的连续优化”

`optimize_no_sparse.py` 的角色是：

- 接过 pruning 产出的固定支撑；
- 在这个支撑上做连续参数优化；
- 返回一个真正可用的稀疏幺正结果。

因此 sparse 层是唯一需要同时知道：

- loss callback；
- fixed support；
- unitarity repair。

## 5.4 runner 层只串流程

`optimize_no_flow_runner.py` 不重新定义 loss，也不重新定义 pruning。

它只负责：

- 调谁；
- 先后顺序；
- 汇总结果。

这样以后你换 loss、换 pruning 策略、换 sparse 优化器，都不用改总流程的其他层。

---

## 6. 推荐的落地顺序

### M0：先做一轮轻清理

优先把这些公共工具从 `optimize_no.py` 挪出来：

- `_hermitian_part`
- `_require_square_matrix`
- `_resolve_distance_matrix`
- `_resolve_interaction_matrix`
- `extract_one_body_hopping_matrix`
- `extract_interaction_matrix`
- `rotate_one_body_matrix`
- `rotate_hamiltonian_to_basis`
- `rotate_hamiltonian_to_natural_orbitals`
- `build_occupation_blocks`

目的：

- 让 `core/losses` 不再反向依赖 monolithic `optimize_no.py`。

### M1：先落地 pruning，不动现有 dense API

先新增 `optimize_no_pruning.py`，但先不碰 `soft/barrier` 的外部接口。

这样你可以很快得到：

- `dense_result`
- `pruning_result`
- `stable_support`

### M2：再落地 sparse masked refinement

新增 `optimize_no_sparse.py`，先用最朴素的“支撑投影 + polar retraction”方案。

不要一上来就写 Givens edge 参数化。

原因：

- 先把流程跑通更重要；
- 你要先看到 stable support 长什么样；
- 只有看过几轮真实 support，才知道是不是值得进一步做 block / band / Givens edge 压缩。

### M3：最后加 NetKet benchmark 文件

新增 `optimize_no_netket_eval.py`。

重点是：

- 用 `rotated_hamiltonian` 做 benchmark；
- 不把 `MCState/VMC` 塞进 basis loss loop；
- 支持可选 `VMC_SR` 和老式 `VMC + SR`。

### M4：如果以后要做 joint loop，再单独做

如果以后要做：

- basis 更新
- 旋转 Hamiltonian
- 用 NQS 重新训练
- 再由 NQS 得到新的 rdm
- 再继续 basis update

那么这是一个单独的新模块，建议另建文件，不要直接写进 `core`。

可以参考你仓库里 `graph_sample/iteration_occ_func_simple.py` 的分段更新思路。

---

## 7. 一个推荐的最小工程组合

如果你现在只想把 sparse flow 真正跑起来，我建议第一版只做下面 5 个文件：

- `optimize_no_utils.py`
- `optimize_no_core.py`
- `optimize_no_losses.py`
- `optimize_no_pruning.py`
- `optimize_no_sparse.py`
- `optimize_no_flow_runner.py`

其中：

- `core/losses` 尽量沿用现有代码；
- 重点新增 `pruning/sparse/runner`；
- `soft/barrier` 继续保留做 dense stage 的入口；
- NetKet benchmark 先可以在 notebook 里手调，等 flow 稳定后再补单独文件。

---

## 8. 最后的建议

从代码工程角度看，这个目录现在最不应该做的事，是再新复制一份 `optimize_no_xxx.py`。

最应该做的是：

1. 保住 `core + losses + wrapper` 这条已经成型的主线；
2. 把 `flow` 真正缺的 `pruning + sparse + runner` 三块补上；
3. 把 NetKet 的 `VMC/VMC_SR` 放到独立 benchmark 层，而不是混进 orbital loss 本体。

一句话压缩：

> orbital basis 优化本体继续走 JAX + Optax；NetKet 负责 Hamiltonian 旋转后的 NQS benchmark；SR 只放在 NQS 训练层，不放在 basis 参数层。
