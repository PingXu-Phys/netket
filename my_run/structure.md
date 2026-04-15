# NetKet `run/` 统一结构、调用链与参数流

更新日期: 2026-04-15

这份文档的目标只有一个: 把现在 `run/` 目录里的职责边界讲清楚。

现在的组织原则是:

- 每个 `run/*.py` 只保留它自己的物理或算法特性。
- 所有跨 runner 都会重复出现的 plumbing 都集中到 `run/_common.py`。
- 参数必须按职责分组，明确说明它们最后流向哪里: Hamiltonian、model、sampler、`MCState`、optimizer、SR、driver、输出文件。

---

## 1. 哪些文件负责什么

| 文件 | 角色 | 应该负责什么 | 不应该负责什么 |
| --- | --- | --- | --- |
| `run/_common.py` | 共用层 | CLI 分组、参数打包、`MCState` 构造、通用 VMC/VMC_SR optimizer-driver helper、driver 运行、JSON/msgpack 落盘、SIAM 共用 builder/model/optimizer/SR | 具体 Hamiltonian 的物理细节、具体 sampler 分支策略 |
| `run/run_siam_no_proposal.py` | SIAM baseline 入口 | 只保留 baseline 的特点: 标准 `MetropolisFermionHop`、不更新 occupation、不转动 Hamiltonian | 不再手写通用 CLI 和 optimizer/SR 样板 |
| `run/run_siam_iteration_occ.py` | SIAM occupation-update 入口 | 只保留“每隔若干步重估 proposal occupation”这条特性 | 不再手写通用参数读取和模型摘要 |
| `run/run_siam_iteration_occ_1rdm.py` | SIAM 1-RDM / NO 入口 | 只保留 full 1-RDM、自然轨道、Hamiltonian rotation 这条特性 | 不再手写通用 optimizer/SR 和参数包装 |
| `run/run_kondoheisenbergchain_transformers.py` | Kondo-Heisenberg transformer 入口 | 只保留 KHC 特有的 system/model/sampler 决策，以及对 common helper 的薄封装 | 不再手写通用 CLI 分组、optimizer/driver 样板、`MCState` 样板、结果落盘 |
| `Hamiltonian/SIAM.py` | SIAM Hamiltonian 构造层 | 根据 SIAM 物理参数返回 `(H, hi)` | 不负责 VMC 执行 |
| `Hamiltonian/KondoHeisenbergChainSpinFermion.py` | KHC Hamiltonian 构造层 | 构造 joint Hilbert、Hamiltonian、sector helper | 不负责 CLI 和输出文件 |
| `graph_sample/hamiltonian_graph.py` | 图提取层 | 从费米子 Hamiltonian 抽取 hopping graph | 不负责优化器 |
| `graph_sample/iteration_occ_func_simple.py` | SIAM 迭代逻辑层 | occupation 更新、1-RDM 估计、自然轨道、Hamiltonian rotation | 不负责 CLI |
| `nqs_state/slater.py` / `nqs_state/AGP.py` | SIAM 模型层 | 提供 Slater/AGP 系 ansatz 类 | 不负责 sampler / driver |
| `transformer_joint_nnbf.py` | KHC 模型层 | 提供 Transformer+NNBF 类与 `resolve_spin_fermion_layout(...)` | 不负责 CLI |

---

## 2. `run/_common.py` 现在到底负责什么

`_common.py` 现在分成三层。

### 2.1 真正跨 runner 的共用层

这些函数不关心你是 SIAM 还是 Kondo-Heisenberg，只关心 runner 的共通结构:

- `create_argument_parser(...)`
  统一 parser 风格。
- `add_runtime_sampling_arguments(...)`
  统一 `n_iter`、`n_samples`、`n_discard_per_chain`、`d_max`、`seed`、`n_chains_per_rank`、`sweep_size` 这类运行时参数。
- `add_progress_arguments(...)`
  统一 `--show-progress/--no-progress`。
- `add_output_argument(...)`
  统一 `--out-dir`。
- `namespace_snapshot(...)`
  从 `argparse.Namespace` 里抽出一组字段，打包成 dict。
- `bool_or_auto(...)`
  把可选布尔开关统一打包成 `auto|True|False`。
- `LINEAR_SOLVER_BUILDERS`
  统一暴露 `nk.optimizer.SR` 和 `nk.driver.VMC_SR` 共用的线性求解器 registry。
- `print_mapping(...)`
  用稳定格式把参数块打印出来。
- `build_mcstate(...)`
  把 sampler、model、`n_samples`、`n_discard_per_chain`、`seed` 统一装配成 `nk.vqs.MCState`。
- `run_driver(...)`
  统一把 `n_iter` 和 `show_progress` 流进 NetKet driver。
- `energy_summary(...)`
  统一收集最终 energy、variance、acceptance。
- `write_json(...)` / `save_vstate(...)` / `save_model_artifacts(...)`
  统一结果落盘。

### 2.2 通用 VMC / VMC_SR helper 层

这些函数现在服务于 KHC 这类轻量 runner，也为以后其他 runner 复用:

- `add_vmc_driver_arguments(...)`
  统一加上 `--driver`、`--optimizer`、`--learning-rate`、`--diag-shift`、`--proj-reg`、`--momentum`、`--linear-solver`、`--mode`、`--use-ntk`、`--on-the-fly`。
- `validate_vmc_driver_args(args)`
  统一拒绝 plain `vmc` 和 `VMC_SR` 专用参数的错误组合。
- `collect_vmc_driver_config(args)`
  统一把 driver 参数打包成终端摘要和 JSON。
- `build_basic_optimizer(args)`
  统一构造轻量 runner 使用的 `adam` / `sgd`。
- `build_vmc_or_vmc_sr_driver(...)`
  统一把共享 driver 参数流进 `nk.VMC` 或 `nk.driver.VMC_SR`。

### 2.3 SIAM runner 家族的共用层

这些函数目前主要服务于三支 SIAM runner:

- `base_parser(...)`
  给 SIAM 统一加上 Hamiltonian/system、model、sampling/runtime 这几组参数。
- `add_model_optimizer_sr_arguments(...)`
  给 SIAM 统一加上 model variant、optimizer、SR 参数。
- `build_hamiltonian(args)`
  解析 `--hamiltonian-builder` 和 `--hamiltonian-kw`，最后调用对应的 `Hamiltonian.<module>.<builder>(...)`。
- `build_model(name, hi, args)`
  解析模型 registry 名称，构造 Slater/AGP 类。
- `get_optimizer_and_sr(args)`
  从统一 CLI 生成 outer optimizer 和 `nk.optimizer.SR`。

重点是: 这些“共用层”只做组装和分发，不做具体的物理分支判断。

---

## 3. 统一主调用链

无论是哪支 runner，现在都应该符合下面这条大链路:

```text
parse_args
-> 按职责分组读取参数
-> 用 common 把参数打包成 dict / summary
-> build Hamiltonian/system
-> build model
-> build sampler
-> _common.build_mcstate
-> build optimizer / SR / driver
-> _common.run_driver 或 runner-specific segmented run
-> 收集 summary
-> _common.write_json / save_vstate（如果该 runner 需要落盘）
```

如果某个 runner 不能自然映射到这条链上，通常意味着它又把共用逻辑写回入口文件里了。

---

## 4. 四条实际入口链

### 4.1 `run_siam_no_proposal.py`

```text
parse_args
-> _common.base_parser
-> _common.add_model_optimizer_sr_arguments
-> _common.build_hamiltonian
-> _common.build_model
-> graph_sample.hamiltonian_graph.graph_from_hamiltonian
-> nk.sampler.MetropolisFermionHop
-> _common.build_mcstate
-> _common.get_optimizer_and_sr
-> nk.VMC
-> _common.run_driver
-> _common.energy_summary
```

这条入口只保留一个特性: sampler 永远是标准 `MetropolisFermionHop`。

### 4.2 `run_siam_iteration_occ.py`

```text
parse_args
-> _common.base_parser
-> _common.add_model_optimizer_sr_arguments
-> _common.build_hamiltonian
-> _common.build_model
-> graph_sample.hamiltonian_graph.graph_from_hamiltonian
-> _common.get_optimizer_and_sr
-> graph_sample.iteration_occ_func_simple.run_vmc_with_iteration_occ
   -> segment driver.run
   -> estimate occupations from samples
   -> rebuild proposal distribution
```

这条入口只保留一个特性: segment 之间更新 proposal occupation，但不估 full 1-RDM，也不旋转 Hamiltonian。

### 4.3 `run_siam_iteration_occ_1rdm.py`

```text
parse_args
-> _common.base_parser
-> _common.add_model_optimizer_sr_arguments
-> _common.build_hamiltonian
-> _common.build_model
-> _common.get_optimizer_and_sr
-> graph_sample.iteration_occ_func_simple.run_vmc_with_iteration_natural_orbitals
   -> segment driver.run
   -> estimate full 1-RDM
   -> diagonalize 1-RDM
   -> rotate Hamiltonian
   -> rebuild graph + proposal occupations
```

这条入口只保留一个特性: full 1-RDM / natural orbital 这条重逻辑。

### 4.4 `run_kondoheisenbergchain_transformers.py`

```text
parse_args
-> _common.create_argument_parser
-> _common.add_vmc_driver_arguments
-> _common.add_runtime_sampling_arguments
-> _common.add_output_argument
-> validate_args
   -> _common.validate_vmc_driver_args
-> build_system
-> resolve_spin_fermion_layout
-> build_model
-> build_sampler
-> _common.build_mcstate
-> maybe_seed_joint_sector
-> build_optimizer
   -> _common.build_basic_optimizer
-> build_driver
   -> _common.build_vmc_or_vmc_sr_driver
-> _common.run_driver
-> _common.energy_summary
-> _common.save_model_artifacts
-> _common.write_json(all_results)
```

这条入口只保留三类 KHC 特有逻辑:

- `KondoHeisenbergChainSpinFermion(...)` 的参数和 sector 约束。
- Transformer registry。
- `hamiltonian` / `factorized` sampler 分支。

optimizer / driver 的参数定义、校验、组装已经上提到 common。

---

## 5. 参数如何流动: SIAM

### 5.1 Hamiltonian 参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--hamiltonian-builder` | `_common.base_parser` | `_common._parse_builder_ref` | `Hamiltonian.<module>.<builder>` |
| `--hamiltonian-kw KEY=VALUE` | `_common.base_parser` | `_common._parse_builder_kwargs` | builder 的额外 kwargs |
| `--L --U --V --ti --pbc --n-fermions-per-spin --penalty-*` | `_common.base_parser` | `_common._default_siam_kwargs` | `Hamiltonian.SIAM.SIAM(...)` 默认 kwargs |

结果是:

```text
args
-> _common.build_hamiltonian(args)
-> builder_kwargs
-> Hamiltonian builder
-> (H, hi)
```

### 5.2 模型参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--models` | `_common.base_parser` | `_common.resolve_model_name` | registry key |
| `--n-layers` | `_common.add_model_optimizer_sr_arguments` | `_common.resolve_model_name` | shallow/deep variant 选择 |
| `--hidden-units` | `_common.base_parser` | `_common.build_model` | Slater/AGP 构造参数 |
| `--symmetric` | `_common.base_parser` | `_common.build_model` | AGP family 构造参数 |
| `--init-backflow-scale` | `_common.add_model_optimizer_sr_arguments` | `_common.build_model` | AGP backflow 构造参数 |

结果是:

```text
args
-> _common.summarize_model_config(...)
-> _common.build_model(...)
-> nqs_state.slater / nqs_state.AGP 中的具体类
-> model object
```

### 5.3 Optimizer / SR 参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--optimizer*` | `_common.add_model_optimizer_sr_arguments` | `_common._optimizer_learning_rate` / `_common.get_optimizer_and_sr` | optax outer optimizer |
| `--diag-shift*` | `_common.add_model_optimizer_sr_arguments` | `_common._sr_diag_shift` | `nk.optimizer.SR(diag_shift=...)` |
| `--sr-solver --sr-qgt --sr-holomorphic` | `_common.add_model_optimizer_sr_arguments` | `_common.get_optimizer_and_sr` | `nk.optimizer.SR(...)` |

结果是:

```text
args
-> _common.get_optimizer_and_sr(args)
-> (optimizer, sr)
-> nk.VMC(..., optimizer=optimizer, preconditioner=sr)
```

### 5.4 Sampling / runtime 参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--n-samples --n-discard-per-chain` | `_common.base_parser` | `_common.build_mcstate` | `nk.vqs.MCState(...)` |
| `--d-max` | `_common.base_parser` | runner-specific sampler builder | `MetropolisFermionHop*` |
| `--n-iter --show-progress` | `_common.base_parser` | `_common.run_driver` 或 segmented run helper | `driver.run(...)` |
| `--occ-update-every` | `run_siam_iteration_occ.py` | runner 自己保留 | `run_vmc_with_iteration_occ(...)` |
| `--orbital-update-every --hamiltonian-cutoff` | `run_siam_iteration_occ_1rdm.py` | runner 自己保留 | `run_vmc_with_iteration_natural_orbitals(...)` |

---

## 6. 参数如何流动: Kondo-Heisenberg

### 6.1 Hamiltonian / sector 参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--Lx --t --J_K --J1 --J2 --delta-z --pbc` | `run_kondoheisenbergchain_transformers.py::parse_args` | `collect_system_config` / `build_system` | `KondoHeisenbergChainSpinFermion(...)` |
| `--n-fermions` | 同上 | `validate_args` / `build_system` | generalized determinant 路径 |
| `--n-fermions-per-spin` | 同上 | `validate_args` / `tuple_or_none(...)` / `build_system` | blocked determinant 路径 |
| `--joint-two-sz --local-total-sz` | 同上 | `validate_args` / `maybe_seed_joint_sector` | conserved-sector 初始化 |

### 6.2 模型参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--models` | `parse_args` | `model_metadata(...)` / `build_model(...)` | Transformer registry |
| `--d-model --n-heads --n-layers --mlp-ratio` | `parse_args` | `collect_model_hyperparameters(...)` / `build_model(...)` | `LogSiteTokenTransformerNNBF` 或 `LogSpinChannelTransformerNNBF` |

### 6.3 Sampler 参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--joint-sampler` | `parse_args` | `build_sampler(...)` | `MetropolisHamiltonian` 或 `MetropolisSampler(TensorRule)` |
| `--fermion-sampler` | `parse_args` | `validate_args` / `build_factorized_sampler(...)` | `FermionHopRule` 或 `FermionHopRule_with_proposal` |
| `--proposal-*` | `parse_args` | `resolve_proposal_occupations(...)` | factorized fermion proposal |
| `--d-max --n-chains-per-rank --sweep-size` | `_common.add_runtime_sampling_arguments` | `build_factorized_sampler(...)` / `build_hamiltonian_sampler(...)` | sampler 运行参数 |

### 6.4 `MCState` / optimizer / driver / 输出参数

| 参数 | 读取位置 | 打包/转换位置 | 最终流向 |
| --- | --- | --- | --- |
| `--n-samples --n-discard-per-chain --seed` | `_common.add_runtime_sampling_arguments` | `_common.build_mcstate(...)` | `nk.vqs.MCState(...)` |
| `--driver --optimizer --learning-rate --diag-shift --proj-reg --momentum --linear-solver --mode --use-ntk --on-the-fly` | `_common.add_vmc_driver_arguments` | `collect_driver_config(...)` / `build_driver(...)` | `nk.VMC` 或 `nk.driver.VMC_SR` |
| `--n-iter --show-progress` | `_common.add_runtime_sampling_arguments` / `_common.add_progress_arguments` | `_common.run_driver(...)` | `driver.run(...)` |
| `--out-dir` | `_common.add_output_argument` | `_common.save_model_artifacts(...)` / `_common.write_json(...)` | `summary.json` / `final_vstate.msgpack` / `all_results.json` |

---

## 7. `MCState`、optimizer、SR、driver 的公共分工

这是最容易在 runner 里反复写样板代码的地方，所以现在统一要求如下。

### 7.1 `MCState`

统一入口:

```text
sampler + model + args(n_samples, n_discard_per_chain, seed)
-> _common.build_mcstate(...)
-> nk.vqs.MCState
```

这意味着 runner 不再自己手写 `MCState` 的样板 kwargs，除非它真的有额外的特殊字段。

对于 KHC 的 `vmc-sr` 分支，`MCState.n_parameters` 还会继续影响 `VMC_SR` 对 `use_ntk` 与 `on_the_fly` 的自动决策。

### 7.2 optimizer / SR / driver

- SIAM runner:
  - optimizer 和 SR 都由 `_common.get_optimizer_and_sr(args)` 统一产生。
  - 所以所有 SIAM 入口现在都能共享一套 optimizer/SR CLI。
- KHC runner:
  - optimizer / driver 的参数定义、打包和真正的 `VMC/VMC_SR` 组装已经上提到 `_common.py`。
  - runner 里保留的只是不属于 common 的 system/model/sampler 决策，以及一个薄的 wrapper。

### 7.3 driver.run

统一入口:

```text
driver + args(n_iter, show_progress)
-> _common.run_driver(...)
```

例外只有 SIAM 的 segmented helper，因为它要在 segment 之间插入 occupation / 1-RDM 更新。

---

## 8. 以后新增一个 runner 应该怎么写

建议严格按下面的顺序来。

1. 先确定这个 runner 的“独有特性”到底是什么。
2. 除这些独有特性外，其余 CLI、参数打包、`MCState`、driver.run、输出，都优先复用 `_common.py`。
3. 入口文件里只保留四类函数:
   - `parse_args()`
   - `validate_args()`（如果需要）
   - `build_*()` 里真正和该 Hamiltonian / model / sampler / driver 分支有关的函数
   - `main()`
4. 任何 summary dict 都先用 `namespace_snapshot(...)` 打包，再传给打印或 JSON。
5. 如果一个逻辑会在两支以上 runner 里重复出现，就应该上提到 `_common.py`。

按这个标准看，现在 `run/` 目录已经比较接近“入口薄、共用层厚、调用链清楚”的状态。
