# run_kondoheisenbergchain_transformers.py 接口说明

更新日期: 2026-04-15  
对应脚本: `netket/run/run_kondoheisenbergchain_transformers.py`

## 1. 这份文档讲什么

这份文档只讲这一支 `run` 脚本当前的实际接口，重点说明:

1. 物理参数和约束参数怎么传。
2. 脚本什么时候走 blocked determinant，什么时候走 generalized determinant。
3. `hamiltonian` 与 `factorized` 两种采样路径分别适合什么问题。
4. 共享 optimizer / driver 参数现在怎样从 `_common.py` 流到 `VMC_SR`。

## 2. 当前接口的核心结论

- 如果只固定总电子数 `n_fermions`，模型走 generalized determinant。
- 如果固定 `n_fermions_per_spin=(N_dn, N_up)`，模型走 blocked determinant。
- `blocked determinant` 只在 `J_K = 0` 的情形下使用。
- full Kondo 主线推荐入口是 `n_fermions + joint_two_sz + joint-sampler=hamiltonian`。
- 当前推荐 driver 是 `vmc-sr`，optimizer 先用 `adam`。
- `momentum=0.8` 现在已经由 runner 显式暴露。
- `use_ntk` 和 `on_the_fly` 如果不显式传参，就会保留 `VMC_SR` 的自动策略。
- 当前推荐路径里先不要使用 proposal 方案。

## 3. 共享参数和调用链

这一版里，Kondo-Heisenberg runner 已经把共享 optimizer / driver 逻辑上提到 `_common.py`:

- `_common.add_vmc_driver_arguments(...)`
  统一加上 `--driver`、`--optimizer`、`--learning-rate`、`--diag-shift`、`--proj-reg`、`--momentum`、`--linear-solver`、`--mode`、`--use-ntk`、`--on-the-fly`。
- `_common.validate_vmc_driver_args(args)`
  统一拦截 `--driver vmc` 却还传 `momentum` / `proj_reg` / `linear_solver` 这类 `VMC_SR` 专用参数的错误组合。
- `_common.collect_vmc_driver_config(args)`
  统一把 driver 参数打包成 `summary.json` 和终端摘要。
- `_common.build_basic_optimizer(args)`
  统一构造轻量 runner 使用的 `adam` / `sgd`。
- `_common.build_vmc_or_vmc_sr_driver(...)`
  统一把这些参数真正流进 `nk.VMC` 或 `nk.driver.VMC_SR`。

因此当前 KHC 的主调用链已经变成:

```text
parse_args
-> _common.add_vmc_driver_arguments
-> validate_args
   -> _common.validate_vmc_driver_args
-> build_system
-> resolve_spin_fermion_layout
-> build_model
-> build_sampler
-> _common.build_mcstate
-> build_optimizer
   -> _common.build_basic_optimizer
-> build_driver
   -> _common.build_vmc_or_vmc_sr_driver
-> _common.run_driver
-> _common.save_model_artifacts / _common.write_json
```

## 4. 参数接口

### 4.1 Hamiltonian / sector

这些参数直接流向 `KondoHeisenbergChainSpinFermion(...)`:

- `--Lx`
- `--t`
- `--J_K`
- `--J1`
- `--J2`
- `--delta-z`
- `--pbc`
- `--n-fermions`
- `--n-fermions-per-spin`
- `--joint-two-sz`
- `--local-total-sz`

其中边界最重要的一条是:

- `J_K != 0` 时走 `--n-fermions`，也就是 generalized determinant。
- `J_K = 0` 时才考虑 `--n-fermions-per-spin`，也就是 blocked determinant。

### 4.2 Model

这些参数决定 Transformer+NNBF ansatz 的结构:

- `--models`
- `--d-model`
- `--n-heads`
- `--n-layers`
- `--mlp-ratio`

### 4.3 Sampler

- `--joint-sampler hamiltonian|factorized`
- `--fermion-sampler standard|with-proposal`
- `--proposal-occupations-file`
- `--proposal-occupation-value`
- `--proposal-noise-strength`
- `--proposal-mixing`
- `--d-max`
- `--n-chains-per-rank`
- `--sweep-size`

当前推荐口径是:

- full Kondo 先用 `joint-sampler = hamiltonian`
- 当前版本先不要打开 proposal 方案

### 4.4 Optimizer / driver

这些参数现在统一由 `_common.add_vmc_driver_arguments(...)` 暴露:

- `--driver vmc|vmc-sr`
- `--optimizer sgd|adam`
- `--learning-rate`
- `--diag-shift`
- `--proj-reg`
- `--momentum`
- `--linear-solver`
- `--mode`
- `--use-ntk` / `--no-use-ntk`
- `--on-the-fly` / `--no-on-the-fly`

这里要分清两层:

- `optimizer` 是外层参数更新器。当前推荐先用 `adam`。
- `momentum` 是 `VMC_SR` 内部的 spring 参数，不是 `adam` 的动量。

如果你不显式传 `--use-ntk` / `--no-use-ntk`，脚本会把 `None` 直接传给 `VMC_SR`。
同理，如果你不显式传 `--on-the-fly` / `--no-on-the-fly`，脚本也会保留 `VMC_SR` 的自动逻辑。

## 5. validate_args 现在会前置拦截什么

脚本会在真正构造系统之前直接拒绝以下参数组合:

- 同时传 `--n-fermions` 和 `--n-fermions-per-spin`
- `J_K != 0` 时再传 `--n-fermions-per-spin`
- `J_K != 0` 时再传 `--local-total-sz`
- `--joint-sampler hamiltonian` 时还传 `--fermion-sampler with-proposal`
- `--driver vmc` 时还传 `--proj-reg`、`--momentum`、`--linear-solver`、`--use-ntk/--no-use-ntk`、`--on-the-fly/--no-on-the-fly`

最后这一条现在已经实际验证过，入口会直接报:

```text
ValueError: --momentum only applies to --driver vmc-sr.
```

## 6. determinant backend 如何选择

- `layout.uses_block_determinant == True`
  说明 Hilbert 固定了 `n_dn` 和 `n_up`，模型走 blocked determinant。
- `layout.uses_block_determinant == False`
  说明只固定了总电子数 `n_fermions`，模型走 generalized determinant。

当前建议非常明确:

- full Kondo 主线用 generalized determinant
- blocked determinant 只保留给 `J_K = 0` 的参考入口

## 7. 推荐命令

### 7.1 full Kondo 主线

```powershell
python run/run_kondoheisenbergchain_transformers.py \
  --Lx 2 \
  --J_K 1.0 \
  --n-fermions 2 \
  --joint-two-sz 0 \
  --joint-sampler hamiltonian \
  --models site-token \
  --d-model 8 \
  --n-heads 2 \
  --n-layers 1 \
  --mlp-ratio 2 \
  --driver vmc-sr \
  --optimizer adam \
  --learning-rate 1e-2 \
  --momentum 0.8 \
  --n-iter 1 \
  --n-samples 16 \
  --n-discard-per-chain 0 \
  --n-chains-per-rank 1 \
  --no-progress
```

### 7.2 `J_K = 0` 的 blocked 参考入口

```powershell
python run/run_kondoheisenbergchain_transformers.py \
  --Lx 2 \
  --J_K 0.0 \
  --n-fermions-per-spin 1 1 \
  --local-total-sz 0 \
  --joint-sampler factorized \
  --fermion-sampler standard \
  --models site-token \
  --d-model 8 \
  --n-heads 2 \
  --n-layers 1 \
  --mlp-ratio 2 \
  --driver vmc-sr \
  --optimizer adam \
  --learning-rate 1e-2 \
  --momentum 0.8 \
  --n-iter 1 \
  --n-samples 16 \
  --n-discard-per-chain 0 \
  --n-chains-per-rank 1 \
  --no-progress
```

## 8. 终端摘要和输出文件里现在能看到什么

脚本启动后，终端和 `summary.json` 里现在会明确记录:

- `determinant_backend = blocked | generalized`
- `n_parameters`
- `momentum`
- `proj_reg`
- `linear_solver`
- `use_ntk = auto|True|False`
- `on_the_fly = auto|True|False`

因此，主线 example 里“哪些参数是显式写的，哪些是没写吃默认”的状态，现在终端和 json 都能对应得上。
