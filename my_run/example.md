# run_kondoheisenbergchain_transformers.py Example

更新日期: 2026-04-15
对应脚本: `netket/run/run_kondoheisenbergchain_transformers.py`

## 1. 当前确认的口径

- `blocked determinant` 只在 `J_K = 0` 的情形下使用。
- 一般主线计算优先使用 `generalized determinant`，也就是传 `--n-fermions`，不要传 `--n-fermions-per-spin`。
- 当前推荐的 driver 是 `--driver vmc-sr`。
- optimizer 先用 `--optimizer adam`。
- `spring` 先按 `--momentum 0.8` 使用。
- `diag_shift`、`proj_reg`、`linear_solver`、`mode` 先保持脚本默认值。
- `use_ntk` 和 `on_the_fly` 先保持自动模式，不要显式传 `--use-ntk`、`--no-use-ntk`、`--on-the-fly`、`--no-on-the-fly`。
- 当前推荐路径先不要使用 proposal 方案。
- full Kondo 主线先用 `--joint-sampler hamiltonian`。

## 2. 已实际跑通的两个 example

### 2.1 full Kondo 主线 example

```powershell
python run/run_kondoheisenbergchain_transformers.py \
  --Lx 2 \
  --J_K 1.0 \
  --J1 1.0 \
  --J2 0.0 \
  --delta-z 1.0 \
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
  --d-max 1 \
  --seed 1234 \
  --no-progress \
  --out-dir C:\Users\10783\run_outputs\khc_mainline_common_smoke
```

这条命令已经实际跑通，关键信息如下:

- `determinant_backend = generalized`
- `n_parameters = 724`
- `Automatic SR implementation choice = NTK`
- `final_energy = -0.9362769026`
- 输出目录: `C:\Users\10783\run_outputs\khc_mainline_common_smoke`

### 2.2 `J_K = 0` 的 blocked 参考 example

```powershell
python run/run_kondoheisenbergchain_transformers.py \
  --Lx 2 \
  --J_K 0.0 \
  --J1 1.0 \
  --J2 0.0 \
  --delta-z 1.0 \
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
  --d-max 1 \
  --seed 1234 \
  --no-progress \
  --out-dir C:\Users\10783\run_outputs\khc_blocked_common_smoke
```

这条命令也已经实际跑通，关键信息如下:

- `determinant_backend = blocked`
- `n_parameters = 702`
- `Automatic SR implementation choice = NTK`
- `final_energy = 1.2728040581`
- 输出目录: `C:\Users\10783\run_outputs\khc_blocked_common_smoke`

## 3. 全部参数名、默认值、显式/隐式情况

下面把当前脚本的全部参数按分组列出来。表里的“显式”表示在上面的 example 命令里明确写了这个参数；“默认”表示命令里没写，脚本直接吃 parser 默认值。

### 3.1 Hamiltonian / system

| 参数 | 脚本默认值 | full Kondo 主线值 | 主线状态 | `J_K=0` 参考值 | 参考状态 |
| --- | --- | --- | --- | --- | --- |
| `--Lx` | `50` | `2` | 显式 | `2` | 显式 |
| `--t` | `1.0` | `1.0` | 默认 | `1.0` | 默认 |
| `--J_K` | `1.0` | `1.0` | 显式 | `0.0` | 显式 |
| `--J1` | `1.0` | `1.0` | 显式 | `1.0` | 显式 |
| `--J2` | `0.0` | `0.0` | 显式 | `0.0` | 显式 |
| `--delta-z` | `1.0` | `1.0` | 显式 | `1.0` | 显式 |
| `--pbc` | `False` | `False` | 默认 | `False` | 默认 |
| `--n-fermions` | `None` | `2` | 显式 | `None` | 默认 |
| `--joint-two-sz` | `0` | `0` | 显式 | `0` | 默认 |
| `--local-total-sz` | `None` | `None` | 默认 | `0` | 显式 |
| `--n-fermions-per-spin` | `None` | `None` | 默认 | `1 1` | 显式 |

### 3.2 Model

| 参数 | 脚本默认值 | full Kondo 主线值 | 主线状态 | `J_K=0` 参考值 | 参考状态 |
| --- | --- | --- | --- | --- | --- |
| `--models` | `site-token spin-channel` | `site-token` | 显式 | `site-token` | 显式 |
| `--d-model` | `64` | `8` | 显式 | `8` | 显式 |
| `--n-heads` | `4` | `2` | 显式 | `2` | 显式 |
| `--n-layers` | `2` | `1` | 显式 | `1` | 显式 |
| `--mlp-ratio` | `4` | `2` | 显式 | `2` | 显式 |

### 3.3 Sampler

| 参数 | 脚本默认值 | full Kondo 主线值 | 主线状态 | `J_K=0` 参考值 | 参考状态 |
| --- | --- | --- | --- | --- | --- |
| `--joint-sampler` | `hamiltonian` | `hamiltonian` | 显式 | `factorized` | 显式 |
| `--fermion-sampler` | `standard` | `standard` | 默认 | `standard` | 显式 |
| `--proposal-occupations-file` | `None` | `None` | 默认 | `None` | 默认 |
| `--proposal-occupation-value` | `0.5` | `0.5` | 默认 | `0.5` | 默认 |
| `--proposal-noise-strength` | `100.0` | `100.0` | 默认 | `100.0` | 默认 |
| `--proposal-mixing` | `0.05` | `0.05` | 默认 | `0.05` | 默认 |

### 3.4 Optimizer / driver

这里特别注意三件事:

- `--optimizer` 和 `--learning-rate` 现在已经上提到 `_common.add_vmc_driver_arguments(...)`。
- `--momentum` 现在也由 common 暴露，并最终流到 `_common.build_vmc_or_vmc_sr_driver(...)`。
- `--use-ntk` 和 `--on-the-fly` 的默认值不是简单的 `True/False`，而是 `auto`，也就是 parser 保留 `None`，再交给 `VMC_SR` 自己决定。

| 参数 | 脚本默认值 | full Kondo 主线值 | 主线状态 | `J_K=0` 参考值 | 参考状态 |
| --- | --- | --- | --- | --- | --- |
| `--driver` | `vmc-sr` | `vmc-sr` | 显式 | `vmc-sr` | 显式 |
| `--optimizer` | `adam` | `adam` | 显式 | `adam` | 显式 |
| `--learning-rate` | `1.0e-2` | `1.0e-2` | 显式 | `1.0e-2` | 显式 |
| `--diag-shift` | `5.0e-2` | `5.0e-2` | 默认 | `5.0e-2` | 默认 |
| `--proj-reg` | `None` | `None` | 默认 | `None` | 默认 |
| `--momentum` | `None` | `0.8` | 显式 | `0.8` | 显式 |
| `--linear-solver` | `cholesky` | `cholesky` | 默认 | `cholesky` | 默认 |
| `--mode` | `real` | `real` | 默认 | `real` | 默认 |
| `--use-ntk` / `--no-use-ntk` | `auto` | `auto` | 默认 | `auto` | 默认 |
| `--on-the-fly` / `--no-on-the-fly` | `auto` | `auto` | 默认 | `auto` | 默认 |

### 3.5 Sampling / runtime / output

| 参数 | 脚本默认值 | full Kondo 主线值 | 主线状态 | `J_K=0` 参考值 | 参考状态 |
| --- | --- | --- | --- | --- | --- |
| `--n-iter` | `200` | `1` | 显式 | `1` | 显式 |
| `--n-samples` | `2048` | `16` | 显式 | `16` | 显式 |
| `--n-discard-per-chain` | `32` | `0` | 显式 | `0` | 显式 |
| `--d-max` | `1` | `1` | 显式 | `1` | 显式 |
| `--n-chains-per-rank` | `16` | `1` | 显式 | `1` | 显式 |
| `--sweep-size` | `None` | `None` | 默认 | `None` | 默认 |
| `--seed` | `1234` | `1234` | 显式 | `1234` | 显式 |
| `--out-dir` | `D:\Seafile\PHD\NQS\NetKet\netket\run_outputs\kondoheisenbergchain_transformers` | `C:\Users\10783\run_outputs\khc_mainline_common_smoke` | 显式 | `C:\Users\10783\run_outputs\khc_blocked_common_smoke` | 显式 |
| `--show-progress` / `--no-progress` | `show-progress=True` | `False` | 显式，传 `--no-progress` | `False` | 显式，传 `--no-progress` |

## 4. 现在推荐你怎么用

如果你现在就是要按这套口径继续算，可以直接记成下面这句:

- full Kondo: `generalized + hamiltonian + vmc-sr + adam + momentum 0.8 + no proposal`
- `J_K = 0`: 才允许把 `blocked` 当参考入口拿出来用
- `use_ntk` / `on_the_fly` 先不要手动钉死，先让 `VMC_SR` 自动判断
- 真正开始生产计算时，只要在这两个 example 的基础上把 `Lx`、`n_iter`、`n_samples` 放大即可
