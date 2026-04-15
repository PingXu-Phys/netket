﻿# run_kondoheisenbergchain_transformers.py Interface Guide

Updated: 2026-04-15
Script: `netket/my_run/run_kondoheisenbergchain_transformers.py`

## 1. Scope

This document describes the actual current interface of the Kondo-Heisenberg runner:

1. how physical and sector parameters are passed
2. when the model uses blocked or generalized determinants
3. what `hamiltonian` and `hamiltonian-with-proposal` mean
4. how proposal parameters flow into `MetropolisHamiltonianWithProposal(...)`
5. how shared optimizer and driver options flow through `_common.py`

## 2. Core rules

- If only total fermion number `n_fermions` is fixed, the model uses the generalized determinant path.
- If `n_fermions_per_spin=(N_dn, N_up)` is fixed, the model uses the blocked determinant path.
- The blocked determinant path is only intended for `J_K = 0`.
- The default full Kondo baseline is `n_fermions + joint_two_sz + joint-sampler=hamiltonian`.
- To test the new sampler against the original rule, use `joint-sampler=hamiltonian-with-proposal`, no occupation input, and `proposal-balance-beta=0`.

## 3. Main call chain

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
-> maybe_seed_joint_sector
-> build_optimizer
   -> _common.build_basic_optimizer
-> build_driver
   -> _common.build_vmc_or_vmc_sr_driver
-> _common.run_driver
-> _common.save_model_artifacts / _common.write_json
```

## 4. Parameter groups

### 4.1 Hamiltonian and sector arguments

These flow into `KondoHeisenbergChainSpinFermion(...)`:

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

Important constraint:

- use `--n-fermions` when `J_K != 0`
- use `--n-fermions-per-spin` only for the blocked, `J_K = 0` path

### 4.2 Model arguments

These define the Transformer+NNBF ansatz:

- `--models`
- `--d-model`
- `--n-heads`
- `--n-layers`
- `--mlp-ratio`

### 4.3 Sampler arguments

- `--joint-sampler hamiltonian|hamiltonian-with-proposal`
- `--proposal-occupations-file`
- `--proposal-occupation-value`
- `--proposal-noise-strength`
- `--proposal-mixing`
- `--proposal-balance-beta`
- `--d-max`
- `--n-chains-per-rank`
- `--sweep-size`

Meaning:

- `hamiltonian`
  use the original `MetropolisHamiltonian(...)` with the exact full joint connectivity
- `hamiltonian-with-proposal`
  keep the same exact connectivity, then add fermion occupation bias plus `ss / ff / sf` soft balancing
- `proposal-occupations-file`
  read occupations from file; supported lengths are `L` and `2L`
- `proposal-occupation-value`
  build a constant length-`L` occupation vector for smoke tests
- `proposal-balance-beta`
  controls move-type balancing
  - `0`: no extra balancing
  - `1`: strongest balancing across non-empty move classes

### 4.4 Optimizer and driver arguments

These are exposed by `_common.add_vmc_driver_arguments(...)`:

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

Notes:

- `optimizer` is the outer parameter update rule
- `momentum` is the internal spring parameter of `VMC_SR`, not Adam momentum

## 5. Validation rules

The runner rejects the following combinations before system construction:

- both `--n-fermions` and `--n-fermions-per-spin`
- `--n-fermions-per-spin` when `J_K != 0`
- `--local-total-sz` when `J_K != 0`
- `--proposal-balance-beta` outside `[0, 1]`
- `--driver vmc` together with SR-only options such as `--proj-reg`, `--momentum`, `--linear-solver`, `--use-ntk`, or `--on-the-fly`

## 6. Determinant backend selection

- `layout.uses_block_determinant == True`
  means fixed `n_dn` and `n_up`, therefore blocked determinant
- `layout.uses_block_determinant == False`
  means only fixed total fermion number, therefore generalized determinant

Recommended use:

- generalized determinant for full Kondo calculations
- blocked determinant only as a `J_K = 0` reference path

## 7. Minimal command set

### 7.1 Baseline

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --n-fermions 2 `
  --joint-two-sz 0 `
  --joint-sampler hamiltonian `
  --models site-token `
  --d-model 8 `
  --n-heads 2 `
  --n-layers 1 `
  --mlp-ratio 2 `
  --driver vmc-sr `
  --optimizer adam `
  --learning-rate 1e-2 `
  --momentum 0.8 `
  --n-iter 1 `
  --n-samples 16 `
  --n-discard-per-chain 0 `
  --n-chains-per-rank 1 `
  --no-progress
```

### 7.2 Neutral check of the new sampler

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --n-fermions 2 `
  --joint-two-sz 0 `
  --joint-sampler hamiltonian-with-proposal `
  --proposal-balance-beta 0 `
  --models site-token `
  --d-model 8 `
  --n-heads 2 `
  --n-layers 1 `
  --mlp-ratio 2 `
  --driver vmc-sr `
  --optimizer adam `
  --learning-rate 1e-2 `
  --momentum 0.8 `
  --n-iter 1 `
  --n-samples 16 `
  --n-discard-per-chain 0 `
  --n-chains-per-rank 1 `
  --no-progress
```

### 7.3 Proposal-enabled smoke test

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --n-fermions 2 `
  --joint-two-sz 0 `
  --joint-sampler hamiltonian-with-proposal `
  --proposal-occupation-value 0.5 `
  --proposal-noise-strength 100 `
  --proposal-mixing 0.05 `
  --proposal-balance-beta 0.5 `
  --models site-token `
  --d-model 8 `
  --n-heads 2 `
  --n-layers 1 `
  --mlp-ratio 2 `
  --driver vmc-sr `
  --optimizer adam `
  --learning-rate 1e-2 `
  --momentum 0.8 `
  --n-iter 1 `
  --n-samples 16 `
  --n-discard-per-chain 0 `
  --n-chains-per-rank 1 `
  --no-progress
```

## 8. Occupation file order

If the file contains length `2L`, the required order is:

```text
[dn_1, ..., dn_L, up_1, ..., up_L]
```

## 9. What appears in terminal summaries and JSON

The runner records at least:

- `determinant_backend = blocked | generalized`
- `joint_sampler = hamiltonian | hamiltonian-with-proposal`
- `proposal_occupations_source`
- `proposal_occupations_length`
- `proposal_noise_strength`
- `proposal_mixing`
- `proposal_balance_beta`
- `n_parameters`
- `momentum`
- `proj_reg`
- `linear_solver`
- `use_ntk = auto|True|False`
- `on_the_fly = auto|True|False`

## 10. Repeated connected entries

This sampler uses the effective connected entries returned by `get_conn_padded(...)` as the proposal support. It does not deduplicate equal target states before assigning proposal mass.

For a current state `?`, let the valid connected entries be `E(?)`, and let each entry `e` point to a target state `T_?(e)=?`. The actual proposal implemented by the code is

```math
q_{code}(?\mid?)
=
\sum_{e:T_?(e)=?}
\frac{W_e(?)}{\sum_{e'\in E(?)} W_{e'}(?)}.
```

This means that repeated connected entries change the proposal shape: a target state that appears multiple times gets the sum of those entry probabilities.

This does not invalidate the Metropolis sampler. The implementation computes the forward and backward proposal probabilities using the same effective `q_{code}`, including duplicates, by summing repeated entries with `_log_state_probability(...)`. Therefore the acceptance rule remains

```math
A(?\to?)
=
\min\left(
1,
\frac{|\Psi(?)|^2\,q_{code}(?\mid?)}{|\Psi(?)|^2\,q_{code}(?\mid?)}
\right),
```

and the final stationary distribution is still `|?|^2`.

The only thing duplicates change is the proposal itself. If a future version must realize an ideal proposal defined on unique target states, deduplication or an explicit multiplicity correction must be added before sampling.
