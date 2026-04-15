# NetKet `run/` Structure, Call Chain, and Parameter Flow

Updated: 2026-04-15

This document explains the current responsibility boundaries inside `run/`.

## 1. General principles

- Each `run/*.py` file should keep only its physical or algorithmic specifics.
- Repeated plumbing belongs in `run/_common.py`.
- Parameters should be grouped by responsibility and clearly mapped to Hamiltonian, model, sampler, `MCState`, optimizer, driver, and output files.

## 2. File responsibilities

| File | Role | Should handle | Should not handle |
| --- | --- | --- | --- |
| `my_run/_common.py` | common layer | CLI grouping, parameter packaging, `MCState` construction, VMC/VMC_SR helpers, driver execution, JSON/msgpack output | Hamiltonian-specific physics, sampler-specific branching |
| `my_run/run_siam_no_proposal.py` | SIAM baseline entry | standard `MetropolisFermionHop`, no occupation update | duplicated CLI and optimizer/SR boilerplate |
| `my_run/run_siam_iteration_occ.py` | SIAM occupation-update entry | periodic proposal-occupation updates | duplicated shared runtime helpers |
| `my_run/run_siam_iteration_occ_1rdm.py` | SIAM 1-RDM / NO entry | 1-RDM, natural orbitals, Hamiltonian rotation | duplicated optimizer/SR and packaging code |
| `my_run/run_kondoheisenbergchain_transformers.py` | Kondo-Heisenberg transformer entry | system / model / sampler decisions plus thin wrappers around `_common.py` | duplicated generic CLI, driver, and output code |
| `Hamiltonian/KondoHeisenbergChainSpinFermion.py` | KHC Hamiltonian layer | joint Hilbert, Hamiltonian, sector helpers | CLI and output handling |
| `nqs_state/transformer_joint_nnbf.py` | KHC model layer | Transformer+NNBF classes and `resolve_spin_fermion_layout(...)` | CLI handling |

## 3. Unified top-level call chain

```text
parse_args
-> read grouped parameters
-> package summary information
-> build Hamiltonian/system
-> build model
-> build sampler
-> _common.build_mcstate
-> build optimizer / driver
-> _common.run_driver or runner-specific segmented run
-> collect summary
-> _common.write_json / save_vstate
```

## 4. Actual Kondo-Heisenberg runner chain

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

KHC-specific logic is intentionally limited to:

- `KondoHeisenbergChainSpinFermion(...)` arguments and sector constraints
- the Transformer registry
- the `hamiltonian` / `hamiltonian-with-proposal` sampler branches

## 5. Current sampler architecture

### 5.1 `hamiltonian`

- uses `nk.sampler.MetropolisHamiltonian(...)`
- proposal support comes from the exact full joint Hamiltonian connectivity
- this is the baseline for full Kondo calculations

### 5.2 `hamiltonian-with-proposal`

- uses `netket.sampler.metropolis.MetropolisHamiltonianWithProposal(...)`
- proposal support still comes from the same exact full Hamiltonian connectivity
- extra proposal logic is added only on top of that support:
  - fermion occupation bias
  - `ss / ff / sf` move-type soft balancing
- if
  - `occupations=None`
  - `balance_beta=0`
  then the proposal reduces to the original `HamiltonianRule`

## 6. Parameter flow for Kondo-Heisenberg

### 6.1 Hamiltonian and sector parameters

| Argument | Read in | Processed in | Final target |
| --- | --- | --- | --- |
| `--Lx --t --J_K --J1 --J2 --delta-z --pbc` | `parse_args` | `collect_system_config` / `build_system` | `KondoHeisenbergChainSpinFermion(...)` |
| `--n-fermions` | `parse_args` | `validate_args` / `build_system` | generalized determinant path |
| `--n-fermions-per-spin` | `parse_args` | `validate_args` / `tuple_or_none(...)` / `build_system` | blocked determinant path |
| `--joint-two-sz --local-total-sz` | `parse_args` | `validate_args` / `maybe_seed_joint_sector` | conserved-sector initialization |

### 6.2 Model parameters

| Argument | Read in | Processed in | Final target |
| --- | --- | --- | --- |
| `--models` | `parse_args` | `model_metadata(...)` / `build_model(...)` | Transformer registry |
| `--d-model --n-heads --n-layers --mlp-ratio` | `parse_args` | `collect_model_hyperparameters(...)` / `build_model(...)` | `LogSiteTokenTransformerNNBF` or `LogSpinChannelTransformerNNBF` |

### 6.3 Sampler parameters

| Argument | Read in | Processed in | Final target |
| --- | --- | --- | --- |
| `--joint-sampler` | `parse_args` | `build_sampler(...)` | `MetropolisHamiltonian` or `MetropolisHamiltonianWithProposal` |
| `--proposal-occupations-file --proposal-occupation-value` | `parse_args` | `resolve_proposal_occupations(...)` | occupation input of `HamiltonianRuleWithProposal` |
| `--proposal-noise-strength --proposal-mixing --proposal-balance-beta` | `parse_args` | `build_hamiltonian_with_proposal_sampler(...)` | proposal controls of `MetropolisHamiltonianWithProposal(...)` |
| `--d-max --n-chains-per-rank --sweep-size` | `_common.add_runtime_sampling_arguments` | `build_hamiltonian_sampler(...)` / `build_hamiltonian_with_proposal_sampler(...)` | sampler runtime settings |

### 6.4 `MCState`, optimizer, driver, and outputs

| Argument | Read in | Processed in | Final target |
| --- | --- | --- | --- |
| `--n-samples --n-discard-per-chain --seed` | `_common.add_runtime_sampling_arguments` | `_common.build_mcstate(...)` | `nk.vqs.MCState(...)` |
| `--driver --optimizer --learning-rate --diag-shift --proj-reg --momentum --linear-solver --mode --use-ntk --on-the-fly` | `_common.add_vmc_driver_arguments` | `collect_driver_config(...)` / `build_driver(...)` | `nk.VMC` or `nk.driver.VMC_SR` |
| `--n-iter --show-progress` | `_common.add_runtime_sampling_arguments` / `_common.add_progress_arguments` | `_common.run_driver(...)` | `driver.run(...)` |
| `--out-dir` | `_common.add_output_argument` | `_common.save_model_artifacts(...)` / `_common.write_json(...)` | `summary.json` / `final_vstate.msgpack` / `all_results.json` |

## 7. Rule for future runners

When adding a new runner:

1. define the truly runner-specific feature first
2. reuse `_common.py` for everything else whenever possible
3. keep only the following in the entry file:
   - `parse_args()`
   - `validate_args()`
   - `build_*()` helpers
   - `main()`
4. if logic is duplicated across two or more runners, move it into `_common.py`
