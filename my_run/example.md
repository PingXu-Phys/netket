# run_kondoheisenbergchain_transformers.py Example

Updated: 2026-04-15
Script: `netket/my_run/run_kondoheisenbergchain_transformers.py`

## 1. Active sampler options

The runner now exposes only two joint sampler branches:

- `--joint-sampler hamiltonian`
- `--joint-sampler hamiltonian-with-proposal`

Meaning:

- `hamiltonian`: use the original `MetropolisHamiltonian(...)` on the full joint Hamiltonian connectivity.
- `hamiltonian-with-proposal`: keep the same full Hamiltonian connectivity, then add fermion occupation bias plus `ss / ff / sf` move-type balancing.

## 2. Most important correctness check

If you want to check that the new sampler reproduces the original behavior under neutral settings, use:

- `--joint-sampler hamiltonian-with-proposal`
- no `--proposal-occupations-file`
- no `--proposal-occupation-value`
- `--proposal-balance-beta 0`

This still goes through the new CLI branch and the new factory, but the proposal reduces to the original `HamiltonianRule` behavior.

## 3. Recommended commands

### 3.1 Full Kondo baseline

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --J1 1.0 `
  --J2 0.0 `
  --delta-z 1.0 `
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
  --d-max 1 `
  --seed 1234 `
  --no-progress `
  --out-dir C:\Users\10783\run_outputs\khc_hamiltonian_smoke
```

### 3.2 New sampler shell, neutral settings

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --J1 1.0 `
  --J2 0.0 `
  --delta-z 1.0 `
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
  --d-max 1 `
  --seed 1234 `
  --no-progress `
  --out-dir C:\Users\10783\run_outputs\khc_hamiltonian_with_proposal_neutral_smoke
```

### 3.3 Proposal enabled with a constant occupation profile

```powershell
python .\my_run\run_kondoheisenbergchain_transformers.py `
  --Lx 2 `
  --J_K 1.0 `
  --J1 1.0 `
  --J2 0.0 `
  --delta-z 1.0 `
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
  --d-max 1 `
  --seed 1234 `
  --no-progress `
  --out-dir C:\Users\10783\run_outputs\khc_hamiltonian_with_proposal_smoke
```

## 4. Occupation file conventions

If `--proposal-occupations-file` is used, the file may contain either:

- length `L`: shared orbital occupations for both spin sectors
- length `2L`: explicit order `[dn_1, ..., dn_L, up_1, ..., up_L]`

## 5. Sampler defaults

| Argument | Default | Meaning |
| --- | --- | --- |
| `--joint-sampler` | `hamiltonian` | full-Hamiltonian baseline |
| `--proposal-occupations-file` | `None` | no file-based occupation input |
| `--proposal-occupation-value` | `None` | no constant occupation vector |
| `--proposal-noise-strength` | `100.0` | proposal noise strength |
| `--proposal-mixing` | `0.05` | proposal mixing |
| `--proposal-balance-beta` | `0.5` | `ss / ff / sf` balancing strength |

## 6. Suggested test order

1. `hamiltonian`
2. `hamiltonian-with-proposal` with `occupations=None` and `balance_beta=0`
3. `hamiltonian-with-proposal` with `occupations=None` and `balance_beta=0.5`
4. `hamiltonian-with-proposal` with non-empty occupations and `balance_beta=0.5`
