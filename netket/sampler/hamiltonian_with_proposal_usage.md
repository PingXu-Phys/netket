# Hamiltonian With Proposal Usage

## 1. What This Adds

This implementation adds a Hamiltonian-connected sampler that stays on the exact off-diagonal connectivity of the full mixed Kondo-Heisenberg operator, but reweights the connected moves using:
- fermion occupation-biased proposals
- a soft balance between `ss`, `ff`, and `sf` move classes

It corresponds to the pair:
- rule file: `netket/sampler/rules/hamiltonian_with_proposal.py`
- factory: `netket/sampler/metropolis.py::MetropolisHamiltonianWithProposal(...)`

Conceptually it is the Hamiltonian analogue of:
- `fermion_2nd.py` -> `fermion_2nd_proposal.py`
- `MetropolisFermionHop(...)` -> `MetropolisFermionHopWithProposal(...)`

## 2. Current Scope

The current implementation is intentionally narrow and matches the Kondo-Heisenberg use case discussed in the design notes.

It currently expects:
- a JAX operator, i.e. `DiscreteJaxOperator`
- a tensor Hilbert layout `(SpinOrbitalFermions) x (local spin)`
- exactly two fermion spin subsectors
- one local spin site per fermion orbital

It does not currently provide a Numba / non-JAX implementation.

## 3. Call Chain

The runtime call chain is:

1. User code imports `MetropolisHamiltonianWithProposal` from `netket.sampler.metropolis` and calls it.
2. The factory builds `HamiltonianRuleWithProposal(...)`.
3. The thin wrapper dispatches to `HamiltonianRuleWithProposalJax`.
4. `transition(...)` calls `operator.get_conn_padded(x)` to enumerate all Hamiltonian-connected states.
5. Each connected state is classified as one of:
   - `ss`: spin-spin only
   - `ff`: fermion-fermion only
   - `sf`: spin-fermion joint change
6. If `occupations` is provided, one noisy occupation profile is sampled per chain using the same mixing + Beta scheme as `fermion_2nd_proposal.py`.
7. A fermion-only proposal bias is computed for every connected state.
8. The three move classes are rebalanced using `balance_beta`.
9. One connected state is sampled.
10. The same noisy occupation profile is reused on the proposed state to compute the backward proposal probability.
11. `transition(...)` returns:
    - the proposed state `x_proposed`
    - the Metropolis-Hastings log correction `log q(x' -> x) - log q(x -> x')`
12. `MetropolisSampler` uses that correction in the usual accept/reject step.

## 4. What The Rule Actually Does

### 4.1 Support set

The support set is exactly the Hamiltonian off-diagonal connectivity:

```math
\mathcal C_H^{\mathrm{off}}(x)=\{x'\neq x : H_{x,x'}\neq 0\}
```

No extra hop clusters or hand-written Kondo move lists are introduced.

### 4.2 Fermion bias

If occupations are provided, the fermion part of the proposal weight is:

```math
B_f(x\to x')
=
\prod_{m:0\to1}\tilde v_m
\prod_{m:1\to0}(1-\tilde v_m)
```

This depends only on fermion mode occupation changes, so it supports:
- same-spin hopping
- on-site fermion spin flips
- basis-rotated one-body processes `c_a^\dagger c_b`

### 4.3 Move-type balance

Connected states are split into three groups:
- `ss`
- `ff`
- `sf`

The parameter `balance_beta` controls how strongly the sampler pushes the proposal mass toward comparable activity in the three groups.

Interpretation:
- `balance_beta = 0`: no explicit move-type balancing
- `balance_beta = 1`: strongest balancing among the non-empty move classes
- `0 < balance_beta < 1`: soft interpolation

This is the parameter that controls the `ss / sf / ff` ratio.

## 5. Interface

### 5.1 Factory

```python
from netket.sampler.metropolis import MetropolisHamiltonianWithProposal

MetropolisHamiltonianWithProposal(
    hilbert,
    hamiltonian,
    *,
    occupations=None,
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    aggregate_duplicate_entries=False,
    dtype=np.int8,
    **kwargs,
)
```

### 5.2 Parameters

- `hilbert`
  The joint Hilbert space.

- `hamiltonian`
  The full mixed operator whose connectivity defines the proposal support.

- `occupations`
  Optional fermion occupation prior.

  Allowed input shapes:
  - length `L`: one value per orbital, shared by both spin blocks
  - length `2L`: one value per fermion mode

  If length `2L` is used, the order must be:

```text
[occ_dn(1), ..., occ_dn(L), occ_up(1), ..., occ_up(L)]
```

- `balance_beta`
  Controls the relative proposal mass assigned to `ss`, `ff`, and `sf` move classes.

- `noise_strength`
  Beta concentration parameter used when occupations are provided.

- `mixing`
  Uniform mixing coefficient applied before Beta resampling.

- `aggregate_duplicate_entries`
  Controls whether repeated connected entries leading to the same target state are explicitly summed in the MH proposal correction.

  Interpretation:
  - `False`: entry-level treatment, closer to the original `HamiltonianRule` behavior
  - `True`: state-level aggregation by summing repeated entries with `logsumexp`

- `dtype`, `**kwargs`
  Passed through to `MetropolisSampler` exactly like other sampler factories.

## 6. Default Behavior And Degenerate Limits

### 6.1 No occupations

If `occupations=None`, the fermion prior is neutral. The implementation does **not** convert this to a literal all-ones occupation array. Instead it uses a dedicated neutral branch:

```math
B_f(x\to x') \equiv 1
```

### 6.2 Recovery of the original Hamiltonian rule

If:

```python
occupations=None
balance_beta=0.0
aggregate_duplicate_entries=False
```

then the proposal becomes uniform over the Hamiltonian-connected states, i.e. it reduces to the original `HamiltonianRule` proposal.

If instead `aggregate_duplicate_entries=True`, the sampler stays in the neutral proposal limit but computes MH corrections with explicit duplicate-target aggregation instead of the original entry-level shortcut.

### 6.3 Only move-type balancing

If:

```python
occupations=None
balance_beta>0
```

then the sampler is still Hamiltonian-connected and fermion-neutral, but it redistributes proposal mass between `ss`, `ff`, and `sf`.

## 7. Efficiency Notes

This implementation is JAX-friendly:
- it keeps the connectivity backend in `get_conn_padded(...)`
- it uses JAX arrays and `vmap`
- it avoids Python loops inside `transition(...)`

However, it is **not** as cheap as the original uniform `HamiltonianRule`.

Reason:
- the original rule only needs uniform sampling over connected states and the forward/backward connected counts
- this rule must compute candidate weights for all connected states and rescore the proposed states again for the backward proposal probability

A useful complexity summary is:
- baseline `HamiltonianRule`: two connectivity-count evaluations plus a uniform connected-entry draw
- neutral fast path (`occupations=None`, `balance_beta=0`, `aggregate_duplicate_entries=False`): same cost class as the baseline
- general proposal path: two full connected-support evaluations plus forward/backward rescoring over all connected candidates

If `B` is the number of chains, `M` is the typical number of connected entries, `D` is the full state size, and `F` is the fermion-block size, the extra proposal work is roughly of order `O(B M (D + F))` on top of the connectivity backend.

The `aggregate_duplicate_entries` switch adds another pass that compares candidate states against the selected target states when computing `log q`. This is an extra cost, but it is not the dominant one: the main cost increase comes from the need to build and rescore the full forward and backward proposal distributions.

So the relationship is:
- same support-set backend as `HamiltonianRule`
- same JAX execution style
- higher per-step cost because the proposal is genuinely more structured

## 8. Example

```python
import numpy as np
import netket as nk

from Hamiltonian.KondoHeisenbergChainSpinFermion import KondoHeisenbergChainSpinFermion

system = KondoHeisenbergChainSpinFermion(
    Lx=8,
    t=1.0,
    J_K=1.0,
    J1=1.0,
    J2=0.0,
    delta_z=1.0,
    pbc=False,
    n_fermions=8,
)

occ_dn = np.full(8, 0.45)
occ_up = np.full(8, 0.55)
occupations = np.concatenate([occ_dn, occ_up])

sampler = MetropolisHamiltonianWithProposal(
    system.joint_hilbert,
    hamiltonian=system.hamiltonian,
    occupations=occupations,
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    aggregate_duplicate_entries=False,
    n_chains_per_rank=16,
)
```

Because `netket/sampler/__init__.py` was intentionally left unchanged, this factory is not exported as `nk.sampler.MetropolisHamiltonianWithProposal`. Import it directly from `netket.sampler.metropolis` as shown above.

If you want the original Hamiltonian proposal back, use:

```python
sampler = MetropolisHamiltonianWithProposal(
    system.joint_hilbert,
    hamiltonian=system.hamiltonian,
    occupations=None,
    balance_beta=0.0,
    aggregate_duplicate_entries=False,
    n_chains_per_rank=16,
)
```

## 9. Repeated Entries And The Aggregation Switch

Repeated connected entries can matter in two different ways:
- they change the proposal shape if one thinks in terms of unique target states rather than connected entries;
- they may or may not need explicit correction depending on whether one wants to mimic the original `HamiltonianRule` entry-level treatment or compute a state-level proposal probability.

The new switch is:

```python
aggregate_duplicate_entries=False
```

Recommended interpretation:
- default `False`: closer to the original `HamiltonianRule` philosophy and cheaper
- optional `True`: safer if one wants the MH correction to use the fully aggregated state-level proposal `q(target | source)`

For standard Hermitian Hamiltonians, repeated-entry multiplicities are usually symmetric between forward and backward moves, so the default `False` is typically acceptable. The `True` branch is kept as an explicit, more conservative option.
