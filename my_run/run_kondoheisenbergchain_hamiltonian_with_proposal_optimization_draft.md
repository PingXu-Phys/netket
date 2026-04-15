# Hamiltonian With Proposal Optimization Draft

## 1. Purpose

This document records a concrete optimization draft for `HamiltonianRuleWithProposal`.

The current implementation is correct and robust, but its current cost model is conservative because it works from full target configurations returned by `get_conn_padded(...)`.

The goal of the next optimization stage is:
- keep the proposal formula unchanged;
- keep JAX support and code robustness;
- reduce per-candidate work from full-state scans to local-delta processing whenever possible.

## 1.1 Hard Constraints

The optimization work must obey two hard constraints.

1. Performance-first
   - reduce full-state scans;
   - reduce repeated post-processing after `get_conn_padded(...)`;
   - avoid work that scales with the full state length when the move is local;
   - avoid extra host/device synchronization.

2. JAX-first
   - prefer pure JAX tensor paths;
   - keep data structures `jit` / `vmap` friendly;
   - prefer fixed-shape arrays over Python objects or variable-length containers;
   - avoid Python loops, Python callbacks, and per-entry Python-side metadata.

Any interface that conflicts with these two constraints should be rejected, even if it looks architecturally cleaner on paper.

## 2. Current Situation

The present implementation uses full connected target states and therefore does the following for each candidate:
- compare the full fermion block to determine whether a fermion move happened;
- compare the full spin block to determine whether a spin move happened;
- scan the full fermion block to build the occupation bias;
- compare full target states again when duplicate connected entries must be aggregated.

A safe upper-bound view of the current move-dependent cost is:

```math
O(B M (D + F))
```

where:
- `B`: number of chains
- `M`: number of connected entries per chain
- `D`: full state length
- `F`: fermion-block length

This is a robust generic cost model, but not the fundamental lower bound of the proposal idea.

## 3. Why A Better Complexity Should Be Possible

In the Kondo-Heisenberg setting, one connected move usually changes only a small local subset of the configuration.

Typical examples:
- a fermion hop changes two fermion modes;
- a fermion spin flip changes one source mode and one target mode;
- a local spin move changes a small number of spin entries;
- a joint spin-fermion move changes one local fermion pattern plus one local spin.

Therefore, an ideal implementation should scale with the number of changed local degrees of freedom, not with the full state length.

A better target cost model is:

```math
O(B M (k_move + k_f))
```

where:
- `k_move`: typical number of changed local degrees of freedom in one move
- `k_f`: typical number of changed fermion modes in one move

In the intended use case, both are small.

## 4. What Information Is Missing Today

The current sampler only receives full target states from `get_conn_padded(...)`.

That means it does not directly know, for each connected entry:
- which indices changed;
- whether the move is `ss`, `ff`, or `sf` without scanning the state;
- which fermion modes went from `1->0` and `0->1`;
- which local spin sites changed;
- a local canonical signature for duplicate detection.

Any serious optimization should therefore focus on exposing this local move information.

## 5. Minimal Local Information Needed Per Connected Entry

A useful per-entry local descriptor can be described as:

```python
changed_count: int
changed_indices: int[k_max]
old_values: int[k_max]
new_values: int[k_max]
fermion_add_count: int
fermion_add_indices: int[f_max]
fermion_remove_count: int
fermion_remove_indices: int[f_max]
spin_change_count: int
spin_change_indices: int[s_max]
move_type: int   # 0=ss, 1=ff, 2=sf
duplicate_signature: int[p]
```

Not every field is required for every optimization stage.

In practice, the most valuable first fields are:
- `move_type`
- `fermion_add_indices`
- `fermion_remove_indices`
- `duplicate_signature`

## 6. Candidate Interface Designs

### 6.1 Option A: New operator method returning local deltas

Add a new operator-side method, conceptually:

```python
xp, mels, delta = operator.get_conn_padded_with_delta(x)
```

where `delta` is a pytree carrying local move descriptors for each connected entry.

Pros:
- best long-term structure;
- sampler becomes clean and fast;
- duplicate handling can use local signatures instead of full-state equality.

Cons:
- requires operator API extension;
- touches backend code in more than one place;
- larger implementation scope.

Assessment:
- best engineering direction;
- not the smallest first step.

### 6.2 Option B: Add an optional JAX-friendly metadata helper alongside `get_conn_padded(...)`

Keep `get_conn_padded(...)` unchanged, but optionally support an additional operator helper such as:

```python
delta = operator.get_conn_delta_metadata(x, xp, mels)
```

This helper should not return Python objects, Python lists, or variable-length structures. It should return a fixed-shape JAX-friendly pytree of integer arrays that can be batched, jitted, and vmapped.

A concrete draft is:

```python
delta = {
    "move_type": int32[B, M],                 # 0=invalid, 1=ss, 2=ff, 3=sf
    "changed_count": int32[B, M],
    "changed_indices": int32[B, M, K],
    "old_values": int8[B, M, K],
    "new_values": int8[B, M, K],
    "fermion_add_count": int32[B, M],
    "fermion_add_indices": int32[B, M, KF],
    "fermion_remove_count": int32[B, M],
    "fermion_remove_indices": int32[B, M, KF],
    "duplicate_signature": int32[B, M, P],
}
```

where:
- `B`: number of chains
- `M`: padded number of connected entries
- `K`: fixed upper bound on changed coordinates per move
- `KF`: fixed upper bound on changed fermion modes per move
- `P`: fixed signature width for duplicate handling

The key requirement is that `K`, `KF`, and `P` must be static and small. Padding is acceptable; Python-side variable-length metadata is not.

With this interface, the sampler can:
- classify `ss / ff / sf` directly from `move_type`;
- compute fermion proposal factors from `fermion_add_indices` and `fermion_remove_indices` only;
- perform duplicate aggregation or representative-entry selection from `duplicate_signature` without full-state equality checks.

Pros:
- preserves the existing main connectivity API;
- supports incremental adoption with `hasattr` fallback;
- can be made fully JAX-friendly;
- directly targets the real cost centers without redesigning the proposal.

Cons:
- still requires operator-side implementation work;
- requires careful choice of static bounds `K`, `KF`, and `P`;
- some duplication with internal connectivity logic may remain.

Assessment:
- best practical compromise;
- best next step if performance and JAX compatibility are the main priorities.

Additional design rule:
- if `get_conn_delta_metadata(...)` is unavailable, the sampler must fall back to the current generic full-state implementation;
- if it is available, all fast-path logic should stay inside pure JAX array code.

### 6.3 Option C: Sampler-only heuristic delta extraction from full states

Do not change operator APIs. Instead, let the sampler derive local deltas by scanning the difference between `x` and `xp` and compressing it.

Pros:
- smallest code-scope change;
- no operator API changes;
- can be added entirely on sampler side.

Cons:
- still starts from full-state comparisons;
- helps constants more than asymptotics;
- duplicate handling still needs full-state work before compression.

Assessment:
- useful as a transitional cleanup;
- not the real optimization target.

### 6.4 Option D: Kondo-specific specialized fast path

Add a Kondo-specific sampler-side optimization path that assumes the current tensor layout and infers local move structure directly from known operator semantics.

Pros:
- potentially strong speedup for the current project;
- limited scope;
- can be tuned to the exact mixed Hilbert layout used here.

Cons:
- less general;
- harder to maintain if the operator family changes;
- risks code duplication between generic and specialized paths.

Assessment:
- attractive for project-specific performance work;
- should come after a cleaner generic metadata design unless short-term speed is the only goal.

## 7. What Can Be Optimized First

The optimization work can be staged.

### 7.1 Stage 1: Local move classification

Current cost center:
- `_classify_move_types(...)` checks the whole fermion and spin blocks.

Optimization target:
- classify `ss / ff / sf` from local delta metadata only.

Expected effect:
- removes full-block comparisons for move classification.

### 7.2 Stage 2: Local fermion-bias computation

Current cost center:
- `_compute_candidate_log_bias(...)` scans the whole fermion block.

Optimization target:
- compute

```math
B_f(x\to x')
=
\prod_{m:0\to1}\tilde v_m
\prod_{m:1\to0}(1-\tilde v_m)
```

from changed fermion modes only.

Expected effect:
- moves the dominant proposal-scoring work from `F`-scale to `k_f`-scale.

### 7.3 Stage 3: Local duplicate handling

Current cost center:
- duplicate aggregation uses full-state equality tests.

Optimization target:
- define a local `duplicate_signature` so that repeated target states can be grouped or represented without full-state equality checks.

Expected effect:
- reduces overhead of `aggregate_duplicate_entries=True`;
- also makes `aggregate_duplicate_entries=False` cleaner because a representative entry can be selected by signature.

### 7.4 Stage 4: Backward-path reuse

Current cost center:
- backward proposal is recomputed from scratch on the proposed state.

Optimization target:
- reuse local move metadata or partial summaries where possible.

Expected effect:
- largest structural speedup if achieved safely;
- also the hardest stage.

## 8. Duplicate Handling Policy Draft

The current code already exposes:

```python
aggregate_duplicate_entries=False
```

Recommended policy for now:
- keep default `False`;
- only enable `True` when a conservative state-level proposal correction is explicitly desired;
- if local duplicate signatures become available, keep the same user-facing switch but make both branches cheaper internally.

This preserves interface stability.

## 9. Recommended Implementation Roadmap

Recommended order:

1. Add an operator metadata hook or optional delta helper.
2. Use it first for move classification.
3. Use it next for local fermion-bias computation.
4. Add local duplicate signatures.
5. Only after that, revisit backward-path reuse.

This order gives the best ratio of speedup to implementation risk.

## 10. Recommended Interface Draft

The most balanced proposal is Option B.

Suggested shape:

```python
def get_conn_delta_metadata(self, x, xp, mels):
    return {
        "move_type": ...,
        "fermion_add_indices": ...,
        "fermion_remove_indices": ...,
        "spin_change_indices": ...,
        "duplicate_signature": ...,
    }
```

Sampler behavior:
- if metadata is available, use local-delta scoring;
- otherwise fall back to the current generic implementation.

This keeps backward compatibility and allows gradual optimization.

## 11. Main Recommendation

The proposal formula itself does not need to change.

The best next step is to enrich the connectivity backend with optional local-delta metadata, then optimize the sampler using that metadata.

In short:
- do not redesign the proposal;
- redesign the information flow between operator and sampler.
