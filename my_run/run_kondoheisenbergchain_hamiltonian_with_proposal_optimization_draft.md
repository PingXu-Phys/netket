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


## 12. Stronger Decision Criteria

The next-stage optimization should be evaluated against a stricter set of criteria than just asymptotic notation.

A useful decision checklist is:
- does it reduce full-state scans or only improve constants;
- does it preserve pure JAX array execution;
- does it keep fixed-shape metadata and avoid Python-side objects;
- does it preserve backward compatibility with existing operators;
- does it keep a robust generic fallback path;
- does it reduce both forward scoring cost and duplicate-handling cost;
- how hard is it to verify detailed balance after the change.

A design that looks elegant but weakens JAX-friendliness or makes correctness hard to audit should be ranked lower.

## 13. Refined Optimization Tiers

The existing options can be reorganized into a more practical tiered roadmap.

### 13.1 Tier 0: Sampler-only tensor cleanup on top of the current API

Scope:
- keep `get_conn_padded(...)` as the only operator interface;
- keep the proposal formula unchanged;
- reduce redundant scans and intermediate tensors inside the sampler.

Concrete ideas:
- fuse move classification and fermion-difference extraction into one tensor pass;
- avoid recomputing boolean masks that can be derived from one shared delta tensor;
- cheapen the `aggregate_duplicate_entries=False` branch by selecting a representative matching entry with less post-processing;
- hoist layout constants and avoid repeated slice construction.

Expected benefit:
- better constants;
- lower memory traffic;
- no asymptotic change in the generic case.

Complexity class:

```math
O(B M (D + F))
```

Pros:
- smallest code scope;
- no operator changes;
- easiest to land and benchmark quickly.

Cons:
- does not attack the main asymptotic bottleneck;
- duplicate handling remains fundamentally full-state based;
- likely insufficient if `M` and `D` both grow.

Assessment:
- good as immediate cleanup;
- not sufficient as the main long-term answer.

### 13.2 Tier 1: Optional minimal metadata helper

Scope:
- keep `get_conn_padded(...)` unchanged;
- add an optional helper such as `get_conn_delta_metadata(x, xp, mels)`;
- let the sampler branch on metadata availability.

Recommended minimal payload:
- `move_type`;
- `fermion_add_indices` and count;
- `fermion_remove_indices` and count;
- optional `duplicate_signature`.

Expected benefit:
- move classification drops from full-block scanning to local metadata lookup;
- fermion-bias evaluation drops from `F`-scale work to `k_f`-scale work;
- duplicate handling can become much cheaper once a signature is available.

Target complexity view:

```math
O(B M (k_move + k_f))
```

for the proposal-dependent part, while keeping fallback compatibility.

Pros:
- best balance between speedup and implementation scope;
- keeps backward compatibility;
- can be adopted incrementally by operator families;
- stays compatible with pure JAX array code.

Cons:
- some operator-side duplication may remain because connectivity and metadata are computed in separate steps;
- requires careful design of fixed bounds and padding.

Assessment:
- strongest practical next step;
- should be the default recommendation.

### 13.3 Tier 2: Native connectivity-plus-delta operator API

Scope:
- introduce an operator method conceptually like

```python
xp, mels, delta = operator.get_conn_padded_with_delta(x)
```

- compute target states and local descriptors together inside the operator backend.

Expected benefit:
- best constant factors and the cleanest information flow;
- avoids rebuilding local move descriptors from already materialized targets;
- most natural place to generate duplicate signatures.

Pros:
- best long-term architecture;
- avoids split logic between connectivity and metadata;
- likely the cleanest route if several operators need the same optimization.

Cons:
- larger API change;
- touches more backend code;
- higher review and migration cost.

Assessment:
- strongest long-term design;
- should follow Tier 1 unless several operators already need a common delta API.

### 13.4 Tier 3: Capability-flagged Kondo-specific fast path

Scope:
- add a project-specific optimized branch for the current Kondo-Heisenberg operator family;
- exploit the known tensor layout and small local move patterns directly.

Expected benefit:
- potentially very strong speedup on the current workload;
- can bypass generic machinery for the most common local moves.

Pros:
- highest short-term upside for this project;
- can be tuned to exact move semantics and fixed small local supports.

Cons:
- less general;
- higher maintenance burden;
- risks diverging from the generic sampler path;
- correctness arguments become more operator-specific.

Assessment:
- attractive only if benchmark pressure is high and generic metadata work is too slow to land.

### 13.5 Tier 4: Backward-path reuse or partial caching

Scope:
- try to reuse information from the forward selected move when evaluating the reverse proposal.

Why it is hard:
- the reverse normalization depends on the full connected neighborhood of the proposed state;
- forward local information is not generally enough to recover the reverse grouped normalization exactly;
- duplicate aggregation complicates reuse further.

Pros:
- could produce the largest structural speedup if done safely.

Cons:
- highest correctness risk;
- easiest place to break detailed balance subtly;
- likely not worth attempting before local metadata exists.

Assessment:
- explicitly lower priority than Tiers 0-3.

## 14. Comparative Assessment Table

| Tier | Main idea | Expected gain | Scope | JAX fit | Generality | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier 0 | Sampler-only cleanup | Low to medium constant-factor gain | Small | Excellent | High | Low | Do now as cleanup |
| Tier 1 | Optional metadata helper | Medium to high | Medium | Excellent if fixed-shape | High | Medium | Primary next step |
| Tier 2 | Native delta API | High | Large | Excellent | Very high | Medium | Long-term target |
| Tier 3 | Kondo-specific fast path | Medium to very high on this project | Medium | Good | Low | Medium to high | Only if benchmarks demand it |
| Tier 4 | Backward reuse / caching | Potentially high | Large | Unclear | Medium | High | Defer |

This table makes the main point explicit: Tier 1 gives the best ratio of speedup, scope control, and compatibility.

## 15. Minimal Metadata Contract

A useful design principle is to separate core metadata from optional metadata.

### 15.1 Core metadata

Core fields should be enough to speed up the proposal-dependent scoring path:
- `move_type[B, M]`;
- `fermion_add_count[B, M]`;
- `fermion_add_indices[B, M, KF]`;
- `fermion_remove_count[B, M]`;
- `fermion_remove_indices[B, M, KF]`.

With just these fields, the sampler can already:
- classify `ss / ff / sf` without scanning full blocks;
- compute the fermion proposal bias from changed modes only.

### 15.2 Optional metadata

Optional fields are mainly for duplicate handling and future extensions:
- `spin_change_count[B, M]`;
- `spin_change_indices[B, M, KS]`;
- `duplicate_signature[B, M, P]`.

This separation matters because it allows a cheap first rollout:
- first land core metadata and harvest the largest speedups;
- only then extend to duplicate signatures if needed.

## 16. Recommended Two-Track Roadmap

A practical roadmap should separate immediate cleanup from structural optimization.

### 16.1 Track A: Immediate low-risk work

1. Land Tier 0 cleanup.
2. Add focused microbenchmarks for:
   - forward proposal scoring;
   - backward proposal scoring;
   - duplicate handling with `aggregate_duplicate_entries=True/False`.
3. Measure how much time is currently spent in:
   - move classification;
   - fermion-bias construction;
   - duplicate matching;
   - reverse recomputation.

Goal:
- establish where the real wall-clock bottleneck is before touching operator APIs.

### 16.2 Track B: Main structural optimization

1. Add Tier 1 metadata helper behind capability detection.
2. Use it first for local move classification.
3. Use it next for local fermion-bias evaluation.
4. Add duplicate signatures only if duplicate handling remains measurable.
5. Reconsider Tier 2 native API only after Tier 1 adoption experience.

Goal:
- get most of the asymptotic and constant-factor benefit without forcing an immediate API redesign.

## 17. What Should Not Be Changed Yet

The following changes should remain explicitly out of scope for the next stage:
- do not redesign the proposal formula;
- do not make `aggregate_duplicate_entries=True` the default;
- do not introduce Python-side variable-length metadata containers;
- do not optimize backward reuse first;
- do not let a specialized fast path replace the generic fallback path.

These exclusions keep the work focused on the actual bottleneck and protect correctness.

## 18. Benchmark And Acceptance Criteria

The optimization work should be considered successful only if it is tied to clear benchmarks.

Recommended benchmark axes:
- `B`: number of chains;
- `M`: typical connected-entry count;
- system size `L`;
- neutral path versus occupation-biased path;
- `aggregate_duplicate_entries=False/True`.

Recommended measured quantities:
- compile time;
- per-step runtime after compilation;
- relative time spent in forward versus backward proposal work;
- memory pressure / peak tensor size;
- acceptance rate stability.

Suggested acceptance criteria:
- Tier 0 should reduce runtime without changing outputs or code complexity substantially;
- Tier 1 should show a clear runtime win on occupation-biased workloads;
- any metadata path must preserve exact agreement with the generic path in sampled proposal probabilities and MH corrections.

## 19. Updated Main Recommendation

The proposal mathematics is already in good shape.

The strongest next move is:
- do a small Tier 0 cleanup immediately;
- then implement Tier 1 optional local-delta metadata;
- postpone native API redesign and backward reuse until benchmark data justifies them.

In short:
- keep the proposal;
- improve the locality of the information supplied to the sampler;
- optimize in the order of lowest correctness risk first.

## 20. Current Workflow Reference

This section records the actual current execution flow. It should be used as a reference when deciding where an optimization belongs.

### 20.1 Operator-side workflow today

For the current Kondo-Heisenberg use case, the operator stack is not a single custom class. It is a composition of:
- primitive fermion or spin operators;
- `EmbedOperator` for subspace lifting;
- `ProductOperator` for joint terms;
- `SumOperator` for the full Hamiltonian.

A useful mental model is:

1. Primitive term application is already local.
   For example, the JAX fermion backend can:
   - pack occupation bits;
   - flip only the acted-on local bits;
   - compute the Jordan-Wigner sign from local masks.

2. Primitive term output is then materialized as a full target configuration.
   Even if the move only changes a few local degrees of freedom, `get_conn_padded(...)` still returns a full target-state tensor `xp`.

3. Composition layers lift local changes back to the full joint Hilbert space.
   `EmbedOperator`, `ProductOperator`, and `SumOperator` all operate in terms of full target states at their public interface.

Therefore the current operator stack already contains local work internally, but its public output is still dense full-state data.

### 20.2 Sampler-side workflow today

The current `HamiltonianRuleWithProposal` weighted path does the following:

1. Call `operator.get_conn_padded(x)`.
2. Receive full connected target states `xp` and matrix elements `mels`.
3. In the sampler, re-scan `xp` against `x` to classify `ss / ff / sf`.
4. Re-scan the fermion block to construct the proposal bias.
5. Sample one entry from the resulting proposal distribution.
6. Recompute `operator.get_conn_padded(x_proposed)`.
7. Repeat the same proposal analysis on the backward neighborhood.
8. Apply duplicate handling and return `log_q_bwd - log_q_fwd`.

This means the weighted proposal currently pays two costs:
- dense target-state materialization inside the operator API;
- dense target-state post-processing inside the sampler.

### 20.3 Where the current bottleneck really is

The key distinction is:
- the primitive operator kernels are not fully naive; several of them already exploit local structure internally;
- the current public operator API still exposes dense full target states only;
- the sampler therefore has to rediscover local information by comparing full states.

This implies that future optimization should focus on the interface between operator and sampler, not only on low-level primitive kernels.

## 21. Data-Flow Map By Tier

### 21.1 Current data-flow map

```text
current state x
    |
    v
operator.get_conn_padded(x)
    |
    |-- primitive fermion/spin term kernels
    |     - local bit flip / local sign computation
    |     - local legality check
    |
    |-- composition layers
    |     - EmbedOperator lifts subspace result to full joint state
    |     - ProductOperator combines full-state branches
    |     - SumOperator concatenates full-state branches
    |
    v
full connected targets xp, mels
    |
    v
sampler-side weighted proposal logic
    |-- classify move type from full xp vs x
    |-- compute fermion bias from full fermion block scan
    |-- sample selected entry
    |-- duplicate handling on full targets if needed
    |
    v
proposed state x'
    |
    v
operator.get_conn_padded(x')
    |
    v
sampler-side backward proposal recomputation
    |-- same full-state classification and scoring again
    |-- final log correction
```

### 21.2 What is already local and what is still dense

Already local or partially local:
- primitive fermion term application;
- primitive legality checks;
- primitive Jordan-Wigner sign evaluation.

Still dense at the current interface boundary:
- `xp` returned by `get_conn_padded(...)`;
- full-state lifting in `EmbedOperator`;
- full-state branch composition in `ProductOperator` and `SumOperator`;
- sampler-side move classification;
- sampler-side fermion-bias evaluation;
- duplicate matching by full-state equality.

### 21.3 Tier-by-tier interpretation of the data flow

Tier 0:
- do not change the operator boundary;
- accept `xp, mels` as dense outputs;
- only reduce redundant sampler-side rescans and tensor traffic.

Tier 1:
- keep `get_conn_padded(...)` unchanged;
- add optional local metadata alongside the dense output;
- let the sampler stop rediscovering local deltas from full-state comparisons.

Tier 2:
- upgrade the operator boundary itself;
- return dense targets and local deltas together from one native operator call;
- reduce split logic between connectivity generation and metadata reconstruction.

Tier 3:
- for the Kondo family, consider a capability-flagged specialized path;
- exploit known joint-layout structure and move semantics directly;
- use this only if the benchmark value is strong enough to justify reduced generality.

Tier 4:
- try to reduce or partially reuse the backward recomputation;
- this is only meaningful after the forward path has access to richer local summaries.

### 21.4 Modification guidance derived from the map

The map above suggests the following practical rule.

If the goal is only to reduce engineering risk and code churn:
- start with Tier 0 sampler-only cleanup.

If the goal is to reduce the main proposal-dependent cost class:
- eventually local information must cross the operator-to-sampler boundary.

In other words:
- sampler-only work can improve constants;
- crossing the boundary with explicit local metadata is the main route to a real structural speedup.

## 22. Style Requirements For This Optimization Work

The optimization work should obey the following coding-style requirements.

1. Keep changes as small and additive as possible.
   - Prefer adding a new file or a thin helper over editing multiple existing library files.
   - Modify existing core files only when necessary to wire a capability into the current architecture.
   - Avoid broad refactors that are not directly tied to the measured bottleneck.

2. Stay fully JAX-compatible and efficient.
   - Use fixed-shape arrays and pytrees only.
   - Keep hot paths `jit` / `vmap` friendly.
   - Avoid Python objects, Python callbacks, variable-length metadata containers, or host-side branching in the critical path.

3. Match the style of the existing library.
   - Keep functions short, precise, and minimally layered.
   - Reuse existing helpers and data structures whenever possible.
   - Do not add verbose abstractions, verbose comments, or redundant wrapper logic.

### 22.1 Consequences of these style rules for the tier choice

These rules have immediate design consequences.

- Tier 0 is strongly aligned with the style rules because it mostly touches one sampler file and keeps the current operator contract unchanged.
- Tier 1 is the best structural next step only if it is implemented as a narrow optional capability, not as a broad operator refactor.
- Tier 2 should be delayed until there is clear benchmark evidence that a larger API change is justified.
- Tier 3 should be gated carefully because project-specific fast paths can violate the “small and additive” rule if introduced too early.
- Tier 4 should remain last because it has the highest correctness risk and the weakest alignment with minimal-change discipline.

## 23. Updated Practical Decision Rule

Use the following decision rule during implementation.

1. First ask whether the bottleneck is caused by repeated sampler-side rescans of dense `xp`.
   If yes, try Tier 0 first.

2. Then ask whether the remaining bottleneck is caused by the sampler not knowing local deltas.
   If yes, Tier 1 becomes the main target.

3. Only after that ask whether the current optional-helper design is too awkward or duplicative.
   If yes, then consider Tier 2.

4. Only consider Tier 3 when Kondo-specific benchmarks clearly justify reduced generality.

5. Only consider Tier 4 after the forward path already has a robust local-summary representation.
