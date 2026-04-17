# Tier 0.5 Implementation Notes 2026-04-16

## Scope

This note records the first implementation step of the sampler-side local descriptor path.

The implementation keeps the public sampler API unchanged and focuses on the internal proposal pipeline.

## New Internal Interface

Added a new internal helper module:
- `netket/sampler/rules/_hamiltonian_with_proposal_local_descriptor.py`

Main internal interfaces added there:
- `LocalMoveDescriptor`
- `extract_local_descriptor_from_dense_targets(...)`
- `build_transition_signature(...)`
- `compute_move_masks_from_descriptor(...)`
- `compute_log_bias_from_descriptor(...)`
- `log_state_probability_from_signature(...)`
- `log_representative_probability_from_signature(...)`

## Existing File Updated

Updated:
- `netket/sampler/rules/hamiltonian_with_proposal.py`

Main changes:
- the weighted proposal path now builds a fixed-shape local descriptor from `x` and `xp`;
- move classification no longer scans dense target states directly in several separate helpers;
- fermion proposal bias is now computed from `fermion_add_indices` and `fermion_remove_indices`;
- duplicate handling now uses `duplicate_signature` instead of full-state equality checks;
- backward correction now compares target signatures instead of full target states.

## Tests Updated

Updated:
- `test/sampler/test_hamiltonian_with_proposal.py`

Added tests for:
- descriptor extraction on handcrafted Kondo-like moves;
- signature-based duplicate aggregation;
- one integration check that the descriptor-driven weighted path runs end-to-end.

## Files Changed In This Step

Added:
- `netket/sampler/rules/_hamiltonian_with_proposal_local_descriptor.py`
- `my_run/run_kondoheisenbergchain_hamiltonian_with_proposal_tier05_implementation_notes_20260416.md`

Modified:
- `netket/sampler/rules/hamiltonian_with_proposal.py`
- `test/sampler/test_hamiltonian_with_proposal.py`

## What Was Deliberately Not Changed

Not changed:
- `MetropolisHamiltonianWithProposal(...)` public interface;
- operator public interfaces;
- Kondo Hamiltonian construction files;
- duplicate policy semantics of `aggregate_duplicate_entries`.

## Current Position Relative To The Draft

This implementation corresponds to Tier 0.5 / Tier 1-sampler:
- dense `xp` is still produced by the operator;
- the sampler immediately compresses it into a local descriptor;
- proposal logic is descriptor-driven after that point.
