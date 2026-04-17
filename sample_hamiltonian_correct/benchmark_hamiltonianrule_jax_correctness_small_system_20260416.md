# HamiltonianRuleJax correctness benchmark (small system, 20260416)

## System

- `Lx = 2`
- `t = 1.0`
- `J_K = 1.0`
- `J1 = 1.0`
- `J2 = 0.0`
- `delta_z = 1.0`
- `n_fermions = 2`
- `fermion_size = 4`
- `spin_size = 2`
- `joint_size = 6`
- `n_basis_states = 24`

## Summary

- support mismatch states: `10`
- zero-state risk states: `6`
- max zero-state probability: `0.25`
- max TV distance to entry-uniform reference: `0.375`
- mean TV distance to entry-uniform reference: `0.18402777777777776`
- one-way code edges: `4`
- max abs correction residual: `0.0`
- max abs DB residual under uniform target: `0.010416666666666666`
- neutral withproposal path matches original direct path: `True`

## Support Mismatch Examples

| x | reported_n_conn | actual_nonzero_entries | tv_distance |
| --- | --- | --- | --- |
| [0, 0, 1, 1, -1, -1] | 4 | 5 | 0.19999999999999996 |
| [0, 0, 1, 1, 1, -1] | 4 | 5 | 0.19999999999999996 |
| [0, 1, 1, 0, -1, -1] | 4 | 6 | 0.16666666666666669 |
| [0, 1, 1, 0, -1, 1] | 4 | 8 | 0.375 |
| [0, 1, 1, 0, 1, 1] | 4 | 6 | 0.16666666666666669 |
| [1, 0, 0, 1, -1, -1] | 4 | 6 | 0.16666666666666669 |
| [1, 0, 0, 1, 1, -1] | 4 | 8 | 0.375 |
| [1, 0, 0, 1, 1, 1] | 4 | 6 | 0.16666666666666669 |

## Zero-State Risk Examples

| x | reported_n_conn | actual_nonzero_entries | zero_state_probability |
| --- | --- | --- | --- |
| [0, 0, 1, 1, 1, 1] | 4 | 3 | 0.25 |
| [0, 1, 0, 1, -1, -1] | 4 | 3 | 0.25 |
| [0, 1, 0, 1, 1, 1] | 4 | 3 | 0.25 |
| [1, 0, 1, 0, -1, -1] | 4 | 3 | 0.25 |
| [1, 0, 1, 0, 1, 1] | 4 | 3 | 0.25 |
| [1, 1, 0, 0, -1, -1] | 4 | 3 | 0.25 |

## Counterexample A: one-way edge

- `x = [0, 0, 1, 1, -1, 1]`
- `y = [1, 0, 0, 1, 1, 1]`
- `q_code(y|x) = 0.25`
- `q_code(x|y) = 0.0`
- `log_prob_corr_code(x->y) = 0.0`

### x nonzero entries

| entry | target_state | mel |
| --- | --- | --- |
| 4 | [0, 0, 1, 1, -1, 1] | -0.25 |
| 5 | [0, 0, 1, 1, 1, -1] | 0.5 |
| 6 | [0, 0, 1, 1, -1, 1] | -0.25 |
| 7 | [1, 0, 0, 1, 1, 1] | 0.5 |
| 9 | [0, 0, 1, 1, -1, 1] | 0.25 |

### y nonzero entries

| entry | target_state | mel |
| --- | --- | --- |
| 0 | [1, 0, 1, 0, 1, 1] | -1.0 |
| 1 | [0, 1, 0, 1, 1, 1] | -1.0 |
| 4 | [1, 0, 0, 1, 1, 1] | 0.25 |
| 6 | [1, 0, 0, 1, 1, 1] | -0.25 |
| 8 | [0, 0, 1, 1, -1, 1] | 0.5 |
| 9 | [1, 0, 0, 1, 1, 1] | 0.25 |

## Counterexample B: invalid zero state risk

- `x0 = [0, 0, 1, 1, 1, 1]`
- `reported_n_conn = 4`
- `actual_nonzero_entries = 3`
- `p_zero = 0.25`

| entry | target_state | mel |
| --- | --- | --- |
| 4 | [0, 0, 1, 1, 1, 1] | 0.25 |
| 6 | [0, 0, 1, 1, 1, 1] | 0.25 |
| 9 | [0, 0, 1, 1, 1, 1] | 0.25 |

## Stationary Distribution Check

- computed: `False`
- note: Skipped because the code kernel leaks probability mass to an invalid zero state.

## Neutral Path Consistency

- checked steps: `8`
- all steps match: `True`
