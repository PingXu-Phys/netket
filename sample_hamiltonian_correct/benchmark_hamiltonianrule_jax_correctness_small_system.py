from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / 'Hamiltonian'):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from Hamiltonian.KondoHeisenbergChainSpinFermion import KondoHeisenbergChainSpinFermion
from netket.sampler.rules.hamiltonian import HamiltonianRule
from netket.sampler.rules.hamiltonian_with_proposal import HamiltonianRuleWithProposal

DATE_TAG = date.today().strftime('%Y%m%d')
REPORT_BASENAME = f'benchmark_hamiltonianrule_jax_correctness_small_system_{DATE_TAG}'
ZERO_LABEL = '__invalid_zero_state__'
TOL = 1.0e-12


@dataclass(frozen=True)
class SmallSystem:
    hamiltonian: object
    joint_hilbert: object
    fermion_size: int
    spin_size: int
    parameters: dict[str, float | int]


def build_small_system() -> SmallSystem:
    system = KondoHeisenbergChainSpinFermion(
        Lx=2,
        t=1.0,
        J_K=1.0,
        J1=1.0,
        J2=0.0,
        delta_z=1.0,
        n_fermions=2,
    )
    return SmallSystem(
        hamiltonian=system.hamiltonian,
        joint_hilbert=system.joint_hilbert,
        fermion_size=int(system.fermion_hilbert.size),
        spin_size=int(system.local_spin_hilbert.size),
        parameters={
            'Lx': 2,
            't': 1.0,
            'J_K': 1.0,
            'J1': 1.0,
            'J2': 0.0,
            'delta_z': 1.0,
            'n_fermions': 2,
        },
    )


def state_key(state: np.ndarray | jax.Array | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(state, tuple):
        return tuple(int(v) for v in state)
    return tuple(int(v) for v in np.asarray(state).tolist())


def state_list(key: tuple[int, ...]) -> list[int]:
    return [int(v) for v in key]


def nonzero_positions(mels_row: jax.Array) -> list[int]:
    return [int(i) for i, flag in enumerate((jnp.abs(mels_row) > 0).tolist()) if flag]


def serialize_distribution(prob_map: dict[tuple[int, ...], float]) -> list[dict[str, object]]:
    return [
        {'target_state': state_list(target), 'probability': float(prob)}
        for target, prob in sorted(prob_map.items())
    ]


def analyze_state(operator, x_key: tuple[int, ...]) -> dict[str, object]:
    x = jnp.asarray([x_key], dtype=jnp.int8)
    xp, mels = operator.get_conn_padded(x)
    positions = nonzero_positions(mels[0])
    reported_n_conn = int(operator.n_conn(x)[0])
    actual_nonzero_entries = len(positions)

    all_nonzero_entries = []
    reference_state_probs: dict[tuple[int, ...], float] = {}
    for pos in positions:
        target = state_key(xp[0, pos])
        mel = complex(np.asarray(mels[0, pos]).item())
        mel_value: object
        if abs(mel.imag) < TOL:
            mel_value = float(mel.real)
        else:
            mel_value = {'real': float(mel.real), 'imag': float(mel.imag)}
        all_nonzero_entries.append(
            {
                'entry_index': pos,
                'target_state': state_list(target),
                'mel': mel_value,
            }
        )
        reference_state_probs[target] = reference_state_probs.get(target, 0.0) + (
            1.0 / actual_nonzero_entries
        )

    code_state_probs: dict[tuple[int, ...], float] = {}
    direct_path_entries = []
    if reported_n_conn > 0:
        prefix_positions = positions[: min(reported_n_conn, actual_nonzero_entries)]
        entry_probability = 1.0 / reported_n_conn
        for pos in prefix_positions:
            target = state_key(xp[0, pos])
            code_state_probs[target] = code_state_probs.get(target, 0.0) + entry_probability
            direct_path_entries.append(
                {
                    'entry_index': pos,
                    'target_state': state_list(target),
                    'entry_probability': entry_probability,
                }
            )
        zero_state_probability = (
            max(reported_n_conn - actual_nonzero_entries, 0) / reported_n_conn
        )
    else:
        zero_state_probability = 0.0

    all_targets = set(reference_state_probs) | set(code_state_probs)
    tv_distance_to_reference = 0.5 * sum(
        abs(code_state_probs.get(target, 0.0) - reference_state_probs.get(target, 0.0))
        for target in all_targets
    ) + 0.5 * zero_state_probability

    return {
        'state': state_list(x_key),
        'reported_n_conn': reported_n_conn,
        'actual_nonzero_entry_count': actual_nonzero_entries,
        'all_nonzero_entries': all_nonzero_entries,
        'direct_path_entries': direct_path_entries,
        'reference_state_distribution': serialize_distribution(reference_state_probs),
        'code_state_distribution': serialize_distribution(code_state_probs),
        'reference_state_probs_map': reference_state_probs,
        'code_state_probs_map': code_state_probs,
        'reference_support': [state_list(target) for target in sorted(reference_state_probs)],
        'code_support': [state_list(target) for target in sorted(code_state_probs)],
        'support_mismatch': set(reference_state_probs) != set(code_state_probs),
        'zero_state_probability': float(zero_state_probability),
        'tv_distance_to_reference': float(tv_distance_to_reference),
    }


def run_neutral_path_consistency_check(system: SmallSystem) -> dict[str, object]:
    states = system.joint_hilbert.all_states().astype(jnp.int8)
    reference_rule = HamiltonianRule(operator=system.hamiltonian)
    proposal_rule = HamiltonianRuleWithProposal(
        operator=system.hamiltonian,
        occupations=None,
        balance_beta=0.0,
        aggregate_duplicate_entries=False,
    )

    reference_states = states
    proposal_states = states
    first_mismatch = None
    keys = jax.random.split(jax.random.PRNGKey(0), 8)

    for step, key in enumerate(keys.tolist()):
        key = jnp.asarray(key, dtype=jnp.uint32)
        ref_next, ref_corr = reference_rule.transition(
            None, None, None, None, key, reference_states
        )
        prop_next, prop_corr = proposal_rule.transition(
            None, None, None, None, key, proposal_states
        )
        if (
            not np.array_equal(np.asarray(ref_next), np.asarray(prop_next))
            or not np.allclose(np.asarray(ref_corr), np.asarray(prop_corr))
        ):
            first_mismatch = {
                'step': int(step),
                'reference_state': np.asarray(ref_next).tolist(),
                'proposal_state': np.asarray(prop_next).tolist(),
                'reference_corr': np.asarray(ref_corr).tolist(),
                'proposal_corr': np.asarray(prop_corr).tolist(),
            }
            break
        reference_states = ref_next
        proposal_states = prop_next

    return {
        'checked_steps': len(keys),
        'all_steps_match': first_mismatch is None,
        'first_mismatch': first_mismatch,
    }


def build_summary(state_info_map: dict[tuple[int, ...], dict[str, object]]) -> dict[str, object]:
    state_keys = sorted(state_info_map)
    n_states = len(state_keys)
    uniform_pi = 1.0 / n_states

    support_mismatch_examples = []
    zero_state_examples = []
    one_way_edges = []
    delta_corr_examples = []
    max_abs_delta_corr = 0.0
    max_abs_db_residual = 0.0
    sum_abs_db_residual = 0.0
    max_db_example = None

    for x_key in state_keys:
        info_x = state_info_map[x_key]
        if info_x['support_mismatch'] and len(support_mismatch_examples) < 8:
            support_mismatch_examples.append(
                {
                    'x': info_x['state'],
                    'reported_n_conn': info_x['reported_n_conn'],
                    'actual_nonzero_entry_count': info_x['actual_nonzero_entry_count'],
                    'reference_support': info_x['reference_support'],
                    'code_support': info_x['code_support'],
                    'tv_distance_to_reference': info_x['tv_distance_to_reference'],
                }
            )
        if info_x['zero_state_probability'] > 0.0 and len(zero_state_examples) < 8:
            zero_state_examples.append(
                {
                    'x': info_x['state'],
                    'reported_n_conn': info_x['reported_n_conn'],
                    'actual_nonzero_entry_count': info_x['actual_nonzero_entry_count'],
                    'zero_state_probability': info_x['zero_state_probability'],
                }
            )

    for x_key in state_keys:
        info_x = state_info_map[x_key]
        c_x = info_x['reported_n_conn']
        code_probs_x = info_x['code_state_probs_map']
        for y_key, q_xy in code_probs_x.items():
            info_y = state_info_map.get(y_key)
            if info_y is None:
                continue
            c_y = info_y['reported_n_conn']
            q_yx = info_y['code_state_probs_map'].get(x_key, 0.0)
            assumed_log_ratio = math.log(c_x) - math.log(c_y)
            if q_yx <= 0.0:
                if len(one_way_edges) < 12:
                    one_way_edges.append(
                        {
                            'x': state_list(x_key),
                            'y': state_list(y_key),
                            'q_xy': float(q_xy),
                            'q_yx': 0.0,
                            'reported_n_conn_x': c_x,
                            'reported_n_conn_y': c_y,
                            'assumed_log_ratio': float(assumed_log_ratio),
                        }
                    )
                accept_xy = min(1.0, math.exp(assumed_log_ratio))
                db_residual = uniform_pi * q_xy * accept_xy
            else:
                actual_log_ratio = math.log(q_yx) - math.log(q_xy)
                delta_corr = actual_log_ratio - assumed_log_ratio
                if abs(delta_corr) > max_abs_delta_corr:
                    max_abs_delta_corr = abs(delta_corr)
                if abs(delta_corr) > TOL and len(delta_corr_examples) < 12:
                    delta_corr_examples.append(
                        {
                            'x': state_list(x_key),
                            'y': state_list(y_key),
                            'q_xy': float(q_xy),
                            'q_yx': float(q_yx),
                            'assumed_log_ratio': float(assumed_log_ratio),
                            'actual_log_ratio': float(actual_log_ratio),
                            'delta_corr': float(delta_corr),
                        }
                    )
                accept_xy = min(1.0, math.exp(assumed_log_ratio))
                accept_yx = min(1.0, math.exp(-assumed_log_ratio))
                db_residual = uniform_pi * q_xy * accept_xy - uniform_pi * q_yx * accept_yx
            abs_residual = abs(db_residual)
            sum_abs_db_residual += abs_residual
            if abs_residual > max_abs_db_residual:
                max_abs_db_residual = abs_residual
                max_db_example = {
                    'x': state_list(x_key),
                    'y': state_list(y_key),
                    'db_residual': float(db_residual),
                    'q_xy': float(q_xy),
                    'q_yx': float(q_yx),
                    'reported_n_conn_x': c_x,
                    'reported_n_conn_y': c_y,
                }

    n_support_mismatch_states = sum(
        1 for info in state_info_map.values() if info['support_mismatch']
    )
    n_zero_state_risk_states = sum(
        1 for info in state_info_map.values() if info['zero_state_probability'] > 0.0
    )
    max_zero_state_probability = max(
        (info['zero_state_probability'] for info in state_info_map.values()), default=0.0
    )
    max_tv_distance = max(
        (info['tv_distance_to_reference'] for info in state_info_map.values()), default=0.0
    )
    mean_tv_distance = float(
        np.mean([info['tv_distance_to_reference'] for info in state_info_map.values()])
    )

    if n_zero_state_risk_states > 0:
        stationary_distribution = {
            'computed': False,
            'reason': 'Skipped because the code kernel leaks probability mass to an invalid zero state.',
        }
    else:
        stationary_distribution = {
            'computed': False,
            'reason': 'Not implemented in this minimal script because zero-state leakage is the first blocking correctness failure to resolve.',
        }

    return {
        'n_states': n_states,
        'n_support_mismatch_states': n_support_mismatch_states,
        'n_zero_state_risk_states': n_zero_state_risk_states,
        'max_zero_state_probability': float(max_zero_state_probability),
        'max_tv_distance_to_reference': float(max_tv_distance),
        'mean_tv_distance_to_reference': mean_tv_distance,
        'n_one_way_edges': len(one_way_edges),
        'max_abs_delta_corr': float(max_abs_delta_corr),
        'n_delta_corr_nonzero_examples': len(delta_corr_examples),
        'max_abs_db_residual_uniform_pi': float(max_abs_db_residual),
        'sum_abs_db_residual_uniform_pi': float(sum_abs_db_residual),
        'support_mismatch_examples': support_mismatch_examples,
        'zero_state_examples': zero_state_examples,
        'one_way_edge_examples': one_way_edges,
        'delta_corr_examples': delta_corr_examples,
        'max_db_residual_example': max_db_example,
        'stationary_distribution': stationary_distribution,
    }


def build_counterexamples(state_info_map: dict[tuple[int, ...], dict[str, object]]) -> dict[str, object]:
    x_a = (0, 0, 1, 1, -1, 1)
    y_a = (1, 0, 0, 1, 1, 1)
    x_b = (0, 0, 1, 1, 1, 1)

    info_x_a = state_info_map[x_a]
    info_y_a = state_info_map[y_a]
    info_x_b = state_info_map[x_b]
    q_xy = info_x_a['code_state_probs_map'].get(y_a, 0.0)
    q_yx = info_y_a['code_state_probs_map'].get(x_a, 0.0)
    c_x = info_x_a['reported_n_conn']
    c_y = info_y_a['reported_n_conn']

    return {
        'missing_reverse_edge': {
            'x': info_x_a['state'],
            'y': info_y_a['state'],
            'x_analysis': {
                'reported_n_conn': c_x,
                'actual_nonzero_entry_count': info_x_a['actual_nonzero_entry_count'],
                'all_nonzero_entries': info_x_a['all_nonzero_entries'],
                'direct_path_entries': info_x_a['direct_path_entries'],
                'code_state_distribution': info_x_a['code_state_distribution'],
            },
            'y_analysis': {
                'reported_n_conn': c_y,
                'actual_nonzero_entry_count': info_y_a['actual_nonzero_entry_count'],
                'all_nonzero_entries': info_y_a['all_nonzero_entries'],
                'direct_path_entries': info_y_a['direct_path_entries'],
                'code_state_distribution': info_y_a['code_state_distribution'],
            },
            'pair_summary': {
                'q_code_y_given_x': float(q_xy),
                'q_code_x_given_y': float(q_yx),
                'code_log_prob_corr_x_to_y': float(math.log(c_x) - math.log(c_y)),
                'reverse_move_missing_in_code_path': q_yx == 0.0,
            },
        },
        'invalid_zero_state': {
            'x': info_x_b['state'],
            'reported_n_conn': info_x_b['reported_n_conn'],
            'actual_nonzero_entry_count': info_x_b['actual_nonzero_entry_count'],
            'all_nonzero_entries': info_x_b['all_nonzero_entries'],
            'zero_state_probability': info_x_b['zero_state_probability'],
        },
    }


def markdown_table(entries: list[dict[str, object]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    lines.append('| ' + ' | '.join(title for title, _ in columns) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in columns) + ' |')
    for entry in entries:
        lines.append('| ' + ' | '.join(str(entry[key]) for _, key in columns) + ' |')
    return lines


def build_markdown_report(system: SmallSystem, summary: dict[str, object], counterexamples: dict[str, object], neutral_check: dict[str, object]) -> str:
    example_a = counterexamples['missing_reverse_edge']
    example_b = counterexamples['invalid_zero_state']

    lines: list[str] = []
    lines.append(f'# HamiltonianRuleJax correctness benchmark (small system, {DATE_TAG})')
    lines.append('')
    lines.append('## System')
    lines.append('')
    lines.append(f"- `Lx = {system.parameters['Lx']}`")
    lines.append(f"- `t = {system.parameters['t']}`")
    lines.append(f"- `J_K = {system.parameters['J_K']}`")
    lines.append(f"- `J1 = {system.parameters['J1']}`")
    lines.append(f"- `J2 = {system.parameters['J2']}`")
    lines.append(f"- `delta_z = {system.parameters['delta_z']}`")
    lines.append(f"- `n_fermions = {system.parameters['n_fermions']}`")
    lines.append(f'- `fermion_size = {system.fermion_size}`')
    lines.append(f'- `spin_size = {system.spin_size}`')
    lines.append(f'- `joint_size = {system.fermion_size + system.spin_size}`')
    lines.append(f'- `n_basis_states = {summary["n_states"]}`')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f'- support mismatch states: `{summary["n_support_mismatch_states"]}`')
    lines.append(f'- zero-state risk states: `{summary["n_zero_state_risk_states"]}`')
    lines.append(f'- max zero-state probability: `{summary["max_zero_state_probability"]}`')
    lines.append(f'- max TV distance to entry-uniform reference: `{summary["max_tv_distance_to_reference"]}`')
    lines.append(f'- mean TV distance to entry-uniform reference: `{summary["mean_tv_distance_to_reference"]}`')
    lines.append(f'- one-way code edges: `{summary["n_one_way_edges"]}`')
    lines.append(f'- max abs correction residual: `{summary["max_abs_delta_corr"]}`')
    lines.append(f'- max abs DB residual under uniform target: `{summary["max_abs_db_residual_uniform_pi"]}`')
    lines.append(f'- neutral withproposal path matches original direct path: `{neutral_check["all_steps_match"]}`')
    lines.append('')
    lines.append('## Support Mismatch Examples')
    lines.append('')
    if summary['support_mismatch_examples']:
        lines.extend(
            markdown_table(
                summary['support_mismatch_examples'],
                [
                    ('x', 'x'),
                    ('reported_n_conn', 'reported_n_conn'),
                    ('actual_nonzero_entries', 'actual_nonzero_entry_count'),
                    ('tv_distance', 'tv_distance_to_reference'),
                ],
            )
        )
    else:
        lines.append('- none')
    lines.append('')
    lines.append('## Zero-State Risk Examples')
    lines.append('')
    if summary['zero_state_examples']:
        lines.extend(
            markdown_table(
                summary['zero_state_examples'],
                [
                    ('x', 'x'),
                    ('reported_n_conn', 'reported_n_conn'),
                    ('actual_nonzero_entries', 'actual_nonzero_entry_count'),
                    ('zero_state_probability', 'zero_state_probability'),
                ],
            )
        )
    else:
        lines.append('- none')
    lines.append('')
    lines.append('## Counterexample A: one-way edge')
    lines.append('')
    lines.append(f"- `x = {example_a['x']}`")
    lines.append(f"- `y = {example_a['y']}`")
    lines.append(f"- `q_code(y|x) = {example_a['pair_summary']['q_code_y_given_x']}`")
    lines.append(f"- `q_code(x|y) = {example_a['pair_summary']['q_code_x_given_y']}`")
    lines.append(f"- `log_prob_corr_code(x->y) = {example_a['pair_summary']['code_log_prob_corr_x_to_y']}`")
    lines.append('')
    lines.append('### x nonzero entries')
    lines.append('')
    lines.extend(
        markdown_table(
            example_a['x_analysis']['all_nonzero_entries'],
            [('entry', 'entry_index'), ('target_state', 'target_state'), ('mel', 'mel')],
        )
    )
    lines.append('')
    lines.append('### y nonzero entries')
    lines.append('')
    lines.extend(
        markdown_table(
            example_a['y_analysis']['all_nonzero_entries'],
            [('entry', 'entry_index'), ('target_state', 'target_state'), ('mel', 'mel')],
        )
    )
    lines.append('')
    lines.append('## Counterexample B: invalid zero state risk')
    lines.append('')
    lines.append(f"- `x0 = {example_b['x']}`")
    lines.append(f"- `reported_n_conn = {example_b['reported_n_conn']}`")
    lines.append(f"- `actual_nonzero_entries = {example_b['actual_nonzero_entry_count']}`")
    lines.append(f"- `p_zero = {example_b['zero_state_probability']}`")
    lines.append('')
    lines.extend(
        markdown_table(
            example_b['all_nonzero_entries'],
            [('entry', 'entry_index'), ('target_state', 'target_state'), ('mel', 'mel')],
        )
    )
    lines.append('')
    lines.append('## Stationary Distribution Check')
    lines.append('')
    lines.append(f"- computed: `{summary['stationary_distribution']['computed']}`")
    lines.append(f"- note: {summary['stationary_distribution']['reason']}")
    lines.append('')
    lines.append('## Neutral Path Consistency')
    lines.append('')
    lines.append(f"- checked steps: `{neutral_check['checked_steps']}`")
    lines.append(f"- all steps match: `{neutral_check['all_steps_match']}`")
    if neutral_check['first_mismatch'] is not None:
        lines.append(f"- first mismatch: `{neutral_check['first_mismatch']}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    system = build_small_system()
    states = [state_key(row) for row in np.asarray(system.joint_hilbert.all_states())]
    state_info_map = {
        key: analyze_state(system.hamiltonian, key)
        for key in sorted(states)
    }
    summary = build_summary(state_info_map)
    counterexamples = build_counterexamples(state_info_map)
    neutral_check = run_neutral_path_consistency_check(system)

    report = {
        'report_name': REPORT_BASENAME,
        'system': {
            'parameters': system.parameters,
            'fermion_size': system.fermion_size,
            'spin_size': system.spin_size,
            'joint_size': system.fermion_size + system.spin_size,
            'n_basis_states': len(states),
        },
        'summary': summary,
        'counterexamples': counterexamples,
        'neutral_withproposal_consistency': neutral_check,
        'state_diagnostics': [
            {
                field: value
                for field, value in state_info_map[key].items()
                if field not in ('reference_state_probs_map', 'code_state_probs_map')
            }
            for key in sorted(state_info_map)
        ],
    }

    output_dir = Path(__file__).resolve().parent
    json_path = output_dir / f'{REPORT_BASENAME}.json'
    md_path = output_dir / f'{REPORT_BASENAME}.md'

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8-sig',
    )
    md_path.write_text(
        build_markdown_report(system, summary, counterexamples, neutral_check),
        encoding='utf-8-sig',
    )

    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
