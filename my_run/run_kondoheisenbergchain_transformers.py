"""Run Transformer+NNBF ansatzes on a Kondo-Heisenberg chain.

This entry point keeps only Kondo-Heisenberg specific decisions:
- which Hamiltonian class to instantiate;
- which Transformer registry to expose;
- how to build the joint sampler;
- whether the driver path is `VMC` or `VMC_SR`.
All shared CLI grouping, argument packaging, `MCState` construction,
driver execution, and artifact writing live in `_common.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import netket as nk
import numpy as np
from netket.sampler.metropolis import MetropolisHamiltonianWithProposal
from netket.sampler.rules.fermion_2nd_proposal import FermionHopRule_with_proposal

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / 'Hamiltonian', REPO_ROOT / 'nqs_state'):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from Hamiltonian.KondoHeisenbergChainSpinFermion import (
    KondoHeisenbergChainSpinFermion,
    seed_joint_sz_sector,
)
from _common import (
    add_output_argument,
    add_progress_arguments,
    add_runtime_sampling_arguments,
    add_vmc_driver_arguments,
    build_basic_optimizer,
    build_mcstate,
    build_vmc_or_vmc_sr_driver,
    collect_vmc_driver_config,
    create_argument_parser,
    energy_summary,
    list_or_none,
    namespace_snapshot,
    print_mapping,
    run_driver,
    save_model_artifacts,
    tuple_or_none,
    validate_vmc_driver_args,
    write_json,
)
from transformer_joint_nnbf import (
    LogSiteTokenTransformerNNBF,
    LogSpinChannelTransformerNNBF,
    resolve_spin_fermion_layout,
)


MODEL_REGISTRY = {
    'site-token': LogSiteTokenTransformerNNBF,
    'spin-channel': LogSpinChannelTransformerNNBF,
}


def parse_args() -> argparse.Namespace:
    """Read CLI arguments and group them by their role in the runner pipeline."""
    parser = create_argument_parser(
        'Run Transformer+NNBF variational states on a Kondo-Heisenberg chain '
        'with explicitly separated fermion and local-spin subspaces.'
    )

    system_group = parser.add_argument_group('Hamiltonian / system')
    system_group.add_argument('--Lx', type=int, default=50)
    system_group.add_argument('--t', type=float, default=1.0)
    system_group.add_argument('--J_K', type=float, default=1.0)
    system_group.add_argument('--J1', type=float, default=1.0)
    system_group.add_argument('--J2', type=float, default=0.0)
    system_group.add_argument('--delta-z', dest='delta_z', type=float, default=1.0)
    system_group.add_argument('--pbc', action='store_true')
    system_group.add_argument('--n-fermions', type=int, default=None)
    system_group.add_argument('--joint-two-sz', type=int, default=0)
    system_group.add_argument('--local-total-sz', type=float, default=None)
    system_group.add_argument(
        '--n-fermions-per-spin',
        type=int,
        nargs=2,
        default=None,
        metavar=('N_DN', 'N_UP'),
    )

    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--models',
        nargs='+',
        choices=sorted(MODEL_REGISTRY),
        default=['site-token', 'spin-channel'],
    )
    model_group.add_argument('--d-model', type=int, default=64)
    model_group.add_argument('--n-heads', type=int, default=4)
    model_group.add_argument('--n-layers', type=int, default=2)
    model_group.add_argument('--mlp-ratio', type=int, default=4)

    sampler_group = parser.add_argument_group('Sampler')
    sampler_group.add_argument('--joint-sampler', choices=('hamiltonian', 'hamiltonian-with-proposal'), default='hamiltonian')
    sampler_group.add_argument('--proposal-occupations-file', type=str, default=None)
    sampler_group.add_argument('--proposal-occupation-value', type=float, default=None)
    sampler_group.add_argument('--proposal-noise-strength', type=float, default=100.0)
    sampler_group.add_argument('--proposal-mixing', type=float, default=0.05)
    sampler_group.add_argument('--proposal-balance-beta', type=float, default=0.5)

    driver_group = parser.add_argument_group('Optimizer / driver')
    add_vmc_driver_arguments(
        driver_group,
        driver_default='vmc-sr',
        optimizer_choices=('sgd', 'adam'),
        optimizer_default='adam',
        learning_rate_default=1.0e-2,
        diag_shift_default=5.0e-2,
        linear_solver_default='cholesky',
        mode_default='real',
    )

    runtime_group = parser.add_argument_group('Sampling / runtime')
    add_runtime_sampling_arguments(
        runtime_group,
        n_iter_default=200,
        n_samples_default=2048,
        n_discard_per_chain_default=32,
        d_max_default=1,
        n_chains_per_rank_default=16,
        sweep_size_default=None,
        seed_default=1234,
    )
    add_output_argument(
        runtime_group,
        default_out_dir=str(REPO_ROOT / 'run_outputs' / 'kondoheisenbergchain_transformers'),
    )
    add_progress_arguments(runtime_group)
    return parser.parse_args()



def validate_args(args: argparse.Namespace) -> None:
    """Reject CLI combinations that would define inconsistent sectors or samplers."""
    if args.n_fermions is not None and args.n_fermions_per_spin is not None:
        raise ValueError('Pass either --n-fermions or --n-fermions-per-spin, not both.')

    if float(args.J_K) != 0.0 and args.n_fermions_per_spin is not None:
        raise ValueError(
            '--n-fermions-per-spin is not compatible with full Kondo spin flips when J_K != 0. '
            'Use --n-fermions and optionally --joint-two-sz instead.'
        )

    if float(args.J_K) != 0.0 and args.local_total_sz is not None:
        raise ValueError(
            '--local-total-sz is not compatible with full Kondo spin flips when J_K != 0. '
            'Use --joint-two-sz to seed the conserved joint sector instead.'
        )

    if not 0.0 <= float(args.proposal_balance_beta) <= 1.0:
        raise ValueError('--proposal-balance-beta must lie in the interval [0, 1].')

    validate_vmc_driver_args(args)



def model_metadata(name: str) -> dict[str, str]:
    """Record which model registry entry and Python class the runner selected."""
    cls = MODEL_REGISTRY[name]
    return {
        'requested_model': name,
        'model_module': cls.__module__,
        'model_class': cls.__name__,
    }



def resolve_determinant_backend(layout) -> str:
    """Explain which determinant backend the resolved layout implies."""
    return 'blocked' if layout.uses_block_determinant else 'generalized'



def collect_system_config(args) -> dict[str, object]:
    """Package the Hamiltonian-specific CLI fields into one summary dict."""
    config = {
        'hamiltonian_module': KondoHeisenbergChainSpinFermion.__module__,
        'hamiltonian_class': KondoHeisenbergChainSpinFermion.__name__,
    }
    config.update(
        namespace_snapshot(
            args,
            'Lx',
            't',
            'J_K',
            'J1',
            'J2',
            'delta_z',
            'pbc',
            'n_fermions',
            'n_fermions_per_spin',
            'local_total_sz',
            'joint_two_sz',
            transforms={
                'pbc': bool,
                'n_fermions_per_spin': list_or_none,
            },
        )
    )
    return config



def collect_model_hyperparameters(args) -> dict[str, object]:
    """Package the Transformer hyperparameters into one summary dict."""
    return namespace_snapshot(args, 'd_model', 'n_heads', 'n_layers', 'mlp_ratio')



def collect_driver_config(args) -> dict[str, object]:
    """Package the optimizer/driver knobs into one summary dict."""
    return collect_vmc_driver_config(args)



def collect_runtime_config(args) -> dict[str, object]:
    """Package the runtime knobs that feed samplers, `MCState`, and driver.run."""
    return namespace_snapshot(
        args,
        'n_iter',
        'n_samples',
        'n_discard_per_chain',
        'n_chains_per_rank',
        'd_max',
        'sweep_size',
        'seed',
        'show_progress',
        transforms={'show_progress': bool},
    )



def collect_output_config(args) -> dict[str, object]:
    """Package the file-output location into one summary dict."""
    return {'out_dir': str(args.out_dir)}



def collect_layout_summary(layout) -> dict[str, object]:
    """Package the resolved fermion/spin layout that the model and sampler see."""
    return {
        'n_sites': layout.n_sites,
        'n_fermions': layout.n_fermions,
        'n_dn': layout.n_dn,
        'n_up': layout.n_up,
        'fermion_size': layout.fermion_size,
        'spin_block_sizes': list(layout.spin_block_sizes),
        'uses_block_determinant': layout.uses_block_determinant,
        'determinant_backend': resolve_determinant_backend(layout),
    }



def build_system(args: argparse.Namespace):
    """Instantiate the Kondo-Heisenberg Hamiltonian and its joint Hilbert space."""
    return KondoHeisenbergChainSpinFermion(
        Lx=args.Lx,
        t=args.t,
        J_K=args.J_K,
        J1=args.J1,
        J2=args.J2,
        delta_z=args.delta_z,
        pbc=args.pbc,
        n_fermions=args.n_fermions,
        n_fermions_per_spin=tuple_or_none(args.n_fermions_per_spin),
        local_total_sz=args.local_total_sz,
    )



def build_model(name: str, layout, args: argparse.Namespace):
    """Instantiate the selected Transformer+NNBF ansatz."""
    cls = MODEL_REGISTRY[name]
    return cls(
        layout=layout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
    )



def _build_spin_rule(spin_hilbert, n_sites: int, nn_edges):
    """Choose the local-spin proposal rule for one spin subspace."""
    if getattr(spin_hilbert, 'constrained', False):
        spin_graph = nk.graph.Graph(edges=nn_edges, n_nodes=n_sites)
        return nk.sampler.rules.ExchangeRule(graph=spin_graph, d_max=1), 'ExchangeRule'
    return nk.sampler.rules.LocalRule(), 'LocalRule'



def _parse_occupations_text(payload: str) -> np.ndarray:
    """Parse a proposal-occupation file written either as JSON or plain numbers."""
    stripped = payload.strip()
    if not stripped:
        raise ValueError('Proposal occupations file is empty.')

    if stripped[0] == '[':
        data = json.loads(stripped)
        return np.asarray(data, dtype=float)

    tokens = stripped.replace(',', ' ').split()
    return np.asarray([float(tok) for tok in tokens], dtype=float)



def resolve_proposal_occupations(layout, args: argparse.Namespace) -> tuple[np.ndarray | None, dict]:
    """Load, synthesize, or disable the proposal occupations for HamiltonianWithProposal."""
    if args.proposal_occupations_file is not None:
        path = Path(args.proposal_occupations_file)
        occupations = _parse_occupations_text(path.read_text(encoding='utf-8'))
        source = str(path)
    elif args.proposal_occupation_value is not None:
        occupations = np.full(layout.n_sites, float(args.proposal_occupation_value), dtype=float)
        source = f'uniform:{args.proposal_occupation_value}'
    else:
        occupations = None
        source = None

    if occupations is not None:
        if occupations.ndim != 1:
            raise ValueError('Proposal occupations must be a 1D array.')
        if occupations.shape[0] not in (layout.n_sites, 2 * layout.n_sites):
            raise ValueError(
                'Proposal occupations must have length n_sites or 2*n_sites. '
                f'Got {occupations.shape[0]}, expected {layout.n_sites} or {2 * layout.n_sites}.'
            )
        occ_len = int(occupations.shape[0])
    else:
        occ_len = None

    return occupations, {
        'proposal_occupations_source': source,
        'proposal_occupations_length': occ_len,
        'proposal_noise_strength': float(args.proposal_noise_strength),
        'proposal_mixing': float(args.proposal_mixing),
        'proposal_balance_beta': float(args.proposal_balance_beta),
    }



def build_factorized_sampler(system, layout, args: argparse.Namespace):
    """Build the explicit tensor-product sampler used for sampler experiments."""
    fermion_graph = nk.graph.Graph(edges=system.geometry['nn'], n_nodes=layout.n_sites)

    if args.fermion_sampler == 'with-proposal':
        occupations, proposal_info = resolve_proposal_occupations(layout, args)
        fermion_rule = FermionHopRule_with_proposal(
            system.fermion_hilbert,
            occupations=occupations,
            graph=fermion_graph,
            d_max=args.d_max,
            spin_symmetric=True,
            noise_strength=args.proposal_noise_strength,
            mixing=args.proposal_mixing,
        )
        fermion_info = {
            'fermion_sampler': 'with-proposal',
            **proposal_info,
        }
    else:
        fermion_rule = nk.sampler.rules.FermionHopRule(
            system.fermion_hilbert,
            graph=fermion_graph,
            d_max=args.d_max,
            spin_symmetric=True,
        )
        fermion_info = {
            'fermion_sampler': 'standard',
            'proposal_occupations_source': None,
            'proposal_occupations_length': None,
            'proposal_noise_strength': None,
            'proposal_mixing': None,
        }

    spin_subspaces = tuple(getattr(system.joint_hilbert, 'subspaces', ())[1:])
    spin_pairs = tuple(
        _build_spin_rule(spin_hilbert, layout.n_sites, system.geometry['nn'])
        for spin_hilbert in spin_subspaces
    )
    spin_rules = tuple(pair[0] for pair in spin_pairs)
    spin_rule_names = [pair[1] for pair in spin_pairs]

    rule = nk.sampler.rules.TensorRule(system.joint_hilbert, (fermion_rule, *spin_rules))
    sampler = nk.sampler.MetropolisSampler(
        system.joint_hilbert,
        rule,
        n_chains_per_rank=args.n_chains_per_rank,
        sweep_size=args.sweep_size,
        reset_chains=False,
    )

    note = None
    if float(args.J_K) != 0.0 and not layout.uses_block_determinant:
        note = (
            'Factorized sampling proposes fermion and local-spin moves on separate '
            'subspaces. It is fast and explicit, but it does not generate a single '
            'coupled Kondo spin-flip move. Use --joint-sampler hamiltonian if you '
            'want symmetry-preserving proposals from the full operator.'
        )

    return sampler, {
        'joint_sampler': 'factorized',
        'spin_rules': spin_rule_names,
        'sampler_note': note,
        **fermion_info,
    }



def build_hamiltonian_sampler(system, args: argparse.Namespace):
    """Build the sampler that follows the full Hamiltonian connectivity."""
    sampler = nk.sampler.MetropolisHamiltonian(
        system.joint_hilbert,
        hamiltonian=system.hamiltonian,
        n_chains_per_rank=args.n_chains_per_rank,
        sweep_size=args.sweep_size,
        reset_chains=False,
    )
    return sampler, {
        'joint_sampler': 'hamiltonian',
        'fermion_sampler': None,
        'spin_rules': None,
        'proposal_occupations_source': None,
        'proposal_occupations_length': None,
        'proposal_noise_strength': None,
        'proposal_mixing': None,
        'proposal_balance_beta': None,
        'sampler_note': (
            'HamiltonianRule proposes moves from the full Kondo-Heisenberg operator '
            'and therefore preserves its exact connectivity and conserved sectors.'
        ),
    }



def build_hamiltonian_with_proposal_sampler(system, layout, args: argparse.Namespace):
    """Build the Hamiltonian-connected sampler with fermion proposal bias."""
    occupations, proposal_info = resolve_proposal_occupations(layout, args)
    sampler = MetropolisHamiltonianWithProposal(
        system.joint_hilbert,
        hamiltonian=system.hamiltonian,
        occupations=occupations,
        balance_beta=args.proposal_balance_beta,
        noise_strength=args.proposal_noise_strength,
        mixing=args.proposal_mixing,
        n_chains_per_rank=args.n_chains_per_rank,
        sweep_size=args.sweep_size,
        reset_chains=False,
    )
    return sampler, {
        'joint_sampler': 'hamiltonian-with-proposal',
        'fermion_sampler': None,
        'spin_rules': None,
        **proposal_info,
        'sampler_note': (
            'HamiltonianRuleWithProposal keeps the full Kondo-Heisenberg connectivity, '
            'biases fermion changes using the supplied occupations, and balances '
            'ss/ff/sf proposal mass through --proposal-balance-beta.'
        ),
    }



def build_sampler(system, layout, args: argparse.Namespace):
    """Select the sampler branch requested by the CLI."""
    if args.joint_sampler == 'hamiltonian':
        return build_hamiltonian_sampler(system, args)
    return build_hamiltonian_with_proposal_sampler(system, layout, args)



def build_optimizer(args: argparse.Namespace):
    """Build the outer optimizer chosen by the CLI."""
    return build_basic_optimizer(args)



def build_driver(system, vstate, optimizer, args: argparse.Namespace):
    """Build either the `VMC` or `VMC_SR` driver branch selected by the CLI."""
    return build_vmc_or_vmc_sr_driver(system.hamiltonian, vstate, optimizer, args)



def maybe_seed_joint_sector(system, layout, vstate, args: argparse.Namespace):
    """Seed the conserved joint-Sz sector for the full-Hamiltonian samplers."""
    if layout.uses_block_determinant:
        return None
    if float(args.J_K) == 0.0:
        return None
    if args.joint_sampler not in ('hamiltonian', 'hamiltonian-with-proposal'):
        return None
    return seed_joint_sz_sector(vstate, two_sz=args.joint_two_sz)



def print_tokenization_summary(system, layout, args: argparse.Namespace) -> None:
    """Explain how the resolved layout feeds the model and sampler."""
    backend = resolve_determinant_backend(layout)
    print('System layout:')
    print(f'  joint_hilbert = {type(system.joint_hilbert).__name__}')
    print(f'  fermion_hilbert = {type(system.fermion_hilbert).__name__}')
    print(f'  local_spin_hilbert = {type(system.local_spin_hilbert).__name__}')
    print(
        '  sizes = '
        f'fermion_size={layout.fermion_size}, spin_block_sizes={layout.spin_block_sizes}, '
        f'n_sites={layout.n_sites}, n_fermions={layout.n_fermions}, '
        f'n_dn={layout.n_dn}, n_up={layout.n_up}'
    )
    print('Token path:')
    print('  flattened state = [fermion down block | fermion up block | local-spin block]')
    print('  site-token model = (n_up[i], n_dn[i]) -> one 4-state fermion token per site, then add spin embeddings')
    print('  spin-channel model = down/up/spin channels share site embedding and differ by channel embedding')
    print('Determinant path:')
    print(f'  determinant_backend={backend}')
    if layout.uses_block_determinant:
        print(
            '  fixed spin sectors detected -> use blocked up/down determinants '
            f'with widths n_dn={layout.n_dn}, n_up={layout.n_up}'
        )
    else:
        print('  only total fermion number fixed -> use one generalized determinant over all spin orbitals')
    print('Sampler path:')
    print(f'  joint_sampler={args.joint_sampler}')
    if args.joint_sampler == 'hamiltonian':
        print('  hamiltonian = HamiltonianRule on the full joint operator')
    else:
        print('  hamiltonian-with-proposal = HamiltonianRuleWithProposal on the full joint operator')
    print('Driver path:')
    driver_info = collect_driver_config(args)
    print('  ' + ', '.join(f'{key}={value}' for key, value in driver_info.items()))



def run_single_model(name: str, system, layout, args: argparse.Namespace, base_out_dir: Path) -> dict:
    """Run one model end-to-end and persist the standard artifact set."""
    model = build_model(name, layout, args)
    sampler, sampler_info = build_sampler(system, layout, args)
    vstate = build_mcstate(sampler, model, args)

    model_identity = model_metadata(name)
    model_print_config = {
        **model_identity,
        **collect_model_hyperparameters(args),
        'n_parameters': int(vstate.n_parameters),
    }
    print_mapping(f'\n[{name}] model settings:', model_print_config)
    print_mapping(f'[{name}] sampler settings:', sampler_info)

    seeded_reference = maybe_seed_joint_sector(system, layout, vstate, args)
    optimizer = build_optimizer(args)
    driver = build_driver(system, vstate, optimizer, args)
    run_driver(driver, args)

    summary = {
        'model': model_identity,
        'model_hyperparameters': collect_model_hyperparameters(args),
        'system': collect_system_config(args),
        'layout': collect_layout_summary(layout),
        'sampler': sampler_info,
        'driver': collect_driver_config(args),
        'runtime': collect_runtime_config(args),
        'output': collect_output_config(args),
        'seeded_reference_state': (None if seeded_reference is None else seeded_reference.tolist()),
    }
    summary.update(energy_summary(system.hamiltonian, vstate, driver))

    model_dir = save_model_artifacts(base_out_dir, name, summary=summary, vstate=vstate)
    print(f"[{name}] final_energy = {summary['energy_mean_real']:.10f}, acceptance = {summary['acceptance']}")
    print(f"[{name}] saved state -> {model_dir / 'final_vstate.msgpack'}")
    print(f"[{name}] saved summary -> {model_dir / 'summary.json'}")
    return summary



def main() -> int:
    """Build the Kondo-Heisenberg system, then run one calculation per model."""
    args = parse_args()
    validate_args(args)
    base_out_dir = Path(args.out_dir)

    print_mapping('System arguments:', collect_system_config(args))
    print_mapping('Model hyperparameters:', collect_model_hyperparameters(args))
    print_mapping('Driver settings:', collect_driver_config(args))
    print_mapping('Sampling / runtime:', collect_runtime_config(args))
    print_mapping('Output settings:', collect_output_config(args))

    system = build_system(args)
    layout = resolve_spin_fermion_layout(system)
    print_tokenization_summary(system, layout, args)

    all_results = []
    for name in args.models:
        all_results.append(run_single_model(name, system, layout, args, base_out_dir))

    write_json(
        base_out_dir / 'all_results.json',
        {
            'results': all_results,
            'argv': sys.argv[1:],
            'output': collect_output_config(args),
        },
    )
    print(f'Combined summary -> {base_out_dir / "all_results.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



