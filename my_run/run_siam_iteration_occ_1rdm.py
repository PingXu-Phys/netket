"""Hamiltonian VMC with iterative 1-RDM / natural-orbital updates.

This entry point keeps only the 1-RDM specific logic:
- split the run into segments;
- estimate the full 1-RDM after each segment;
- diagonalize the 1-RDM into natural orbitals;
- rotate the Hamiltonian and rebuild the proposal graph.
All shared CLI, model resolution, and optimizer/SR wiring live in `_common.py`.
"""
from __future__ import annotations

import netket as nk
import numpy as np

from _common import (
    add_model_optimizer_sr_arguments,
    base_parser,
    build_hamiltonian,
    build_mcstate,
    build_model,
    fmt,
    get_optimizer_and_sr,
    graph_from_hamiltonian,
    namespace_snapshot,
    print_mapping,
    summarize_model_config,
    summarize_optimizer_and_sr_config,
)
from graph_sample.iteration_occ_func_simple import run_vmc_with_iteration_natural_orbitals



def _preview_parameter_count(hi, H, model, args) -> int:
    """Instantiate a matching `MCState` once so we can report the parameter count."""
    sampler = nk.sampler.MetropolisFermionHopWithProposal(
        hi,
        graph=graph_from_hamiltonian(H),
        occupations=np.full(hi.n_orbitals, 0.5),
        d_max=args.d_max,
        spin_symmetric=True,
    )
    vstate = build_mcstate(sampler, model, args)
    return int(vstate.n_parameters)



def parse_args():
    """Read CLI arguments for the SIAM 1-RDM / natural-orbital runner."""
    epilog = """Examples:
  python run/run_siam_iteration_occ_1rdm.py
  python run/run_siam_iteration_occ_1rdm.py --models backflow-deep --n-hidden 64 --n-iter 200
  python run/run_siam_iteration_occ_1rdm.py --models agp-backflow --n-layers 2 --optimizer adamw --learning-rate 3e-3 --sr-solver pinv_smooth --diag-shift 0.02
"""
    parser = base_parser(
        "Hamiltonian VMC with iterative 1-RDM/natural-orbital updates. "
        "Select the initial (H, hi) with --hamiltonian-builder.",
        epilog=epilog,
    )
    add_model_optimizer_sr_arguments(parser)
    parser.add_argument(
        "--orbital-update-every",
        type=int,
        default=20,
        help="Number of VMC steps between full 1-RDM / natural-orbital updates.",
    )
    parser.add_argument(
        "--hamiltonian-cutoff",
        type=float,
        default=None,
        help="Drop rotated Hamiltonian terms smaller than this cutoff after each orbital rotation.",
    )
    return parser.parse_args()



def collect_runner_config(args) -> dict[str, object]:
    """Package the 1-RDM runner-specific runtime arguments."""
    return namespace_snapshot(
        args,
        "orbital_update_every",
        "hamiltonian_cutoff",
        "n_iter",
        "n_samples",
        "n_discard_per_chain",
        "d_max",
        "show_progress",
    )



def main() -> int:
    """Build the SIAM system, then run one natural-orbital VMC calculation per model."""
    args = parse_args()
    H, hi = build_hamiltonian(args)

    print_mapping("Runner settings:", collect_runner_config(args))
    print_mapping("Optimizer/SR settings:", summarize_optimizer_and_sr_config(args))

    for requested_name in args.models:
        model = build_model(requested_name, hi, args)
        model_config = {
            **summarize_model_config(requested_name, args),
            "n_parameters": _preview_parameter_count(hi, H, model, args),
        }
        print_mapping(f"\n[{requested_name}] model settings:", model_config)

        op, sr = get_optimizer_and_sr(args)
        result = run_vmc_with_iteration_natural_orbitals(
            hi=hi,
            H=H,
            model=model,
            occ_init=np.full(hi.n_orbitals, 0.5),
            op=op,
            sr=sr,
            N_ITER=args.n_iter,
            orbital_update_every=args.orbital_update_every,
            n_samples=args.n_samples,
            n_discard_per_chain=args.n_discard_per_chain,
            d_max=args.d_max,
            spin_symmetric=True,
            show_progress=args.show_progress,
            hamiltonian_cutoff=args.hamiltonian_cutoff,
        )

        energy = result["energy_history"]
        final_rdm = result["final_rdm"]
        resolved_name = model_config["resolved_model"]
        print(f"\n[{resolved_name}] steps={result['driver'].step_count}")
        print(f"  n_parameters={model_config['n_parameters']}")
        print(f"  energy={fmt(energy)}")
        print(f"  final_occ={fmt(result['final_occ'])}")
        print(f"  final_natural_occupations={fmt(result['final_natural_occupations'])}")
        print(f"  final_rdm_diag={fmt(np.real(np.diag(final_rdm)))}")
        print(f"  final_rdm=\n{fmt(final_rdm)}")
        print(f"  final_energy={energy[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())