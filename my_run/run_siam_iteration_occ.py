"""Hamiltonian VMC with periodic proposal-occupation updates.

This entry point keeps only the occupation-update specific logic:
- split the run into segments;
- estimate diagonal occupations from samples;
- rebuild the proposal distribution after each segment.
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
from graph_sample.iteration_occ_func_simple import run_vmc_with_iteration_occ



def _preview_parameter_count(hi, graph, model, args) -> int:
    """Instantiate a matching `MCState` once so we can report the parameter count."""
    sampler = nk.sampler.MetropolisFermionHopWithProposal(
        hi,
        graph=graph,
        occupations=np.full(hi.n_orbitals, 0.5),
        d_max=args.d_max,
        spin_symmetric=True,
    )
    vstate = build_mcstate(sampler, model, args)
    return int(vstate.n_parameters)



def parse_args():
    """Read CLI arguments for the SIAM occupation-update runner."""
    parser = base_parser(
        "Hamiltonian VMC with occupation-updated proposal. "
        "Select the initial (H, hi) with --hamiltonian-builder."
    )
    add_model_optimizer_sr_arguments(parser)
    parser.add_argument(
        "--occ-update-every",
        type=int,
        default=10,
        help="Number of VMC steps between proposal occupation updates.",
    )
    return parser.parse_args()



def collect_runner_config(args) -> dict[str, object]:
    """Package the occupation-update specific runtime arguments."""
    return namespace_snapshot(
        args,
        "occ_update_every",
        "n_iter",
        "n_samples",
        "n_discard_per_chain",
        "d_max",
        "show_progress",
    )



def main() -> int:
    """Build the SIAM system, then run one occupation-update VMC calculation per model."""
    args = parse_args()
    H, hi = build_hamiltonian(args)
    graph = graph_from_hamiltonian(H)

    print_mapping("Runner settings:", collect_runner_config(args))
    print_mapping("Optimizer/SR settings:", summarize_optimizer_and_sr_config(args))

    for requested_name in args.models:
        model = build_model(requested_name, hi, args)
        model_config = {
            **summarize_model_config(requested_name, args),
            "n_parameters": _preview_parameter_count(hi, graph, model, args),
        }
        print_mapping(f"\n[{requested_name}] model settings:", model_config)

        op, sr = get_optimizer_and_sr(args)
        result = run_vmc_with_iteration_occ(
            hi=hi,
            H=H,
            model=model,
            graph_sampler=graph,
            occ_init=np.full(hi.n_orbitals, 0.5),
            op=op,
            sr=sr,
            N_ITER=args.n_iter,
            occ_update_every=args.occ_update_every,
            n_samples=args.n_samples,
            n_discard_per_chain=args.n_discard_per_chain,
            d_max=args.d_max,
            spin_symmetric=True,
            show_progress=args.show_progress,
        )

        energy = result["energy_history"]
        resolved_name = model_config["resolved_model"]
        print(f"\n[{resolved_name}] steps={result['driver'].step_count}")
        print(f"  n_parameters={model_config['n_parameters']}")
        print(f"  energy={fmt(energy)}")
        print(f"  final_occ={fmt(result['final_occ'])}")
        print(f"  final_energy={energy[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())