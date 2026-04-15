"""Hamiltonian VMC baseline with standard MetropolisFermionHop.

This entry point keeps only the baseline-specific choice:
- standard fermion hopping proposals;
- no occupation update;
- no 1-RDM / natural-orbital rotation.
All shared CLI, model resolution, and optimizer/SR wiring live in `_common.py`.
"""
from __future__ import annotations

import netket as nk

from _common import (
    add_model_optimizer_sr_arguments,
    base_parser,
    build_hamiltonian,
    build_mcstate,
    build_model,
    energy_summary,
    get_optimizer_and_sr,
    graph_from_hamiltonian,
    namespace_snapshot,
    print_mapping,
    run_driver,
    summarize_model_config,
    summarize_optimizer_and_sr_config,
)


def parse_args():
    """Read CLI arguments for the baseline SIAM runner."""
    parser = base_parser(
        "Hamiltonian VMC with standard MetropolisFermionHop. "
        "Select the initial (H, hi) with --hamiltonian-builder."
    )
    add_model_optimizer_sr_arguments(parser)
    return parser.parse_args()



def collect_runner_config(args) -> dict[str, object]:
    """Package the baseline-specific runtime arguments for console output."""
    return namespace_snapshot(
        args,
        "n_iter",
        "n_samples",
        "n_discard_per_chain",
        "d_max",
        "show_progress",
    )



def main() -> int:
    """Build the SIAM system, then run one baseline VMC calculation per model."""
    args = parse_args()
    H, hi = build_hamiltonian(args)
    graph = graph_from_hamiltonian(H)

    print_mapping("Runner settings:", collect_runner_config(args))
    print_mapping("Optimizer/SR settings:", summarize_optimizer_and_sr_config(args))

    for requested_name in args.models:
        model = build_model(requested_name, hi, args)
        sampler = nk.sampler.MetropolisFermionHop(
            hi,
            graph=graph,
            d_max=args.d_max,
            spin_symmetric=True,
        )
        vstate = build_mcstate(sampler, model, args)
        model_config = {
            **summarize_model_config(requested_name, args),
            "n_parameters": int(vstate.n_parameters),
        }
        print_mapping(f"\n[{requested_name}] model settings:", model_config)

        op, sr = get_optimizer_and_sr(args)
        driver = nk.VMC(
            hamiltonian=H,
            optimizer=op,
            preconditioner=sr,
            variational_state=vstate,
        )
        run_driver(driver, args)

        summary = energy_summary(H, vstate, driver)
        resolved_name = model_config["resolved_model"]
        print(f"\n[{resolved_name}] steps={summary['step_count']}")
        print(f"  acceptance={summary['acceptance']}")
        print(f"  final_energy={summary['energy_mean_real']}")
        print(f"  energy_error={summary['energy_error_real']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())