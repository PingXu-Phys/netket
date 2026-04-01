# Copyright 2026 Ping Xu - All rights reserved.

from __future__ import annotations

import netket as nk
import numpy as np

from hamiltonian_graph import graph_from_hamiltonian
from iteration_occ import iterate_occupations


def build_demo_hamiltonian():
    hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
    create = nk.operator.fermion.create
    destroy = nk.operator.fermion.destroy

    H = (
        -1.0 * (create(hi, 0) @ destroy(hi, 1) + create(hi, 1) @ destroy(hi, 0))
        - 0.7 * (create(hi, 1) @ destroy(hi, 2) + create(hi, 2) @ destroy(hi, 1))
        - 0.4 * (create(hi, 2) @ destroy(hi, 3) + create(hi, 3) @ destroy(hi, 2))
    )
    return hi, H


def main():
    hi, H = build_demo_hamiltonian()
    graph = graph_from_hamiltonian(H)

    model = nk.models.RBM(alpha=1)
    occ_init = np.full(hi.n_orbitals, 0.5)

    result = iterate_occupations(
        hilbert=hi,
        graph=graph,
        occ_init=occ_init,
        model=model,
        n_stages=3,
        n_samples=256,
        n_discard_per_chain=16,
        d_max=1,
        spin_symmetric=True,
    )

    print("Graph edges:", list(graph.edges()))
    print("Occupation history:")
    for stage, occ in enumerate(result.occupations_history):
        print(f"  stage {stage}: {occ}")


if __name__ == "__main__":
    main()
