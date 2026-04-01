# Copyright 2026 Ping Xu - All rights reserved.

"""
Minimal helper to build a hopping graph directly from a FermionOperator2nd Hamiltonian.
"""

from __future__ import annotations

import netket as nk

from netket.hilbert import SpinOrbitalFermions
from netket.operator._fermion2nd.base import FermionOperator2ndBase


Pair = tuple[int, int]


def _validate_fermion_operator(H: FermionOperator2ndBase) -> SpinOrbitalFermions:
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "H must be a netket FermionOperator2nd-compatible operator, "
            f"got {type(H)}."
        )
    if not isinstance(H.hilbert, SpinOrbitalFermions):
        raise TypeError(
            "H.hilbert must be SpinOrbitalFermions, "
            f"got {type(H.hilbert)}."
        )
    return H.hilbert


def _canonical_pair(i: int, j: int) -> Pair:
    if i == j:
        raise ValueError("self-loops are not valid hopping pairs")
    return (i, j) if i < j else (j, i)


def _extract_hopping_pairs(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
) -> list[Pair]:
    _validate_fermion_operator(H)
    op = H.to_normal_order() if normal_order else H

    pairs: set[Pair] = set()
    for term, weight in op.operators.items():
        if abs(weight) <= cutoff:
            continue
        if len(term) != 2:
            continue

        (i, d1), (j, d2) = term
        if d1 == 1 and d2 == 0 and i != j:
            pairs.add(_canonical_pair(int(i), int(j)))

    return sorted(pairs)


def graph_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
    full_mode_graph: bool | None = None,
    require_identical_spin_blocks: bool = False,
) -> nk.graph.Graph:
    """
    Build the hopping graph associated with ``H``.

    For spinless systems, the graph nodes are the Hilbert modes.
    For spinful systems, the default is an orbital graph suitable for
    ``MetropolisFermionHop(..., spin_symmetric=True)``. Set
    ``full_mode_graph=True`` to keep the full mode indexing instead.
    """
    hi = _validate_fermion_operator(H)

    if full_mode_graph is None:
        full_mode_graph = hi.n_spin_subsectors == 1

    pairs = _extract_hopping_pairs(H, cutoff=cutoff, normal_order=normal_order)

    if full_mode_graph or hi.n_spin_subsectors == 1:
        return nk.graph.Graph(edges=pairs, n_nodes=hi.size)

    orbital_pairs: set[Pair] = set()
    per_spin_pairs: dict[int, set[Pair]] = {}

    for i, j in pairs:
        spin_i, orb_i = divmod(i, hi.n_orbitals)
        spin_j, orb_j = divmod(j, hi.n_orbitals)
        if spin_i != spin_j:
            raise ValueError(
                "Found spin-mixing hopping terms. Cannot compress to a single orbital "
                "graph. Use full_mode_graph=True instead."
            )

        pair = _canonical_pair(orb_i, orb_j)
        orbital_pairs.add(pair)
        per_spin_pairs.setdefault(spin_i, set()).add(pair)

    if require_identical_spin_blocks and per_spin_pairs:
        reference = next(iter(per_spin_pairs.values()))
        for spin_idx, spin_pairs in per_spin_pairs.items():
            if spin_pairs != reference:
                raise ValueError(
                    "Different spin subsectors induce different orbital graphs. "
                    f"Spin block {spin_idx} differs from the reference block."
                )

    return nk.graph.Graph(edges=sorted(orbital_pairs), n_nodes=hi.n_orbitals)
