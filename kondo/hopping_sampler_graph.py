"""
Utilities to build MetropolisFermionHop sampler graphs/clusters from a
FermionOperator2nd Hamiltonian.

This script focuses on the common case where the sampler proposal should follow
the one-body hopping structure of the Hamiltonian. It supports:

1. Spinless fermions:
   Build a graph/clusters directly on the orbital indices.

2. Spinful fermions, full mode graph:
   Build a graph/clusters on the full Hilbert-mode indices
   ``0 .. hilbert.size - 1``. Use this with ``spin_symmetric=False``.

3. Spinful fermions, orbital graph:
   Compress spin-conserving hopping into an orbital-level graph with
   ``0 .. hilbert.n_orbitals - 1`` nodes. Use this with ``spin_symmetric=True``.
   This only works when the Hamiltonian contains no spin-mixing hopping terms.

Example
-------
Spinless:

    hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
    H = -1.0 * (
        nk.operator.fermion.create(hi, 0) @ nk.operator.fermion.destroy(hi, 1)
        + nk.operator.fermion.create(hi, 1) @ nk.operator.fermion.destroy(hi, 0)
    )
    g_sampler, clusters = sampler_graph_from_hamiltonian(H)
    sa = nk.sampler.MetropolisFermionHop(hi, graph=g_sampler, d_max=1)

Spinful, full-mode graph:

    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    g_sampler, clusters = sampler_graph_from_hamiltonian(H, full_mode_graph=True)
    sa = nk.sampler.MetropolisFermionHop(
        hi, graph=g_sampler, d_max=1, spin_symmetric=False
    )

Spinful, orbital graph:

    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    g_sampler, clusters = sampler_graph_from_hamiltonian(H, full_mode_graph=False)
    sa = nk.sampler.MetropolisFermionHop(
        hi, graph=g_sampler, d_max=1, spin_symmetric=True
    )

Typical call chain in this module
---------------------------------
The usual path used by callers is:

1. ``sampler_graph_from_hamiltonian(H)``
2. ``full_mode_connectivity_from_hamiltonian`` or
   ``orbital_connectivity_from_hamiltonian``
3. ``extract_hopping_pairs(H)``
4. build ``nk.graph.Graph`` and ``clusters``
5. pass the result to ``nk.sampler.MetropolisFermionHop``

Inside NetKet, the next stage is roughly:

``MetropolisFermionHop -> FermionHopRule -> ExchangeRule/compute_clusters``

This module therefore does not perform Monte Carlo updates itself. It converts
the Hamiltonian into the graph/clusters/kwargs that the sampler rule will use.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import netket as nk
import numpy as np

from netket.hilbert import SpinOrbitalFermions
from netket.operator._fermion2nd.base import FermionOperator2ndBase


Pair = tuple[int, int]


@dataclass(frozen=True)
class SamplerConnectivity:
    """
    Bundle the connectivity objects produced by this module.

    Field summary:
    - ``edges``: Python ``list[tuple[int, int]]`` describing the undirected edges.
    - ``clusters``: ``np.ndarray`` with shape usually ``(n_edges, 2)``. It stores the
      same two-site moves as ``edges``, but in array form.
    - ``g_sampler``: ``nk.graph.Graph`` ready to be passed to
      ``nk.sampler.MetropolisFermionHop(graph=...)``.
    """

    edges: list[Pair]
    clusters: np.ndarray
    g_sampler: nk.graph.Graph


def _validate_fermion_operator(H: FermionOperator2ndBase) -> SpinOrbitalFermions:
    """
    Validate that ``H`` is a supported fermionic operator and return its Hilbert space.

    Purpose:
    - Centralize the type checks used by the public helper functions in this file.

    Call chain:
    - Called first by ``extract_hopping_pairs``.
    - Called first by ``full_mode_connectivity_from_hamiltonian``.
    - Called first by ``orbital_connectivity_from_hamiltonian``.
    - Called first by ``sampler_graph_from_hamiltonian``.
    - Called first by ``sampler_kwargs_from_hamiltonian``.

    Output:
    - Returns ``H.hilbert`` as a ``SpinOrbitalFermions`` instance.
    - Raises ``TypeError`` if the operator or Hilbert type is unsupported.
    """
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
    """
    Canonicalize an edge as ``(min(i, j), max(i, j))``.

    Purpose:
    - Remove the directional ambiguity between ``(i, j)`` and ``(j, i)``.
    - Reject self-loops, because a hopping edge must connect two different nodes.

    Call chain:
    - Used by ``extract_hopping_pairs`` for full-mode pairs.
    - Used by ``orbital_connectivity_from_hamiltonian`` for orbital-level pairs.

    Output:
    - Returns a two-integer ``Pair``.
    - Raises ``ValueError`` if ``i == j``.
    """
    if i == j:
        raise ValueError("self-loops are not valid hopping pairs")
    return (i, j) if i < j else (j, i)


def extract_hopping_pairs(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
) -> list[Pair]:
    """
    Extract all one-body hopping pairs ``(i, j)`` from a fermionic Hamiltonian.

    Purpose:
    - Scan ``H.operators`` and keep only terms of the form ``c_i^dag c_j`` with
      ``i != j``.
    - Filter out small couplings with ``abs(weight) <= cutoff``.
    - Canonicalize and deduplicate the pairs before graph construction.

    Call chain:
    - This is the core low-level routine used to derive a sampler graph from ``H``.
    - Called directly by ``full_mode_connectivity_from_hamiltonian``.
    - Called directly by ``orbital_connectivity_from_hamiltonian``.
    - Reached indirectly from ``sampler_graph_from_hamiltonian`` and
      ``sampler_kwargs_from_hamiltonian``.

    Output:
    - Returns a sorted ``list[Pair]``.
    - The indices are full Hilbert mode indices by default, not compressed orbital
      indices.
    """
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


def _spin_orbital_labels(hi: SpinOrbitalFermions) -> list[tuple[int, int | None]]:
    """
    Build a human-readable label for each Hilbert mode.

    Purpose:
    - Help a reader map a flat mode index back to its physical meaning.
    - For spinless systems the labels are ``(orbital, None)``.
    - For spinful systems the labels are ``(orbital, sz)``.

    Call chain:
    - Currently used only by ``demo()`` for printed diagnostics.

    Output:
    - Returns a list whose element ``k`` describes the physical meaning of mode ``k``.
    """
    if hi.spin is None:
        return [(orb, None) for orb in range(hi.n_orbitals)]

    labels: list[tuple[int, int]] = []
    for spin_idx in range(hi.n_spin_subsectors):
        sz = -int(2 * hi.spin) + 2 * spin_idx
        for orb in range(hi.n_orbitals):
            labels.append((orb, sz))
    return labels


def full_mode_connectivity_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
) -> SamplerConnectivity:
    """
    Build sampler connectivity in the full Hilbert mode index space.

    Purpose:
    - Keep the graph at the explicit mode level without compressing spin sectors.
    - This is the natural representation for spinless systems, or for spinful systems
      where each spin-mode should remain explicit.

    Call chain:
    - Can be called directly by users.
    - Called by ``sampler_graph_from_hamiltonian`` when ``full_mode_graph=True``.
    - Also reached from ``orbital_connectivity_from_hamiltonian`` for the spinless case.
    - Internally it calls ``extract_hopping_pairs``.

    Output:
    - Returns ``SamplerConnectivity``.
    - ``edges`` and ``clusters`` use full mode indices.
    - ``g_sampler`` has ``hi.size`` nodes.
    - The result is meant to be paired with
      ``MetropolisFermionHop(..., spin_symmetric=False)``.
    """
    hi = _validate_fermion_operator(H)
    edges = extract_hopping_pairs(H, cutoff=cutoff, normal_order=normal_order)
    clusters = np.asarray(edges, dtype=np.int64)
    g_sampler = nk.graph.Graph(edges=edges, n_nodes=hi.size)
    return SamplerConnectivity(edges=edges, clusters=clusters, g_sampler=g_sampler)


def orbital_connectivity_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
    require_identical_spin_blocks: bool = False,
) -> SamplerConnectivity:
    """
    Build an orbital-level sampler graph for spinful, spin-conserving Hamiltonians.

    Purpose:
    - Compress the connectivity so that only orbital-to-orbital edges remain.
    - This is more compact when all spin subsectors share the same hopping pattern.
    - When later used with ``MetropolisFermionHop(..., spin_symmetric=True)``,
      NetKet replicates this orbital graph across spin subsectors internally.

    Call chain:
    - Can be called directly by users.
    - Called by ``sampler_graph_from_hamiltonian`` when ``full_mode_graph=False``.
    - Internally it calls ``extract_hopping_pairs`` and then compresses full-mode
      edges down to orbital edges.

    Output:
    - Returns ``SamplerConnectivity``.
    - ``edges`` and ``clusters`` contain orbital index pairs in
      ``0 .. hi.n_orbitals - 1``.
    - ``g_sampler`` has ``hi.n_orbitals`` nodes.
    - Raises ``ValueError`` if spin-mixing hopping is present.
    - Raises ``ValueError`` if ``require_identical_spin_blocks=True`` and different
      spin sectors induce different orbital connectivities.
    """
    hi = _validate_fermion_operator(H)
    if hi.n_spin_subsectors == 1:
        return full_mode_connectivity_from_hamiltonian(
            H, cutoff=cutoff, normal_order=normal_order
        )

    pairs = extract_hopping_pairs(H, cutoff=cutoff, normal_order=normal_order)
    orbital_pairs: set[Pair] = set()
    per_spin_pairs: dict[int, set[Pair]] = defaultdict(set)

    for i, j in pairs:
        spin_i, orb_i = divmod(i, hi.n_orbitals)
        spin_j, orb_j = divmod(j, hi.n_orbitals)
        if spin_i != spin_j:
            raise ValueError(
                "Found spin-mixing hopping terms. Cannot compress to a single orbital "
                "graph for spin_symmetric=True. Use full_mode_connectivity_from_hamiltonian "
                "and spin_symmetric=False instead."
            )

        pair = _canonical_pair(orb_i, orb_j)
        orbital_pairs.add(pair)
        per_spin_pairs[spin_i].add(pair)

    if require_identical_spin_blocks and per_spin_pairs:
        reference = next(iter(per_spin_pairs.values()))
        for spin_idx, spin_pairs in per_spin_pairs.items():
            if spin_pairs != reference:
                raise ValueError(
                    "Different spin subsectors induce different orbital graphs. "
                    "Cannot safely use a single graph with spin_symmetric=True. "
                    f"Spin block {spin_idx} differs from the reference block."
                )

    edges = sorted(orbital_pairs)
    clusters = np.asarray(edges, dtype=np.int64)
    g_sampler = nk.graph.Graph(edges=edges, n_nodes=hi.n_orbitals)
    return SamplerConnectivity(edges=edges, clusters=clusters, g_sampler=g_sampler)


def sampler_graph_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
    full_mode_graph: bool | None = None,
    require_identical_spin_blocks: bool = False,
) -> SamplerConnectivity:
    """
    Choose between a full-mode graph and an orbital-level graph automatically.

    Purpose:
    - Provide a single public entry point for most callers.
    - Default policy:
      - spinless -> full-mode graph
      - spinful -> orbital graph, intended for ``spin_symmetric=True``

    Call chain:
    - This is the main convenience entry point in the module.
    - Called directly by ``sampler_kwargs_from_hamiltonian``.
    - Can also be called directly by user code before constructing a sampler.

    Output:
    - Returns ``SamplerConnectivity``.
    - The exact representation depends on whether it dispatches to
      ``full_mode_connectivity_from_hamiltonian`` or
      ``orbital_connectivity_from_hamiltonian``.
    """
    hi = _validate_fermion_operator(H)

    if full_mode_graph is None:
        full_mode_graph = hi.n_spin_subsectors == 1

    if full_mode_graph:
        return full_mode_connectivity_from_hamiltonian(
            H, cutoff=cutoff, normal_order=normal_order
        )
    else:
        return orbital_connectivity_from_hamiltonian(
            H,
            cutoff=cutoff,
            normal_order=normal_order,
            require_identical_spin_blocks=require_identical_spin_blocks,
        )


def hopping_graph_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
    full_mode_graph: bool | None = None,
    require_identical_spin_blocks: bool = False,
) -> nk.graph.Graph:
    """
    Build and return only the hopping graph extracted from ``H``.

    Purpose:
    - Provide the shortest public interface in this module for the common case
      "input a FermionOperator2nd Hamiltonian, output the graph connecting all
      orbitals/modes that have hopping terms".
    - Reuse ``sampler_graph_from_hamiltonian`` so the graph convention stays
      identical to the rest of this file.

    Call chain:
    - Thin wrapper over ``sampler_graph_from_hamiltonian``.
    - For spinless systems, the default is a full-mode graph.
    - For spinful systems, the default is an orbital graph suitable for
      ``spin_symmetric=True``.

    Output:
    - Returns an ``nk.graph.Graph``.
    - To also access ``edges`` and ``clusters``, call
      ``sampler_graph_from_hamiltonian`` instead.
    """
    return sampler_graph_from_hamiltonian(
        H,
        cutoff=cutoff,
        normal_order=normal_order,
        full_mode_graph=full_mode_graph,
        require_identical_spin_blocks=require_identical_spin_blocks,
    ).g_sampler


def clusters_from_graph(graph: nk.graph.AbstractGraph, d_max: int = 1) -> np.ndarray:
    """
    Explicitly generate the two-site clusters used by the sampler rules.

    Purpose:
    - Expand an ``nk.graph.Graph`` into the actual ``(i, j)`` cluster array used by
      ``ExchangeRule`` and ``FermionHopRule``.
    - Useful for debugging or for checking what the sampler will really see.

    Call chain:
    - Thin wrapper over ``netket.sampler.rules.exchange.compute_clusters``.
    - In this file it is used mainly by ``demo()``.

    Output:
    - Returns an ``np.ndarray`` of shape ``(n_clusters, 2)``.
    """
    from netket.sampler.rules.exchange import compute_clusters

    return compute_clusters(graph, d_max)


def sampler_kwargs_from_hamiltonian(
    H: FermionOperator2ndBase,
    *,
    cutoff: float = 1.0e-10,
    normal_order: bool = True,
    full_mode_graph: bool | None = None,
    require_identical_spin_blocks: bool = False,
) -> dict:
    """
    Build keyword arguments ready to be unpacked into ``MetropolisFermionHop``.

    Purpose:
    - Provide a one-call way to obtain the graph convention and matching sampler flags.

    Call chain:
    - This is another public convenience entry point.
    - Internally it first calls ``sampler_graph_from_hamiltonian`` and then assembles
      the sampler kwargs.

    Output:
    - Returns a ``dict`` containing:
      - ``graph``: an ``nk.graph.Graph``
      - ``d_max``: currently fixed to ``1``
      - ``spin_symmetric``: chosen to match the selected graph convention
    - The return value can be used as
      ``nk.sampler.MetropolisFermionHop(hi, **sampler_kwargs_from_hamiltonian(H))``.
    """
    hi = _validate_fermion_operator(H)
    conn = sampler_graph_from_hamiltonian(
        H,
        cutoff=cutoff,
        normal_order=normal_order,
        full_mode_graph=full_mode_graph,
        require_identical_spin_blocks=require_identical_spin_blocks,
    )

    if full_mode_graph is None:
        full_mode_graph = hi.n_spin_subsectors == 1

    return {
        "graph": conn.g_sampler,
        "d_max": 1,
        "spin_symmetric": False if full_mode_graph else True,
    }


def _print_connectivity_summary(title: str, conn: SamplerConnectivity) -> None:
    """
    Print a readable summary of a ``SamplerConnectivity`` object.

    Purpose:
    - Make the key connectivity outputs easy to inspect in demos and debugging.

    Call chain:
    - Currently used only by ``demo()``.

    Output:
    - Returns ``None``.
    - Prints the edge count, edge list, graph size, graph edges, and clusters array.
    """
    print(f"\n{title}")
    print(f"  n_edges     = {len(conn.edges)}")
    print(f"  edges       = {conn.edges}")
    print(f"  graph nodes = {conn.g_sampler.n_nodes}")
    print(f"  graph edges = {list(conn.g_sampler.edges())}")
    print(f"  clusters    =\n{conn.clusters}")


def _build_demo_spinless() -> tuple[SpinOrbitalFermions, FermionOperator2ndBase]:
    """
    Construct a small spinless example system for the demos.

    Purpose:
    - Provide a simple ``hi`` and hopping Hamiltonian ``H`` used to show how the
      graph extraction works.

    Call chain:
    - Used only by ``demo()``.

    Output:
    - Returns ``(hi, H)`` where ``hi`` is a ``SpinOrbitalFermions`` instance and
      ``H`` is a compatible ``FermionOperator2ndBase`` Hamiltonian.
    """
    hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
    c = nk.operator.fermion.create
    a = nk.operator.fermion.destroy

    H = (
        -1.0 * (c(hi, 0) @ a(hi, 1) + c(hi, 1) @ a(hi, 0))
        - 0.7 * (c(hi, 1) @ a(hi, 2) + c(hi, 2) @ a(hi, 1))
        - 0.3 * (c(hi, 0) @ a(hi, 3) + c(hi, 3) @ a(hi, 0))
    )
    return hi, H


def _build_demo_spinful() -> tuple[SpinOrbitalFermions, FermionOperator2ndBase]:
    """
    Construct a spin-conserving spinful example system for the demos.

    Purpose:
    - Give each spin subsector the same hopping pattern so that both the full-mode
      and orbital-level graph constructions can be demonstrated clearly.

    Call chain:
    - Used only by ``demo()``.

    Output:
    - Returns ``(hi, H)`` where ``hi`` is spinful and ``H`` contains the hopping
      terms for both spin blocks.
    """
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    c = nk.operator.fermion.create
    a = nk.operator.fermion.destroy

    H = 0.0
    for sz in (-1, +1):
        H += -1.0 * (c(hi, 0, sz=sz) @ a(hi, 1, sz=sz) + c(hi, 1, sz=sz) @ a(hi, 0, sz=sz))
        H += -0.7 * (c(hi, 1, sz=sz) @ a(hi, 2, sz=sz) + c(hi, 2, sz=sz) @ a(hi, 1, sz=sz))
        H += -0.3 * (c(hi, 0, sz=sz) @ a(hi, 3, sz=sz) + c(hi, 3, sz=sz) @ a(hi, 0, sz=sz))
    return hi, H


def hopping_matrix_to_fermion_operator(
    hi: SpinOrbitalFermions,
    hopping: np.ndarray,
    *,
    sz: int | None = None,
) -> FermionOperator2ndBase:
    """
    Convert a single-particle hopping matrix into a second-quantized fermion operator.

    Purpose:
    - Implement ``sum_ij hopping[i,j] c_i^dag c_j`` in NetKet operator form.
    - Make it easy to go from a matrix representation to a Hamiltonian that this
      module can process.

    Call chain:
    - Used by ``transformed_chain_demo()``.
    - Also useful as a general helper when constructing test Hamiltonians.

    Output:
    - Returns a ``FermionOperator2ndBase``-compatible operator ``H``.
    - For spinless systems, use ``sz=None``.
    - For spinful systems, passing a concrete ``sz`` fills only that spin subsector.
    """
    if hopping.shape != (hi.n_orbitals, hi.n_orbitals):
        raise ValueError(
            "hopping must have shape (n_orbitals, n_orbitals), got "
            f"{hopping.shape} for n_orbitals={hi.n_orbitals}."
        )

    H = nk.operator.fermion.zero(hi, dtype=complex)
    for i in range(hi.n_orbitals):
        for j in range(hi.n_orbitals):
            tij = hopping[i, j]
            if abs(tij) > 0:
                H = H + tij * (
                    nk.operator.fermion.create(hi, i, sz=sz)
                    @ nk.operator.fermion.destroy(hi, j, sz=sz)
                )
    return H


def random_unitary(n: int, seed: int = 1234) -> np.ndarray:
    """
    Generate a deterministic random unitary matrix using QR factorization.

    Purpose:
    - Create a reproducible orbital basis rotation for the transformed-chain demo.

    Call chain:
    - Used by ``transformed_chain_demo()``.

    Output:
    - Returns a complex ``np.ndarray`` of shape ``(n, n)`` that is approximately
      unitary.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q * np.conjugate(phases)


def nearest_neighbor_chain_hopping(n_orbitals: int, t: float = 1.0) -> np.ndarray:
    """
    Build the open-boundary nearest-neighbor hopping matrix of a 1D chain.

    Purpose:
    - Provide the standard tight-binding single-particle matrix used in the demo.

    Call chain:
    - Used by ``transformed_chain_demo()``.

    Output:
    - Returns a complex matrix ``h`` with shape ``(n_orbitals, n_orbitals)``.
    """
    h = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for i in range(n_orbitals - 1):
        h[i, i + 1] = -t
        h[i + 1, i] = -t
    return h


def transformed_chain_demo(
    *,
    n_orbitals: int = 4,
    t: float = 1.0,
    seed: int = 7,
    cutoff: float = 0.15,
) -> None:
    """
    Demonstrate the full workflow from a chain Hamiltonian to an extracted sampler graph.

    Purpose:
    - Start from a nearest-neighbor hopping matrix ``h_old``.
    - Rotate it with a random unitary to obtain ``h_new`` in a new orbital basis.
    - Convert ``h_new`` into a fermionic Hamiltonian and extract the sampler graph.

    Call chain:
    - Calls ``nearest_neighbor_chain_hopping``.
    - Then calls ``random_unitary``.
    - Then calls ``hopping_matrix_to_fermion_operator``.
    - Then calls ``sampler_graph_from_hamiltonian``.
    - Invoked at the end of ``demo()``.

    Output:
    - Returns ``None``.
    - Prints ``h_old``, ``U``, ``h_new``, the transformed operator string, the
      extracted sampler ``edges / graph / clusters``, and an example sampler call.
    """
    print("\n=== Random-unitary transformed chain demo ===")
    h_old = nearest_neighbor_chain_hopping(n_orbitals, t=t)
    U = random_unitary(n_orbitals, seed=seed)
    h_new = U.conj().T @ h_old @ U

    print("Original hopping matrix h_old:")
    print(np.array2string(h_old, precision=3, suppress_small=True))
    print("\nRandom unitary U:")
    print(np.array2string(U, precision=3, suppress_small=True))
    print("\nTransformed hopping matrix h_new = U^dag h_old U:")
    print(np.array2string(h_new, precision=3, suppress_small=True))

    hi = nk.hilbert.SpinOrbitalFermions(n_orbitals, n_fermions=n_orbitals // 2)
    H_new = hopping_matrix_to_fermion_operator(hi, h_new)

    print("\nTransformed Hamiltonian operator string:")
    print(H_new.operator_string())

    conn = sampler_graph_from_hamiltonian(H_new, cutoff=cutoff, full_mode_graph=True)
    print(f"\nSampler graph extracted with cutoff={cutoff}:")
    print("  edges =", conn.edges)
    print("  graph =", conn.g_sampler)
    print("  clusters =")
    print(conn.clusters)
    print(
        "\nSampler usage:"
        "\n  sa = nk.sampler.MetropolisFermionHop("
        "hi, graph=conn.g_sampler, d_max=1)"
    )


def demo() -> None:
    """
    Run all main demonstrations contained in this file.

    Purpose:
    - Show the spinless example.
    - Show the spinful full-mode example.
    - Show the spinful orbital-level example.
    - Print the mode labels.
    - Show explicit clusters from a graph.
    - Run the transformed-chain example.

    Call chain:
    - This is the top-level script entry point.
    - It is executed by ``if __name__ == "__main__":``.

    Output:
    - Returns ``None``.
    - Prints the intermediate connectivity objects and example sampler usages.
    """
    print("=== Spinless example ===")
    hi_sl, H_sl = _build_demo_spinless()
    conn_sl = sampler_graph_from_hamiltonian(H_sl)
    _print_connectivity_summary("spinless connectivity", conn_sl)
    print(
        "  sampler usage = nk.sampler.MetropolisFermionHop("
        "hi_sl, graph=conn_sl.g_sampler, d_max=1)"
    )

    print("\n=== Spinful example ===")
    hi_sf, H_sf = _build_demo_spinful()

    conn_sf_full = full_mode_connectivity_from_hamiltonian(H_sf)
    _print_connectivity_summary("spinful full-mode connectivity", conn_sf_full)
    print(
        "  sampler usage = nk.sampler.MetropolisFermionHop("
        "hi_sf, graph=conn_sf_full.g_sampler, d_max=1, spin_symmetric=False)"
    )

    conn_sf_orb = orbital_connectivity_from_hamiltonian(H_sf)
    _print_connectivity_summary("spinful orbital connectivity", conn_sf_orb)
    print(
        "  sampler usage = nk.sampler.MetropolisFermionHop("
        "hi_sf, graph=conn_sf_orb.g_sampler, d_max=1, spin_symmetric=True)"
    )

    print("\nMode index labels for the spinful example:")
    for mode, (orb, sz) in enumerate(_spin_orbital_labels(hi_sf)):
        print(f"  mode {mode}: orbital={orb}, sz={sz}")

    print("\nExplicit clusters from the orbital graph with d_max=1:")
    print(clusters_from_graph(conn_sf_orb.g_sampler, d_max=1))

    transformed_chain_demo()


if __name__ == "__main__":
    demo()
