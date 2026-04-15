"""
Generalized 2D Kondo-Heisenberg ladder for NetKet.

Modeling choice:
- The real-space geometry is a true ``Lx x n_legs`` rectangular ladder.
- Each spatial site ``(x, leg)`` carries one spinful conduction orbital and
  one local spin-1/2 moment.
- For NetKet indexing we only flatten ``(x, leg) -> x * n_legs + leg``.

Bond shells:
- ``shell_1``: nearest neighbors ``(dx, dy) = (1, 0)`` and ``(0, 1)``
- ``shell_2``: plaquette diagonals ``(dx, dy) = (1, 1)`` and ``(1, -1)``
- ``shell_3``: axial third neighbors ``(dx, dy) = (2, 0)`` and ``(0, 2)``

Boundary conditions:
- ``pbc_x`` can be open or periodic.
- ``pbc_y`` is allowed only when ``n_legs > 2``.

Sector note:
- For ``J_K != 0``, the full Kondo term contains spin-flip pieces and therefore
  does not conserve ``n_fermions_per_spin`` or the local-spin ``total_sz``
  separately. In that regime, use ``n_fermions`` to fix only the total
  conduction-electron number, and seed the sampler in the desired joint
  ``S^z`` sector with ``seed_joint_sz_sector(..., two_sz=0)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp
import netket as nk

from netket.jax.sharding import shard_along_axis
from netket.operator import EmbedOperator, SumOperator
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc
from netket.operator.spin import sigmam, sigmap, sigmaz


Bond = tuple[int, int]


def ladder_index(x: int, leg: int, n_legs: int) -> int:
    """Flatten ``(x, leg)`` into a single spatial-site index."""
    if n_legs < 1:
        raise ValueError(f"n_legs must be >= 1, got {n_legs}.")
    if x < 0:
        raise ValueError(f"x must be >= 0, got {x}.")
    if leg < 0 or leg >= n_legs:
        raise ValueError(f"leg must satisfy 0 <= leg < {n_legs}, got {leg}.")
    return x * n_legs + leg


def _resolve_fermion_hilbert_kwargs(
    n_sites: int,
    n_fermions: int | None,
    n_fermions_per_spin: tuple[int, int] | None,
) -> dict[str, Any]:
    if n_fermions is not None and n_fermions_per_spin is not None:
        raise ValueError(
            "Pass either n_fermions or n_fermions_per_spin, not both."
        )

    if n_fermions_per_spin is not None:
        if len(n_fermions_per_spin) != 2:
            raise ValueError(
                "n_fermions_per_spin must have length 2, "
                f"got {n_fermions_per_spin}."
            )
        n_up = int(n_fermions_per_spin[0])
        n_dn = int(n_fermions_per_spin[1])
        if not (0 <= n_up <= n_sites and 0 <= n_dn <= n_sites):
            raise ValueError(
                "Each spin-sector particle number must satisfy "
                f"0 <= N_sigma <= n_sites={n_sites}, got {(n_up, n_dn)}."
            )
        return {"n_fermions_per_spin": (n_up, n_dn)}

    if n_fermions is not None:
        n_total = int(n_fermions)
        if not (0 <= n_total <= 2 * n_sites):
            raise ValueError(
                "n_fermions must satisfy "
                f"0 <= n_fermions <= 2 * n_sites = {2 * n_sites}, got {n_total}."
            )
        return {"n_fermions": n_total}

    if n_sites % 2 != 0:
        raise ValueError(
            "n_sites must be even to use the default half-filled choice "
            "n_fermions = n_sites. Pass n_fermions explicitly for odd systems."
        )

    return {"n_fermions": n_sites}


def _validate_geometry(Lx: int, n_legs: int, *, pbc_y: bool = False) -> None:
    if Lx < 1:
        raise ValueError(f"Lx must be >= 1, got {Lx}.")
    if n_legs < 1:
        raise ValueError(f"n_legs must be >= 1, got {n_legs}.")
    if pbc_y and n_legs <= 2:
        raise ValueError(
            "Periodic boundary conditions along the leg direction are only "
            f"allowed when n_legs > 2. Got n_legs={n_legs}."
        )


def _generate_bonds(
    Lx: int,
    n_legs: int,
    *,
    dx: int,
    dy: int,
    pbc_x: bool,
    pbc_y: bool,
) -> list[Bond]:
    bonds: set[Bond] = set()

    for x in range(Lx):
        for leg in range(n_legs):
            x2 = x + dx
            leg2 = leg + dy

            if pbc_x:
                x2 %= Lx
            elif not (0 <= x2 < Lx):
                continue

            if pbc_y:
                leg2 %= n_legs
            elif not (0 <= leg2 < n_legs):
                continue

            i = ladder_index(x, leg, n_legs)
            j = ladder_index(x2, leg2, n_legs)
            if i != j:
                bonds.add((i, j) if i < j else (j, i))

    return sorted(bonds)


def _merge_bonds(*groups: list[Bond]) -> list[Bond]:
    merged: set[Bond] = set()
    for group in groups:
        merged.update(group)
    return sorted(merged)


def kondo_heisenberg_geometry(
    *,
    Lx: int,
    n_legs: int,
    pbc_x: bool = False,
    pbc_y: bool = False,
) -> dict[str, Any]:
    """
    Return the shared 2D geometry used by the Hamiltonian and by the plotter.

    The shell convention is:
    - ``shell_1``: nearest neighbors along ``x`` and along ``leg``
    - ``shell_2``: plaquette diagonals
    - ``shell_3``: axial third neighbors, i.e. two steps along ``x`` or ``leg``
    """
    _validate_geometry(Lx, n_legs, pbc_y=pbc_y)

    shell_1_x = _generate_bonds(Lx, n_legs, dx=1, dy=0, pbc_x=pbc_x, pbc_y=False)
    shell_1_y = _generate_bonds(Lx, n_legs, dx=0, dy=1, pbc_x=False, pbc_y=pbc_y)
    shell_2_up = _generate_bonds(Lx, n_legs, dx=1, dy=1, pbc_x=pbc_x, pbc_y=pbc_y)
    shell_2_dn = _generate_bonds(Lx, n_legs, dx=1, dy=-1, pbc_x=pbc_x, pbc_y=pbc_y)
    shell_3_x = _generate_bonds(Lx, n_legs, dx=2, dy=0, pbc_x=pbc_x, pbc_y=False)
    shell_3_y = _generate_bonds(Lx, n_legs, dx=0, dy=2, pbc_x=False, pbc_y=pbc_y)

    return {
        "Lx": Lx,
        "n_legs": n_legs,
        "n_sites": Lx * n_legs,
        "pbc_x": pbc_x,
        "pbc_y": pbc_y,
        "shell_1_x": shell_1_x,
        "shell_1_y": shell_1_y,
        "shell_1": _merge_bonds(shell_1_x, shell_1_y),
        "shell_2_up": shell_2_up,
        "shell_2_dn": shell_2_dn,
        "shell_2": _merge_bonds(shell_2_up, shell_2_dn),
        "shell_3_x": shell_3_x,
        "shell_3_y": shell_3_y,
        "shell_3": _merge_bonds(shell_3_x, shell_3_y),
    }


def build_kondo_heisenberg_hilbert(
    *,
    Lx: int,
    n_legs: int,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[
    nk.hilbert.TensorHilbert,
    nk.hilbert.SpinOrbitalFermions,
    nk.hilbert.Spin,
]:
    """Build the tensor-product Hilbert space of the generalized ladder."""
    _validate_geometry(Lx, n_legs)

    n_sites = Lx * n_legs
    fermion_kwargs = _resolve_fermion_hilbert_kwargs(
        n_sites,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    fermion_hi = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        **fermion_kwargs,
    )
    local_spin_hi = nk.hilbert.Spin(
        s=1 / 2,
        N=n_sites,
        total_sz=local_total_sz,
    )
    joint_hi = fermion_hi * local_spin_hi

    return joint_hi, fermion_hi, local_spin_hi


def _xxz_spin_bond(
    hi: nk.hilbert.Spin,
    i: int,
    j: int,
    *,
    delta_z: float,
) -> Any:
    return (
        0.5 * (sigmap(hi, i) @ sigmam(hi, j) + sigmam(hi, i) @ sigmap(hi, j))
        + 0.25 * float(delta_z) * (sigmaz(hi, i) @ sigmaz(hi, j))
    )


def _sum_terms(terms: list[Any], coefficients: list[float] | None = None) -> Any | None:
    if not terms:
        return None
    if coefficients is None:
        return terms[0] if len(terms) == 1 else SumOperator(*terms)
    if len(terms) == 1:
        return coefficients[0] * terms[0]
    return SumOperator(*terms, coefficients=coefficients)


def joint_total_two_sz_operator(
    joint_hi: nk.hilbert.TensorHilbert,
    fermion_hi: nk.hilbert.SpinOrbitalFermions,
    local_spin_hi: nk.hilbert.Spin,
) -> Any:
    """Return the operator for ``2 S^z_total`` of electrons plus local spins."""
    total = 0.0
    for site in range(fermion_hi.n_orbitals):
        total += EmbedOperator(
            joint_hi,
            nc(fermion_hi, site, sz=+1) - nc(fermion_hi, site, sz=-1),
            subspace=0,
        )
        total += EmbedOperator(joint_hi, sigmaz(local_spin_hi, site), subspace=1)
    return total


def joint_total_two_sz_of_states(samples: Any, n_sites: int) -> Any:
    """Compute ``2 S^z_total`` for raw TensorHilbert samples."""
    samples = jnp.asarray(samples)
    n_dn = samples[..., :n_sites]
    n_up = samples[..., n_sites : 2 * n_sites]
    sigma_loc = samples[..., 2 * n_sites : 3 * n_sites]
    return jnp.sum(n_up - n_dn, axis=-1) + jnp.sum(sigma_loc, axis=-1)


def _resolve_joint_sector_composition(
    n_sites: int,
    n_fermions: int,
    two_sz: int,
) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []

    min_m_e = max(-n_fermions, n_fermions - 2 * n_sites)
    max_m_e = min(n_fermions, 2 * n_sites - n_fermions)

    start = min_m_e
    if (n_fermions + start) % 2 != 0:
        start += 1

    for m_e in range(start, max_m_e + 1, 2):
        n_up = (n_fermions + m_e) // 2
        n_dn = n_fermions - n_up
        if not (0 <= n_up <= n_sites and 0 <= n_dn <= n_sites):
            continue

        m_loc = int(two_sz) - m_e
        if abs(m_loc) > n_sites:
            continue
        if (n_sites + m_loc) % 2 != 0:
            continue

        n_loc_up = (n_sites + m_loc) // 2
        candidates.append((abs(m_e), abs(m_loc), n_up, n_dn, n_loc_up))

    if not candidates:
        raise ValueError(
            "Could not realize the requested joint sector with the given system size "
            f"and fermion number: n_sites={n_sites}, n_fermions={n_fermions}, two_sz={two_sz}."
        )

    _, _, n_up, n_dn, n_loc_up = min(candidates)
    return n_up, n_dn, n_loc_up


def joint_sector_reference_state(
    *,
    n_sites: int,
    n_fermions: int,
    two_sz: int = 0,
    dtype: Any = np.int8,
) -> np.ndarray:
    """
    Return a deterministic product configuration in the chosen joint ``2 S^z`` sector.

    State layout follows the TensorHilbert ordering used here:
    - ``[0:n_sites]``: electron ``down`` occupations
    - ``[n_sites:2*n_sites]``: electron ``up`` occupations
    - ``[2*n_sites:3*n_sites]``: local-spin ``sigma^z`` values ``(+1/-1)``
    """
    n_up, n_dn, n_loc_up = _resolve_joint_sector_composition(
        n_sites,
        int(n_fermions),
        int(two_sz),
    )

    state = np.empty(3 * n_sites, dtype=dtype)
    state[:n_sites] = 0
    state[n_sites : 2 * n_sites] = 0
    state[2 * n_sites :] = -1

    state[:n_dn] = 1
    state[n_sites : n_sites + n_up] = 1
    state[2 * n_sites : 2 * n_sites + n_loc_up] = 1

    return state


def seed_joint_sz_sector(vstate: Any, *, two_sz: int = 0) -> np.ndarray:
    """
    Seed an MCState in a fixed joint ``2 S^z`` sector without changing the Hilbert space.

    This is the practical way to work in a fixed joint ``S^z`` sector for the full
    Kondo model in the current TensorHilbert setup: use a Hilbert space constrained
    only by ``n_fermions``, then start the Hamiltonian sampler from a configuration
    with the desired joint ``2 S^z``. Because the Hamiltonian preserves that quantum
    number, subsequent Hamiltonian-driven moves remain in the same sector.
    """
    sampler = vstate.sampler
    if getattr(sampler, "reset_chains", False):
        raise ValueError(
            "seed_joint_sz_sector requires sampler.reset_chains=False, otherwise "
            "the seeded sector would be overwritten on reset()."
        )

    joint_hi = vstate.hilbert
    if not isinstance(joint_hi, nk.hilbert.TensorHilbert) or len(joint_hi.subspaces) != 2:
        raise TypeError(
            "seed_joint_sz_sector expects a TensorHilbert with fermion and local-spin subspaces."
        )

    fermion_hi, local_spin_hi = joint_hi.subspaces
    if not isinstance(fermion_hi, nk.hilbert.SpinOrbitalFermions):
        raise TypeError("The first TensorHilbert subspace must be SpinOrbitalFermions.")
    if not isinstance(local_spin_hi, nk.hilbert.Spin):
        raise TypeError("The second TensorHilbert subspace must be Spin.")
    if fermion_hi.n_fermions is None:
        raise ValueError(
            "seed_joint_sz_sector requires a fixed total conduction-electron number. "
            "Construct the Hilbert space with n_fermions=... ."
        )

    reference = joint_sector_reference_state(
        n_sites=fermion_hi.n_orbitals,
        n_fermions=int(fermion_hi.n_fermions),
        two_sz=two_sz,
        dtype=np.asarray(0, dtype=vstate.sampler_state.σ.dtype).dtype,
    )
    sigma = np.repeat(reference[None, :], sampler.n_batches, axis=0)
    sigma = shard_along_axis(jnp.asarray(sigma, dtype=vstate.sampler_state.σ.dtype), axis=0)

    vstate.sampler_state = vstate.sampler_state.replace(σ=sigma)
    vstate.reset()

    return reference


def KondoHeisenberg(
    *,
    Lx: int,
    n_legs: int = 2,
    t1: float = 1.0,
    t2: float = 0.0,
    t3: float = 0.0,
    J_K: float = 1.0,
    J1: float = 1.0,
    J2: float = 0.0,
    J3: float = 0.0,
    delta_z: float = 1.0,
    pbc_x: bool = False,
    pbc_y: bool = False,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[Any, nk.hilbert.TensorHilbert]:
    """
    Construct the generalized 2D Kondo-Heisenberg ladder and return ``(H, hi)``.

    Shell convention:
    - ``t1`` / ``J1``: nearest neighbors ``(1, 0)`` and ``(0, 1)``
    - ``t2`` / ``J2``: plaquette diagonals ``(1, 1)`` and ``(1, -1)``
    - ``t3`` / ``J3``: axial third neighbors ``(2, 0)`` and ``(0, 2)``

    Hamiltonian:

        H = -sum_n t_n sum_{<ij>_n,sigma} (c^dag_{i,sigma} c_{j,sigma} + h.c.)
            + J_K sum_i s_i . S_i
            + sum_n J_n sum_{<ij>_n} (S_i^x S_j^x + S_i^y S_j^y + delta_z S_i^z S_j^z)

    Sector guidance:
    - For ``J_K = 0``, the optional ``n_fermions_per_spin`` and ``local_total_sz``
      constraints are still valid.
    - For ``J_K != 0``, use ``n_fermions`` instead. To work in a fixed joint
      ``S^z`` sector, seed the sampler with ``seed_joint_sz_sector(vstate, two_sz=0)``.
    """
    J_K = float(J_K)
    if J_K != 0.0 and n_fermions_per_spin is not None:
        raise ValueError(
            "n_fermions_per_spin is not compatible with the full Kondo spin-flip term. "
            "For J_K != 0, use n_fermions=... and seed the desired joint S^z sector "
            "with seed_joint_sz_sector(..., two_sz=0)."
        )
    if J_K != 0.0 and local_total_sz is not None:
        raise ValueError(
            "local_total_sz is not compatible with the full Kondo spin-flip term. "
            "For J_K != 0, constrain only n_fermions and seed the desired joint S^z sector."
        )

    joint_hi, fermion_hi, local_spin_hi = build_kondo_heisenberg_hilbert(
        Lx=Lx,
        n_legs=n_legs,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )

    geometry = kondo_heisenberg_geometry(Lx=Lx, n_legs=n_legs, pbc_x=pbc_x, pbc_y=pbc_y)
    n_sites = geometry["n_sites"]

    fermion_terms: list[Any] = []
    fermion_coeffs: list[float] = []
    for bonds, hopping in (
        (geometry["shell_1"], float(t1)),
        (geometry["shell_2"], float(t2)),
        (geometry["shell_3"], float(t3)),
    ):
        if hopping == 0.0 or not bonds:
            continue
        for i, j in bonds:
            for sz in (+1, -1):
                fermion_terms.append(cdag(fermion_hi, i, sz=sz) @ c(fermion_hi, j, sz=sz))
                fermion_coeffs.append(-hopping)
                fermion_terms.append(cdag(fermion_hi, j, sz=sz) @ c(fermion_hi, i, sz=sz))
                fermion_coeffs.append(-hopping)

    local_terms: list[Any] = []
    local_coeffs: list[float] = []
    for bonds, coupling in (
        (geometry["shell_1"], float(J1)),
        (geometry["shell_2"], float(J2)),
        (geometry["shell_3"], float(J3)),
    ):
        if coupling == 0.0 or not bonds:
            continue
        for i, j in bonds:
            local_terms.append(_xxz_spin_bond(local_spin_hi, i, j, delta_z=delta_z))
            local_coeffs.append(coupling)

    terms: list[Any] = []
    H_fermion = _sum_terms(fermion_terms, fermion_coeffs)
    if H_fermion is not None:
        terms.append(EmbedOperator(joint_hi, H_fermion, subspace=0))

    H_local = _sum_terms(local_terms, local_coeffs)
    if H_local is not None:
        terms.append(EmbedOperator(joint_hi, H_local, subspace=1))

    if J_K != 0.0:
        kondo_terms: list[Any] = []
        kondo_coeffs: list[float] = []
        for site in range(n_sites):
            s_z = 0.5 * (nc(fermion_hi, site, sz=+1) - nc(fermion_hi, site, sz=-1))
            s_plus = cdag(fermion_hi, site, sz=+1) @ c(fermion_hi, site, sz=-1)
            s_minus = cdag(fermion_hi, site, sz=-1) @ c(fermion_hi, site, sz=+1)

            S_z = 0.5 * sigmaz(local_spin_hi, site)
            S_plus = sigmap(local_spin_hi, site)
            S_minus = sigmam(local_spin_hi, site)

            kondo_terms.extend(
                [
                    EmbedOperator(joint_hi, s_z, subspace=0)
                    @ EmbedOperator(joint_hi, S_z, subspace=1),
                    EmbedOperator(joint_hi, s_plus, subspace=0)
                    @ EmbedOperator(joint_hi, S_minus, subspace=1),
                    EmbedOperator(joint_hi, s_minus, subspace=0)
                    @ EmbedOperator(joint_hi, S_plus, subspace=1),
                ]
            )
            kondo_coeffs.extend([J_K, 0.5 * J_K, 0.5 * J_K])

        H_kondo = _sum_terms(kondo_terms, kondo_coeffs)
        if H_kondo is not None:
            terms.append(H_kondo)

    if not terms:
        H = 0.0
    elif len(terms) == 1:
        H = terms[0]
    else:
        H = SumOperator(*terms)

    return H, joint_hi


def kondo_heisenberg_chain_geometry(
    *,
    Lx: int,
    pbc: bool = False,
) -> dict[str, Any]:
    """
    Convenience wrapper for the ``n_legs = 1`` chain geometry.

    For a dedicated 1D interface, prefer ``KondoHeisenbergChain.py``.
    """
    return kondo_heisenberg_geometry(Lx=Lx, n_legs=1, pbc_x=pbc, pbc_y=False)


def KondoHeisenbergChain(
    *,
    Lx: int = 8,
    t: float = 1.0,
    J_K: float = 1.0,
    J1: float = 1.0,
    J2: float = 0.0,
    pbc: bool = False,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[Any, nk.hilbert.TensorHilbert]:
    """
    Convenience wrapper for the simplest 1D chain.

    In the 2D shell convention, the chain next-nearest-neighbor coupling maps
    to ``shell_3`` because ``shell_2`` is purely diagonal and vanishes for
    ``n_legs = 1``.
    """
    return KondoHeisenberg(
        Lx=Lx,
        n_legs=1,
        t1=t,
        t2=0.0,
        t3=0.0,
        J_K=J_K,
        J1=J1,
        J2=0.0,
        J3=J2,
        pbc_x=pbc,
        pbc_y=False,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )


build_kondo_heisenberg = KondoHeisenberg
build_kondo_heisenberg_chain = KondoHeisenbergChain


__all__ = [
    "KondoHeisenberg",
    "KondoHeisenbergChain",
    "build_kondo_heisenberg",
    "build_kondo_heisenberg_chain",
    "build_kondo_heisenberg_hilbert",
    "joint_sector_reference_state",
    "joint_total_two_sz_of_states",
    "joint_total_two_sz_operator",
    "kondo_heisenberg_chain_geometry",
    "kondo_heisenberg_geometry",
    "ladder_index",
    "seed_joint_sz_sector",
]
