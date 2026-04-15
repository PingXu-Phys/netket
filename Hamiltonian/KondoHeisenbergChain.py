"""
Dedicated 1D Kondo-Heisenberg chain for NetKet.

This file is intentionally separate from the multileg ladder version.
It describes the physically minimal 1-leg case:

- one conduction-electron chain;
- one local-spin chain;
- onsite Kondo exchange between the two chains at the same spatial site.

Hamiltonian:

    H = -t  sum_{x,sigma} (c^dag_{x+1,sigma} c_{x,sigma} + h.c.)
        + J_K sum_x s_x . S_x
        + J1  sum_x S_x . S_{x+1}
        + J2  sum_x S_x . S_{x+2}

Only the chain parameters are exposed: ``Lx, t, J_K, J1, J2, pbc``.

Sector note:
- For ``J_K != 0``, use ``n_fermions`` to fix only the total conduction-electron
  number and seed the joint ``S^z`` sector with ``seed_joint_sz_sector``.
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


def _resolve_fermion_hilbert_kwargs(
    Lx: int,
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
        if not (0 <= n_up <= Lx and 0 <= n_dn <= Lx):
            raise ValueError(
                "Each spin-sector particle number must satisfy "
                f"0 <= N_sigma <= Lx={Lx}, got {(n_up, n_dn)}."
            )
        return {"n_fermions_per_spin": (n_up, n_dn)}

    if n_fermions is not None:
        n_total = int(n_fermions)
        if not (0 <= n_total <= 2 * Lx):
            raise ValueError(
                "n_fermions must satisfy "
                f"0 <= n_fermions <= 2 * Lx = {2 * Lx}, got {n_total}."
            )
        return {"n_fermions": n_total}

    if Lx % 2 != 0:
        raise ValueError(
            "Lx must be even to use the default half-filled choice n_fermions = Lx. "
            "Pass n_fermions explicitly for odd systems."
        )

    return {"n_fermions": Lx}


def kondo_heisenberg_chain_geometry(
    *,
    Lx: int,
    pbc: bool = False,
) -> dict[str, Any]:
    """
    Geometry of the 1D Kondo-Heisenberg chain.

    The two visible lines in a plot are not two spatial legs. They are the
    conduction chain and the local-spin chain living on the same 1D sites.
    """
    if Lx < 1:
        raise ValueError(f"Lx must be >= 1, got {Lx}.")

    nn: list[Bond] = []
    nnn: list[Bond] = []

    for x in range(Lx):
        x1 = (x + 1) % Lx if pbc else x + 1
        if 0 <= x1 < Lx:
            nn.append((x, x1) if x < x1 else (x1, x))

        x2 = (x + 2) % Lx if pbc else x + 2
        if 0 <= x2 < Lx:
            nnn.append((x, x2) if x < x2 else (x2, x))

    return {
        "Lx": Lx,
        "n_sites": Lx,
        "pbc": pbc,
        "nn": sorted(set(nn)),
        "nnn": sorted(set(nnn)),
    }


def build_kondo_heisenberg_chain_hilbert(
    *,
    Lx: int,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[
    nk.hilbert.TensorHilbert,
    nk.hilbert.SpinOrbitalFermions,
    nk.hilbert.Spin,
]:
    """
    Hilbert space of the 1D Kondo-Heisenberg chain.

    At each spatial site x there are two degrees of freedom:
    - one spinful conduction orbital;
    - one local spin-1/2 moment.
    """
    if Lx < 1:
        raise ValueError(f"Lx must be >= 1, got {Lx}.")

    fermion_kwargs = _resolve_fermion_hilbert_kwargs(
        Lx,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    fermion_hi = nk.hilbert.SpinOrbitalFermions(
        Lx,
        s=1 / 2,
        **fermion_kwargs,
    )
    local_spin_hi = nk.hilbert.Spin(
        s=1 / 2,
        N=Lx,
        total_sz=local_total_sz,
    )

    return fermion_hi * local_spin_hi, fermion_hi, local_spin_hi


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
    """Return a deterministic product configuration in the chosen joint ``2 S^z`` sector."""
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
    """Seed an MCState in a fixed joint ``2 S^z`` sector."""
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


def KondoHeisenbergChain(
    *,
    Lx: int = 8,
    t: float = 1.0,
    J_K: float = 1.0,
    J1: float = 1.0,
    J2: float = 0.0,
    delta_z: float = 1.0,
    pbc: bool = False,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[Any, nk.hilbert.TensorHilbert]:
    """
    Construct the dedicated 1D Kondo-Heisenberg chain and return ``(H, hi)``.

    Parameters:
    - ``t``: conduction-electron hopping along the chain.
    - ``J_K``: onsite Kondo exchange.
    - ``J1``: local-spin nearest-neighbor exchange.
    - ``J2``: local-spin next-nearest-neighbor exchange.
    - ``delta_z``: XXZ anisotropy in the local-spin sector.
    - ``pbc``: periodic boundary condition along the chain.
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

    joint_hi, fermion_hi, local_spin_hi = build_kondo_heisenberg_chain_hilbert(
        Lx=Lx,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )

    geometry = kondo_heisenberg_chain_geometry(Lx=Lx, pbc=pbc)
    nn = geometry["nn"]
    nnn = geometry["nnn"]

    H_fermion = 0.0
    if float(t) != 0.0:
        for i, j in nn:
            for sz in (+1, -1):
                H_fermion += -float(t) * (cdag(fermion_hi, i, sz=sz) @ c(fermion_hi, j, sz=sz))
                H_fermion += -float(t) * (cdag(fermion_hi, j, sz=sz) @ c(fermion_hi, i, sz=sz))

    H_local = 0.0
    for bonds, coupling in ((nn, float(J1)), (nnn, float(J2))):
        if coupling == 0.0:
            continue
        for i, j in bonds:
            H_local += coupling * (
                0.5 * (sigmap(local_spin_hi, i) @ sigmam(local_spin_hi, j) + sigmam(local_spin_hi, i) @ sigmap(local_spin_hi, j))
                + 0.25 * float(delta_z) * (sigmaz(local_spin_hi, i) @ sigmaz(local_spin_hi, j))
            )

    terms = []
    if float(t) != 0.0:
        terms.append(EmbedOperator(joint_hi, H_fermion, subspace=0))
    if float(J1) != 0.0 or float(J2) != 0.0:
        terms.append(EmbedOperator(joint_hi, H_local, subspace=1))

    if J_K != 0.0:
        for site in range(Lx):
            s_z = 0.5 * (nc(fermion_hi, site, sz=+1) - nc(fermion_hi, site, sz=-1))
            s_plus = cdag(fermion_hi, site, sz=+1) @ c(fermion_hi, site, sz=-1)
            s_minus = cdag(fermion_hi, site, sz=-1) @ c(fermion_hi, site, sz=+1)

            S_z = 0.5 * sigmaz(local_spin_hi, site)
            S_plus = sigmap(local_spin_hi, site)
            S_minus = sigmam(local_spin_hi, site)

            terms.append(
                SumOperator(
                    EmbedOperator(joint_hi, s_z, subspace=0) @ EmbedOperator(joint_hi, S_z, subspace=1),
                    EmbedOperator(joint_hi, s_plus, subspace=0) @ EmbedOperator(joint_hi, S_minus, subspace=1),
                    EmbedOperator(joint_hi, s_minus, subspace=0) @ EmbedOperator(joint_hi, S_plus, subspace=1),
                    coefficients=[J_K, 0.5 * J_K, 0.5 * J_K],
                )
            )

    if not terms:
        H = 0.0
    elif len(terms) == 1:
        H = terms[0]
    else:
        H = SumOperator(*terms)

    return H, joint_hi


build_kondo_heisenberg_chain = KondoHeisenbergChain


__all__ = [
    "KondoHeisenbergChain",
    "build_kondo_heisenberg_chain",
    "build_kondo_heisenberg_chain_hilbert",
    "joint_sector_reference_state",
    "joint_total_two_sz_of_states",
    "joint_total_two_sz_operator",
    "kondo_heisenberg_chain_geometry",
    "seed_joint_sz_sector",
]
