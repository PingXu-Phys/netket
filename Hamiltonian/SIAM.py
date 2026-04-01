from __future__ import annotations

from typing import Any

import numpy as np
import netket as nk

from netket.graph.common_lattices import Chain
from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc


def total_spin_squared(
    hi: nk.hilbert.SpinOrbitalFermions,
) -> Any:
    """
    Construct the total-spin operator ``S^2``.

    We use the normal-ordered identity
        S^2 = (S^z)^2 + S^z + S^- S^+
    with
        S^z = 1/2 sum_i (n_{i,up} - n_{i,dn})
        S^+ = sum_i c^dag_{i,up} c_{i,dn}
        S^- = sum_i c^dag_{i,dn} c_{i,up}
    """
    sz = 0.0
    s_plus = 0.0
    s_minus = 0.0

    for site in range(hi.n_orbitals):
        n_up = nc(hi, site, sz=+1)
        n_dn = nc(hi, site, sz=-1)
        sz += 0.5 * (n_up - n_dn)
        s_plus += cdag(hi, site, sz=+1) @ c(hi, site, sz=-1)
        s_minus += cdag(hi, site, sz=-1) @ c(hi, site, sz=+1)

    return sz @ sz + sz + s_minus @ s_plus


def add_spin_penalty(
    H: Any,
    hi: nk.hilbert.SpinOrbitalFermions,
    *,
    penalty_strength: float,
    target_spin: float | None = None,
) -> Any:
    """
    Add an optional total-spin penalty to the Hamiltonian.

    If ``target_spin is None``, add the singlet penalty
        H_eff = H + lambda * S^2

    If ``target_spin`` is specified, add the more general penalty
        H_eff = H + lambda * [S^2 - S_target(S_target + 1)]^2
    """
    if penalty_strength < 0:
        raise ValueError(
            f"penalty_strength must be non-negative, got {penalty_strength}."
        )
    if penalty_strength == 0:
        return H

    s2 = total_spin_squared(hi)

    if target_spin is None:
        return H + penalty_strength * s2

    target_eigenvalue = float(target_spin) * (float(target_spin) + 1.0)
    shifted = s2 - target_eigenvalue
    return H + penalty_strength * (shifted @ shifted)


def _resolve_spin_sector(
    n_orbitals: int,
    n_fermions_per_spin: tuple[int, int] | None,
) -> tuple[int, int]:
    if n_fermions_per_spin is not None:
        if len(n_fermions_per_spin) != 2:
            raise ValueError(
                "n_fermions_per_spin must have length 2 for spin-1/2 SIAM, "
                f"got {n_fermions_per_spin}."
            )
        return int(n_fermions_per_spin[0]), int(n_fermions_per_spin[1])

    if n_orbitals % 2 != 0:
        raise ValueError(
            "n_orbitals must be even to use the default half-filled choice "
            "N_up = N_dn = n_orbitals // 2."
        )
    half = n_orbitals // 2
    return half, half


def _resolve_hopping_profile(L: int, ti: float | list[float] | np.ndarray | None) -> np.ndarray:
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}.")

    if L == 1:
        return np.zeros((0,), dtype=float)

    if ti is None:
        return np.ones(L - 1, dtype=float)

    if np.isscalar(ti):
        return np.full(L - 1, float(ti), dtype=float)

    ti_arr = np.asarray(ti, dtype=float)
    if ti_arr.shape != (L - 1,):
        raise ValueError(
            "ti must be scalar or an array of shape (L - 1,). "
            f"Got shape {ti_arr.shape} for L={L}."
        )
    return ti_arr


def add_impurity_pinning_field(
    H: Any,
    hi: nk.hilbert.SpinOrbitalFermions,
    *,
    impurity_site: int = 0,
    h_pin: float,
) -> Any:
    """
    Add the impurity spin pinning field used in SIAM_bench2.ipynb.

    This follows the notebook code exactly:
        H_pinned = H + h_pin * 0.5 * (n_d_up - n_d_dn)
    """
    n_d_up = nc(hi, impurity_site, sz=+1)
    n_d_dn = nc(hi, impurity_site, sz=-1)
    return H + h_pin * 0.5 * (n_d_up - n_d_dn)


def SIAM(
    *,
    L: int = 19,
    U: float = 4.0,
    V: float = 0.15,
    ti: float | list[float] | np.ndarray | None = None,
    pbc: bool = False,
    impurity_site: int = 0,
    conduction_offset: int = 1,
    n_fermions_per_spin: tuple[int, int] | None = None,
    penalty_strength: float = 0.0,
    penalty_target_spin: float | None = None,
) -> tuple[Any, nk.hilbert.SpinOrbitalFermions]:
    """
    Construct the single-impurity Anderson model and return ``(H, hi)``.

    The geometry is:
        impurity site 0  --  conduction site 1  --  ...  --  conduction site L

    The Hamiltonian matches the notebook logic:
      1. U * (n_d_up - 1/2) (n_d_dn - 1/2) on the impurity.
      2. V hybridization between the impurity and the first conduction site.
      3. Nearest-neighbor hopping along the conduction chain with amplitudes ti.
      4. Optional spin penalty controlled by ``penalty_strength``.
    """
    if impurity_site != 0:
        raise ValueError(
            "This constructor currently assumes impurity_site=0 to match the "
            f"bench2 indexing convention. Got impurity_site={impurity_site}."
        )
    if conduction_offset != 1:
        raise ValueError(
            "This constructor currently assumes conduction_offset=1 to match the "
            f"bench2 indexing convention. Got conduction_offset={conduction_offset}."
        )

    chain = Chain(L, pbc=pbc)
    n_orbitals = L + conduction_offset
    n_up, n_dn = _resolve_spin_sector(n_orbitals, n_fermions_per_spin)
    ti_arr = _resolve_hopping_profile(L, ti)

    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals,
        s=1 / 2,
        n_fermions_per_spin=(n_up, n_dn),
    )

    H = 0.0

    n_d_up = nc(hi, impurity_site, sz=+1)
    n_d_dn = nc(hi, impurity_site, sz=-1)
    H += U * (n_d_up - 0.5) @ (n_d_dn - 0.5)

    c0_site = conduction_offset
    for sz in (+1, -1):
        H += V * (cdag(hi, impurity_site, sz=sz) @ c(hi, c0_site, sz=sz))
        H += V * (cdag(hi, c0_site, sz=sz) @ c(hi, impurity_site, sz=sz))

    for i_local, j_local in chain.edges():
        left_local, right_local = sorted((i_local, j_local))
        t_bond = float(ti_arr[left_local])
        left_site = left_local + conduction_offset
        right_site = right_local + conduction_offset

        for sz in (+1, -1):
            H += t_bond * (cdag(hi, right_site, sz=sz) @ c(hi, left_site, sz=sz))
            H += t_bond * (cdag(hi, left_site, sz=sz) @ c(hi, right_site, sz=sz))

    H = add_spin_penalty(
        H,
        hi,
        penalty_strength=penalty_strength,
        target_spin=penalty_target_spin,
    )

    return H, hi


build_siam = SIAM


__all__ = [
    "SIAM",
    "add_impurity_pinning_field",
    "add_spin_penalty",
    "build_siam",
    "total_spin_squared",
]
