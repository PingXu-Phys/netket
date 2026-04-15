"""
2D Fermi-Hubbard model for NetKet.

Modeling choice:
- The real-space geometry is a rectangular ``Lx x Ly`` lattice.
- Each spatial site carries one spinful fermionic orbital.
- For NetKet indexing we flatten ``(x, y) -> x * Ly + y``.

Bond shells:
- ``shell_1``: nearest neighbors ``(dx, dy) = (1, 0)`` and ``(0, 1)``
- ``shell_2``: diagonal next-nearest neighbors ``(dx, dy) = (1, 1)`` and ``(1, -1)``

Boundary conditions:
- ``pbc_x`` can be open or periodic.
- ``pbc_y`` can be open or periodic.
"""

from __future__ import annotations

from typing import Any

import netket as nk

from netket.operator.fermion import create as cdag
from netket.operator.fermion import destroy as c
from netket.operator.fermion import number as nc


Bond = tuple[int, int]


def hubbard_index(x: int, y: int, Ly: int) -> int:
    """Flatten ``(x, y)`` into a single spatial-site index."""
    if Ly < 1:
        raise ValueError(f"Ly must be >= 1, got {Ly}.")
    if x < 0:
        raise ValueError(f"x must be >= 0, got {x}.")
    if y < 0 or y >= Ly:
        raise ValueError(f"y must satisfy 0 <= y < Ly, got {y}.")
    return x * Ly + y


def _validate_geometry(Lx: int, Ly: int) -> None:
    if Lx < 1:
        raise ValueError(f"Lx must be >= 1, got {Lx}.")
    if Ly < 1:
        raise ValueError(f"Ly must be >= 1, got {Ly}.")


def _generate_bonds(
    Lx: int,
    Ly: int,
    *,
    dx: int,
    dy: int,
    pbc_x: bool,
    pbc_y: bool,
) -> list[Bond]:
    bonds: set[Bond] = set()

    for x in range(Lx):
        for y in range(Ly):
            x2 = x + dx
            y2 = y + dy

            if pbc_x:
                x2 %= Lx
            elif not (0 <= x2 < Lx):
                continue

            if pbc_y:
                y2 %= Ly
            elif not (0 <= y2 < Ly):
                continue

            i = hubbard_index(x, y, Ly)
            j = hubbard_index(x2, y2, Ly)
            if i != j:
                bonds.add((i, j) if i < j else (j, i))

    return sorted(bonds)


def _merge_bonds(*groups: list[Bond]) -> list[Bond]:
    merged: set[Bond] = set()
    for group in groups:
        merged.update(group)
    return sorted(merged)


def _resolve_hilbert_kwargs(
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
            "n_up = n_dn = n_sites // 2. Pass n_fermions or "
            "n_fermions_per_spin explicitly for odd lattices."
        )

    half = n_sites // 2
    return {"n_fermions_per_spin": (half, half)}


def hubbard_geometry(
    *,
    Lx: int,
    Ly: int,
    pbc_x: bool = False,
    pbc_y: bool = False,
) -> dict[str, Any]:
    """
    Return the shared 2D geometry used by the Hubbard Hamiltonian.

    The shell convention is:
    - ``shell_1``: nearest neighbors along ``x`` and along ``y``
    - ``shell_2``: diagonal next-nearest neighbors
    """
    _validate_geometry(Lx, Ly)

    shell_1_x = _generate_bonds(Lx, Ly, dx=1, dy=0, pbc_x=pbc_x, pbc_y=False)
    shell_1_y = _generate_bonds(Lx, Ly, dx=0, dy=1, pbc_x=False, pbc_y=pbc_y)
    shell_2_up = _generate_bonds(Lx, Ly, dx=1, dy=1, pbc_x=pbc_x, pbc_y=pbc_y)
    shell_2_dn = _generate_bonds(Lx, Ly, dx=1, dy=-1, pbc_x=pbc_x, pbc_y=pbc_y)

    return {
        "Lx": Lx,
        "Ly": Ly,
        "n_sites": Lx * Ly,
        "pbc_x": pbc_x,
        "pbc_y": pbc_y,
        "shell_1_x": shell_1_x,
        "shell_1_y": shell_1_y,
        "shell_1": _merge_bonds(shell_1_x, shell_1_y),
        "shell_2_up": shell_2_up,
        "shell_2_dn": shell_2_dn,
        "shell_2": _merge_bonds(shell_2_up, shell_2_dn),
    }


def build_hubbard_hilbert(
    *,
    Lx: int,
    Ly: int,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
) -> nk.hilbert.SpinOrbitalFermions:
    """Build the spinful fermionic Hilbert space of the 2D Hubbard model."""
    _validate_geometry(Lx, Ly)

    n_sites = Lx * Ly
    hilbert_kwargs = _resolve_hilbert_kwargs(
        n_sites,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    return nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        **hilbert_kwargs,
    )


def Hubbard(
    *,
    Lx: int,
    Ly: int,
    t1: float = 1.0,
    t2: float = 0.0,
    U: float = 4.0,
    pbc_x: bool = False,
    pbc_y: bool = False,
    n_fermions: int | None = None,
    n_fermions_per_spin: tuple[int, int] | None = None,
) -> tuple[Any, nk.hilbert.SpinOrbitalFermions]:
    """
    Construct the 2D Fermi-Hubbard model and return ``(H, hi)``.

    Hamiltonian:

        H = -t1 sum_{<ij>_1,sigma} (c^dag_{i,sigma} c_{j,sigma} + h.c.)
            -t2 sum_{<ij>_2,sigma} (c^dag_{i,sigma} c_{j,sigma} + h.c.)
            + U sum_i n_{i,up} n_{i,dn}

    Particle-number sector:
    - Pass ``n_fermions_per_spin=(N_up, N_dn)`` to fix both spin sectors.
    - Pass ``n_fermions=N`` to fix only the total particle number.
    - If neither is given, default to half filling with ``N_up = N_dn = n_sites // 2``.
    """
    hi = build_hubbard_hilbert(
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    geometry = hubbard_geometry(Lx=Lx, Ly=Ly, pbc_x=pbc_x, pbc_y=pbc_y)
    n_sites = geometry["n_sites"]

    H = 0.0

    for bonds, hopping in (
        (geometry["shell_1"], float(t1)),
        (geometry["shell_2"], float(t2)),
    ):
        if hopping == 0.0 or not bonds:
            continue
        for i, j in bonds:
            for sz in (+1, -1):
                H += -hopping * (cdag(hi, i, sz=sz) @ c(hi, j, sz=sz))
                H += -hopping * (cdag(hi, j, sz=sz) @ c(hi, i, sz=sz))

    U = float(U)
    if U != 0.0:
        for site in range(n_sites):
            H += U * nc(hi, site, sz=+1) @ nc(hi, site, sz=-1)

    return H, hi


build_hubbard = Hubbard


__all__ = [
    "Hubbard",
    "build_hubbard",
    "build_hubbard_hilbert",
    "hubbard_geometry",
    "hubbard_index",
]
