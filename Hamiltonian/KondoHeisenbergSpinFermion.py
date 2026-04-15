"""Structured spin-fermion interface for the generalized Kondo-Heisenberg ladder.

This module does not alter the physics relative to `KondoHeisenberg.py`.
Instead, it exposes the already separate fermion and local-spin Hilbert spaces
through a single structured return object, which is more convenient for ansatzes
that process the two sectors explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import netket as nk

from KondoHeisenberg import KondoHeisenberg as _KondoHeisenberg
from KondoHeisenberg import build_kondo_heisenberg_hilbert as _build_kondo_heisenberg_hilbert
from KondoHeisenberg import kondo_heisenberg_geometry


@dataclass(frozen=True)
class SpinFermionKondoHeisenbergSystem:
    hamiltonian: Any
    joint_hilbert: nk.hilbert.TensorHilbert
    fermion_hilbert: nk.hilbert.SpinOrbitalFermions
    local_spin_hilbert: nk.hilbert.Spin
    geometry: dict[str, Any]


def build_kondo_heisenberg_spin_fermion_hilbert(
    *,
    Lx: int,
    n_legs: int,
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> tuple[
    nk.hilbert.TensorHilbert,
    nk.hilbert.SpinOrbitalFermions,
    nk.hilbert.Spin,
]:
    """Return the explicit tensor-product Hilbert space `(fermion) x (local spin)`."""
    return _build_kondo_heisenberg_hilbert(
        Lx=Lx,
        n_legs=n_legs,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )


def KondoHeisenbergSpinFermion(
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
    n_fermions_per_spin: tuple[int, int] | None = None,
    local_total_sz: float | None = None,
) -> SpinFermionKondoHeisenbergSystem:
    """Build the generalized ladder together with its separated subspace metadata."""
    joint_hi, fermion_hi, local_spin_hi = build_kondo_heisenberg_spin_fermion_hilbert(
        Lx=Lx,
        n_legs=n_legs,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )
    geometry = kondo_heisenberg_geometry(Lx=Lx, n_legs=n_legs, pbc_x=pbc_x, pbc_y=pbc_y)
    hamiltonian, joint_hi_ref = _KondoHeisenberg(
        Lx=Lx,
        n_legs=n_legs,
        t1=t1,
        t2=t2,
        t3=t3,
        J_K=J_K,
        J1=J1,
        J2=J2,
        J3=J3,
        delta_z=delta_z,
        pbc_x=pbc_x,
        pbc_y=pbc_y,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )

    if joint_hi_ref.size != joint_hi.size:
        raise RuntimeError('Unexpected Hilbert mismatch while building structured interface.')

    return SpinFermionKondoHeisenbergSystem(
        hamiltonian=hamiltonian,
        joint_hilbert=joint_hi,
        fermion_hilbert=fermion_hi,
        local_spin_hilbert=local_spin_hi,
        geometry=geometry,
    )


build_kondo_heisenberg_spin_fermion = KondoHeisenbergSpinFermion


__all__ = [
    'KondoHeisenbergSpinFermion',
    'SpinFermionKondoHeisenbergSystem',
    'build_kondo_heisenberg_spin_fermion',
    'build_kondo_heisenberg_spin_fermion_hilbert',
]
