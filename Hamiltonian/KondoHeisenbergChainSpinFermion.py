"""Structured spin-fermion interface for the dedicated 1D Kondo-Heisenberg chain.

This module mirrors `KondoHeisenbergChain.py`, but returns the already separate
fermion and local-spin Hilbert spaces together with the joint Hamiltonian.
It is intended for variational ansatzes that want to process the two sectors
explicitly while staying exactly compatible with the existing chain model.

Unlike the original lightweight wrapper, this version keeps parity with the
underlying chain module for particle-number control in the full Kondo regime and
re-exports the joint-sector helper utilities, so callers can stay in a single
namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import netket as nk

from KondoHeisenbergChain import KondoHeisenbergChain as _KondoHeisenbergChain
from KondoHeisenbergChain import build_kondo_heisenberg_chain_hilbert as _build_kondo_heisenberg_chain_hilbert
from KondoHeisenbergChain import joint_sector_reference_state
from KondoHeisenbergChain import joint_total_two_sz_of_states
from KondoHeisenbergChain import joint_total_two_sz_operator
from KondoHeisenbergChain import kondo_heisenberg_chain_geometry
from KondoHeisenbergChain import seed_joint_sz_sector


@dataclass(frozen=True)
class SpinFermionKondoHeisenbergChainSystem:
    hamiltonian: Any
    joint_hilbert: nk.hilbert.TensorHilbert
    fermion_hilbert: nk.hilbert.SpinOrbitalFermions
    local_spin_hilbert: nk.hilbert.Spin
    geometry: dict[str, Any]


def build_kondo_heisenberg_chain_spin_fermion_hilbert(
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
    """Return the explicit tensor-product Hilbert space `(fermion) x (local spin)`."""
    return _build_kondo_heisenberg_chain_hilbert(
        Lx=Lx,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )


def KondoHeisenbergChainSpinFermion(
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
) -> SpinFermionKondoHeisenbergChainSystem:
    """Build the dedicated chain together with its separated subspace metadata."""
    joint_hi, fermion_hi, local_spin_hi = build_kondo_heisenberg_chain_spin_fermion_hilbert(
        Lx=Lx,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )
    geometry = kondo_heisenberg_chain_geometry(Lx=Lx, pbc=pbc)
    hamiltonian, joint_hi_ref = _KondoHeisenbergChain(
        Lx=Lx,
        t=t,
        J_K=J_K,
        J1=J1,
        J2=J2,
        delta_z=delta_z,
        pbc=pbc,
        n_fermions=n_fermions,
        n_fermions_per_spin=n_fermions_per_spin,
        local_total_sz=local_total_sz,
    )

    if joint_hi_ref.size != joint_hi.size:
        raise RuntimeError("Unexpected Hilbert mismatch while building structured interface.")

    return SpinFermionKondoHeisenbergChainSystem(
        hamiltonian=hamiltonian,
        joint_hilbert=joint_hi,
        fermion_hilbert=fermion_hi,
        local_spin_hilbert=local_spin_hi,
        geometry=geometry,
    )


build_kondo_heisenberg_chain_spin_fermion = KondoHeisenbergChainSpinFermion


__all__ = [
    "KondoHeisenbergChainSpinFermion",
    "SpinFermionKondoHeisenbergChainSystem",
    "build_kondo_heisenberg_chain_spin_fermion",
    "build_kondo_heisenberg_chain_spin_fermion_hilbert",
    "joint_sector_reference_state",
    "joint_total_two_sz_of_states",
    "joint_total_two_sz_operator",
    "kondo_heisenberg_chain_geometry",
    "seed_joint_sz_sector",
]