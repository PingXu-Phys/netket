"""Shared utilities for orbital optimisation modules.

This module holds the reusable helper functions that are shared across the
optimizer, losses, and later pruning / sparse layers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from netket.operator._fermion2nd.base import FermionOperator2ndBase

try:
    from graph_sample.iteration_occ_func_simple import (
        _expand_orbital_rotation_to_modes,
        natural_orbitals_from_rdm,
        rotate_fermion_hamiltonian,
    )
except ImportError:
    from iteration_occ_func_simple import (
        _expand_orbital_rotation_to_modes,
        natural_orbitals_from_rdm,
        rotate_fermion_hamiltonian,
    )

EPS = 1.0e-15

OCCUPATION_SORT_TOL = 1.0e-10

NO_CONSISTENCY_RTOL = 1.0e-8

NO_CONSISTENCY_ATOL = 1.0e-10

def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)

def _require_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}.")
    return matrix

def _resolve_distance_matrix(
    n_orbitals: int,
    distance_matrix: np.ndarray | None = None,
) -> np.ndarray:
    if distance_matrix is None:
        indices = np.arange(n_orbitals, dtype=float)
        return np.abs(indices[:, None] - indices[None, :])

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.shape != (n_orbitals, n_orbitals):
        raise ValueError(
            "distance_matrix must have shape "
            f"({n_orbitals}, {n_orbitals}), got {distance_matrix.shape}."
        )
    return distance_matrix

def _resolve_interaction_matrix(
    interaction_matrix: np.ndarray | None,
    n_sites: int,
) -> np.ndarray | None:
    """Return a validated (n_sites, n_sites) interaction matrix, or None.

    Accepts:
      None          → disabled (returns None)
      1-D (n,)      → treated as diagonal S = diag(J); shorthand for Kondo J_i
      2-D (n, n)    → used as-is (general site-to-site coupling matrix)
    """
    if interaction_matrix is None:
        return None
    interaction_matrix = np.asarray(interaction_matrix, dtype=float)
    if interaction_matrix.ndim == 1:
        if interaction_matrix.shape != (n_sites,):
            raise ValueError(
                f"1-D interaction_matrix must have shape ({n_sites},), "
                f"got {interaction_matrix.shape}."
            )
        return np.diag(interaction_matrix)
    if interaction_matrix.shape != (n_sites, n_sites):
        raise ValueError(
            f"interaction_matrix must have shape ({n_sites}, {n_sites}), "
            f"got {interaction_matrix.shape}."
        )
    return interaction_matrix

def _validate_no_reference_data(
    rdm: np.ndarray,
    natural_occupations: np.ndarray,
    natural_orbitals: np.ndarray,
) -> None:
    if natural_orbitals.shape != rdm.shape:
        raise ValueError(
            "natural_orbitals and rdm must have identical shapes, got "
            f"{natural_orbitals.shape} and {rdm.shape}."
        )
    if natural_occupations.shape != (rdm.shape[0],):
        raise ValueError(
            "natural_occupations must have shape "
            f"({rdm.shape[0]},), got {natural_occupations.shape}."
        )
    if np.any(np.diff(natural_occupations) > OCCUPATION_SORT_TOL):
        raise ValueError(
            "natural_occupations must be sorted in non-increasing order to "
            "define NO occupation blocks."
        )

    gamma_no = rotate_one_body_matrix(rdm, natural_orbitals)
    offdiag = gamma_no.copy()
    np.fill_diagonal(offdiag, 0.0)
    rdm_norm = max(float(np.linalg.norm(rdm, ord="fro")), EPS)
    offdiag_ratio = float(np.linalg.norm(offdiag, ord="fro")) / rdm_norm
    if offdiag_ratio > NO_CONSISTENCY_RTOL:
        raise ValueError(
            "natural_orbitals must diagonalize rdm within relative Frobenius "
            f"tolerance {NO_CONSISTENCY_RTOL:.1e}; got {offdiag_ratio:.3e}."
        )

    diagonal_occupations = np.real(np.diag(gamma_no))
    occupation_scale = max(
        float(np.max(np.abs(diagonal_occupations))) if diagonal_occupations.size else 0.0,
        float(np.max(np.abs(natural_occupations))) if natural_occupations.size else 0.0,
        1.0,
    )
    occupation_diff = (
        float(np.max(np.abs(diagonal_occupations - natural_occupations)))
        if diagonal_occupations.size
        else 0.0
    )
    allowed_diff = NO_CONSISTENCY_ATOL + NO_CONSISTENCY_RTOL * occupation_scale
    if occupation_diff > allowed_diff:
        raise ValueError(
            "natural_occupations must match diag(U^dagger rdm U) within "
            f"tolerance {allowed_diff:.3e}; got {occupation_diff:.3e}."
        )

def extract_one_body_hopping_matrix(
    H,
    *,
    spin_symmetric: bool = True,
    cutoff: float | None = None,
    require_identical_spin_blocks: bool = False,
) -> np.ndarray:
    """Extract the one-body hopping matrix from a FermionOperator2nd Hamiltonian."""
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "H must be a FermionOperator2nd-compatible Hamiltonian, got "
            f"{type(H)}."
        )

    hi = H.hilbert
    inferred_cutoff = getattr(H, "cutoff", 0.0)
    cutoff = 0.0 if cutoff is None and inferred_cutoff is None else (
        float(inferred_cutoff) if cutoff is None else float(cutoff)
    )

    op = H.to_normal_order()
    full_matrix = np.zeros((hi.size, hi.size), dtype=np.complex128)
    for term, weight in op.operators.items():
        if abs(weight) <= cutoff or len(term) != 2:
            continue
        (i, d1), (j, d2) = term
        if d1 == 1 and d2 == 0:
            full_matrix[int(i), int(j)] += weight

    full_matrix = _hermitian_part(full_matrix)
    if not spin_symmetric or hi.n_spin_subsectors == 1:
        return full_matrix

    n_orbitals = hi.n_orbitals
    blocks = []
    for spin_i in range(hi.n_spin_subsectors):
        lo = spin_i * n_orbitals
        hi_idx = lo + n_orbitals
        blocks.append(full_matrix[lo:hi_idx, lo:hi_idx].copy())

    for spin_i in range(hi.n_spin_subsectors):
        lo_i = spin_i * n_orbitals
        hi_i = lo_i + n_orbitals
        for spin_j in range(hi.n_spin_subsectors):
            if spin_i == spin_j:
                continue
            lo_j = spin_j * n_orbitals
            hi_j = lo_j + n_orbitals
            off_block = full_matrix[lo_i:hi_i, lo_j:hi_j]
            if np.max(np.abs(off_block)) > cutoff:
                raise ValueError(
                    "Found spin-mixing one-body terms while spin_symmetric=True. "
                    "Use spin_symmetric=False instead."
                )

    if require_identical_spin_blocks and blocks:
        reference = blocks[0]
        for spin_idx, block in enumerate(blocks[1:], start=1):
            if np.max(np.abs(block - reference)) > cutoff:
                raise ValueError(
                    "Different spin subsectors induce different one-body orbital "
                    f"blocks. Spin block {spin_idx} differs from the reference."
                )

    return _hermitian_part(sum(blocks) / float(len(blocks)))

def extract_interaction_matrix(
    H,
    *,
    spin_symmetric: bool = True,
    cutoff: float | None = None,
) -> np.ndarray:
    """Build the site-to-site interaction strength matrix S from H's two-body terms.

    For each normal-ordered term  V * c†_i c†_j c_k c_l  in H, the pair (i, j)
    contributes |V| to  S[orb_i, orb_j]  (and symmetrically S[orb_j, orb_i]).

    When spin_symmetric=True the spin index is stripped so S is indexed by
    orbital only; contributions from all spin sectors are summed.

    The resulting S is used as the interaction_matrix in the loss:
        Lambda_ab = sum_ij  S_ij |U_ia|^2 |U_jb|^2

    Parameters
    ----------
    H : FermionOperator2nd
    spin_symmetric : bool
        Strip spin index; return an (n_orb, n_orb) matrix.
        When False return a (n_modes, n_modes) matrix.
    cutoff : float, optional
        Terms with |weight| <= cutoff are ignored.

    Returns
    -------
    S : (n, n) float array  (non-negative, symmetric)
    """
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "H must be a FermionOperator2nd-compatible Hamiltonian, got "
            f"{type(H)}."
        )

    hi = H.hilbert
    inferred_cutoff = getattr(H, "cutoff", 0.0)
    eff_cutoff = 0.0 if cutoff is None and inferred_cutoff is None else (
        float(inferred_cutoff) if cutoff is None else float(cutoff)
    )

    op = H.to_normal_order()

    if spin_symmetric and hi.n_spin_subsectors > 1:
        n = hi.n_orbitals          # orbital dimension (per spin)
        S = np.zeros((n, n), dtype=float)
        for term, weight in op.operators.items():
            if len(term) != 4 or abs(weight) <= eff_cutoff:
                continue
            (m1, d1), (m2, d2), (m3, d3), (m4, d4) = term
            if d1 != 1 or d2 != 1 or d3 != 0 or d4 != 0:
                continue
            i = int(m1) % n        # orbital index (strip spin)
            j = int(m2) % n
            val = abs(weight)
            S[i, j] += val
            if i != j:
                S[j, i] += val
    else:
        n = hi.size                # full mode dimension
        S = np.zeros((n, n), dtype=float)
        for term, weight in op.operators.items():
            if len(term) != 4 or abs(weight) <= eff_cutoff:
                continue
            (m1, d1), (m2, d2), (m3, d3), (m4, d4) = term
            if d1 != 1 or d2 != 1 or d3 != 0 or d4 != 0:
                continue
            i, j = int(m1), int(m2)
            val = abs(weight)
            S[i, j] += val
            if i != j:
                S[j, i] += val

    return S

def rotate_one_body_matrix(
    matrix: np.ndarray,
    site_to_orbital: np.ndarray,
) -> np.ndarray:
    """Rotate a one-body matrix with the convention gamma' = U^dag gamma U."""
    matrix = _require_square_matrix(matrix, name="matrix").astype(np.complex128)
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name="site_to_orbital",
    ).astype(np.complex128)
    if matrix.shape != site_to_orbital.shape:
        raise ValueError(
            "matrix and site_to_orbital must have identical shapes, got "
            f"{matrix.shape} and {site_to_orbital.shape}."
        )
    return _hermitian_part(site_to_orbital.conj().T @ matrix @ site_to_orbital)

def extract_hamiltonian_terms(
    H,
    *,
    cutoff: float | None = None,
    normal_order: bool = True,
) -> dict[str, Any]:
    """Read out the explicit terms of a FermionOperator2nd Hamiltonian."""
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "H must be a FermionOperator2nd-compatible Hamiltonian, got "
            f"{type(H)}."
        )

    inferred_cutoff = getattr(H, "cutoff", 0.0)
    if cutoff is None:
        cutoff = 0.0 if inferred_cutoff is None else float(inferred_cutoff)
    else:
        cutoff = float(cutoff)

    op = H.to_normal_order() if normal_order else H
    constant = 0.0 + 0.0j
    terms: list[dict[str, Any]] = []
    counts_by_order: dict[int, int] = {}

    for term, weight in op.operators.items():
        if abs(weight) <= cutoff:
            continue
        if len(term) == 0:
            constant += weight
            continue

        order = len(term)
        counts_by_order[order] = counts_by_order.get(order, 0) + 1
        terms.append(
            {
                "operators": tuple((int(mode), int(dagger)) for mode, dagger in term),
                "weight": complex(weight),
                "order": order,
            }
        )

    return {
        "constant": complex(constant),
        "n_terms": len(terms),
        "counts_by_order": counts_by_order,
        "terms": terms,
    }

def rotate_hamiltonian_to_basis(
    H,
    site_to_orbital: np.ndarray,
    *,
    spin_symmetric: bool = True,
    cutoff: float | None = None,
):
    """Rotate a Hamiltonian to a user-supplied one-particle basis."""
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "H must be a FermionOperator2nd-compatible Hamiltonian, got "
            f"{type(H)}."
        )

    site_to_orbital = _require_square_matrix(
        np.asarray(site_to_orbital, dtype=np.complex128),
        name="site_to_orbital",
    )
    mode_rotation = (
        _expand_orbital_rotation_to_modes(H.hilbert, site_to_orbital)
        if spin_symmetric
        else site_to_orbital
    )
    return rotate_fermion_hamiltonian(H, mode_rotation, cutoff=cutoff)

def rotate_hamiltonian_to_natural_orbitals(
    H,
    rdm: np.ndarray,
    *,
    spin_symmetric: bool = True,
    cutoff: float | None = None,
) -> dict[str, Any]:
    """Diagonalize a 1-RDM, then rotate the Hamiltonian into the NO basis."""
    occupations, natural_orbitals = natural_orbitals_from_rdm(rdm)
    rotated = rotate_hamiltonian_to_basis(
        H,
        natural_orbitals,
        spin_symmetric=spin_symmetric,
        cutoff=cutoff,
    )
    return {
        "natural_occupations": occupations.copy(),
        "natural_orbitals": natural_orbitals.copy(),
        "rotated_hamiltonian": rotated,
    }

def build_occupation_blocks(
    occupations: np.ndarray,
    *,
    filled_tol: float = 0.05,
    empty_tol: float = 0.05,
    degeneracy_tol: float = 0.02,
) -> list[np.ndarray]:
    """Split occupations sorted in descending order into NO occupation blocks."""
    occupations = np.asarray(occupations, dtype=float)
    if occupations.ndim != 1:
        raise ValueError(
            f"occupations must be a one-dimensional array, got {occupations.shape}."
        )
    if np.any(np.diff(occupations) > OCCUPATION_SORT_TOL):
        raise ValueError(
            "occupations must be sorted in non-increasing order before building "
            "occupation blocks."
        )

    def sector(value: float) -> str:
        if value >= 1.0 - filled_tol:
            return "filled"
        if value <= empty_tol:
            return "empty"
        return "active"

    blocks: list[np.ndarray] = []
    if occupations.size == 0:
        return blocks

    start = 0
    current_sector = sector(float(occupations[0]))
    for idx in range(1, occupations.size):
        next_sector = sector(float(occupations[idx]))
        gap = float(occupations[idx - 1]) - float(occupations[idx])
        if next_sector != current_sector or gap > degeneracy_tol:
            blocks.append(np.arange(start, idx, dtype=int))
            start = idx
            current_sector = next_sector

    blocks.append(np.arange(start, occupations.size, dtype=int))
    return blocks

def occupation_phase_space(diagonal_occupations: np.ndarray) -> np.ndarray:
    """Return Xi_ab = n_a (1 - n_b) + n_b (1 - n_a)."""
    n = np.clip(np.real(np.asarray(diagonal_occupations, dtype=float)), 0.0, 1.0)
    return n[:, None] * (1.0 - n[None, :]) + n[None, :] * (1.0 - n[:, None])


def interaction_footprint(
    site_to_orbital: np.ndarray,
    interaction_matrix: np.ndarray,
) -> np.ndarray:
    """Return Lambda_ab = sum_ij S_ij |U_ia|^2 |U_jb|^2."""
    U_sq = np.abs(np.asarray(site_to_orbital)) ** 2
    S = np.asarray(interaction_matrix, dtype=float)
    return U_sq.T @ S @ U_sq


def _empty_structure_metrics() -> dict[str, float]:
    return {
        'structure_cost': 0.0,
        'participation_ratio_norm': 0.0,
        'decay_cost': 0.0,
        'effective_support': 0.0,
        'total_weight': 0.0,
    }


def pair_structure_metrics(
    weights: np.ndarray,
    *,
    distance_matrix: np.ndarray | None = None,
    power: float = 2.0,
    metric: str = 'hybrid',
) -> dict[str, float]:
    """Quantify how concentrated / banded a pair-weight distribution is."""
    weights = np.abs(np.asarray(weights, dtype=float))
    weights = _require_square_matrix(weights, name='weights')
    n_orbitals = weights.shape[0]

    if n_orbitals < 2:
        return _empty_structure_metrics()

    upper = np.triu(weights, k=1)
    total_weight = float(upper.sum())
    if total_weight <= EPS:
        return _empty_structure_metrics()

    sq_sum = float(np.square(upper).sum())
    max_support = max((n_orbitals * (n_orbitals - 1)) // 2, 1)
    effective_support = total_weight * total_weight / max(sq_sum, EPS)
    if max_support <= 1:
        participation_ratio_norm = 0.0
    else:
        participation_ratio_norm = (effective_support - 1.0) / (max_support - 1.0)

    distance_matrix = _resolve_distance_matrix(n_orbitals, distance_matrix)
    dist_upper = np.triu(np.asarray(distance_matrix, dtype=float), k=1)
    dist_max = float(np.max(dist_upper)) if dist_upper.size > 0 else 0.0
    if dist_max <= EPS:
        decay_cost = 0.0
    else:
        decay_cost = float(
            np.sum(upper * np.power(dist_upper / dist_max, power)) / total_weight
        )

    if metric == 'participation':
        structure_cost = participation_ratio_norm
    elif metric == 'decay':
        structure_cost = decay_cost
    elif metric == 'hybrid':
        structure_cost = 0.5 * (participation_ratio_norm + decay_cost)
    else:
        raise ValueError(
            "metric must be one of {'participation', 'decay', 'hybrid'}, got "
            f'{metric!r}.'
        )

    return {
        'structure_cost': float(structure_cost),
        'participation_ratio_norm': float(participation_ratio_norm),
        'decay_cost': float(decay_cost),
        'effective_support': float(effective_support),
        'total_weight': total_weight,
    }


def orbital_locality_metrics(
    site_to_orbital: np.ndarray,
    *,
    site_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | None]:
    """Measure orbital localization through IPR and optional real-space spread."""
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name='site_to_orbital',
    ).astype(np.complex128)
    weights = np.abs(site_to_orbital) ** 2

    ipr = np.sum(weights * weights, axis=0).real
    participation_ratio = 1.0 / np.clip(ipr, EPS, None)
    n_sites = float(site_to_orbital.shape[0])
    if n_sites <= 1:
        participation_ratio_norm = 0.0
    else:
        participation_ratio_norm = float(
            np.mean((participation_ratio - 1.0) / max(n_sites - 1.0, 1.0))
        )

    spreads = None
    spread_norm = participation_ratio_norm
    if site_positions is not None:
        site_positions = np.asarray(site_positions, dtype=float)
        if site_positions.ndim == 1:
            site_positions = site_positions[:, None]
        if site_positions.shape[0] != site_to_orbital.shape[0]:
            raise ValueError(
                'site_positions must have one entry per original orbital/site, got '
                f'{site_positions.shape[0]} positions for '
                f'{site_to_orbital.shape[0]} sites.'
            )

        centers = weights.T @ site_positions
        delta = site_positions[:, None, :] - centers[None, :, :]
        spreads = np.sum(weights[:, :, None] * delta * delta, axis=(0, 2)).real
        pair_delta = site_positions[:, None, :] - site_positions[None, :, :]
        scale = float(np.max(np.sum(pair_delta * pair_delta, axis=-1)))
        if scale <= EPS:
            spread_norm = 0.0
        else:
            spread_norm = float(np.mean(spreads / scale))

    locality_cost = (
        participation_ratio_norm
        if site_positions is None
        else 0.5 * (participation_ratio_norm + spread_norm)
    )

    return {
        'ipr': ipr,
        'participation_ratio': participation_ratio,
        'participation_ratio_norm': participation_ratio_norm,
        'spreads': spreads,
        'spread_norm': spread_norm,
        'locality_cost': float(locality_cost),
    }


def _config_value(config: Any | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def evaluate_basis_metrics(
    rdm: np.ndarray,
    site_to_orbital: np.ndarray,
    *,
    hopping_matrix: np.ndarray | None = None,
    interaction_matrix: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    config: Any | None = None,
) -> dict[str, object]:
    """Evaluate basis diagnostics without depending on a monolithic config file."""
    rdm = _hermitian_part(_require_square_matrix(rdm, name='rdm'))
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name='site_to_orbital',
    ).astype(np.complex128)

    if rdm.shape != site_to_orbital.shape:
        raise ValueError(
            'rdm and site_to_orbital must have identical shapes, got '
            f'{rdm.shape} and {site_to_orbital.shape}.'
        )

    distance_power = float(_config_value(config, 'distance_power', 2.0))
    structure_metric = _config_value(config, 'structure_metric', 'hybrid')
    lambda_occupancy = float(_config_value(config, 'lambda_occupancy', 1.0))
    lambda_hopping = float(_config_value(config, 'lambda_hopping', 1.0))
    lambda_interaction = float(_config_value(config, 'lambda_interaction', 1.0))
    lambda_locality = float(_config_value(config, 'lambda_locality', 0.1))

    gamma_rot = rotate_one_body_matrix(rdm, site_to_orbital)
    diagonal_occupations = np.clip(np.real(np.diag(gamma_rot)), 0.0, 1.0)
    offdiag = gamma_rot.copy()
    np.fill_diagonal(offdiag, 0.0)
    occupation_offdiag_cost = float(
        np.linalg.norm(offdiag, ord='fro') ** 2
        / max(np.linalg.norm(rdm, ord='fro') ** 2, EPS)
    )

    phase_space = occupation_phase_space(diagonal_occupations)
    locality_metrics = orbital_locality_metrics(
        site_to_orbital,
        site_positions=site_positions,
    )

    rotated_hopping = None
    hopping_metrics = _empty_structure_metrics()
    if hopping_matrix is not None:
        hopping_matrix = _hermitian_part(
            _require_square_matrix(hopping_matrix, name='hopping_matrix')
        )
        if hopping_matrix.shape != rdm.shape:
            raise ValueError(
                'hopping_matrix and rdm must have identical shapes, got '
                f'{hopping_matrix.shape} and {rdm.shape}.'
            )
        rotated_hopping = rotate_one_body_matrix(hopping_matrix, site_to_orbital)
        hopping_metrics = pair_structure_metrics(
            np.abs(rotated_hopping) ** 2,
            distance_matrix=distance_matrix,
            power=distance_power,
            metric=structure_metric,
        )

    lambda_ab = None
    effective_scattering = None
    interaction_metrics = _empty_structure_metrics()
    if interaction_matrix is not None:
        resolved_interaction = _resolve_interaction_matrix(
            interaction_matrix,
            site_to_orbital.shape[0],
        )
        lambda_ab = interaction_footprint(site_to_orbital, resolved_interaction)
        effective_scattering = lambda_ab * phase_space
        interaction_metrics = pair_structure_metrics(
            effective_scattering,
            distance_matrix=distance_matrix,
            power=distance_power,
            metric=structure_metric,
        )

    total_loss = (
        lambda_occupancy * occupation_offdiag_cost
        + lambda_hopping * hopping_metrics['structure_cost']
        + lambda_interaction * interaction_metrics['structure_cost']
        + lambda_locality * locality_metrics['locality_cost']
    )

    return {
        'loss': float(total_loss),
        'occupation_offdiag_cost': occupation_offdiag_cost,
        'rdm_in_basis': gamma_rot,
        'diag_occupations': diagonal_occupations,
        'phase_space': phase_space,
        'rotated_hopping': rotated_hopping,
        'hopping_metrics': hopping_metrics,
        'interaction_footprint': lambda_ab,
        'effective_scattering': effective_scattering,
        'interaction_metrics': interaction_metrics,
        'locality_metrics': locality_metrics,
    }


__all__ = [
    'EPS',
    'OCCUPATION_SORT_TOL',
    'NO_CONSISTENCY_RTOL',
    'NO_CONSISTENCY_ATOL',
    '_hermitian_part',
    '_require_square_matrix',
    '_resolve_distance_matrix',
    '_resolve_interaction_matrix',
    '_validate_no_reference_data',
    'build_occupation_blocks',
    'evaluate_basis_metrics',
    'extract_hamiltonian_terms',
    'extract_interaction_matrix',
    'extract_one_body_hopping_matrix',
    'interaction_footprint',
    'occupation_phase_space',
    'orbital_locality_metrics',
    'pair_structure_metrics',
    'rotate_hamiltonian_to_basis',
    'rotate_hamiltonian_to_natural_orbitals',
    'rotate_one_body_matrix',
]
