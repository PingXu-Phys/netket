"""Post-NO orbital optimization with a JAX/Optax optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import optax

from netket.operator._fermion2nd.base import FermionOperator2ndBase

try:
    from .iteration_occ_func_simple import (
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


@dataclass(slots=True)
class NOOptimizationConfig:
    """Hyperparameters for the post-NO orbital optimisation.

    Loss weights
    ------------
    lambda_occupancy : float
        Weight of C_γ — penalises off-diagonal 1-RDM elements.
        Acts as a *regulariser* that keeps the result close to the NO basis;
        keep small (≪ lambda_kondo) to allow the optimiser to explore.
    lambda_hopping : float
        Weight of C_t — penalises delocalised / slowly-decaying hopping.
        Requires hopping_matrix or H to be supplied.
    lambda_kondo : float
        Weight of C_K — penalises large W_ab = Λ_ab · Ξ_ab (Kondo footprint
        × phase space).  The main physical target. Requires exchange_profile.
    lambda_locality : float
        Weight of C_loc — IPR-based orbital localisation regulariser.

    Occupation block splitting
    --------------------------
    filled_tol  : orbitals with n > 1 − filled_tol are "near-full"  (frozen block).
    empty_tol   : orbitals with n < empty_tol are "near-empty" (frozen block).
    degeneracy_tol : further splits a sector into sub-blocks when adjacent
                     occupation gap exceeds this threshold.

    Optimiser
    ---------
    optimizer_name : "adam" | "adamw" | "sgd"
    learning_rate, weight_decay, gradient_clip, n_steps, log_every

    Structure metric
    ----------------
    structure_metric : "participation" | "decay" | "hybrid"
        How the pair-weight matrix is scored.  "hybrid" averages both.
    distance_power : exponent p in the decay term  Σ w_ab (d_ab/d_max)^p.

    Miscellaneous
    -------------
    hamiltonian_cutoff : terms below this threshold are dropped when rotating H.
    real_orbitals : if True (default), the block rotations are real orthogonal
                    matrices; set False only when the problem is genuinely complex.
    """

    lambda_occupancy: float = 1.0
    lambda_hopping: float = 1.0
    lambda_kondo: float = 1.0
    lambda_locality: float = 0.1
    filled_tol: float = 0.05
    empty_tol: float = 0.05
    degeneracy_tol: float = 0.02
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    n_steps: int = 200
    log_every: int = 25
    structure_metric: str = "hybrid"
    distance_power: float = 2.0
    hamiltonian_cutoff: float | None = None
    real_orbitals: bool = True


def _coerce_config(config: NOOptimizationConfig | None) -> NOOptimizationConfig:
    if config is None:
        config = NOOptimizationConfig()

    if config.optimizer_name not in {"adam", "adamw", "sgd"}:
        raise ValueError(
            "optimizer_name must be one of {'adam', 'adamw', 'sgd'}, got "
            f"{config.optimizer_name!r}."
        )
    if config.structure_metric not in {"participation", "decay", "hybrid"}:
        raise ValueError(
            "structure_metric must be one of {'participation', 'decay', 'hybrid'}, "
            f"got {config.structure_metric!r}."
        )
    if config.n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {config.n_steps}.")
    if config.log_every <= 0:
        raise ValueError(f"log_every must be positive, got {config.log_every}.")
    if config.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be strictly positive, got {config.learning_rate}."
        )
    if config.gradient_clip is not None and config.gradient_clip <= 0:
        raise ValueError(
            "gradient_clip must be positive when provided, got "
            f"{config.gradient_clip}."
        )

    return config

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


def _resolve_exchange_profile(
    exchange_profile: float | np.ndarray | None,
    n_sites: int,
) -> np.ndarray:
    if exchange_profile is None:
        return np.ones(n_sites, dtype=float)

    if np.isscalar(exchange_profile):
        return np.full(n_sites, float(exchange_profile), dtype=float)

    exchange_profile = np.asarray(exchange_profile, dtype=float)
    if exchange_profile.shape != (n_sites,):
        raise ValueError(
            "exchange_profile must be scalar or have shape "
            f"({n_sites},), got {exchange_profile.shape}."
        )
    return exchange_profile


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


def _validate_occupation_blocks(
    blocks: list[np.ndarray],
    *,
    n_orbitals: int,
) -> list[np.ndarray]:
    validated_blocks: list[np.ndarray] = []
    seen = np.zeros(n_orbitals, dtype=bool)

    for block_id, block in enumerate(blocks):
        block_array = np.asarray(block, dtype=int)
        if block_array.ndim != 1:
            raise ValueError(
                "Each occupation block must be one-dimensional, got "
                f"{block_array.shape} for block {block_id}."
            )
        if block_array.size == 0:
            raise ValueError(f"occupation_blocks[{block_id}] must not be empty.")
        if np.any(block_array < 0) or np.any(block_array >= n_orbitals):
            raise ValueError(
                f"occupation_blocks[{block_id}] contains indices outside "
                f"[0, {n_orbitals - 1}]."
            )
        if np.unique(block_array).size != block_array.size:
            raise ValueError(
                f"occupation_blocks[{block_id}] contains duplicate orbital indices."
            )
        if np.any(seen[block_array]):
            raise ValueError("occupation_blocks must not overlap.")

        seen[block_array] = True
        validated_blocks.append(block_array.copy())

    if not np.all(seen):
        missing = np.flatnonzero(~seen).tolist()
        raise ValueError(
            "occupation_blocks must cover each orbital exactly once; missing "
            f"indices {missing}."
        )

    return validated_blocks


def _validate_no_context(context: dict[str, Any]) -> dict[str, Any]:
    required_keys = (
        "config",
        "rdm",
        "natural_occupations",
        "natural_orbitals",
        "occupation_blocks",
        "distance_matrix",
        "hopping_matrix",
        "hopping_no",
        "exchange_profile",
        "site_positions",
    )
    missing = [key for key in required_keys if key not in context]
    if missing:
        raise KeyError(f"context is missing required keys: {missing}.")

    config = _coerce_config(context["config"])
    rdm = _hermitian_part(
        _require_square_matrix(np.asarray(context["rdm"], dtype=np.complex128), name="rdm")
    )
    natural_orbitals = _require_square_matrix(
        np.asarray(context["natural_orbitals"], dtype=np.complex128),
        name="natural_orbitals",
    )
    natural_occupations = np.asarray(context["natural_occupations"], dtype=float)
    _validate_no_reference_data(rdm, natural_occupations, natural_orbitals)

    n_orbitals = natural_orbitals.shape[0]
    occupation_blocks = _validate_occupation_blocks(
        context["occupation_blocks"],
        n_orbitals=n_orbitals,
    )
    distance_matrix = _resolve_distance_matrix(n_orbitals, context["distance_matrix"])

    hopping_matrix = None
    if context["hopping_matrix"] is not None:
        hopping_matrix = _hermitian_part(
            _require_square_matrix(
                np.asarray(context["hopping_matrix"], dtype=np.complex128),
                name="hopping_matrix",
            )
        )
        if hopping_matrix.shape != (n_orbitals, n_orbitals):
            raise ValueError(
                "hopping_matrix and natural_orbitals must have identical shapes, got "
                f"{hopping_matrix.shape} and {(n_orbitals, n_orbitals)}."
            )

    hopping_no = None
    if context["hopping_no"] is not None:
        hopping_no = _hermitian_part(
            _require_square_matrix(
                np.asarray(context["hopping_no"], dtype=np.complex128),
                name="hopping_no",
            )
        )
        if hopping_no.shape != (n_orbitals, n_orbitals):
            raise ValueError(
                "hopping_no and natural_orbitals must have identical shapes, got "
                f"{hopping_no.shape} and {(n_orbitals, n_orbitals)}."
            )

    exchange_profile = (
        None
        if context["exchange_profile"] is None
        else _resolve_exchange_profile(context["exchange_profile"], n_orbitals)
    )

    site_positions = None
    if context["site_positions"] is not None:
        site_positions = np.asarray(context["site_positions"], dtype=float)
        if site_positions.ndim == 1:
            site_positions = site_positions[:, None]
        if site_positions.shape[0] != n_orbitals:
            raise ValueError(
                "site_positions must have one entry per original site/orbital, got "
                f"{site_positions.shape[0]} for {n_orbitals}."
            )

    return {
        "config": config,
        "rdm": rdm,
        "natural_occupations": natural_occupations.copy(),
        "natural_orbitals": natural_orbitals.copy(),
        "occupation_blocks": occupation_blocks,
        "distance_matrix": distance_matrix.copy(),
        "hopping_matrix": None if hopping_matrix is None else hopping_matrix.copy(),
        "hopping_no": None if hopping_no is None else hopping_no.copy(),
        "exchange_profile": None
        if exchange_profile is None
        else exchange_profile.copy(),
        "site_positions": None if site_positions is None else site_positions.copy(),
    }


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


def occupation_phase_space(diagonal_occupations: np.ndarray) -> np.ndarray:
    """Return Xi_ab = n_a (1 - n_b) + n_b (1 - n_a)."""
    n = np.clip(np.real(np.asarray(diagonal_occupations, dtype=float)), 0.0, 1.0)
    return n[:, None] * (1.0 - n[None, :]) + n[None, :] * (1.0 - n[:, None])


def kondo_tensor_from_site_rotation(
    site_to_orbital: np.ndarray,
    *,
    exchange_profile: float | np.ndarray | None = None,
) -> np.ndarray:
    """Build K_iab = J_i U_ia^* U_ib in the rotated basis."""
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name="site_to_orbital",
    ).astype(np.complex128)
    exchange_profile = _resolve_exchange_profile(
        exchange_profile,
        site_to_orbital.shape[0],
    )
    return (
        exchange_profile[:, None, None]
        * site_to_orbital.conj()[:, :, None]
        * site_to_orbital[:, None, :]
    )


def kondo_footprint(
    site_to_orbital: np.ndarray,
    *,
    exchange_profile: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return Lambda_ab = sum_i |J_i U_ia^* U_ib|."""
    return np.sum(
        np.abs(
            kondo_tensor_from_site_rotation(
                site_to_orbital,
                exchange_profile=exchange_profile,
            )
        ),
        axis=0,
    )

def _empty_structure_metrics() -> dict[str, float]:
    return {
        "structure_cost": 0.0,
        "participation_ratio_norm": 0.0,
        "decay_cost": 0.0,
        "effective_support": 0.0,
        "total_weight": 0.0,
    }


def pair_structure_metrics(
    weights: np.ndarray,
    *,
    distance_matrix: np.ndarray | None = None,
    power: float = 2.0,
    metric: str = "hybrid",
) -> dict[str, float]:
    """Quantify how concentrated / banded a pair-weight distribution is."""
    weights = np.abs(np.asarray(weights, dtype=float))
    weights = _require_square_matrix(weights, name="weights")
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

    if metric == "participation":
        structure_cost = participation_ratio_norm
    elif metric == "decay":
        structure_cost = decay_cost
    elif metric == "hybrid":
        structure_cost = 0.5 * (participation_ratio_norm + decay_cost)
    else:
        raise ValueError(
            "metric must be one of {'participation', 'decay', 'hybrid'}, got "
            f"{metric!r}."
        )

    return {
        "structure_cost": float(structure_cost),
        "participation_ratio_norm": float(participation_ratio_norm),
        "decay_cost": float(decay_cost),
        "effective_support": float(effective_support),
        "total_weight": total_weight,
    }


def orbital_locality_metrics(
    site_to_orbital: np.ndarray,
    *,
    site_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | None]:
    """Measure orbital localization through IPR and optional real-space spread."""
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name="site_to_orbital",
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
                "site_positions must have one entry per original orbital/site, got "
                f"{site_positions.shape[0]} positions for "
                f"{site_to_orbital.shape[0]} sites."
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
        "ipr": ipr,
        "participation_ratio": participation_ratio,
        "participation_ratio_norm": participation_ratio_norm,
        "spreads": spreads,
        "spread_norm": spread_norm,
        "locality_cost": float(locality_cost),
    }


def evaluate_basis_metrics(
    rdm: np.ndarray,
    site_to_orbital: np.ndarray,
    *,
    hopping_matrix: np.ndarray | None = None,
    exchange_profile: float | np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    config: NOOptimizationConfig | None = None,
) -> dict[str, object]:
    """Evaluate the multi-objective hardness diagnostics for one basis."""
    config = _coerce_config(config)
    rdm = _hermitian_part(_require_square_matrix(rdm, name="rdm"))
    site_to_orbital = _require_square_matrix(
        site_to_orbital,
        name="site_to_orbital",
    ).astype(np.complex128)

    if rdm.shape != site_to_orbital.shape:
        raise ValueError(
            "rdm and site_to_orbital must have identical shapes, got "
            f"{rdm.shape} and {site_to_orbital.shape}."
        )

    gamma_rot = rotate_one_body_matrix(rdm, site_to_orbital)
    diagonal_occupations = np.clip(np.real(np.diag(gamma_rot)), 0.0, 1.0)
    offdiag = gamma_rot.copy()
    np.fill_diagonal(offdiag, 0.0)
    occupation_offdiag_cost = float(
        np.linalg.norm(offdiag, ord="fro") ** 2
        / max(np.linalg.norm(rdm, ord="fro") ** 2, EPS)
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
            _require_square_matrix(hopping_matrix, name="hopping_matrix")
        )
        if hopping_matrix.shape != rdm.shape:
            raise ValueError(
                "hopping_matrix and rdm must have identical shapes, got "
                f"{hopping_matrix.shape} and {rdm.shape}."
            )
        rotated_hopping = rotate_one_body_matrix(hopping_matrix, site_to_orbital)
        hopping_metrics = pair_structure_metrics(
            np.abs(rotated_hopping) ** 2,
            distance_matrix=distance_matrix,
            power=config.distance_power,
            metric=config.structure_metric,
        )

    lambda_ab = None
    effective_scattering = None
    kondo_metrics = _empty_structure_metrics()
    if exchange_profile is not None:
        lambda_ab = kondo_footprint(
            site_to_orbital,
            exchange_profile=exchange_profile,
        )
        effective_scattering = lambda_ab * phase_space
        kondo_metrics = pair_structure_metrics(
            effective_scattering,
            distance_matrix=distance_matrix,
            power=config.distance_power,
            metric=config.structure_metric,
        )

    total_loss = (
        config.lambda_occupancy * occupation_offdiag_cost
        + config.lambda_hopping * hopping_metrics["structure_cost"]
        + config.lambda_kondo * kondo_metrics["structure_cost"]
        + config.lambda_locality * locality_metrics["locality_cost"]
    )

    return {
        "loss": float(total_loss),
        "occupation_offdiag_cost": occupation_offdiag_cost,
        "rdm_in_basis": gamma_rot,
        "diag_occupations": diagonal_occupations,
        "phase_space": phase_space,
        "rotated_hopping": rotated_hopping,
        "hopping_metrics": hopping_metrics,
        "kondo_footprint": lambda_ab,
        "effective_scattering": effective_scattering,
        "kondo_metrics": kondo_metrics,
        "locality_metrics": locality_metrics,
    }


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


def inspect_occupation_structure(
    *,
    rdm: np.ndarray | None = None,
    occupations: np.ndarray | None = None,
    natural_orbitals: np.ndarray | None = None,
    config: NOOptimizationConfig | None = None,
) -> dict[str, Any]:
    """Inspect NO occupations, blocks, and phase-space structure."""
    config = _coerce_config(config)

    if occupations is None:
        if rdm is None:
            raise ValueError("Provide either occupations or rdm.")
        occupations, natural_orbitals = natural_orbitals_from_rdm(rdm)
    else:
        occupations = np.asarray(occupations, dtype=float)
        if occupations.ndim != 1:
            raise ValueError(
                f"occupations must be a one-dimensional array, got {occupations.shape}."
            )

    occupations = np.asarray(occupations, dtype=float)
    blocks = build_occupation_blocks(
        occupations,
        filled_tol=config.filled_tol,
        empty_tol=config.empty_tol,
        degeneracy_tol=config.degeneracy_tol,
    )

    return {
        "occupations": occupations.copy(),
        "natural_orbitals": None
        if natural_orbitals is None
        else np.asarray(natural_orbitals, dtype=np.complex128).copy(),
        "occupation_blocks": [block.copy() for block in blocks],
        "phase_space": occupation_phase_space(occupations).copy(),
        "active_weight": float(np.sum(2.0 * occupations * (1.0 - occupations))),
    }


def prepare_no_optimization_context(
    *,
    H=None,
    rdm: np.ndarray,
    natural_occupations: np.ndarray | None = None,
    natural_orbitals: np.ndarray | None = None,
    hopping_matrix: np.ndarray | None = None,
    exchange_profile: float | np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    config: NOOptimizationConfig | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Bundle all inputs needed to define the post-NO loss and optimization."""
    config = _coerce_config(config)
    rdm = _hermitian_part(_require_square_matrix(rdm, name="rdm"))

    if natural_occupations is None or natural_orbitals is None:
        natural_occupations, natural_orbitals = natural_orbitals_from_rdm(rdm)

    natural_occupations = np.asarray(natural_occupations, dtype=float)
    natural_orbitals = _require_square_matrix(
        np.asarray(natural_orbitals, dtype=np.complex128),
        name="natural_orbitals",
    )
    _validate_no_reference_data(rdm, natural_occupations, natural_orbitals)

    if hopping_matrix is None and H is not None:
        hopping_matrix = extract_one_body_hopping_matrix(
            H,
            spin_symmetric=spin_symmetric,
        )

    if hopping_matrix is not None:
        hopping_matrix = _hermitian_part(
            _require_square_matrix(hopping_matrix, name="hopping_matrix")
        )
        hopping_no = rotate_one_body_matrix(hopping_matrix, natural_orbitals)
    else:
        hopping_no = None

    resolved_distance = _resolve_distance_matrix(
        natural_orbitals.shape[0],
        distance_matrix,
    )
    resolved_exchange = (
        None
        if exchange_profile is None
        else _resolve_exchange_profile(exchange_profile, natural_orbitals.shape[0])
    )

    resolved_positions = None
    if site_positions is not None:
        resolved_positions = np.asarray(site_positions, dtype=float)
        if resolved_positions.ndim == 1:
            resolved_positions = resolved_positions[:, None]
        if resolved_positions.shape[0] != natural_orbitals.shape[0]:
            raise ValueError(
                "site_positions must have one entry per original site/orbital, got "
                f"{resolved_positions.shape[0]} for {natural_orbitals.shape[0]}."
            )

    occupation_blocks = build_occupation_blocks(
        natural_occupations,
        filled_tol=config.filled_tol,
        empty_tol=config.empty_tol,
        degeneracy_tol=config.degeneracy_tol,
    )
    initial_metrics = evaluate_basis_metrics(
        rdm,
        natural_orbitals,
        hopping_matrix=hopping_matrix,
        exchange_profile=resolved_exchange,
        distance_matrix=resolved_distance,
        site_positions=resolved_positions,
        config=config,
    )

    return {
        "H": H,
        "rdm": rdm,
        "spin_symmetric": spin_symmetric,
        "config": config,
        "natural_occupations": natural_occupations.copy(),
        "natural_orbitals": natural_orbitals.copy(),
        "occupation_blocks": [block.copy() for block in occupation_blocks],
        "hopping_matrix": None if hopping_matrix is None else hopping_matrix.copy(),
        "hopping_no": None if hopping_no is None else hopping_no.copy(),
        "exchange_profile": None
        if resolved_exchange is None
        else resolved_exchange.copy(),
        "distance_matrix": resolved_distance.copy(),
        "site_positions": None
        if resolved_positions is None
        else resolved_positions.copy(),
        "initial_metrics": initial_metrics,
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
        gap = abs(float(occupations[idx - 1]) - float(occupations[idx]))
        if next_sector != current_sector or gap > degeneracy_tol:
            blocks.append(np.arange(start, idx, dtype=int))
            start = idx
            current_sector = next_sector

    blocks.append(np.arange(start, occupations.size, dtype=int))
    return blocks


def create_optax_optimizer(
    config: NOOptimizationConfig | None = None,
) -> optax.GradientTransformation:
    """Create the Optax optimizer used for post-NO optimization."""
    config = _coerce_config(config)

    transforms: list[optax.GradientTransformation] = []
    if config.gradient_clip is not None:
        transforms.append(optax.clip_by_global_norm(config.gradient_clip))

    if config.optimizer_name == "adam":
        transforms.append(optax.adam(config.learning_rate))
    elif config.optimizer_name == "adamw":
        transforms.append(
            optax.adamw(
                config.learning_rate,
                weight_decay=config.weight_decay,
            )
        )
    else:
        transforms.append(optax.sgd(config.learning_rate))

    return optax.chain(*transforms)


def _init_block_params(
    blocks: list[np.ndarray],
    *,
    real_orbitals: bool = True,
) -> dict[str, dict[str, jax.Array]]:
    params: dict[str, dict[str, jax.Array]] = {}
    for block_id, block in enumerate(blocks):
        if block.size <= 1:
            continue
        shape = (int(block.size), int(block.size))
        entry: dict[str, jax.Array] = {"real": jnp.zeros(shape, dtype=jnp.float64)}
        if not real_orbitals:
            entry["imag"] = jnp.zeros(shape, dtype=jnp.float64)
        params[f"block_{block_id}"] = entry
    return params


def _block_unitary(param: dict[str, jax.Array], *, real_orbitals: bool = True) -> jax.Array:
    if real_orbitals:
        # anti-symmetric generator → real orthogonal matrix
        generator = param["real"] - param["real"].T
    else:
        # anti-Hermitian generator → complex unitary matrix
        raw = param["real"] + 1j * param["imag"]
        generator = raw - jnp.conjugate(raw.T)
    return jsp_linalg.expm(generator)


def _build_post_no_rotation(
    params: dict[str, dict[str, jax.Array]],
    blocks: list[np.ndarray],
    n_orbitals: int,
    *,
    real_orbitals: bool = True,
) -> jax.Array:
    dtype = jnp.float64 if real_orbitals else jnp.complex128
    rotation = jnp.eye(n_orbitals, dtype=dtype)
    for block_id, block in enumerate(blocks):
        if block.size <= 1:
            continue
        block_idx = jnp.asarray(block, dtype=jnp.int32)
        block_rotation = _block_unitary(params[f"block_{block_id}"], real_orbitals=real_orbitals)
        rotation = rotation.at[block_idx[:, None], block_idx[None, :]].set(block_rotation)
    return rotation


def _structure_cost_jax(
    weights: jax.Array,
    *,
    distance_matrix: jax.Array,
    power: float,
    metric: str,
) -> jax.Array:
    weights = jnp.abs(weights)
    upper = jnp.triu(weights, k=1)
    total_weight = jnp.sum(upper)
    total_safe = jnp.maximum(total_weight, EPS)

    n_orbitals = weights.shape[0]
    max_support = max((n_orbitals * (n_orbitals - 1)) // 2, 1)
    sq_sum = jnp.sum(jnp.square(upper))
    effective_support = total_safe * total_safe / (sq_sum + EPS)
    if max_support <= 1:
        participation = jnp.array(0.0)
    else:
        participation = (effective_support - 1.0) / float(max_support - 1)
    participation = jnp.where(total_weight <= EPS, 0.0, participation)

    dist_upper = jnp.triu(distance_matrix, k=1)
    dist_max = jnp.maximum(jnp.max(dist_upper), EPS)
    decay = jnp.sum(upper * jnp.power(dist_upper / dist_max, power)) / total_safe
    decay = jnp.where(total_weight <= EPS, 0.0, decay)

    if metric == "participation":
        return participation
    if metric == "decay":
        return decay
    return 0.5 * (participation + decay)


def _locality_cost_jax(
    site_to_orbital: jax.Array,
    *,
    site_positions: jax.Array | None,
) -> jax.Array:
    weights = jnp.abs(site_to_orbital) ** 2
    ipr = jnp.sum(weights * weights, axis=0)
    participation_ratio = 1.0 / (ipr + EPS)

    n_sites = site_to_orbital.shape[0]
    if n_sites <= 1:
        participation_norm = jnp.array(0.0)
    else:
        participation_norm = jnp.mean(
            (participation_ratio - 1.0) / float(max(n_sites - 1, 1))
        )

    if site_positions is None:
        return participation_norm

    centers = weights.T @ site_positions
    delta = site_positions[:, None, :] - centers[None, :, :]
    spreads = jnp.sum(weights[:, :, None] * delta * delta, axis=(0, 2))
    pair_delta = site_positions[:, None, :] - site_positions[None, :, :]
    scale = jnp.maximum(jnp.max(jnp.sum(pair_delta * pair_delta, axis=-1)), EPS)
    spread_norm = jnp.mean(spreads / scale)
    return 0.5 * (participation_norm + spread_norm)


def _loss_terms_jax(
    post_no_rotation: jax.Array,
    *,
    natural_occupations: jax.Array,
    natural_orbitals: jax.Array,
    hopping_no: jax.Array | None,
    exchange_profile: jax.Array | None,
    distance_matrix: jax.Array,
    site_positions: jax.Array | None,
    config: NOOptimizationConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    diag_n = jnp.diag(natural_occupations)
    gamma_rot = post_no_rotation.conj().T @ diag_n @ post_no_rotation
    offdiag = gamma_rot - jnp.diag(jnp.diag(gamma_rot))
    occ_norm = jnp.maximum(jnp.sum(jnp.square(natural_occupations)), EPS)
    occupancy_cost = jnp.sum(jnp.abs(offdiag) ** 2) / occ_norm

    hopping_cost = jnp.array(0.0)
    if hopping_no is not None:
        hopping_rot = post_no_rotation.conj().T @ hopping_no @ post_no_rotation
        hopping_cost = _structure_cost_jax(
            jnp.abs(hopping_rot) ** 2,
            distance_matrix=distance_matrix,
            power=config.distance_power,
            metric=config.structure_metric,
        )

    site_to_orbital = natural_orbitals @ post_no_rotation
    kondo_cost = jnp.array(0.0)
    if exchange_profile is not None:
        lambda_ab = jnp.sum(
            jnp.abs(
                exchange_profile[:, None, None]
                * jnp.conjugate(site_to_orbital)[:, :, None]
                * site_to_orbital[:, None, :]
            ),
            axis=0,
        )
        diag_occ = jnp.clip(jnp.real(jnp.diag(gamma_rot)), 0.0, 1.0)
        phase_space = (
            diag_occ[:, None] * (1.0 - diag_occ[None, :])
            + diag_occ[None, :] * (1.0 - diag_occ[:, None])
        )
        kondo_cost = _structure_cost_jax(
            lambda_ab * phase_space,
            distance_matrix=distance_matrix,
            power=config.distance_power,
            metric=config.structure_metric,
        )

    locality_cost = _locality_cost_jax(
        site_to_orbital,
        site_positions=site_positions,
    )

    total = (
        config.lambda_occupancy * occupancy_cost
        + config.lambda_hopping * hopping_cost
        + config.lambda_kondo * kondo_cost
        + config.lambda_locality * locality_cost
    )
    return jnp.real(total), (
        jnp.real(occupancy_cost),
        jnp.real(hopping_cost),
        jnp.real(kondo_cost),
        jnp.real(locality_cost),
    )


def build_no_loss_fn(context: dict[str, Any]):
    """Return a JAX loss function for use with Optax/JAX."""
    validated = _validate_no_context(context)
    config = validated["config"]
    real_orbitals = config.real_orbitals
    arr_dtype = jnp.float64 if real_orbitals else jnp.complex128
    blocks = validated["occupation_blocks"]
    n_orbitals = validated["natural_orbitals"].shape[0]
    natural_occupations_jax = jnp.asarray(validated["natural_occupations"])
    _no = validated["natural_orbitals"]
    natural_orbitals_jax = jnp.asarray(np.real(_no) if real_orbitals else _no, dtype=arr_dtype)
    _hop = validated["hopping_no"]
    hopping_no_jax = (
        None if _hop is None
        else jnp.asarray(np.real(_hop) if real_orbitals else _hop, dtype=arr_dtype)
    )
    exchange_profile_jax = (
        None
        if validated["exchange_profile"] is None
        else jnp.asarray(validated["exchange_profile"])
    )
    distance_matrix_jax = jnp.asarray(validated["distance_matrix"])
    site_positions_jax = (
        None
        if validated["site_positions"] is None
        else jnp.asarray(validated["site_positions"])
    )

    def loss_fn(current_params: dict[str, dict[str, jax.Array]]) -> jax.Array:
        post_no_rotation = _build_post_no_rotation(
            current_params,
            blocks,
            n_orbitals,
            real_orbitals=real_orbitals,
        )
        loss, _ = _loss_terms_jax(
            post_no_rotation,
            natural_occupations=natural_occupations_jax,
            natural_orbitals=natural_orbitals_jax,
            hopping_no=hopping_no_jax,
            exchange_profile=exchange_profile_jax,
            distance_matrix=distance_matrix_jax,
            site_positions=site_positions_jax,
            config=config,
        )
        return loss

    return loss_fn


def evaluate_no_loss(
    context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate the loss and diagnostics at a user-supplied post-NO rotation."""
    validated = _validate_no_context(context)
    config = validated["config"]
    real_orbitals = config.real_orbitals
    arr_dtype = np.float64 if real_orbitals else np.complex128
    blocks = validated["occupation_blocks"]
    natural_orbitals = validated["natural_orbitals"]
    n_orbitals = natural_orbitals.shape[0]

    if post_no_rotation is None:
        if params is None:
            post_no_rotation = np.eye(n_orbitals, dtype=arr_dtype)
        else:
            post_no_rotation = np.asarray(
                _build_post_no_rotation(params, blocks, n_orbitals, real_orbitals=real_orbitals)
            )
    else:
        post_no_rotation = _require_square_matrix(
            np.asarray(post_no_rotation, dtype=arr_dtype),
            name="post_no_rotation",
        )

    total, terms = _loss_terms_jax(
        jnp.asarray(post_no_rotation),
        natural_occupations=jnp.asarray(validated["natural_occupations"]),
        natural_orbitals=jnp.asarray(validated["natural_orbitals"]),
        hopping_no=None
        if validated["hopping_no"] is None
        else jnp.asarray(validated["hopping_no"]),
        exchange_profile=None
        if validated["exchange_profile"] is None
        else jnp.asarray(validated["exchange_profile"]),
        distance_matrix=jnp.asarray(validated["distance_matrix"]),
        site_positions=None
        if validated["site_positions"] is None
        else jnp.asarray(validated["site_positions"]),
        config=config,
    )

    site_to_optimized_orbital = natural_orbitals @ post_no_rotation
    metrics = evaluate_basis_metrics(
        validated["rdm"],
        site_to_optimized_orbital,
        hopping_matrix=validated["hopping_matrix"],
        exchange_profile=validated["exchange_profile"],
        distance_matrix=validated["distance_matrix"],
        site_positions=validated["site_positions"],
        config=config,
    )

    return {
        "loss": float(total),
        "loss_terms": {
            "occupancy": float(terms[0]),
            "hopping": float(terms[1]),
            "kondo": float(terms[2]),
            "locality": float(terms[3]),
        },
        "post_no_rotation": post_no_rotation,
        "site_to_optimized_orbital": site_to_optimized_orbital,
        "metrics": metrics,
    }


def optimize_no_from_context(
    context: dict[str, Any],
    *,
    optimizer: optax.GradientTransformation | None = None,
) -> dict[str, Any]:
    """Optimize starting from a precomputed context dictionary."""
    return optimize_no_basis(
        H=context["H"],
        rdm=context["rdm"],
        natural_occupations=context["natural_occupations"],
        natural_orbitals=context["natural_orbitals"],
        hopping_matrix=context["hopping_matrix"],
        exchange_profile=context["exchange_profile"],
        distance_matrix=context["distance_matrix"],
        site_positions=context["site_positions"],
        config=context["config"],
        optimizer=optimizer,
        spin_symmetric=context["spin_symmetric"],
    )

def _metrics_summary(step: int, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "step": int(step),
        "loss": float(metrics["loss"]),
        "occupation_offdiag_cost": float(metrics["occupation_offdiag_cost"]),
        "hopping_structure_cost": float(metrics["hopping_metrics"]["structure_cost"]),
        "kondo_structure_cost": float(metrics["kondo_metrics"]["structure_cost"]),
        "locality_cost": float(metrics["locality_metrics"]["locality_cost"]),
        "diag_occupations": np.asarray(metrics["diag_occupations"], dtype=float).copy(),
    }


def optimize_no_basis(
    *,
    H=None,
    rdm: np.ndarray,
    natural_occupations: np.ndarray | None = None,
    natural_orbitals: np.ndarray | None = None,
    hopping_matrix: np.ndarray | None = None,
    exchange_profile: float | np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    config: NOOptimizationConfig | None = None,
    optimizer: optax.GradientTransformation | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Find an optimised one-particle basis starting from the natural orbitals.

    All matrix inputs must be expressed in the **same current basis**
    (typically the site / impurity basis you are working in).

    Parameters
    ----------
    rdm : (n, n) real-symmetric or Hermitian array
        1-RDM in the current basis.  Diagonalised internally to obtain the
        natural orbitals U₀ and natural occupations nᵢ.
    H : FermionOperator2nd, optional
        Full second-quantised Hamiltonian.  If given, the one-body hopping
        matrix is extracted automatically (overrides ``hopping_matrix``), and
        the returned ``optimised_hamiltonian`` is H rotated into the new basis.
    hopping_matrix : (n, n) array, optional
        One-body hopping/kinetic matrix tᵢⱼ in the current basis.
        Used for the C_t sparsity cost.  Ignored when H is supplied.
    exchange_profile : scalar or (n,) array, optional
        On-site Kondo coupling Jᵢ.  Enables the C_K term.  Pass ``None`` to
        disable the Kondo cost entirely.
    distance_matrix : (n, n) array, optional
        Pairwise distances dₐᵦ used to score hopping / Kondo decay.
        Defaults to |a − b| (1-D index distance).
    site_positions : (n,) or (n, d) array, optional
        Real-space coordinates of the sites.  Activates the real-space spread
        contribution to the locality cost C_loc.
    natural_occupations, natural_orbitals : optional
        Pre-computed NO diagonalisation of ``rdm``.  Computed automatically
        when omitted; supply them to skip re-diagonalisation.
    config : NOOptimizationConfig, optional
        All hyperparameters.  Uses defaults when omitted.
    optimizer : optax.GradientTransformation, optional
        Custom Optax optimiser.  Built from ``config`` when omitted.
    spin_symmetric : bool
        Whether H is spin-symmetric (used only for Hamiltonian rotation).

    What the optimiser does
    -----------------------
    1. Diagonalise rdm → natural orbitals U₀ (columns), occupations nᵢ.
    2. Split orbitals into blocks by occupation sector (near-full / active /
       near-empty) and degeneracy gaps.  Near-full and near-empty blocks with
       a single orbital are frozen (identity rotation).
    3. Within each active block, optimise a real-orthogonal (or unitary) matrix
       V_b via gradient descent on the loss L(V).
    4. Final transformation: U = U₀ @ block_diag(V_b).

    Loss function
    -------------
    L(V) = λ_γ C_γ  +  λ_t C_t  +  λ_K C_K  +  λ_loc C_loc

    C_γ   off-diagonal cost of 1-RDM  (regulariser, keep λ_γ small)
    C_t   hopping sparsity / decay cost
    C_K   W_ab = Λ_ab · Ξ_ab  structure cost  (main physical target)
    C_loc orbital localisation (IPR)

    Returns
    -------
    dict with keys:

    site_to_optimized_orbital : (n, n)
        Full change-of-basis matrix U = U₀ @ V.
        Column j = optimised orbital j expressed in the input basis.
        Use this to rotate the Hamiltonian manually if H was not passed.
    rotation_in_no_basis : (n, n)
        The post-NO block rotation V alone.
    natural_orbitals : (n, n)
        The NO eigenvectors U₀ (columns).
    natural_occupations : (n,)
        Natural orbital occupation numbers nᵢ (descending).
    occupation_blocks : list of index arrays
        Block partition of the n orbitals.
    optimized_rdm : (n, n)
        1-RDM in the optimised basis.
    optimized_diag_occupations : (n,)
        Diagonal occupation numbers in the optimised basis.
    optimized_hopping_matrix : (n, n) or None
        Hopping matrix in the optimised basis.
    kondo_footprint : (n, n) or None
        Λ_ab = Σᵢ |Jᵢ Uᵢₐ* Uᵢᵦ|  in the optimised basis.
    effective_scattering : (n, n) or None
        W_ab = Λ_ab · Ξ_ab  in the optimised basis.
    optimized_hamiltonian : FermionOperator2nd or None
        H rotated into the optimised basis (only when H was supplied).
    initial_metrics, final_metrics : dict
        Full diagnostic snapshots before and after optimisation.
    final_loss_terms : dict
        {"occupancy", "hopping", "kondo", "locality"} breakdown.
    history : list of dicts
        Metrics logged every ``config.log_every`` steps.
    """
    config = _coerce_config(config)
    rdm = _hermitian_part(_require_square_matrix(rdm, name="rdm"))

    if natural_occupations is None or natural_orbitals is None:
        natural_occupations, natural_orbitals = natural_orbitals_from_rdm(rdm)

    natural_occupations = np.asarray(natural_occupations, dtype=float)
    natural_orbitals = _require_square_matrix(
        np.asarray(natural_orbitals, dtype=np.complex128),
        name="natural_orbitals",
    )
    _validate_no_reference_data(rdm, natural_occupations, natural_orbitals)

    if hopping_matrix is None and H is not None:
        hopping_matrix = extract_one_body_hopping_matrix(
            H,
            spin_symmetric=spin_symmetric,
        )

    if hopping_matrix is not None:
        hopping_matrix = _hermitian_part(
            _require_square_matrix(hopping_matrix, name="hopping_matrix")
        )
        hopping_no = rotate_one_body_matrix(hopping_matrix, natural_orbitals)
    else:
        hopping_no = None

    blocks = build_occupation_blocks(
        natural_occupations,
        filled_tol=config.filled_tol,
        empty_tol=config.empty_tol,
        degeneracy_tol=config.degeneracy_tol,
    )

    initial_metrics = evaluate_basis_metrics(
        rdm,
        natural_orbitals,
        hopping_matrix=hopping_matrix,
        exchange_profile=exchange_profile,
        distance_matrix=distance_matrix,
        site_positions=site_positions,
        config=config,
    )
    history = [_metrics_summary(0, initial_metrics)]

    n_orbitals = natural_orbitals.shape[0]
    real_orbitals = config.real_orbitals
    arr_dtype = np.float64 if real_orbitals else np.complex128
    params = _init_block_params(blocks, real_orbitals=real_orbitals)
    if len(params) == 0 or config.n_steps == 0:
        final_post_no_rotation = np.eye(n_orbitals, dtype=arr_dtype)
        current_rotation = natural_orbitals.copy()
        final_metrics = initial_metrics
        final_loss_terms = {
            "occupancy": float(initial_metrics["occupation_offdiag_cost"]),
            "hopping": float(initial_metrics["hopping_metrics"]["structure_cost"]),
            "kondo": float(initial_metrics["kondo_metrics"]["structure_cost"]),
            "locality": float(initial_metrics["locality_metrics"]["locality_cost"]),
        }
    else:
        optimizer = optimizer or create_optax_optimizer(config)
        opt_state = optimizer.init(params)

        resolved_distance = _resolve_distance_matrix(n_orbitals, distance_matrix)
        jax_dtype = jnp.float64 if real_orbitals else jnp.complex128
        natural_occupations_jax = jnp.asarray(natural_occupations)
        natural_orbitals_jax = jnp.asarray(
            np.real(natural_orbitals) if real_orbitals else natural_orbitals, dtype=jax_dtype
        )
        hopping_no_jax = None if hopping_no is None else jnp.asarray(
            np.real(hopping_no) if real_orbitals else hopping_no, dtype=jax_dtype
        )
        exchange_profile_jax = (
            None
            if exchange_profile is None
            else jnp.asarray(
                _resolve_exchange_profile(exchange_profile, natural_orbitals.shape[0])
            )
        )
        site_positions_jax = None
        if site_positions is not None:
            site_positions = np.asarray(site_positions, dtype=float)
            if site_positions.ndim == 1:
                site_positions = site_positions[:, None]
            if site_positions.shape[0] != natural_orbitals.shape[0]:
                raise ValueError(
                    "site_positions must have one entry per original site/orbital, "
                    f"got {site_positions.shape[0]} for {natural_orbitals.shape[0]}."
                )
            site_positions_jax = jnp.asarray(site_positions)

        distance_matrix_jax = jnp.asarray(resolved_distance)

        def loss_fn(
            current_params: dict[str, dict[str, jax.Array]],
        ) -> jax.Array:
            post_no_rotation = _build_post_no_rotation(
                current_params,
                blocks,
                n_orbitals,
                real_orbitals=real_orbitals,
            )
            loss, _ = _loss_terms_jax(
                post_no_rotation,
                natural_occupations=natural_occupations_jax,
                natural_orbitals=natural_orbitals_jax,
                hopping_no=hopping_no_jax,
                exchange_profile=exchange_profile_jax,
                distance_matrix=distance_matrix_jax,
                site_positions=site_positions_jax,
                config=config,
            )
            return loss

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        for step in range(1, config.n_steps + 1):
            loss_value, grads = loss_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % config.log_every == 0 or step == config.n_steps:
                post_no_rotation = np.asarray(
                    _build_post_no_rotation(params, blocks, n_orbitals, real_orbitals=real_orbitals)
                )
                current_rotation = natural_orbitals @ post_no_rotation
                metrics = evaluate_basis_metrics(
                    rdm,
                    current_rotation,
                    hopping_matrix=hopping_matrix,
                    exchange_profile=exchange_profile,
                    distance_matrix=resolved_distance,
                    site_positions=site_positions,
                    config=config,
                )
                entry = _metrics_summary(step, metrics)
                entry["optax_loss"] = float(loss_value)
                history.append(entry)

        final_post_no_rotation = np.asarray(
            _build_post_no_rotation(params, blocks, n_orbitals, real_orbitals=real_orbitals)
        )
        current_rotation = natural_orbitals @ final_post_no_rotation
        final_metrics = evaluate_basis_metrics(
            rdm,
            current_rotation,
            hopping_matrix=hopping_matrix,
            exchange_profile=exchange_profile,
            distance_matrix=resolved_distance,
            site_positions=site_positions,
            config=config,
        )
        _, final_terms = _loss_terms_jax(
            jnp.asarray(final_post_no_rotation),
            natural_occupations=natural_occupations_jax,
            natural_orbitals=natural_orbitals_jax,
            hopping_no=hopping_no_jax,
            exchange_profile=exchange_profile_jax,
            distance_matrix=distance_matrix_jax,
            site_positions=site_positions_jax,
            config=config,
        )
        final_loss_terms = {
            "occupancy": float(final_terms[0]),
            "hopping": float(final_terms[1]),
            "kondo": float(final_terms[2]),
            "locality": float(final_terms[3]),
        }

    optimized_hamiltonian = None
    mode_rotation = None
    if H is not None:
        if not isinstance(H, FermionOperator2ndBase):
            raise TypeError(
                "H must be a FermionOperator2nd-compatible Hamiltonian, got "
                f"{type(H)}."
            )

        mode_rotation = (
            _expand_orbital_rotation_to_modes(H.hilbert, current_rotation)
            if spin_symmetric
            else current_rotation
        )
        optimized_hamiltonian = rotate_fermion_hamiltonian(
            H,
            mode_rotation,
            cutoff=config.hamiltonian_cutoff,
        )

    return {
        "config": asdict(config),
        "optimizer_name": config.optimizer_name,
        "natural_occupations": natural_occupations.copy(),
        "natural_orbitals": natural_orbitals.copy(),
        "occupation_blocks": [block.copy() for block in blocks],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "final_loss_terms": final_loss_terms,
        "history": history,
        "rotation_in_no_basis": final_post_no_rotation,
        "site_to_optimized_orbital": current_rotation,
        "mode_rotation": mode_rotation,
        "optimized_hamiltonian": optimized_hamiltonian,
        "optimized_rdm": final_metrics["rdm_in_basis"].copy(),
        "optimized_diag_occupations": np.asarray(
            final_metrics["diag_occupations"],
            dtype=float,
        ).copy(),
        "optimized_hopping_matrix": None
        if final_metrics["rotated_hopping"] is None
        else final_metrics["rotated_hopping"].copy(),
        "kondo_footprint": None
        if final_metrics["kondo_footprint"] is None
        else final_metrics["kondo_footprint"].copy(),
        "effective_scattering": None
        if final_metrics["effective_scattering"] is None
        else final_metrics["effective_scattering"].copy(),
    }

optimize_no_basis_from_rdm = optimize_no_basis


__all__ = [
    "NOOptimizationConfig",
    "build_occupation_blocks",
    "build_no_loss_fn",
    "create_optax_optimizer",
    "evaluate_basis_metrics",
    "evaluate_no_loss",
    "extract_one_body_hopping_matrix",
    "extract_hamiltonian_terms",
    "inspect_occupation_structure",
    "kondo_footprint",
    "kondo_tensor_from_site_rotation",
    "occupation_phase_space",
    "optimize_no_basis",
    "optimize_no_from_context",
    "optimize_no_basis_from_rdm",
    "orbital_locality_metrics",
    "pair_structure_metrics",
    "prepare_no_optimization_context",
    "rotate_hamiltonian_to_basis",
    "rotate_hamiltonian_to_natural_orbitals",
    "rotate_one_body_matrix",
]
