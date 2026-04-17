"""Pruning utilities for orbital optimisation.

This module owns the threshold schedule, support construction, lightweight
feasibility repair, and stable-support extraction used between the dense and
sparse stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np

from .orbital_utils import _hermitian_part


@dataclass(slots=True)
class OrbitalPruningConfig:
    taus: tuple[float, ...]
    top_m: int = 1
    eta_H: float = 0.03
    eta_gamma: float = 0.08
    f_min: float = 0.7
    repair_steps: int = 12
    tol_unitary: float = 1.0e-6
    tol_mask: float = 1.0e-8
    polar_floor: float = 1.0e-12
    stop_on_reject: bool = True


NOPruningConfig = OrbitalPruningConfig


def _asdict_safe(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def coerce_pruning_config(
    config: OrbitalPruningConfig | dict[str, Any] | None,
) -> OrbitalPruningConfig:
    if config is None:
        config = OrbitalPruningConfig(taus=())
    elif isinstance(config, dict):
        config = OrbitalPruningConfig(**config)
    if not isinstance(config, OrbitalPruningConfig):
        raise TypeError(
            'config must be OrbitalPruningConfig, dict, or None, got '
            f'{type(config)}.'
        )
    taus = tuple(float(tau) for tau in config.taus)
    if any(tau < 0.0 for tau in taus):
        raise ValueError(f'taus must be non-negative, got {taus}.')
    if any(t2 < t1 for t1, t2 in zip(taus, taus[1:])):
        raise ValueError(f'taus must be non-decreasing, got {taus}.')
    if config.top_m <= 0:
        raise ValueError(f'top_m must be positive, got {config.top_m}.')
    if config.repair_steps < 0:
        raise ValueError(
            f'repair_steps must be non-negative, got {config.repair_steps}.'
        )
    if config.f_min < 0.0 or config.f_min > 1.0:
        raise ValueError(f'f_min must lie in [0, 1], got {config.f_min}.')
    if config.tol_unitary <= 0.0 or config.tol_mask <= 0.0:
        raise ValueError('tol_unitary and tol_mask must be positive.')
    if config.polar_floor <= 0.0:
        raise ValueError(f'polar_floor must be positive, got {config.polar_floor}.')
    return OrbitalPruningConfig(
        taus=taus,
        top_m=int(config.top_m),
        eta_H=float(config.eta_H),
        eta_gamma=float(config.eta_gamma),
        f_min=float(config.f_min),
        repair_steps=int(config.repair_steps),
        tol_unitary=float(config.tol_unitary),
        tol_mask=float(config.tol_mask),
        polar_floor=float(config.polar_floor),
        stop_on_reject=bool(config.stop_on_reject),
    )


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f'values must be one-dimensional, got {values.shape}.')
    k = min(max(int(k), 1), values.size)
    if k == values.size:
        return np.arange(values.size, dtype=int)
    idx = np.argpartition(values, -k)[-k:]
    return np.asarray(idx, dtype=int)


def repair_support(
    mask: np.ndarray,
    rotation: np.ndarray,
    *,
    top_m: int = 1,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rotation = np.asarray(rotation)
    if mask.shape != rotation.shape:
        raise ValueError(
            'mask and rotation must have identical shapes, got '
            f'{mask.shape} and {rotation.shape}.'
        )
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f'mask must be square, got {mask.shape}.')
    repaired = mask.copy()
    magnitude = np.abs(rotation)
    n_orbitals = repaired.shape[0]
    keep = min(max(int(top_m), 1), n_orbitals)

    for row in range(n_orbitals):
        if int(np.count_nonzero(repaired[row, :])) >= keep:
            continue
        repaired[row, _top_indices(magnitude[row, :], keep)] = True

    for col in range(n_orbitals):
        if int(np.count_nonzero(repaired[:, col])) >= keep:
            continue
        repaired[_top_indices(magnitude[:, col], keep), col] = True

    return repaired


def build_support_from_rotation(
    post_no_rotation: np.ndarray,
    tau: float,
    *,
    top_m: int = 1,
) -> np.ndarray:
    rotation = np.asarray(post_no_rotation)
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(
            'post_no_rotation must be a square matrix, got '
            f'{rotation.shape}.'
        )
    mask = np.abs(rotation) >= float(tau)
    return repair_support(mask, rotation, top_m=top_m)


def project_support(rotation: np.ndarray, support: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.complex128)
    support = np.asarray(support, dtype=bool)
    if rotation.shape != support.shape:
        raise ValueError(
            'rotation and support must have identical shapes, got '
            f'{rotation.shape} and {support.shape}.'
        )
    return np.where(support, rotation, 0.0)


def polar_retraction(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    gram = _hermitian_part(matrix.conj().T @ matrix)
    evals, evecs = np.linalg.eigh(gram)
    evals = np.clip(np.real(evals), float(floor), None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T
    return matrix @ inv_sqrt


def unitary_error(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.complex128)
    ident = np.eye(rotation.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(rotation.conj().T @ rotation - ident, ord='fro'))


def mask_error(rotation: np.ndarray, support: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.complex128)
    projected = project_support(rotation, support)
    return float(np.linalg.norm(rotation - projected, ord='fro'))


def alternating_support_projection(
    rotation: np.ndarray,
    support: np.ndarray,
    *,
    n_steps: int = 12,
    floor: float = 1.0e-12,
) -> np.ndarray:
    current = project_support(rotation, support)
    if n_steps <= 0:
        return current
    for _ in range(int(n_steps)):
        current = polar_retraction(current, floor=floor)
        current = project_support(current, support)
    return current


def persistence_matrix(supports: list[np.ndarray]) -> np.ndarray:
    if len(supports) == 0:
        raise ValueError('supports must be non-empty to build a persistence matrix.')
    stacked = np.stack([np.asarray(s, dtype=float) for s in supports], axis=0)
    return np.mean(stacked, axis=0)


def build_stable_support(
    supports: list[np.ndarray],
    *,
    f_min: float,
    fallback_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if len(supports) == 0:
        if fallback_shape is None:
            raise ValueError('fallback_shape is required when supports is empty.')
        return np.ones(fallback_shape, dtype=bool)
    persistence = persistence_matrix(supports)
    return persistence >= float(f_min)


def _extract_hamiltonian_cost(metrics: dict[str, Any]) -> float:
    loss_terms = metrics.get('loss_terms', {})
    if 'hamiltonian' in loss_terms:
        return float(loss_terms['hamiltonian'])
    if 'hopping' in loss_terms or 'interaction' in loss_terms:
        return float(loss_terms.get('hopping', 0.0)) + float(
            loss_terms.get('interaction', 0.0)
        )
    if 'hopping_norm' in loss_terms or 'interaction_norm' in loss_terms:
        return float(loss_terms.get('hopping_norm', 0.0)) + float(
            loss_terms.get('interaction_norm', 0.0)
        )
    return 0.0


def _extract_occupancy_cost(metrics: dict[str, Any]) -> float:
    loss_terms = metrics.get('loss_terms', {})
    if 'occupancy' in loss_terms:
        return float(loss_terms['occupancy'])
    if 'occupation_offdiag_cost' in metrics:
        return float(metrics['occupation_offdiag_cost'])
    return 0.0


def _support_density(support: np.ndarray) -> float:
    support = np.asarray(support, dtype=bool)
    return float(np.count_nonzero(support) / support.size)


def _evaluate_acceptance(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    unitary_err: float,
    support_err: float,
    config: OrbitalPruningConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    base_H = _extract_hamiltonian_cost(baseline_metrics)
    cand_H = _extract_hamiltonian_cost(candidate_metrics)
    if base_H > 0.0 and cand_H > (1.0 + config.eta_H) * base_H:
        reasons.append('hamiltonian_cost')

    base_gamma = _extract_occupancy_cost(baseline_metrics)
    cand_gamma = _extract_occupancy_cost(candidate_metrics)
    if base_gamma > 0.0 and cand_gamma > (1.0 + config.eta_gamma) * base_gamma:
        reasons.append('occupancy_cost')

    if unitary_err >= config.tol_unitary:
        reasons.append('unitary_error')
    if support_err >= config.tol_mask:
        reasons.append('mask_error')
    return len(reasons) == 0, reasons


def _history_entry(
    *,
    step: int,
    tau: float,
    support: np.ndarray,
    rotation: np.ndarray,
    metrics: dict[str, Any],
    accepted: bool,
    reject_reasons: list[str],
) -> dict[str, Any]:
    return {
        'step': int(step),
        'tau': float(tau),
        'accepted': bool(accepted),
        'reject_reasons': list(reject_reasons),
        'support': np.asarray(support, dtype=bool).copy(),
        'rho': _support_density(support),
        'unitarity_error': unitary_error(rotation),
        'mask_error': mask_error(rotation, support),
        'hamiltonian_cost': _extract_hamiltonian_cost(metrics),
        'occupancy_cost': _extract_occupancy_cost(metrics),
        'loss': float(metrics.get('loss', 0.0)),
        'metrics': metrics,
        'rotation_in_no_basis': np.asarray(rotation, dtype=np.complex128).copy(),
    }


def run_pruning_schedule(
    *,
    base_context: dict[str, Any],
    loss_module: dict[str, Any],
    dense_result: dict[str, Any],
    config: OrbitalPruningConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    del base_context
    config = coerce_pruning_config(config)
    evaluate_metrics = loss_module['evaluate_metrics']

    current_rotation = np.asarray(
        dense_result.get('rotation_in_no_basis'),
        dtype=np.complex128,
    )
    if current_rotation.ndim != 2 or current_rotation.shape[0] != current_rotation.shape[1]:
        raise ValueError(
            'dense_result["rotation_in_no_basis"] must be a square matrix, got '
            f'{current_rotation.shape}.'
        )

    baseline_metrics = dense_result.get('final_metrics')
    if baseline_metrics is None:
        baseline_metrics = evaluate_metrics(current_rotation)

    accepted_supports: list[np.ndarray] = []
    accepted_records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for step, tau in enumerate(config.taus, start=1):
        support = build_support_from_rotation(current_rotation, tau, top_m=config.top_m)
        candidate_rotation = alternating_support_projection(
            current_rotation,
            support,
            n_steps=config.repair_steps,
            floor=config.polar_floor,
        )
        candidate_metrics = evaluate_metrics(candidate_rotation)
        cand_unitary = unitary_error(candidate_rotation)
        cand_mask = mask_error(candidate_rotation, support)
        accepted, reject_reasons = _evaluate_acceptance(
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            unitary_err=cand_unitary,
            support_err=cand_mask,
            config=config,
        )
        record = _history_entry(
            step=step,
            tau=tau,
            support=support,
            rotation=candidate_rotation,
            metrics=candidate_metrics,
            accepted=accepted,
            reject_reasons=reject_reasons,
        )
        history.append(record)

        if accepted:
            current_rotation = candidate_rotation
            accepted_supports.append(np.asarray(support, dtype=bool).copy())
            accepted_records.append(record)
        elif config.stop_on_reject:
            break

    stable_support = build_stable_support(
        accepted_supports,
        f_min=config.f_min,
        fallback_shape=current_rotation.shape,
    )
    persistence = (
        persistence_matrix(accepted_supports)
        if accepted_supports
        else np.ones(current_rotation.shape, dtype=float)
    )
    final_metrics = evaluate_metrics(current_rotation)

    return {
        'config': _asdict_safe(config),
        'baseline_metrics': baseline_metrics,
        'history': history,
        'accepted_steps': len(accepted_records),
        'accepted_records': accepted_records,
        'accepted_supports': accepted_supports,
        'persistence_matrix': persistence,
        'stable_support': stable_support,
        'final_rotation_in_no_basis': current_rotation,
        'final_metrics': final_metrics,
    }


__all__ = [
    'NOPruningConfig',
    'OrbitalPruningConfig',
    'alternating_support_projection',
    'build_stable_support',
    'build_support_from_rotation',
    'coerce_pruning_config',
    'mask_error',
    'persistence_matrix',
    'polar_retraction',
    'project_support',
    'repair_support',
    'run_pruning_schedule',
    'unitary_error',
]
