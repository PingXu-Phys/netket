from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

try:
    from .optimize_no import (
        NOOptimizationConfig,
        _coerce_config as _coerce_optimize_config,
        _loss_terms_jax as _optimize_loss_terms_jax,
    )
    from .orbital_utils import (
        EPS,
        _resolve_distance_matrix,
        evaluate_basis_metrics as _evaluate_optimize_basis_metrics,
        pair_structure_metrics,
    )
    from .optimize_no_core import _build_post_no_rotation
except ImportError:
    from orbital_optimization.optimize_no import (
        NOOptimizationConfig,
        _coerce_config as _coerce_optimize_config,
        _loss_terms_jax as _optimize_loss_terms_jax,
    )
    from orbital_optimization.orbital_utils import (
        EPS,
        _resolve_distance_matrix,
        evaluate_basis_metrics as _evaluate_optimize_basis_metrics,
        pair_structure_metrics,
    )
    from orbital_optimization.optimize_no_core import _build_post_no_rotation


def _coerce(config: Any, cls: type) -> Any:
    if config is None:
        return cls()
    if isinstance(config, cls):
        return config
    if isinstance(config, dict):
        return cls(**config)
    values = {}
    for field in fields(cls):
        if hasattr(config, field.name):
            values[field.name] = getattr(config, field.name)
    return cls(**values)


@dataclass(slots=True)
class NOSoftOptimizationConfig:
    alpha_hopping: float = 0.5
    alpha_interaction: float = 0.5
    lambda_occupancy: float = 0.05
    lambda_locality: float = 0.0
    filled_tol: float = 0.05
    empty_tol: float = 0.05
    degeneracy_tol: float = 0.02
    distance_power: float = 2.0
    distance_weight_strength: float = 1.0
    smooth_eps: float = 1.0e-8
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    n_steps: int = 300
    log_every: int = 50
    hamiltonian_cutoff: float | None = None
    real_orbitals: bool = True


@dataclass(slots=True)
class NOBarrierOptimizationConfig:
    alpha_hopping: float = 0.5
    alpha_interaction: float = 0.5
    eta_hopping: float = 0.5
    eta_interaction: float = 0.5
    mu_gamma: float = 5.0
    delta_gamma: float = 0.02
    mu_active_frozen: float = 0.0
    delta_active_frozen: float = 0.02
    barrier_tau: float = 0.02
    lambda_locality: float = 0.0
    filled_tol: float = 0.05
    empty_tol: float = 0.05
    degeneracy_tol: float = 0.02
    distance_power: float = 2.0
    distance_weight_strength: float = 1.0
    structure_metric: str = "hybrid"
    smooth_eps: float = 1.0e-8
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    n_steps: int = 300
    log_every: int = 50
    hamiltonian_cutoff: float | None = None
    real_orbitals: bool = True


def coerce_soft_config(config: NOSoftOptimizationConfig | dict[str, Any] | None):
    config = _coerce(config, NOSoftOptimizationConfig)
    if config.optimizer_name not in {"adam", "adamw", "sgd"}:
        raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name!r}.")
    if config.alpha_hopping < 0.0 or config.alpha_interaction < 0.0:
        raise ValueError("alpha_hopping and alpha_interaction must be non-negative.")
    if config.learning_rate <= 0.0 or config.smooth_eps <= 0.0:
        raise ValueError("learning_rate and smooth_eps must be positive.")
    if config.gradient_clip is not None and config.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive when provided.")
    if config.n_steps < 0 or config.log_every <= 0:
        raise ValueError("n_steps must be non-negative and log_every positive.")
    if config.distance_power < 0.0 or config.distance_weight_strength < 0.0:
        raise ValueError("distance_power and distance_weight_strength must be non-negative.")
    return config


def coerce_barrier_config(config: NOBarrierOptimizationConfig | dict[str, Any] | None):
    config = _coerce(config, NOBarrierOptimizationConfig)
    if config.optimizer_name not in {"adam", "adamw", "sgd"}:
        raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name!r}.")
    if config.structure_metric not in {"participation", "decay", "hybrid"}:
        raise ValueError(f"Unsupported structure_metric: {config.structure_metric!r}.")
    if config.barrier_tau <= 0.0 or config.smooth_eps <= 0.0:
        raise ValueError("barrier_tau and smooth_eps must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.gradient_clip is not None and config.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive when provided.")
    if config.n_steps < 0 or config.log_every <= 0:
        raise ValueError("n_steps must be non-negative and log_every positive.")
    for name in ("eta_hopping", "eta_interaction"):
        value = getattr(config, name)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return config


def coerce_optimize_config(config: NOOptimizationConfig | dict[str, Any] | None):
    config = _coerce(config, NOOptimizationConfig)
    return _coerce_optimize_config(config)


def occupation_phase_space(diagonal_occupations: np.ndarray) -> np.ndarray:
    n = np.clip(np.real(np.asarray(diagonal_occupations, dtype=float)), 0.0, 1.0)
    return n[:, None] * (1.0 - n[None, :]) + n[None, :] * (1.0 - n[:, None])


def _pair_weights_np(distance_matrix: np.ndarray, *, strength: float, power: float) -> np.ndarray:
    distance_matrix = _resolve_distance_matrix(distance_matrix.shape[0], np.asarray(distance_matrix, dtype=float))
    dist_max = float(np.max(distance_matrix))
    scaled = np.zeros_like(distance_matrix) if dist_max <= EPS else distance_matrix / dist_max
    return np.triu(1.0 + strength * np.power(scaled, power), k=1)


def _pair_weights_jax(distance_matrix: jax.Array, *, strength: float, power: float) -> jax.Array:
    dist_max = jnp.max(distance_matrix)
    scaled = jnp.where(dist_max <= EPS, jnp.zeros_like(distance_matrix), distance_matrix / dist_max)
    return jnp.triu(1.0 + strength * jnp.power(scaled, power), k=1)


def _weighted_ratio_np(values: np.ndarray, reference: np.ndarray, pair_weights: np.ndarray, *, smooth_eps: float) -> float:
    values = np.triu(np.asarray(np.abs(values), dtype=float), k=1)
    reference = np.triu(np.asarray(np.abs(reference), dtype=float), k=1)
    pair_weights = np.triu(np.asarray(pair_weights, dtype=float), k=1)
    num = np.sum(pair_weights * np.sqrt(values * values + smooth_eps * smooth_eps))
    den = np.sum(pair_weights * np.sqrt(reference * reference + smooth_eps * smooth_eps))
    return float(num / (den + smooth_eps))


def _weighted_ratio_jax(values: jax.Array, reference: jax.Array, pair_weights: jax.Array, *, smooth_eps: float) -> jax.Array:
    values = jnp.triu(jnp.abs(values), k=1)
    reference = jnp.triu(jnp.abs(reference), k=1)
    pair_weights = jnp.triu(pair_weights, k=1)
    num = jnp.sum(pair_weights * jnp.sqrt(values * values + smooth_eps * smooth_eps))
    den = jnp.sum(pair_weights * jnp.sqrt(reference * reference + smooth_eps * smooth_eps))
    return num / (den + smooth_eps)


def _interaction_np(site_to_orbital: np.ndarray, interaction_matrix: np.ndarray) -> np.ndarray:
    weights = np.abs(np.asarray(site_to_orbital, dtype=np.complex128)) ** 2
    return weights.T @ np.asarray(interaction_matrix, dtype=float) @ weights


def _interaction_jax(site_to_orbital: jax.Array, interaction_matrix: jax.Array) -> jax.Array:
    weights = jnp.abs(site_to_orbital) ** 2
    return weights.T @ interaction_matrix @ weights


def _occupancy_np(post_no_rotation: np.ndarray, natural_occupations: np.ndarray) -> tuple[float, np.ndarray]:
    diag_n = np.diag(np.asarray(natural_occupations, dtype=float))
    gamma_rot = post_no_rotation.conj().T @ diag_n @ post_no_rotation
    offdiag = gamma_rot - np.diag(np.diag(gamma_rot))
    norm = max(float(np.sum(np.square(natural_occupations))), EPS)
    return float(np.sum(np.abs(offdiag) ** 2) / norm), gamma_rot


def _occupancy_jax(post_no_rotation: jax.Array, natural_occupations: jax.Array) -> tuple[jax.Array, jax.Array]:
    diag_n = jnp.diag(natural_occupations)
    gamma_rot = post_no_rotation.conj().T @ diag_n @ post_no_rotation
    offdiag = gamma_rot - jnp.diag(jnp.diag(gamma_rot))
    norm = jnp.maximum(jnp.sum(jnp.square(natural_occupations)), EPS)
    return jnp.real(jnp.sum(jnp.abs(offdiag) ** 2) / norm), gamma_rot


def _spread_np(site_to_orbital: np.ndarray, site_positions: np.ndarray | None) -> float:
    if site_positions is None:
        return 0.0
    weights = np.abs(np.asarray(site_to_orbital, dtype=np.complex128)) ** 2
    positions = np.asarray(site_positions, dtype=float)
    centers = weights.T @ positions
    delta = positions[:, None, :] - centers[None, :, :]
    spreads = np.sum(weights[:, :, None] * delta * delta, axis=(0, 2)).real
    pair_delta = positions[:, None, :] - positions[None, :, :]
    scale = float(np.max(np.sum(pair_delta * pair_delta, axis=-1)))
    return 0.0 if scale <= EPS else float(np.mean(spreads / scale))


def _spread_jax(site_to_orbital: jax.Array, site_positions: jax.Array | None) -> jax.Array:
    if site_positions is None:
        return jnp.array(0.0)
    weights = jnp.abs(site_to_orbital) ** 2
    centers = weights.T @ site_positions
    delta = site_positions[:, None, :] - centers[None, :, :]
    spreads = jnp.sum(weights[:, :, None] * delta * delta, axis=(0, 2))
    pair_delta = site_positions[:, None, :] - site_positions[None, :, :]
    scale = jnp.max(jnp.sum(pair_delta * pair_delta, axis=-1))
    return jnp.where(scale <= EPS, 0.0, jnp.mean(spreads / scale))


def _structure_jax(weights: jax.Array, *, distance_matrix: jax.Array, power: float, metric: str) -> jax.Array:
    upper = jnp.triu(jnp.abs(weights), k=1)
    total_weight = jnp.sum(upper)
    total_safe = jnp.maximum(total_weight, EPS)
    n_orbitals = weights.shape[0]
    max_support = max((n_orbitals * (n_orbitals - 1)) // 2, 1)
    sq_sum = jnp.sum(jnp.square(upper))
    effective_support = total_safe * total_safe / jnp.maximum(sq_sum, EPS)
    participation = jnp.where(total_weight <= EPS, 0.0, 0.0 if max_support <= 1 else (effective_support - 1.0) / float(max_support - 1))
    dist_upper = jnp.triu(distance_matrix, k=1)
    dist_max = jnp.max(dist_upper)
    decay = jnp.where(dist_max <= EPS, 0.0, jnp.sum(upper * jnp.power(dist_upper / dist_max, power)) / total_safe)
    if metric == "participation":
        return participation
    if metric == "decay":
        return decay
    return 0.5 * (participation + decay)


def _active_frozen_np(post_no_rotation: np.ndarray, natural_occupations: np.ndarray) -> float:
    active_weight = 4.0 * natural_occupations * (1.0 - natural_occupations)
    pair_scale = np.triu((active_weight[:, None] - active_weight[None, :]) ** 2, k=1)
    mixing = np.triu(np.abs(post_no_rotation) ** 2, k=1)
    return float(np.sum(pair_scale * mixing) / (np.sum(pair_scale) + EPS))


def _active_frozen_jax(post_no_rotation: jax.Array, natural_occupations: jax.Array) -> jax.Array:
    active_weight = 4.0 * natural_occupations * (1.0 - natural_occupations)
    pair_scale = jnp.triu((active_weight[:, None] - active_weight[None, :]) ** 2, k=1)
    mixing = jnp.triu(jnp.abs(post_no_rotation) ** 2, k=1)
    return jnp.sum(pair_scale * mixing) / (jnp.sum(pair_scale) + EPS)


def _barrier_np(excess: float, tau: float) -> float:
    return float((tau * np.logaddexp(0.0, excess / tau)) ** 2)


def _barrier_jax(excess: jax.Array, tau: float) -> jax.Array:
    return jnp.square(tau * jnn.softplus(excess / tau))


def _common_loss_context(base_context: dict[str, Any], *, strength: float, power: float) -> dict[str, Any]:
    pair_weights = _pair_weights_np(base_context["distance_matrix"], strength=strength, power=power)
    reference_phase_space = occupation_phase_space(base_context["natural_occupations"])
    reference_interaction = None
    if base_context["interaction_matrix"] is not None:
        reference_interaction = _interaction_np(base_context["natural_orbitals"], base_context["interaction_matrix"]) * reference_phase_space
    return {
        "pair_weights": pair_weights,
        "reference_phase_space": reference_phase_space,
        "reference_interaction": reference_interaction,
    }


def build_soft_loss_module(base_context: dict[str, Any], config: NOSoftOptimizationConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    config = coerce_soft_config(config)
    core_config = base_context["core_config"]
    loss_context = _common_loss_context(base_context, strength=config.distance_weight_strength, power=config.distance_power)
    n_orbitals = base_context["natural_orbitals"].shape[0]
    arr_dtype = jnp.float64 if core_config.real_orbitals else jnp.complex128
    natural_orbitals = np.asarray(base_context["natural_orbitals"], dtype=np.complex128)
    if core_config.real_orbitals:
        natural_orbitals = np.real(natural_orbitals)
    hopping_no = base_context["hopping_no"]
    if hopping_no is not None and core_config.real_orbitals:
        hopping_no = np.real(hopping_no)
    natural_occupations_jax = jnp.asarray(base_context["natural_occupations"])
    natural_orbitals_jax = jnp.asarray(natural_orbitals, dtype=arr_dtype)
    hopping_no_jax = None if hopping_no is None else jnp.asarray(hopping_no, dtype=arr_dtype)
    interaction_matrix_jax = None if base_context["interaction_matrix"] is None else jnp.asarray(base_context["interaction_matrix"])
    pair_weights_jax = _pair_weights_jax(jnp.asarray(base_context["distance_matrix"]), strength=config.distance_weight_strength, power=config.distance_power)
    reference_phase_space_jax = jnp.asarray(loss_context["reference_phase_space"])
    reference_interaction_jax = None if loss_context["reference_interaction"] is None else jnp.asarray(loss_context["reference_interaction"])
    site_positions_jax = None if base_context["site_positions"] is None else jnp.asarray(base_context["site_positions"])

    def _loss_terms(post_no_rotation: jax.Array):
        occupancy_cost, _ = _occupancy_jax(post_no_rotation, natural_occupations_jax)
        hopping_cost = jnp.array(0.0)
        if hopping_no_jax is not None:
            hopping_rot = post_no_rotation.conj().T @ hopping_no_jax @ post_no_rotation
            hopping_cost = _weighted_ratio_jax(hopping_rot, hopping_no_jax, pair_weights_jax, smooth_eps=config.smooth_eps)
        site_to_orbital = natural_orbitals_jax @ post_no_rotation
        interaction_cost = jnp.array(0.0)
        if interaction_matrix_jax is not None and reference_interaction_jax is not None:
            lambda_ab = _interaction_jax(site_to_orbital, interaction_matrix_jax)
            effective_scattering = lambda_ab * reference_phase_space_jax
            interaction_cost = _weighted_ratio_jax(effective_scattering, reference_interaction_jax, pair_weights_jax, smooth_eps=config.smooth_eps)
        locality_cost = _spread_jax(site_to_orbital, site_positions_jax)
        total = config.alpha_hopping * hopping_cost + config.alpha_interaction * interaction_cost + config.lambda_occupancy * occupancy_cost + config.lambda_locality * locality_cost
        return jnp.real(total), (jnp.real(occupancy_cost), jnp.real(hopping_cost), jnp.real(interaction_cost), jnp.real(locality_cost))

    def build_loss_fn():
        def loss_fn(current_params: dict[str, dict[str, jax.Array]]) -> jax.Array:
            post_no_rotation = _build_post_no_rotation(current_params, base_context["occupation_blocks"], n_orbitals, real_orbitals=core_config.real_orbitals)
            return _loss_terms(post_no_rotation)[0]
        return loss_fn

    def build_rotation_loss_fn():
        def loss_fn(post_no_rotation: jax.Array) -> jax.Array:
            return _loss_terms(post_no_rotation)[0]
        return loss_fn

    def evaluate_metrics(post_no_rotation: np.ndarray) -> dict[str, Any]:
        natural_occupations = np.asarray(base_context["natural_occupations"], dtype=float)
        site_to_orbital = np.asarray(base_context["natural_orbitals"], dtype=np.complex128) @ post_no_rotation
        occupancy_cost, gamma_rot = _occupancy_np(post_no_rotation, natural_occupations)
        diagonal_occupations = np.clip(np.real(np.diag(gamma_rot)), 0.0, 1.0)
        rotated_hopping = None
        hopping_cost = 0.0
        if base_context["hopping_no"] is not None:
            rotated_hopping = post_no_rotation.conj().T @ base_context["hopping_no"] @ post_no_rotation
            hopping_cost = _weighted_ratio_np(rotated_hopping, base_context["hopping_no"], loss_context["pair_weights"], smooth_eps=config.smooth_eps)
        interaction_footprint = None
        effective_scattering = None
        interaction_cost = 0.0
        if base_context["interaction_matrix"] is not None and loss_context["reference_interaction"] is not None:
            interaction_footprint = _interaction_np(site_to_orbital, base_context["interaction_matrix"])
            effective_scattering = interaction_footprint * loss_context["reference_phase_space"]
            interaction_cost = _weighted_ratio_np(effective_scattering, loss_context["reference_interaction"], loss_context["pair_weights"], smooth_eps=config.smooth_eps)
        locality_cost = _spread_np(site_to_orbital, base_context["site_positions"])
        total = config.alpha_hopping * hopping_cost + config.alpha_interaction * interaction_cost + config.lambda_occupancy * occupancy_cost + config.lambda_locality * locality_cost
        return {
            "loss": float(total),
            "loss_terms": {"occupancy": float(occupancy_cost), "hopping": float(hopping_cost), "interaction": float(interaction_cost), "locality": float(locality_cost)},
            "post_no_rotation": post_no_rotation,
            "site_to_optimized_orbital": site_to_orbital,
            "rdm_in_basis": gamma_rot,
            "diag_occupations": diagonal_occupations,
            "rotated_hopping": rotated_hopping,
            "interaction_footprint": interaction_footprint,
            "effective_scattering": effective_scattering,
            "reference_phase_space": loss_context["reference_phase_space"].copy(),
            "pair_weights": loss_context["pair_weights"].copy(),
        }

    def history_entry(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": int(step),
            "loss": float(metrics["loss"]),
            "occupancy_cost": float(metrics["loss_terms"]["occupancy"]),
            "hopping_cost": float(metrics["loss_terms"]["hopping"]),
            "interaction_cost": float(metrics["loss_terms"]["interaction"]),
            "locality_cost": float(metrics["loss_terms"]["locality"]),
            "diag_occupations": np.asarray(metrics["diag_occupations"], dtype=float).copy(),
        }

    return {
        "kind": "soft",
        "config": config,
        "loss_context": loss_context,
        "build_loss_fn": build_loss_fn,
        "build_rotation_loss_fn": build_rotation_loss_fn,
        "evaluate_metrics": evaluate_metrics,
        "history_entry": history_entry,
        "result_fields": lambda final_metrics: {"reference_phase_space": final_metrics["reference_phase_space"].copy()},
    }


def build_barrier_loss_module(base_context: dict[str, Any], config: NOBarrierOptimizationConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    config = coerce_barrier_config(config)
    core_config = base_context["core_config"]
    loss_context = _common_loss_context(base_context, strength=config.distance_weight_strength, power=config.distance_power)
    n_orbitals = base_context["natural_orbitals"].shape[0]
    arr_dtype = jnp.float64 if core_config.real_orbitals else jnp.complex128
    natural_orbitals = np.asarray(base_context["natural_orbitals"], dtype=np.complex128)
    if core_config.real_orbitals:
        natural_orbitals = np.real(natural_orbitals)
    hopping_no = base_context["hopping_no"]
    if hopping_no is not None and core_config.real_orbitals:
        hopping_no = np.real(hopping_no)
    natural_occupations_jax = jnp.asarray(base_context["natural_occupations"])
    natural_orbitals_jax = jnp.asarray(natural_orbitals, dtype=arr_dtype)
    hopping_no_jax = None if hopping_no is None else jnp.asarray(hopping_no, dtype=arr_dtype)
    interaction_matrix_jax = None if base_context["interaction_matrix"] is None else jnp.asarray(base_context["interaction_matrix"])
    distance_matrix_jax = jnp.asarray(base_context["distance_matrix"])
    pair_weights_jax = _pair_weights_jax(distance_matrix_jax, strength=config.distance_weight_strength, power=config.distance_power)
    reference_phase_space_jax = jnp.asarray(loss_context["reference_phase_space"])
    reference_interaction_jax = None if loss_context["reference_interaction"] is None else jnp.asarray(loss_context["reference_interaction"])
    site_positions_jax = None if base_context["site_positions"] is None else jnp.asarray(base_context["site_positions"])

    def _loss_terms(post_no_rotation: jax.Array):
        occupancy_cost, _ = _occupancy_jax(post_no_rotation, natural_occupations_jax)
        occupancy_barrier = _barrier_jax(occupancy_cost - config.delta_gamma, config.barrier_tau)
        hopping_norm = jnp.array(0.0)
        hopping_structure = jnp.array(0.0)
        if hopping_no_jax is not None:
            hopping_rot = post_no_rotation.conj().T @ hopping_no_jax @ post_no_rotation
            hopping_norm = _weighted_ratio_jax(hopping_rot, hopping_no_jax, pair_weights_jax, smooth_eps=config.smooth_eps)
            hopping_structure = _structure_jax(jnp.abs(hopping_rot) ** 2, distance_matrix=distance_matrix_jax, power=config.distance_power, metric=config.structure_metric)
        hopping_objective = config.eta_hopping * hopping_norm + (1.0 - config.eta_hopping) * hopping_structure
        site_to_orbital = natural_orbitals_jax @ post_no_rotation
        interaction_norm = jnp.array(0.0)
        interaction_structure = jnp.array(0.0)
        if interaction_matrix_jax is not None and reference_interaction_jax is not None:
            lambda_ab = _interaction_jax(site_to_orbital, interaction_matrix_jax)
            effective_scattering = lambda_ab * reference_phase_space_jax
            interaction_norm = _weighted_ratio_jax(effective_scattering, reference_interaction_jax, pair_weights_jax, smooth_eps=config.smooth_eps)
            interaction_structure = _structure_jax(effective_scattering, distance_matrix=distance_matrix_jax, power=config.distance_power, metric=config.structure_metric)
        interaction_objective = config.eta_interaction * interaction_norm + (1.0 - config.eta_interaction) * interaction_structure
        hamiltonian_objective = config.alpha_hopping * hopping_objective + config.alpha_interaction * interaction_objective
        active_frozen_cost = _active_frozen_jax(post_no_rotation, natural_occupations_jax)
        active_frozen_barrier = _barrier_jax(active_frozen_cost - config.delta_active_frozen, config.barrier_tau)
        locality_cost = _spread_jax(site_to_orbital, site_positions_jax)
        total = hamiltonian_objective + config.mu_gamma * occupancy_barrier + config.mu_active_frozen * active_frozen_barrier + config.lambda_locality * locality_cost
        return jnp.real(total), {
            "hamiltonian": jnp.real(hamiltonian_objective),
            "hopping_norm": jnp.real(hopping_norm),
            "hopping_structure": jnp.real(hopping_structure),
            "interaction_norm": jnp.real(interaction_norm),
            "interaction_structure": jnp.real(interaction_structure),
            "occupancy": jnp.real(occupancy_cost),
            "occupancy_barrier": jnp.real(occupancy_barrier),
            "active_frozen": jnp.real(active_frozen_cost),
            "active_frozen_barrier": jnp.real(active_frozen_barrier),
            "locality": jnp.real(locality_cost),
        }

    def build_loss_fn():
        def loss_fn(current_params: dict[str, dict[str, jax.Array]]) -> jax.Array:
            post_no_rotation = _build_post_no_rotation(current_params, base_context["occupation_blocks"], n_orbitals, real_orbitals=core_config.real_orbitals)
            return _loss_terms(post_no_rotation)[0]
        return loss_fn

    def build_rotation_loss_fn():
        def loss_fn(post_no_rotation: jax.Array) -> jax.Array:
            return _loss_terms(post_no_rotation)[0]
        return loss_fn

    def evaluate_metrics(post_no_rotation: np.ndarray) -> dict[str, Any]:
        natural_occupations = np.asarray(base_context["natural_occupations"], dtype=float)
        site_to_orbital = np.asarray(base_context["natural_orbitals"], dtype=np.complex128) @ post_no_rotation
        occupancy_cost, gamma_rot = _occupancy_np(post_no_rotation, natural_occupations)
        diagonal_occupations = np.clip(np.real(np.diag(gamma_rot)), 0.0, 1.0)
        occupancy_barrier = _barrier_np(occupancy_cost - config.delta_gamma, config.barrier_tau)
        rotated_hopping = None
        hopping_norm = 0.0
        hopping_structure = 0.0
        if base_context["hopping_no"] is not None:
            rotated_hopping = post_no_rotation.conj().T @ base_context["hopping_no"] @ post_no_rotation
            hopping_norm = _weighted_ratio_np(rotated_hopping, base_context["hopping_no"], loss_context["pair_weights"], smooth_eps=config.smooth_eps)
            hopping_structure = pair_structure_metrics(np.abs(rotated_hopping) ** 2, distance_matrix=base_context["distance_matrix"], power=config.distance_power, metric=config.structure_metric)["structure_cost"]
        hopping_objective = config.eta_hopping * hopping_norm + (1.0 - config.eta_hopping) * hopping_structure
        interaction_footprint = None
        effective_scattering = None
        interaction_norm = 0.0
        interaction_structure = 0.0
        if base_context["interaction_matrix"] is not None and loss_context["reference_interaction"] is not None:
            interaction_footprint = _interaction_np(site_to_orbital, base_context["interaction_matrix"])
            effective_scattering = interaction_footprint * loss_context["reference_phase_space"]
            interaction_norm = _weighted_ratio_np(effective_scattering, loss_context["reference_interaction"], loss_context["pair_weights"], smooth_eps=config.smooth_eps)
            interaction_structure = pair_structure_metrics(effective_scattering, distance_matrix=base_context["distance_matrix"], power=config.distance_power, metric=config.structure_metric)["structure_cost"]
        interaction_objective = config.eta_interaction * interaction_norm + (1.0 - config.eta_interaction) * interaction_structure
        hamiltonian_objective = config.alpha_hopping * hopping_objective + config.alpha_interaction * interaction_objective
        active_frozen_cost = _active_frozen_np(post_no_rotation, natural_occupations)
        active_frozen_barrier = _barrier_np(active_frozen_cost - config.delta_active_frozen, config.barrier_tau)
        locality_cost = _spread_np(site_to_orbital, base_context["site_positions"])
        total = hamiltonian_objective + config.mu_gamma * occupancy_barrier + config.mu_active_frozen * active_frozen_barrier + config.lambda_locality * locality_cost
        return {
            "loss": float(total),
            "loss_terms": {
                "hamiltonian": float(hamiltonian_objective),
                "hopping_norm": float(hopping_norm),
                "hopping_structure": float(hopping_structure),
                "interaction_norm": float(interaction_norm),
                "interaction_structure": float(interaction_structure),
                "occupancy": float(occupancy_cost),
                "occupancy_barrier": float(occupancy_barrier),
                "active_frozen": float(active_frozen_cost),
                "active_frozen_barrier": float(active_frozen_barrier),
                "locality": float(locality_cost),
            },
            "post_no_rotation": post_no_rotation,
            "site_to_optimized_orbital": site_to_orbital,
            "rdm_in_basis": gamma_rot,
            "diag_occupations": diagonal_occupations,
            "rotated_hopping": rotated_hopping,
            "interaction_footprint": interaction_footprint,
            "effective_scattering": effective_scattering,
            "reference_phase_space": loss_context["reference_phase_space"].copy(),
            "pair_weights": loss_context["pair_weights"].copy(),
        }

    def history_entry(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": int(step),
            "loss": float(metrics["loss"]),
            "hamiltonian_objective": float(metrics["loss_terms"]["hamiltonian"]),
            "occupancy_cost": float(metrics["loss_terms"]["occupancy"]),
            "occupancy_barrier": float(metrics["loss_terms"]["occupancy_barrier"]),
            "active_frozen_cost": float(metrics["loss_terms"]["active_frozen"]),
            "locality_cost": float(metrics["loss_terms"]["locality"]),
            "diag_occupations": np.asarray(metrics["diag_occupations"], dtype=float).copy(),
        }

    return {
        "kind": "barrier",
        "config": config,
        "loss_context": loss_context,
        "build_loss_fn": build_loss_fn,
        "build_rotation_loss_fn": build_rotation_loss_fn,
        "evaluate_metrics": evaluate_metrics,
        "history_entry": history_entry,
        "result_fields": lambda final_metrics: {"reference_phase_space": final_metrics["reference_phase_space"].copy()},
    }


def build_optimize_loss_module(
    base_context: dict[str, Any],
    config: NOOptimizationConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = coerce_optimize_config(config)
    real_orbitals = config.real_orbitals
    arr_dtype = jnp.float64 if real_orbitals else jnp.complex128
    n_orbitals = base_context["natural_orbitals"].shape[0]

    natural_orbitals = np.asarray(base_context["natural_orbitals"], dtype=np.complex128)
    if real_orbitals:
        natural_orbitals = np.real(natural_orbitals)
    hopping_no = base_context["hopping_no"]
    if hopping_no is not None and real_orbitals:
        hopping_no = np.real(hopping_no)

    natural_occupations_jax = jnp.asarray(base_context["natural_occupations"])
    natural_orbitals_jax = jnp.asarray(natural_orbitals, dtype=arr_dtype)
    hopping_no_jax = None if hopping_no is None else jnp.asarray(hopping_no, dtype=arr_dtype)
    interaction_matrix_jax = (
        None
        if base_context["interaction_matrix"] is None
        else jnp.asarray(base_context["interaction_matrix"])
    )
    distance_matrix_jax = jnp.asarray(base_context["distance_matrix"])
    site_positions_jax = (
        None
        if base_context["site_positions"] is None
        else jnp.asarray(base_context["site_positions"])
    )

    def build_loss_fn():
        def loss_fn(current_params: dict[str, dict[str, jax.Array]]) -> jax.Array:
            post_no_rotation = _build_post_no_rotation(
                current_params,
                base_context["occupation_blocks"],
                n_orbitals,
                real_orbitals=real_orbitals,
            )
            loss, _ = _optimize_loss_terms_jax(
                post_no_rotation,
                natural_occupations=natural_occupations_jax,
                natural_orbitals=natural_orbitals_jax,
                hopping_no=hopping_no_jax,
                interaction_matrix=interaction_matrix_jax,
                distance_matrix=distance_matrix_jax,
                site_positions=site_positions_jax,
                config=config,
            )
            return loss

        return loss_fn

    def build_rotation_loss_fn():
        def loss_fn(post_no_rotation: jax.Array) -> jax.Array:
            loss, _ = _optimize_loss_terms_jax(
                post_no_rotation,
                natural_occupations=natural_occupations_jax,
                natural_orbitals=natural_orbitals_jax,
                hopping_no=hopping_no_jax,
                interaction_matrix=interaction_matrix_jax,
                distance_matrix=distance_matrix_jax,
                site_positions=site_positions_jax,
                config=config,
            )
            return loss

        return loss_fn

    def evaluate_metrics(post_no_rotation: np.ndarray) -> dict[str, Any]:
        site_to_optimized_orbital = np.asarray(
            base_context["natural_orbitals"],
            dtype=np.complex128,
        ) @ post_no_rotation
        metrics = _evaluate_optimize_basis_metrics(
            base_context["rdm"],
            site_to_optimized_orbital,
            hopping_matrix=base_context["hopping_matrix"],
            interaction_matrix=base_context["interaction_matrix"],
            distance_matrix=base_context["distance_matrix"],
            site_positions=base_context["site_positions"],
            config=config,
        )
        return {
            "loss": float(metrics["loss"]),
            "loss_terms": {
                "occupancy": float(metrics["occupation_offdiag_cost"]),
                "hopping": float(metrics["hopping_metrics"]["structure_cost"]),
                "interaction": float(metrics["interaction_metrics"]["structure_cost"]),
                "locality": float(metrics["locality_metrics"]["locality_cost"]),
            },
            "post_no_rotation": post_no_rotation,
            "site_to_optimized_orbital": site_to_optimized_orbital,
            **metrics,
        }

    def history_entry(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": int(step),
            "loss": float(metrics["loss"]),
            "occupation_offdiag_cost": float(metrics["occupation_offdiag_cost"]),
            "hopping_structure_cost": float(metrics["hopping_metrics"]["structure_cost"]),
            "interaction_structure_cost": float(
                metrics["interaction_metrics"]["structure_cost"]
            ),
            "locality_cost": float(metrics["locality_metrics"]["locality_cost"]),
            "diag_occupations": np.asarray(metrics["diag_occupations"], dtype=float).copy(),
        }

    return {
        "kind": "optimize",
        "config": config,
        "build_loss_fn": build_loss_fn,
        "build_rotation_loss_fn": build_rotation_loss_fn,
        "evaluate_metrics": evaluate_metrics,
        "history_entry": history_entry,
        "result_fields": lambda final_metrics: {"phase_space": final_metrics["phase_space"].copy()},
    }


build_original_loss_module = build_optimize_loss_module


__all__ = [
    "NOOptimizationConfig",
    "NOBarrierOptimizationConfig",
    "NOSoftOptimizationConfig",
    "build_optimize_loss_module",
    "build_original_loss_module",
    "build_barrier_loss_module",
    "build_soft_loss_module",
    "coerce_optimize_config",
    "coerce_barrier_config",
    "coerce_soft_config",
    "occupation_phase_space",
]
