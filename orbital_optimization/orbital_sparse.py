"""Sparse refinement on a fixed orbital support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from .orbital_pruning import (
    alternating_support_projection,
    mask_error,
    polar_retraction,
    project_support,
    unitary_error,
)


@dataclass(slots=True)
class OrbitalSparseConfig:
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    n_steps: int = 100
    log_every: int = 25
    lambda_fit: float = 0.3
    repair_steps: int = 12
    tol_unitary: float = 1.0e-6
    tol_mask: float = 1.0e-8
    polar_floor: float = 1.0e-12
    imag_tol: float = 1.0e-10


NOSparseConfig = OrbitalSparseConfig


def _asdict_safe(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def coerce_sparse_config(
    config: OrbitalSparseConfig | dict[str, Any] | None,
) -> OrbitalSparseConfig:
    if config is None:
        config = OrbitalSparseConfig()
    elif isinstance(config, dict):
        config = OrbitalSparseConfig(**config)
    if not isinstance(config, OrbitalSparseConfig):
        raise TypeError(
            'config must be OrbitalSparseConfig, dict, or None, got '
            f'{type(config)}.'
        )
    if config.optimizer_name not in {'adam', 'adamw', 'sgd'}:
        raise ValueError(
            "optimizer_name must be one of {'adam', 'adamw', 'sgd'}, got "
            f'{config.optimizer_name!r}.'
        )
    if config.learning_rate <= 0.0:
        raise ValueError(
            f'learning_rate must be positive, got {config.learning_rate}.'
        )
    if config.gradient_clip is not None and config.gradient_clip <= 0.0:
        raise ValueError('gradient_clip must be positive when provided.')
    if config.n_steps < 0 or config.log_every <= 0:
        raise ValueError('n_steps must be non-negative and log_every positive.')
    if config.lambda_fit < 0.0:
        raise ValueError(f'lambda_fit must be non-negative, got {config.lambda_fit}.')
    if config.repair_steps < 0:
        raise ValueError(
            f'repair_steps must be non-negative, got {config.repair_steps}.'
        )
    if config.tol_unitary <= 0.0 or config.tol_mask <= 0.0:
        raise ValueError('tol_unitary and tol_mask must be positive.')
    if config.polar_floor <= 0.0 or config.imag_tol < 0.0:
        raise ValueError('polar_floor must be positive and imag_tol non-negative.')
    return config


def _create_optimizer(config: OrbitalSparseConfig) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if config.gradient_clip is not None:
        transforms.append(optax.clip_by_global_norm(config.gradient_clip))
    if config.optimizer_name == 'adam':
        transforms.append(optax.adam(config.learning_rate))
    elif config.optimizer_name == 'adamw':
        transforms.append(
            optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        )
    else:
        transforms.append(optax.sgd(config.learning_rate))
    return optax.chain(*transforms)


def _history_entry(
    *,
    step: int,
    rotation: np.ndarray,
    support: np.ndarray,
    metrics: dict[str, Any],
    optax_loss: float | None,
) -> dict[str, Any]:
    return {
        'step': int(step),
        'loss': float(metrics.get('loss', 0.0)),
        'optax_loss': None if optax_loss is None else float(optax_loss),
        'unitarity_error': unitary_error(rotation),
        'mask_error': mask_error(rotation, support),
        'rotation_in_no_basis': np.asarray(rotation, dtype=np.complex128).copy(),
        'metrics': metrics,
    }


def _projection_only_refine(
    *,
    support: np.ndarray,
    init_rotation: np.ndarray,
    loss_module: dict[str, Any],
    config: OrbitalSparseConfig,
) -> dict[str, Any]:
    final_rotation = alternating_support_projection(
        init_rotation,
        support,
        n_steps=config.repair_steps,
        floor=config.polar_floor,
    )
    final_metrics = loss_module['evaluate_metrics'](final_rotation)
    history = [
        _history_entry(
            step=0,
            rotation=final_rotation,
            support=support,
            metrics=final_metrics,
            optax_loss=None,
        )
    ]
    return {
        'mode': 'projection_only',
        'history': history,
        'final_rotation_in_no_basis': final_rotation,
        'final_metrics': final_metrics,
        'unitarity_error': unitary_error(final_rotation),
        'mask_error': mask_error(final_rotation, support),
    }


def optimize_sparse_with_support(
    *,
    base_context: dict[str, Any],
    loss_module: dict[str, Any],
    support: np.ndarray,
    init_post_no_rotation: np.ndarray,
    config: OrbitalSparseConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    del base_context
    config = coerce_sparse_config(config)
    support = np.asarray(support, dtype=bool)
    init_rotation = np.asarray(init_post_no_rotation, dtype=np.complex128)
    if support.shape != init_rotation.shape:
        raise ValueError(
            'support and init_post_no_rotation must have identical shapes, got '
            f'{support.shape} and {init_rotation.shape}.'
        )
    if init_rotation.ndim != 2 or init_rotation.shape[0] != init_rotation.shape[1]:
        raise ValueError(
            'init_post_no_rotation must be a square matrix, got '
            f'{init_rotation.shape}.'
        )

    build_rotation_loss_fn = loss_module.get('build_rotation_loss_fn')
    imag_max = float(np.max(np.abs(np.imag(init_rotation))))
    if build_rotation_loss_fn is None or imag_max > config.imag_tol:
        result = _projection_only_refine(
            support=support,
            init_rotation=init_rotation,
            loss_module=loss_module,
            config=config,
        )
        result.update(
            {
                'config': _asdict_safe(config),
                'support': support.copy(),
                'used_gradient_optimization': False,
                'fallback_reason': (
                    'missing_rotation_loss_fn'
                    if build_rotation_loss_fn is None
                    else 'complex_rotation_not_supported_yet'
                ),
            }
        )
        return result

    rotation_loss_fn = build_rotation_loss_fn()
    optimizer = _create_optimizer(config)
    init_real = np.real(project_support(init_rotation, support))
    support_jax = jnp.asarray(support)
    reference_rotation = jnp.asarray(np.real(init_rotation))

    def objective(current_rotation: jax.Array) -> jax.Array:
        masked = jnp.where(support_jax, current_rotation, 0.0)
        base_loss = rotation_loss_fn(masked)
        fit_loss = config.lambda_fit * jnp.sum(jnp.square(masked - reference_rotation))
        return jnp.real(base_loss + fit_loss)

    loss_and_grad = jax.jit(jax.value_and_grad(objective))
    params = jnp.asarray(init_real)
    opt_state = optimizer.init(params)

    history: list[dict[str, Any]] = []
    for step in range(1, config.n_steps + 1):
        loss_value, grads = loss_and_grad(params)
        grads = jnp.where(support_jax, grads, 0.0)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = jnp.where(support_jax, params, 0.0)
        params_np = polar_retraction(
            np.asarray(params, dtype=np.float64),
            floor=config.polar_floor,
        )
        params_np = np.real(project_support(params_np, support))
        params = jnp.asarray(params_np)

        if step % config.log_every == 0 or step == config.n_steps:
            rotation_np = np.asarray(params, dtype=np.complex128)
            metrics = loss_module['evaluate_metrics'](rotation_np)
            history.append(
                _history_entry(
                    step=step,
                    rotation=rotation_np,
                    support=support,
                    metrics=metrics,
                    optax_loss=float(loss_value),
                )
            )

    final_rotation = np.asarray(params, dtype=np.complex128)
    final_rotation = alternating_support_projection(
        final_rotation,
        support,
        n_steps=max(config.repair_steps, 1),
        floor=config.polar_floor,
    )
    final_metrics = loss_module['evaluate_metrics'](final_rotation)

    return {
        'config': _asdict_safe(config),
        'support': support.copy(),
        'mode': 'gradient_projection',
        'used_gradient_optimization': True,
        'history': history,
        'final_rotation_in_no_basis': final_rotation,
        'final_metrics': final_metrics,
        'unitarity_error': unitary_error(final_rotation),
        'mask_error': mask_error(final_rotation, support),
    }


__all__ = [
    'NOSparseConfig',
    'OrbitalSparseConfig',
    'coerce_sparse_config',
    'optimize_sparse_with_support',
]
