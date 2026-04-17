"""Minimal soft post-NO orbital optimisation.

Loss
----
This module implements the minimal soft loss described in
``loss_function_revised_minimal.md``:

    L = alpha_hopping * C_t_norm
      + alpha_interaction * C_int_norm
      + lambda_occupancy * C_gamma
      + lambda_locality * C_loc

Important Parameters
--------------------
alpha_hopping
    Weight of the one-body complexity term. Larger values push harder to
    simplify the rotated hopping matrix.
alpha_interaction
    Weight of the interaction complexity term. Larger values push harder to
    compress the effective scattering channels.
lambda_occupancy
    Soft penalty for leaving the NO occupation structure. Larger values keep
    the result closer to the NO basis. In the minimal version this is the main
    regularisation parameter and should usually stay small.
lambda_locality
    Weak spread regulariser. Set to 0.0 for the first pass; increase only if
    the optimised orbitals become too delocalised in real space.
filled_tol / empty_tol
    Define which orbitals are treated as near-full and near-empty before the
    optimisation starts. They determine the hard sector split.
degeneracy_tol
    Further splits a sector when neighbouring occupations differ too much.
    Smaller values create more, smaller blocks; larger values merge more
    orbitals into the same block. Different blocks never mix.
distance_weight_strength / distance_power
    Control how strongly long-range pair couplings are penalised inside the
    weighted smooth-L1 objective.
learning_rate / n_steps
    Standard optimiser controls.

Recommended Start
-----------------
The defaults follow the recommended minimal setup:

    alpha_hopping = alpha_interaction = 0.5
    lambda_occupancy = 0.05
    lambda_locality = 0.0
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import optax

try:
    from .optimize_no_core import (
        create_optimizer,
        core_config_from_config,
        evaluate_with_loss,
        optimize_with_loss,
        prepare_base_context,
        resolve_post_no_rotation,
    )
    from .optimize_no_losses import (
        NOSoftOptimizationConfig,
        build_soft_loss_module,
        coerce_soft_config,
    )
except ImportError:
    from orbital_optimization.optimize_no_core import (
        create_optimizer,
        core_config_from_config,
        evaluate_with_loss,
        optimize_with_loss,
        prepare_base_context,
        resolve_post_no_rotation,
    )
    from orbital_optimization.optimize_no_losses import (
        NOSoftOptimizationConfig,
        build_soft_loss_module,
        coerce_soft_config,
    )


_BASE_CONTEXT_KEYS = (
    "H",
    "rdm",
    "spin_symmetric",
    "core_config",
    "natural_occupations",
    "natural_orbitals",
    "occupation_blocks",
    "hopping_matrix",
    "hopping_no",
    "interaction_matrix",
    "distance_matrix",
    "site_positions",
)


def _base_context_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """Extract the shared optimisation context from a soft context dict."""
    if "base_context" in context:
        base_context = dict(context["base_context"])
    else:
        base_context = {key: context[key] for key in _BASE_CONTEXT_KEYS if key in context}
    if "core_config" not in base_context:
        base_context["core_config"] = context.get("config")
    base_context["core_config"] = core_config_from_config(base_context["core_config"])
    return base_context


def _loss_module_from_context(
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the prepared base context and soft loss module from a context dict."""
    base_context = _base_context_from_context(context)
    loss_module = context.get("loss_module")
    if loss_module is None:
        loss_module = build_soft_loss_module(
            base_context,
            config=context.get("config"),
        )
    return base_context, loss_module


def prepare_soft_optimization_context(
    *,
    H=None,
    rdm: np.ndarray,
    natural_occupations: np.ndarray | None = None,
    natural_orbitals: np.ndarray | None = None,
    hopping_matrix: np.ndarray | None = None,
    interaction_matrix: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    active_indices: list | np.ndarray | None = None,
    config: NOSoftOptimizationConfig | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Prepare the shared base context and attach the soft loss module."""
    config = coerce_soft_config(config)
    base_context = prepare_base_context(
        H=H,
        rdm=rdm,
        natural_occupations=natural_occupations,
        natural_orbitals=natural_orbitals,
        hopping_matrix=hopping_matrix,
        interaction_matrix=interaction_matrix,
        distance_matrix=distance_matrix,
        site_positions=site_positions,
        active_indices=active_indices,
        core_config=config,
        spin_symmetric=spin_symmetric,
    )
    loss_module = build_soft_loss_module(base_context, config=config)
    context = dict(base_context)
    context.update(loss_module["loss_context"])
    context["config"] = config
    context["base_context"] = base_context
    context["loss_module"] = loss_module
    context["initial_metrics"] = loss_module["evaluate_metrics"](
        resolve_post_no_rotation(base_context)
    )
    return context


def build_soft_loss_fn(context: dict[str, Any]):
    """Return a JAX loss function for the minimal soft objective."""
    _, loss_module = _loss_module_from_context(context)
    return loss_module["build_loss_fn"]()


def evaluate_soft_basis_metrics(
    context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate the minimal soft loss and the underlying diagnostic terms."""
    base_context, loss_module = _loss_module_from_context(context)
    resolved_post_no_rotation = resolve_post_no_rotation(
        base_context,
        post_no_rotation=post_no_rotation,
        params=params,
    )
    return loss_module["evaluate_metrics"](resolved_post_no_rotation)


def evaluate_soft_loss(
    context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate the minimal soft loss at a supplied post-NO rotation."""
    base_context, loss_module = _loss_module_from_context(context)
    return evaluate_with_loss(
        base_context=base_context,
        loss_module=loss_module,
        post_no_rotation=post_no_rotation,
        params=params,
    )


def create_soft_optimizer(
    config: NOSoftOptimizationConfig | None = None,
) -> optax.GradientTransformation:
    """Create the Optax optimiser used by the soft variant."""
    return create_optimizer(coerce_soft_config(config))


def optimize_soft_basis(
    *,
    H=None,
    rdm: np.ndarray,
    natural_occupations: np.ndarray | None = None,
    natural_orbitals: np.ndarray | None = None,
    hopping_matrix: np.ndarray | None = None,
    interaction_matrix: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    active_indices: list | np.ndarray | None = None,
    config: NOSoftOptimizationConfig | None = None,
    optimizer: optax.GradientTransformation | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Optimise the post-NO rotation using the minimal soft loss."""
    context = prepare_soft_optimization_context(
        H=H,
        rdm=rdm,
        natural_occupations=natural_occupations,
        natural_orbitals=natural_orbitals,
        hopping_matrix=hopping_matrix,
        interaction_matrix=interaction_matrix,
        distance_matrix=distance_matrix,
        site_positions=site_positions,
        active_indices=active_indices,
        config=config,
        spin_symmetric=spin_symmetric,
    )
    return optimize_soft_from_context(context, optimizer=optimizer)


def optimize_soft_from_context(
    context: dict[str, Any],
    *,
    optimizer: optax.GradientTransformation | None = None,
) -> dict[str, Any]:
    """Optimise starting from a prepared soft context."""
    config = coerce_soft_config(context.get("config"))
    base_context, loss_module = _loss_module_from_context(context)
    result = optimize_with_loss(
        base_context=base_context,
        loss_module=loss_module,
        optimizer=optimizer,
    )
    result["config"] = asdict(config)
    return result


__all__ = [
    "NOSoftOptimizationConfig",
    "build_soft_loss_fn",
    "create_soft_optimizer",
    "evaluate_soft_basis_metrics",
    "evaluate_soft_loss",
    "optimize_soft_basis",
    "optimize_soft_from_context",
    "prepare_soft_optimization_context",
]
