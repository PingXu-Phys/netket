"""Barrier-style post-NO orbital optimisation.

Loss
----
This module implements the barrier loss described in
``loss_function_revised_full.md``:

    L = C_H
      + mu_gamma * barrier(C_gamma - delta_gamma)
      + mu_active_frozen * barrier(C_af - delta_active_frozen)
      + lambda_locality * C_loc

with

    C_H = alpha_hopping * C_t_star + alpha_interaction * C_int_star
    C_*_star = eta * norm + (1 - eta) * structure

Important Parameters
--------------------
alpha_hopping / alpha_interaction
    Relative weights of the hopping and interaction parts inside the main
    Hamiltonian objective C_H.
eta_hopping / eta_interaction
    Mix norm reduction and structure shaping.
    eta = 1.0 means only the weighted norm term.
    eta = 0.0 means only the structure term.
mu_gamma / delta_gamma
    Barrier strength and tolerance for occupation-structure leakage C_gamma.
    delta_gamma is the allowed deviation; mu_gamma is how hard the wall becomes
    after crossing that threshold.
mu_active_frozen / delta_active_frozen
    Optional second barrier for active/frozen mixing. Leave mu_active_frozen at
    0.0 if you do not want this extra constraint.
barrier_tau
    Smoothness of the barrier. Smaller values make the wall sharper.
lambda_locality
    Weak spread regulariser.
filled_tol / empty_tol / degeneracy_tol
    Define the hard occupation blocks before optimisation. Different blocks
    never mix. Smaller degeneracy_tol gives finer blocks; larger values give
    coarser blocks.
distance_weight_strength / distance_power
    Control how strongly long-range pair couplings are penalised.
structure_metric
    Chooses how the structure term is measured: ``participation``, ``decay``,
    or ``hybrid``.
learning_rate / n_steps
    Standard optimiser controls.

Recommended Start
-----------------
The defaults are a practical first pass:

    alpha_hopping = alpha_interaction = 0.5
    eta_hopping = eta_interaction = 0.5
    mu_gamma = 5.0
    delta_gamma = 0.02
    mu_active_frozen = 0.0
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
        NOBarrierOptimizationConfig,
        build_barrier_loss_module,
        coerce_barrier_config,
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
        NOBarrierOptimizationConfig,
        build_barrier_loss_module,
        coerce_barrier_config,
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
    """Extract the shared optimisation context from a barrier context dict."""
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
    """Resolve the prepared base context and barrier loss module from a context dict."""
    base_context = _base_context_from_context(context)
    loss_module = context.get("loss_module")
    if loss_module is None:
        loss_module = build_barrier_loss_module(
            base_context,
            config=context.get("config"),
        )
    return base_context, loss_module


def prepare_barrier_optimization_context(
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
    config: NOBarrierOptimizationConfig | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Prepare the shared base context and attach the barrier loss module."""
    config = coerce_barrier_config(config)
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
    loss_module = build_barrier_loss_module(base_context, config=config)
    context = dict(base_context)
    context.update(loss_module["loss_context"])
    context["config"] = config
    context["base_context"] = base_context
    context["loss_module"] = loss_module
    context["initial_metrics"] = loss_module["evaluate_metrics"](
        resolve_post_no_rotation(base_context)
    )
    return context


def build_barrier_loss_fn(context: dict[str, Any]):
    """Return a JAX loss function for the barrier objective."""
    _, loss_module = _loss_module_from_context(context)
    return loss_module["build_loss_fn"]()


def evaluate_barrier_basis_metrics(
    context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate the barrier loss and all of its components."""
    base_context, loss_module = _loss_module_from_context(context)
    resolved_post_no_rotation = resolve_post_no_rotation(
        base_context,
        post_no_rotation=post_no_rotation,
        params=params,
    )
    return loss_module["evaluate_metrics"](resolved_post_no_rotation)


def evaluate_barrier_loss(
    context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate the barrier loss at a supplied post-NO rotation."""
    base_context, loss_module = _loss_module_from_context(context)
    return evaluate_with_loss(
        base_context=base_context,
        loss_module=loss_module,
        post_no_rotation=post_no_rotation,
        params=params,
    )


def create_barrier_optimizer(
    config: NOBarrierOptimizationConfig | None = None,
) -> optax.GradientTransformation:
    """Create the Optax optimiser used by the barrier variant."""
    return create_optimizer(coerce_barrier_config(config))


def optimize_barrier_basis(
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
    config: NOBarrierOptimizationConfig | None = None,
    optimizer: optax.GradientTransformation | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Optimise the post-NO rotation using the barrier loss."""
    context = prepare_barrier_optimization_context(
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
    return optimize_barrier_from_context(context, optimizer=optimizer)


def optimize_barrier_from_context(
    context: dict[str, Any],
    *,
    optimizer: optax.GradientTransformation | None = None,
) -> dict[str, Any]:
    """Optimise starting from a prepared barrier context."""
    config = coerce_barrier_config(context.get("config"))
    base_context, loss_module = _loss_module_from_context(context)
    result = optimize_with_loss(
        base_context=base_context,
        loss_module=loss_module,
        optimizer=optimizer,
    )
    result["config"] = asdict(config)
    return result


__all__ = [
    "NOBarrierOptimizationConfig",
    "build_barrier_loss_fn",
    "create_barrier_optimizer",
    "evaluate_barrier_basis_metrics",
    "evaluate_barrier_loss",
    "optimize_barrier_basis",
    "optimize_barrier_from_context",
    "prepare_barrier_optimization_context",
]
