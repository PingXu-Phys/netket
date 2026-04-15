"""Common optimisation core for the optimize_no family."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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
    from .optimize_no import (
        EPS,
        _hermitian_part,
        _require_square_matrix,
        _resolve_distance_matrix,
        _resolve_interaction_matrix,
        _validate_no_reference_data,
        build_occupation_blocks,
        extract_hamiltonian_terms,
        extract_interaction_matrix,
        extract_one_body_hopping_matrix,
        rotate_hamiltonian_to_basis,
        rotate_hamiltonian_to_natural_orbitals,
        rotate_one_body_matrix,
    )
except ImportError:
    from iteration_occ_func_simple import (
        _expand_orbital_rotation_to_modes,
        natural_orbitals_from_rdm,
        rotate_fermion_hamiltonian,
    )
    from optimize_no import (
        EPS,
        _hermitian_part,
        _require_square_matrix,
        _resolve_distance_matrix,
        _resolve_interaction_matrix,
        _validate_no_reference_data,
        build_occupation_blocks,
        extract_hamiltonian_terms,
        extract_interaction_matrix,
        extract_one_body_hopping_matrix,
        rotate_hamiltonian_to_basis,
        rotate_hamiltonian_to_natural_orbitals,
        rotate_one_body_matrix,
    )


@dataclass(slots=True)
class NOCoreConfig:
    """Shared optimisation settings used by every optimise-no variant."""

    filled_tol: float = 0.05
    empty_tol: float = 0.05
    degeneracy_tol: float = 0.02
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    n_steps: int = 200
    log_every: int = 25
    hamiltonian_cutoff: float | None = None
    real_orbitals: bool = True


def coerce_core_config(config: NOCoreConfig | None) -> NOCoreConfig:
    """Validate the shared optimisation configuration."""
    if config is None:
        config = NOCoreConfig()
    if config.optimizer_name not in {"adam", "adamw", "sgd"}:
        raise ValueError(
            "optimizer_name must be one of {'adam', 'adamw', 'sgd'}, got "
            f"{config.optimizer_name!r}."
        )
    if config.n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {config.n_steps}.")
    if config.log_every <= 0:
        raise ValueError(f"log_every must be positive, got {config.log_every}.")
    if config.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be strictly positive, got {config.learning_rate}."
        )
    if config.gradient_clip is not None and config.gradient_clip <= 0.0:
        raise ValueError(
            "gradient_clip must be positive when provided, got "
            f"{config.gradient_clip}."
        )
    return config


def core_config_from_config(config_like: Any) -> NOCoreConfig:
    """Extract the shared optimisation settings from any config object."""
    if isinstance(config_like, NOCoreConfig):
        return coerce_core_config(config_like)
    values: dict[str, Any] = {}
    for name in (
        "filled_tol",
        "empty_tol",
        "degeneracy_tol",
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "n_steps",
        "log_every",
        "hamiltonian_cutoff",
        "real_orbitals",
    ):
        if hasattr(config_like, name):
            values[name] = getattr(config_like, name)
    return coerce_core_config(NOCoreConfig(**values))


def create_optimizer(
    config: NOCoreConfig | Any | None = None,
) -> optax.GradientTransformation:
    """Create the shared Optax optimiser."""
    config = core_config_from_config(config) if config is not None else NOCoreConfig()
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
    """Initialise one anti-symmetric generator per non-trivial block."""
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


def _block_unitary(
    param: dict[str, jax.Array],
    *,
    real_orbitals: bool = True,
) -> jax.Array:
    """Exponentiate one block generator into an orthogonal / unitary rotation."""
    if real_orbitals:
        generator = param["real"] - param["real"].T
    else:
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
    """Assemble the full post-NO rotation from block-local parameters."""
    dtype = jnp.float64 if real_orbitals else jnp.complex128
    rotation = jnp.eye(n_orbitals, dtype=dtype)
    for block_id, block in enumerate(blocks):
        if block.size <= 1:
            continue
        block_idx = jnp.asarray(block, dtype=jnp.int32)
        block_rotation = _block_unitary(
            params[f"block_{block_id}"],
            real_orbitals=real_orbitals,
        )
        rotation = rotation.at[block_idx[:, None], block_idx[None, :]].set(block_rotation)
    return rotation


def resolve_post_no_rotation(
    base_context: dict[str, Any],
    *,
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> np.ndarray:
    """Resolve the post-NO rotation from either params or an explicit matrix."""
    core_config = base_context["core_config"]
    n_orbitals = base_context["natural_orbitals"].shape[0]
    dtype = np.float64 if core_config.real_orbitals else np.complex128
    if post_no_rotation is None:
        if params is None:
            return np.eye(n_orbitals, dtype=dtype)
        return np.asarray(
            _build_post_no_rotation(
                params,
                base_context["occupation_blocks"],
                n_orbitals,
                real_orbitals=core_config.real_orbitals,
            )
        )
    post_no_rotation = _require_square_matrix(
        np.asarray(post_no_rotation, dtype=dtype),
        name="post_no_rotation",
    )
    if post_no_rotation.shape != (n_orbitals, n_orbitals):
        raise ValueError(
            "post_no_rotation must have shape "
            f"({n_orbitals}, {n_orbitals}), got {post_no_rotation.shape}."
        )
    return post_no_rotation


def prepare_base_context(
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
    core_config: NOCoreConfig | Any | None = None,
    spin_symmetric: bool = True,
) -> dict[str, Any]:
    """Prepare the shared base context used by every optimise-no variant."""
    core_config = core_config_from_config(core_config)
    rdm = _hermitian_part(_require_square_matrix(rdm, name="rdm"))
    n_full = rdm.shape[0]
    if active_indices is not None:
        active_idx = np.asarray(active_indices, dtype=int)
        if active_idx.ndim != 1 or np.any(active_idx < 0) or np.any(active_idx >= n_full):
            raise ValueError(
                f"active_indices must be a 1-D array of valid orbital indices in "
                f"[0, {n_full - 1}], got {active_idx}."
            )
        ix = np.ix_(active_idx, active_idx)
        rdm = rdm[ix]
        if hopping_matrix is None and H is not None:
            hopping_matrix = extract_one_body_hopping_matrix(H, spin_symmetric=spin_symmetric)
        if hopping_matrix is not None:
            hopping_matrix = np.asarray(hopping_matrix)[ix]
        if interaction_matrix is None and H is not None:
            interaction_matrix = extract_interaction_matrix(H, spin_symmetric=spin_symmetric)
        if interaction_matrix is not None:
            interaction_matrix = _resolve_interaction_matrix(interaction_matrix, n_full)[ix]
        if distance_matrix is not None:
            distance_matrix = np.asarray(distance_matrix)[ix]
        if site_positions is not None:
            positions = np.asarray(site_positions, dtype=float)
            if positions.ndim == 1:
                positions = positions[:, None]
            site_positions = positions[active_idx]
        natural_occupations = None
        natural_orbitals = None

    if natural_occupations is None or natural_orbitals is None:
        natural_occupations, natural_orbitals = natural_orbitals_from_rdm(rdm)

    natural_occupations = np.asarray(natural_occupations, dtype=float)
    natural_orbitals = _require_square_matrix(
        np.asarray(natural_orbitals, dtype=np.complex128),
        name="natural_orbitals",
    )
    _validate_no_reference_data(rdm, natural_occupations, natural_orbitals)

    if hopping_matrix is None and H is not None:
        hopping_matrix = extract_one_body_hopping_matrix(H, spin_symmetric=spin_symmetric)
    if hopping_matrix is not None:
        hopping_matrix = _hermitian_part(
            _require_square_matrix(hopping_matrix, name="hopping_matrix")
        )
        hopping_no = rotate_one_body_matrix(hopping_matrix, natural_orbitals)
    else:
        hopping_no = None

    resolved_distance = _resolve_distance_matrix(natural_orbitals.shape[0], distance_matrix)
    resolved_interaction = _resolve_interaction_matrix(
        interaction_matrix,
        natural_orbitals.shape[0],
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
        filled_tol=core_config.filled_tol,
        empty_tol=core_config.empty_tol,
        degeneracy_tol=core_config.degeneracy_tol,
    )

    return {
        "H": H,
        "rdm": rdm,
        "spin_symmetric": spin_symmetric,
        "core_config": core_config,
        "natural_occupations": natural_occupations.copy(),
        "natural_orbitals": natural_orbitals.copy(),
        "occupation_blocks": [block.copy() for block in occupation_blocks],
        "hopping_matrix": None if hopping_matrix is None else hopping_matrix.copy(),
        "hopping_no": None if hopping_no is None else hopping_no.copy(),
        "interaction_matrix": None
        if resolved_interaction is None
        else resolved_interaction.copy(),
        "distance_matrix": resolved_distance.copy(),
        "site_positions": None
        if resolved_positions is None
        else resolved_positions.copy(),
    }


def _asdict_safe(config: Any) -> Any:
    """Convert dataclasses to plain dicts while leaving other objects untouched."""
    return asdict(config) if is_dataclass(config) else config


def evaluate_with_loss(
    *,
    base_context: dict[str, Any],
    loss_module: dict[str, Any],
    post_no_rotation: np.ndarray | None = None,
    params: dict[str, dict[str, jax.Array]] | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared loss module at a supplied post-NO rotation."""
    post_no_rotation = resolve_post_no_rotation(
        base_context,
        post_no_rotation=post_no_rotation,
        params=params,
    )
    metrics = loss_module["evaluate_metrics"](post_no_rotation)
    return {
        "loss": float(metrics["loss"]),
        "loss_terms": dict(metrics["loss_terms"]),
        "post_no_rotation": post_no_rotation,
        "site_to_optimized_orbital": metrics["site_to_optimized_orbital"],
        "metrics": metrics,
    }


def optimize_with_loss(
    *,
    base_context: dict[str, Any],
    loss_module: dict[str, Any],
    optimizer: optax.GradientTransformation | None = None,
) -> dict[str, Any]:
    """Run the shared optimisation loop for a prepared loss module."""
    core_config = base_context["core_config"]
    blocks = base_context["occupation_blocks"]
    n_orbitals = base_context["natural_orbitals"].shape[0]
    real_orbitals = core_config.real_orbitals
    dtype = np.float64 if real_orbitals else np.complex128

    identity = np.eye(n_orbitals, dtype=dtype)
    initial_metrics = loss_module["evaluate_metrics"](identity)
    history = [loss_module["history_entry"](0, initial_metrics)]

    params = _init_block_params(blocks, real_orbitals=real_orbitals)
    if len(params) == 0 or core_config.n_steps == 0:
        final_post_no_rotation = identity
        current_rotation = base_context["natural_orbitals"] @ final_post_no_rotation
        final_metrics = initial_metrics
    else:
        optimizer = optimizer or create_optimizer(core_config)
        opt_state = optimizer.init(params)
        loss_fn = loss_module["build_loss_fn"]()
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        for step in range(1, core_config.n_steps + 1):
            loss_value, grads = loss_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % core_config.log_every == 0 or step == core_config.n_steps:
                current_post_no_rotation = np.asarray(
                    _build_post_no_rotation(
                        params,
                        blocks,
                        n_orbitals,
                        real_orbitals=real_orbitals,
                    )
                )
                metrics = loss_module["evaluate_metrics"](current_post_no_rotation)
                entry = loss_module["history_entry"](step, metrics)
                entry["optax_loss"] = float(loss_value)
                history.append(entry)

        final_post_no_rotation = current_post_no_rotation
        current_rotation = base_context["natural_orbitals"] @ final_post_no_rotation
        final_metrics = metrics

    optimized_hamiltonian = None
    mode_rotation = None
    H = base_context["H"]
    spin_symmetric = base_context["spin_symmetric"]
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
            cutoff=core_config.hamiltonian_cutoff,
        )

    result = {
        "config": _asdict_safe(loss_module["config"]),
        "core_config": asdict(core_config),
        "optimizer_name": core_config.optimizer_name,
        "natural_occupations": base_context["natural_occupations"].copy(),
        "natural_orbitals": base_context["natural_orbitals"].copy(),
        "occupation_blocks": [block.copy() for block in blocks],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "final_loss_terms": dict(final_metrics["loss_terms"]),
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
        "interaction_footprint": None
        if final_metrics["interaction_footprint"] is None
        else final_metrics["interaction_footprint"].copy(),
        "effective_scattering": None
        if final_metrics["effective_scattering"] is None
        else final_metrics["effective_scattering"].copy(),
    }
    extra_result_fields = loss_module.get("result_fields")
    if extra_result_fields is not None:
        result.update(extra_result_fields(final_metrics))
    return result


__all__ = [
    "EPS",
    "NOCoreConfig",
    "_build_post_no_rotation",
    "_init_block_params",
    "build_occupation_blocks",
    "coerce_core_config",
    "core_config_from_config",
    "create_optimizer",
    "evaluate_with_loss",
    "extract_hamiltonian_terms",
    "extract_interaction_matrix",
    "extract_one_body_hopping_matrix",
    "optimize_with_loss",
    "prepare_base_context",
    "resolve_post_no_rotation",
    "rotate_hamiltonian_to_basis",
    "rotate_hamiltonian_to_natural_orbitals",
    "rotate_one_body_matrix",
]
