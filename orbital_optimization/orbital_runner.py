"""Top-level runner for the orbital optimisation workflow."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .orbital_losses import (
        build_barrier_loss_module,
        build_optimize_loss_module,
        build_original_loss_module,
        build_soft_loss_module,
    )
    from .orbital_optimizer import (
        optimize_with_loss,
        prepare_base_context,
        rotate_hamiltonian_to_basis,
    )
    from .orbital_pruning import run_pruning_schedule
    from .orbital_sparse import optimize_sparse_with_support
except ImportError:
    from orbital_optimization.orbital_losses import (
        build_barrier_loss_module,
        build_optimize_loss_module,
        build_original_loss_module,
        build_soft_loss_module,
    )
    from orbital_optimization.orbital_optimizer import (
        optimize_with_loss,
        prepare_base_context,
        rotate_hamiltonian_to_basis,
    )
    from orbital_optimization.orbital_pruning import run_pruning_schedule
    from orbital_optimization.orbital_sparse import optimize_sparse_with_support


_DENSE_BUILDERS = {
    'soft': build_soft_loss_module,
    'barrier': build_barrier_loss_module,
    'optimize': build_optimize_loss_module,
    'original': build_original_loss_module,
}


def _build_dense_loss_module(
    *,
    base_context: dict[str, Any],
    dense_kind: str,
    dense_config: Any,
) -> dict[str, Any]:
    dense_kind = dense_kind.lower().strip()
    if dense_kind not in _DENSE_BUILDERS:
        raise ValueError(
            "dense_kind must be one of {'soft', 'barrier', 'optimize', 'original'}, "
            f'got {dense_kind!r}.'
        )
    return _DENSE_BUILDERS[dense_kind](base_context, config=dense_config)


def _finalise_sparse_result(
    *,
    base_context: dict[str, Any],
    sparse_result: dict[str, Any],
    spin_symmetric: bool,
) -> dict[str, Any]:
    result = dict(sparse_result)
    final_rotation = np.asarray(result['final_rotation_in_no_basis'], dtype=np.complex128)
    final_basis = np.asarray(base_context['natural_orbitals'], dtype=np.complex128) @ final_rotation
    result['site_to_optimized_orbital'] = final_basis
    result['rotation_in_no_basis'] = final_rotation
    result['optimized_hamiltonian'] = None
    if base_context.get('H') is not None:
        result['optimized_hamiltonian'] = rotate_hamiltonian_to_basis(
            base_context['H'],
            final_basis,
            spin_symmetric=spin_symmetric,
            cutoff=base_context['core_config'].hamiltonian_cutoff,
        )
    return result


def optimize_sparse_no_basis_flow(
    *,
    H=None,
    rdm: np.ndarray,
    dense_kind: str = 'soft',
    dense_config=None,
    pruning_config=None,
    sparse_config=None,
    netket_eval_config=None,
    hopping_matrix=None,
    interaction_matrix=None,
    distance_matrix=None,
    site_positions=None,
    active_indices=None,
    spin_symmetric: bool = True,
    vstate_builder=None,
) -> dict[str, Any]:
    del netket_eval_config, vstate_builder
    base_context = prepare_base_context(
        H=H,
        rdm=rdm,
        hopping_matrix=hopping_matrix,
        interaction_matrix=interaction_matrix,
        distance_matrix=distance_matrix,
        site_positions=site_positions,
        active_indices=active_indices,
        core_config=dense_config,
        spin_symmetric=spin_symmetric,
    )
    loss_module = _build_dense_loss_module(
        base_context=base_context,
        dense_kind=dense_kind,
        dense_config=dense_config,
    )
    dense_result = optimize_with_loss(
        base_context=base_context,
        loss_module=loss_module,
    )
    pruning_result = run_pruning_schedule(
        base_context=base_context,
        loss_module=loss_module,
        dense_result=dense_result,
        config=pruning_config,
    )
    sparse_result = optimize_sparse_with_support(
        base_context=base_context,
        loss_module=loss_module,
        support=pruning_result['stable_support'],
        init_post_no_rotation=pruning_result['final_rotation_in_no_basis'],
        config=sparse_config,
    )
    sparse_result = _finalise_sparse_result(
        base_context=base_context,
        sparse_result=sparse_result,
        spin_symmetric=spin_symmetric,
    )
    return {
        'dense_kind': dense_kind,
        'base_context': base_context,
        'loss_module_kind': loss_module.get('kind'),
        'dense_result': dense_result,
        'pruning_result': pruning_result,
        'sparse_result': sparse_result,
        'netket_eval': None,
    }


__all__ = [
    'optimize_sparse_no_basis_flow',
]
