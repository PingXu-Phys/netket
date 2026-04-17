import numpy as np

from .. import common

pytestmark = common.skipif_distributed

from orbital_optimization.orbital_losses import NOSoftOptimizationConfig
from orbital_optimization.orbital_pruning import OrbitalPruningConfig
from orbital_optimization.orbital_runner import optimize_sparse_no_basis_flow
from orbital_optimization.orbital_sparse import OrbitalSparseConfig


def test_orbital_runner_smoke():
    rdm = np.diag([0.95, 0.55, 0.45, 0.05]).astype(np.complex128)
    hopping = np.array(
        [
            [0.0, 0.2, 0.0, 0.0],
            [0.2, 0.1, 0.15, 0.0],
            [0.0, 0.15, -0.1, 0.18],
            [0.0, 0.0, 0.18, 0.0],
        ],
        dtype=np.float64,
    )
    interaction = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.2, 0.9, 0.15, 0.0],
            [0.0, 0.15, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    site_positions = np.arange(4, dtype=np.float64)[:, None]

    dense_config = NOSoftOptimizationConfig(
        n_steps=0,
        log_every=1,
        learning_rate=1.0e-3,
        lambda_occupancy=0.05,
        lambda_locality=0.0,
    )
    pruning_config = OrbitalPruningConfig(
        taus=(0.5,),
        top_m=1,
        repair_steps=4,
        stop_on_reject=True,
    )
    sparse_config = OrbitalSparseConfig(
        n_steps=4,
        log_every=1,
        learning_rate=1.0e-3,
        lambda_fit=0.1,
        repair_steps=4,
    )

    result = optimize_sparse_no_basis_flow(
        rdm=rdm,
        dense_kind="soft",
        dense_config=dense_config,
        pruning_config=pruning_config,
        sparse_config=sparse_config,
        hopping_matrix=hopping,
        interaction_matrix=interaction,
        site_positions=site_positions,
    )

    assert result["dense_result"]["rotation_in_no_basis"].shape == (4, 4)
    assert result["pruning_result"]["stable_support"].shape == (4, 4)
    assert result["sparse_result"]["final_rotation_in_no_basis"].shape == (4, 4)
    assert result["sparse_result"]["site_to_optimized_orbital"].shape == (4, 4)
    assert result["netket_eval"] is None

    stable_support = result["pruning_result"]["stable_support"]
    assert stable_support.dtype == np.bool_
    assert np.array_equal(stable_support, np.eye(4, dtype=bool))
    assert result["pruning_result"]["accepted_steps"] == 1

    unitary_err = float(result["sparse_result"]["unitarity_error"])
    mask_err = float(result["sparse_result"]["mask_error"])
    assert unitary_err < 1.0e-6
    assert mask_err < 1.0e-8

    final_metrics = result["sparse_result"]["final_metrics"]
    assert "loss" in final_metrics
    assert "diag_occupations" in final_metrics
    assert final_metrics["diag_occupations"].shape == (4,)
