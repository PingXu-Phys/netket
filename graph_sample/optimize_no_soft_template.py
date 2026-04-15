"""Call template for the minimal soft post-NO optimiser.

Most important knobs
--------------------
lambda_occupancy
    Increase if the optimisation smears occupations too much.
alpha_hopping / alpha_interaction
    Rebalance the one-body and interaction objectives.
lambda_locality
    Turn on only when the orbitals become too spread out.
filled_tol / empty_tol / degeneracy_tol
    Control the hard block split before optimisation.
"""

import numpy as np

from netket.graph_sample.optimize_no_soft import (
    NOSoftOptimizationConfig,
    optimize_soft_basis,
)

rdm: np.ndarray = ...                # (n, n) 1-RDM
hop: np.ndarray | None = ...         # (n, n) hopping matrix, or None
interaction: np.ndarray | None = ... # (n,) or (n, n) interaction matrix, or None
positions: np.ndarray | None = ...   # (n,) or (n, d) site positions, or None

config = NOSoftOptimizationConfig(
    alpha_hopping=0.5,
    alpha_interaction=0.5,
    lambda_occupancy=0.05,
    lambda_locality=0.0,
    learning_rate=1.0e-2,
    n_steps=300,
    log_every=50,
)

result = optimize_soft_basis(
    rdm=rdm,
    hopping_matrix=hop,
    interaction_matrix=interaction,
    site_positions=positions,
    config=config,
)

print("initial loss:", result["initial_metrics"]["loss"])
print("final loss:  ", result["final_metrics"]["loss"])
print("final terms: ", result["final_loss_terms"])
print("optimized occupations:", result["optimized_diag_occupations"])
