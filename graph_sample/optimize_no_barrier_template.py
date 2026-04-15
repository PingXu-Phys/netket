"""Call template for the barrier post-NO optimiser.

Most important knobs
--------------------
mu_gamma / delta_gamma
    Main barrier for occupation protection.
mu_active_frozen / delta_active_frozen
    Optional barrier for active/frozen separation.
eta_hopping / eta_interaction
    Mix norm reduction and structure shaping inside the Hamiltonian objective.
filled_tol / empty_tol / degeneracy_tol
    Control the hard block split before optimisation.
"""

import numpy as np

from netket.graph_sample.optimize_no_barrier import (
    NOBarrierOptimizationConfig,
    optimize_barrier_basis,
)

rdm: np.ndarray = ...                # (n, n) 1-RDM
hop: np.ndarray | None = ...         # (n, n) hopping matrix, or None
interaction: np.ndarray | None = ... # (n,) or (n, n) interaction matrix, or None
positions: np.ndarray | None = ...   # (n,) or (n, d) site positions, or None

config = NOBarrierOptimizationConfig(
    alpha_hopping=0.5,
    alpha_interaction=0.5,
    eta_hopping=0.5,
    eta_interaction=0.5,
    mu_gamma=5.0,
    delta_gamma=0.02,
    mu_active_frozen=0.0,
    delta_active_frozen=0.02,
    barrier_tau=0.02,
    lambda_locality=0.0,
    learning_rate=1.0e-2,
    n_steps=300,
    log_every=50,
)

result = optimize_barrier_basis(
    rdm=rdm,
    hopping_matrix=hop,
    interaction_matrix=interaction,
    site_positions=positions,
    config=config,
)

print("initial loss:", result["initial_metrics"]["loss"])
print("final loss:  ", result["final_metrics"]["loss"])
print("final terms: ", result["final_loss_terms"])
