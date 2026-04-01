# Copyright 2026 Ping Xu - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np

from jax import numpy as jnp

from netket.graph import AbstractGraph
from netket.graph import disjoint_union
from netket.hilbert import SpinOrbitalFermions

from .exchange import ExchangeRule, _compute_different_clusters_mask


def _build_mode_occupations(
    hilbert: SpinOrbitalFermions, occupations: np.ndarray | list[float]
) -> np.ndarray:
    if occupations is None:
        raise ValueError("occupations must be provided.")

    occ = np.asarray(occupations, dtype=float)
    if occ.ndim != 1:
        raise ValueError(
            "occupations must be a 1D array with one entry per orbital or mode."
        )
    if occ.size == 0:
        raise ValueError("occupations must not be empty.")
    if not np.all(np.isfinite(occ)):
        raise ValueError("occupations must contain only finite values.")

    if occ.shape[0] == hilbert.size:
        mode_occ = occ
    elif occ.shape[0] == hilbert.n_orbitals:
        mode_occ = np.tile(occ, hilbert.n_spin_subsectors)
    else:
        raise ValueError(
            "occupations must have length equal to the number of orbitals "
            f"({hilbert.n_orbitals}) or the full number of spin-modes "
            f"({hilbert.size} = {hilbert.n_orbitals} x {hilbert.n_spin_subsectors}), "
            f"got {occ.shape[0]}."
        )

    if np.any(mode_occ < 0.0) or np.any(mode_occ > 1.0):
        raise ValueError("occupations entries must lie in the interval [0, 1].")

    return mode_occ


def _safe_log(x, eps: float = 1.0e-12):
    return jnp.log(jnp.clip(x, eps, None))


def _proposal_weights(clusters, sigma, mode_occupations, eps: float = 1.0e-12):
    site_i = clusters[:, 0]
    site_j = clusters[:, 1]

    sigma_i = sigma[site_i]
    sigma_j = sigma[site_j]

    hop_i_to_j = sigma_i > sigma_j
    source = jnp.where(hop_i_to_j, site_i, site_j)
    target = jnp.where(hop_i_to_j, site_j, site_i)

    occ_source = jnp.clip(mode_occupations[source], eps, 1.0 - eps)
    occ_target = jnp.clip(mode_occupations[target], eps, 1.0 - eps)
    weights = occ_target * (1.0 - occ_source)
    hoppable = sigma_i != sigma_j
    return jnp.where(hoppable, weights, 0.0)


def _selected_move_log_weights(
    clusters, cluster_id, sigma, mode_occupations, eps: float = 1.0e-12
):
    site_i = clusters[cluster_id, 0]
    site_j = clusters[cluster_id, 1]

    hop_i_to_j = sigma[site_i] > sigma[site_j]
    source = jnp.where(hop_i_to_j, site_i, site_j)
    target = jnp.where(hop_i_to_j, site_j, site_i)

    occ_source = jnp.clip(mode_occupations[source], eps, 1.0 - eps)
    occ_target = jnp.clip(mode_occupations[target], eps, 1.0 - eps)

    log_w_forward = _safe_log(occ_target * (1.0 - occ_source))
    log_w_backward = _safe_log(
        occ_source * (1.0 - occ_target)
    )
    return log_w_forward, log_w_backward


class FermionHopRule_with_proposal(ExchangeRule):
    """
    Hopping rule with proposal probabilities biased by target occupations.

    The proposal weight for a legal hop ``i -> j`` is
    ``W_{i->j} = occupations[j] * (1 - occupations[i])``.
    """

    def __init__(
        self,
        hilbert,
        *,
        occupations: np.ndarray | list[float],
        clusters: list[tuple[int, int]] | None = None,
        graph: AbstractGraph | None = None,
        d_max: int = 1,
        spin_symmetric: bool = True,
    ):
        r"""
        Constructs the FermionHopRule_with_proposal.

        Args:
            hilbert: The hilbert space to be sampled.
            occupations: Mean occupations for each orbital or mode. The length
                must be either ``hilbert.n_orbitals`` or ``hilbert.size``.
            clusters: Explicit cluster list.
            graph: Graph used to build clusters if ``clusters`` is not passed.
            d_max: Maximum graph distance used when clusters are derived from a graph.
            spin_symmetric: Replicate orbital-level graph/clusters across spin sectors.
        """
        if not isinstance(hilbert, SpinOrbitalFermions):
            raise ValueError(
                "This sampler rule currently only works with SpinOrbitalFermions hilbert spaces."
            )

        self.occupations = jnp.asarray(_build_mode_occupations(hilbert, occupations))

        if spin_symmetric and hilbert.n_spin_subsectors > 1:
            if graph is not None and graph.n_nodes == hilbert.n_orbitals:
                graph = disjoint_union(*[graph] * hilbert.n_spin_subsectors)
            if clusters is not None and np.max(clusters) < hilbert.n_orbitals:
                clusters = np.concatenate(
                    [
                        clusters + i * hilbert.n_orbitals
                        for i in range(hilbert.n_spin_subsectors)
                    ]
                )
        super().__init__(clusters=clusters, graph=graph, d_max=d_max)

    def transition(rule, sampler, machine, parameters, state, key, sigma):
        """
        Custom transition hook.

        Clusters are sampled with proposal weights
        ``W_{i->j} = occupations[j] * (1 - occupations[i])``,
        where ``i`` is the occupied source and ``j`` the empty target.
        The returned log correction matches the corresponding asymmetric
        forward/backward proposal ratio.
        """
        n_chains = sigma.shape[0]

        hoppable_clusters = _compute_different_clusters_mask(rule.clusters, sigma)
        keys = jnp.asarray(jax.random.split(key, n_chains))

        @jax.vmap
        def _update_samples(key, sigma, hoppable_clusters):
            proposal_weights = _proposal_weights(
                rule.clusters, sigma, rule.occupations
            )
            proposal_weights = jnp.where(hoppable_clusters, proposal_weights, 0.0)
            z_sigma = jnp.sum(proposal_weights)
            proposal_probs = proposal_weights / z_sigma

            cluster = jax.random.choice(
                key,
                a=jnp.arange(rule.clusters.shape[0]),
                p=proposal_probs,
                replace=True,
            )

            site_i = rule.clusters[cluster, 0]
            site_j = rule.clusters[cluster, 1]

            sigma_proposed = sigma.at[site_i].set(sigma[site_j])
            sigma_proposed = sigma_proposed.at[site_j].set(sigma[site_i])

            weights_proposed = _proposal_weights(
                rule.clusters, sigma_proposed, rule.occupations
            )
            z_sigma_proposed = jnp.sum(weights_proposed)

            log_w_forward, log_w_backward = _selected_move_log_weights(
                rule.clusters, cluster, sigma, rule.occupations
            )
            log_prob_corr = (
                log_w_backward
                - _safe_log(z_sigma_proposed)
                - log_w_forward
                + _safe_log(z_sigma)
            )
            return sigma_proposed, log_prob_corr

        return _update_samples(keys, sigma, hoppable_clusters)

    def __repr__(self):
        return (
            "FermionHopRule_with_proposal("
            f"# of clusters: {len(self.clusters)}, "
            f"# of occupations: {len(self.occupations)})"
        )
