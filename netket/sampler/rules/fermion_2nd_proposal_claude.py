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
    """Validate and expand user-provided occupations to a per-mode array.

    If `occupations` has length `n_orbitals`, it is tiled across all spin
    subsectors.  If it already has length `hilbert.size`, it is used as-is.
    Returns a 1-D numpy array of shape ``(hilbert.size,)`` with values in [0, 1].
    """
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
    """Numerically safe logarithm: clips input from below by `eps`."""
    return jnp.log(jnp.clip(x, eps, None))


def _proposal_weights(clusters, sigma, mode_occupations, eps: float = 1.0e-12):
    """Compute unnormalised proposal weight for every cluster given state `sigma`.

    For a cluster (i, j) where exactly one site is occupied, the weight is
    ``occ[target] * (1 - occ[source])`` with source being the occupied site
    and target the empty site.  Clusters where both sites have the same
    occupation get weight 0.
    """
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
        """Propose a new configuration by biased hopping and return log-correction.

        1. Compute proposal weights W for all clusters under current sigma.
        2. Sample one cluster proportionally to W.
        3. Swap the two sites of that cluster to get sigma'.
        4. Compute Z' incrementally (only clusters touching the swapped sites
           change their weight), avoiding a full recomputation.
        5. Return (sigma', log_prob_corr) where the correction equals
           log[q(sigma|sigma')] - log[q(sigma'|sigma)].
        """
        n_chains = sigma.shape[0]

        hoppable_clusters = _compute_different_clusters_mask(rule.clusters, sigma)
        keys = jnp.asarray(jax.random.split(key, n_chains))

        @jax.vmap
        def _update_samples(key, sigma, hoppable_clusters):
            eps = 1.0e-12
            clusters = rule.clusters
            occ = rule.occupations

            # --- forward proposal weights & normalisation ---
            weights = _proposal_weights(clusters, sigma, occ, eps)
            weights = jnp.where(hoppable_clusters, weights, 0.0)
            z_fwd = jnp.sum(weights)

            # --- sample a cluster ---
            cluster = jax.random.choice(
                key,
                a=jnp.arange(clusters.shape[0]),
                p=weights / z_fwd,
                replace=True,
            )

            site_i = clusters[cluster, 0]
            site_j = clusters[cluster, 1]

            # --- apply the swap ---
            sigma_proposed = sigma.at[site_i].set(sigma[site_j])
            sigma_proposed = sigma_proposed.at[site_j].set(sigma[site_i])

            # --- forward / backward log-weights for the selected cluster ---
            # Reuse the already-computed forward weight instead of recalculating.
            log_w_fwd = _safe_log(weights[cluster], eps)

            # Backward weight: the reverse hop swaps source <-> target,
            # so w_bwd = occ[source] * (1 - occ[target]).
            hop_i_to_j = sigma[site_i] > sigma[site_j]
            source = jnp.where(hop_i_to_j, site_i, site_j)
            target = jnp.where(hop_i_to_j, site_j, site_i)
            occ_s = jnp.clip(occ[source], eps, 1.0 - eps)
            occ_t = jnp.clip(occ[target], eps, 1.0 - eps)
            log_w_bwd = _safe_log(occ_s * (1.0 - occ_t), eps)

            # --- incremental Z' ---
            # Only clusters that touch site_i or site_j can change weight.
            c0 = clusters[:, 0]
            c1 = clusters[:, 1]
            affected = (
                (c0 == site_i) | (c1 == site_i) |
                (c0 == site_j) | (c1 == site_j)
            )

            old_contribution = jnp.sum(jnp.where(affected, weights, 0.0))

            # Recompute weights only for affected clusters under sigma'.
            sp0 = sigma_proposed[c0]
            sp1 = sigma_proposed[c1]
            hop_new = sp0 > sp1
            src_new = jnp.where(hop_new, c0, c1)
            tgt_new = jnp.where(hop_new, c1, c0)
            occ_src_new = jnp.clip(occ[src_new], eps, 1.0 - eps)
            occ_tgt_new = jnp.clip(occ[tgt_new], eps, 1.0 - eps)
            w_new = jnp.where(sp0 != sp1, occ_tgt_new * (1.0 - occ_src_new), 0.0)
            new_contribution = jnp.sum(jnp.where(affected, w_new, 0.0))

            z_bwd = z_fwd - old_contribution + new_contribution

            log_prob_corr = (
                log_w_bwd - _safe_log(z_bwd, eps)
                - log_w_fwd + _safe_log(z_fwd, eps)
            )
            return sigma_proposed, log_prob_corr

        return _update_samples(keys, sigma, hoppable_clusters)

    def __repr__(self):
        return (
            "FermionHopRule_with_proposal("
            f"# of clusters: {len(self.clusters)}, "
            f"# of occupations: {len(self.occupations)})"
        )
