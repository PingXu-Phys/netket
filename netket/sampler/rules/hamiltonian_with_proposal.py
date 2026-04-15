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
from jax.scipy.special import logsumexp

from netket.hilbert import SpinOrbitalFermions, TensorHilbert
from netket.operator import DiscreteJaxOperator
from netket.utils import struct

from .hamiltonian import HamiltonianRuleBase


def _infer_kondo_layout(hilbert):
    """Validate the mixed Hilbert layout and return block sizes.

    The current implementation is intentionally narrow: it expects a tensor
    product `(SpinOrbitalFermions) x (local spins)` with two fermion spin
    subsectors and one local-spin degree of freedom per orbital.
    """
    if not isinstance(hilbert, TensorHilbert):
        raise ValueError(
            "HamiltonianRuleWithProposal expects a TensorHilbert with "
            "a SpinOrbitalFermions block followed by a local-spin block."
        )

    subspaces = hilbert.subspaces
    if len(subspaces) != 2:
        raise ValueError(
            "HamiltonianRuleWithProposal currently expects exactly two subspaces: "
            "(SpinOrbitalFermions, local spins)."
        )

    fermion_hi = subspaces[0]
    spin_hi = subspaces[1]

    if not isinstance(fermion_hi, SpinOrbitalFermions):
        raise ValueError(
            "The first subspace must be SpinOrbitalFermions for "
            "HamiltonianRuleWithProposal."
        )
    if fermion_hi.n_spin_subsectors != 2:
        raise ValueError(
            "HamiltonianRuleWithProposal currently expects spin-1/2 fermions "
            "with exactly two spin subsectors."
        )
    if spin_hi.size != fermion_hi.n_orbitals:
        raise ValueError(
            "The local-spin block size must equal the number of fermion orbitals. "
            f"Got spin_hi.size={spin_hi.size} and n_orbitals={fermion_hi.n_orbitals}."
        )

    return fermion_hi, fermion_hi.size, spin_hi.size


def _build_mode_occupations_or_none(
    fermion_hi: SpinOrbitalFermions, occupations: np.ndarray | list[float] | None
):
    """Validate user occupations and expand orbital-level inputs to mode layout.

    Accepted input shapes are:
    - ``None``: neutral fermion proposal branch.
    - ``L`` entries: one occupation per orbital, replicated across both spin blocks.
    - ``2L`` entries: one occupation per fermion mode in the internal order used
      by ``SpinOrbitalFermions``. For the present Kondo layout this means
      ``[dn block | up block]``.
    """
    if occupations is None:
        return None

    occ = np.asarray(occupations, dtype=float)
    if occ.ndim != 1:
        raise ValueError(
            "occupations must be a 1D array with one entry per orbital or mode."
        )
    if occ.size == 0:
        raise ValueError("occupations must not be empty.")
    if not np.all(np.isfinite(occ)):
        raise ValueError("occupations must contain only finite values.")

    if occ.shape[0] == fermion_hi.size:
        mode_occ = occ
    elif occ.shape[0] == fermion_hi.n_orbitals:
        mode_occ = np.tile(occ, fermion_hi.n_spin_subsectors)
    else:
        raise ValueError(
            "occupations must have length equal to the number of orbitals "
            f"({fermion_hi.n_orbitals}) or the full number of spin-modes "
            f"({fermion_hi.size}), got {occ.shape[0]}."
        )

    if np.any(mode_occ < 0.0) or np.any(mode_occ > 1.0):
        raise ValueError("occupations entries must lie in the interval [0, 1].")

    return jnp.asarray(mode_occ)


def _sample_noisy_occupations(key, occupations, noise_strength, mixing, eps):
    """Apply the same mixing + Beta-resampling scheme used in with-proposal."""
    occ_safe = (1.0 - mixing) * occupations + mixing * 0.5
    occ_safe = jnp.clip(occ_safe, eps, 1.0 - eps)
    alpha = noise_strength * occ_safe
    beta_param = noise_strength * (1.0 - occ_safe)
    return jax.random.beta(key, alpha, beta_param)


def _classify_move_types(x, xp, valid, fermion_size):
    """Split Hamiltonian-connected states into ss / ff / sf move classes."""
    fermion_changed = jnp.any(xp[:, :fermion_size] != x[None, :fermion_size], axis=-1)
    spin_changed = jnp.any(xp[:, fermion_size:] != x[None, fermion_size:], axis=-1)

    mask_ss = valid & (~fermion_changed) & spin_changed
    mask_ff = valid & fermion_changed & (~spin_changed)
    mask_sf = valid & fermion_changed & spin_changed

    group_idx = jnp.where(mask_ss, 0, jnp.where(mask_ff, 1, jnp.where(mask_sf, 2, -1)))
    return mask_ss, mask_ff, mask_sf, group_idx


def _compute_candidate_log_bias(x, xp, valid, occ, fermion_size, eps):
    """Compute the fermion-only proposal bias for every connected candidate.

    The bias only tracks which fermion modes change occupancy. Therefore it
    automatically covers same-spin hops, on-site spin flips, and basis-rotated
    one-body processes `c_a^dagger c_b`.
    """
    if occ is None:
        return jnp.where(valid, 0.0, -jnp.inf)

    x_f = x[None, :fermion_size]
    xp_f = xp[:, :fermion_size]

    add_mask = xp_f > x_f
    remove_mask = x_f > xp_f

    occ = jnp.clip(occ, eps, 1.0 - eps)
    log_occ = jnp.log(occ)
    log_one_minus_occ = jnp.log1p(-occ)

    log_bias = jnp.sum(jnp.where(add_mask, log_occ[None, :], 0.0), axis=-1)
    log_bias += jnp.sum(
        jnp.where(remove_mask, log_one_minus_occ[None, :], 0.0), axis=-1
    )
    return jnp.where(valid, log_bias, -jnp.inf)


def _compute_log_candidate_probabilities(
    log_bias, mask_ss, mask_ff, mask_sf, balance_beta
):
    """Build the two-stage proposal in log-space.

    Stage 1 assigns proposal mass to the three non-empty move classes.
    Stage 2 normalizes candidates inside the selected class.

    With ``balance_beta = 0`` this reduces to direct candidate-wise weighting.
    With ``balance_beta = 1`` the non-empty classes are chosen approximately
    uniformly before sampling inside the class.
    """
    log_s_ss = logsumexp(jnp.where(mask_ss, log_bias, -jnp.inf))
    log_s_ff = logsumexp(jnp.where(mask_ff, log_bias, -jnp.inf))
    log_s_sf = logsumexp(jnp.where(mask_sf, log_bias, -jnp.inf))

    log_s = jnp.stack([log_s_ss, log_s_ff, log_s_sf])
    nonempty = jnp.stack([jnp.any(mask_ss), jnp.any(mask_ff), jnp.any(mask_sf)])
    log_group_weights = jnp.where(nonempty, (1.0 - balance_beta) * log_s, -jnp.inf)
    log_norm = logsumexp(log_group_weights)

    group_idx = jnp.where(mask_ss, 0, jnp.where(mask_ff, 1, jnp.where(mask_sf, 2, -1)))
    group_idx_clipped = jnp.clip(group_idx, 0, 2)
    log_s_candidate = log_s[group_idx_clipped]

    valid = group_idx >= 0
    log_probs = log_bias - balance_beta * log_s_candidate - log_norm
    return jnp.where(valid, log_probs, -jnp.inf)


def _log_state_probability(candidates, log_candidate_probs, target, valid):
    """Return log q(target) while summing duplicate connected states if needed."""
    same_state = jnp.all(candidates == target[None, :], axis=-1)
    return logsumexp(jnp.where(valid & same_state, log_candidate_probs, -jnp.inf))


def _log_representative_entry_probability(candidates, log_candidate_probs, target, valid):
    """Return one matching entry probability without duplicate-state aggregation.

    This matches the original HamiltonianRule philosophy more closely: connected
    entries are treated as the primitive proposal objects, and repeated target
    states are not explicitly aggregated at the acceptance-correction level.
    For the current proposal construction, repeated entries leading to the same
    target carry identical per-entry log-probabilities, so selecting the maximum
    matching log-probability is equivalent to selecting any one representative
    entry.
    """
    same_state = jnp.all(candidates == target[None, :], axis=-1)
    return jnp.max(jnp.where(valid & same_state, log_candidate_probs, -jnp.inf))


def _compute_chain_log_probs(x, xp, mels, occ, fermion_size, balance_beta, eps):
    valid = jnp.abs(mels) > 0
    mask_ss, mask_ff, mask_sf, group_idx = _classify_move_types(
        x, xp, valid, fermion_size
    )
    log_bias = _compute_candidate_log_bias(x, xp, valid, occ, fermion_size, eps)
    log_probs = _compute_log_candidate_probabilities(
        log_bias, mask_ss, mask_ff, mask_sf, balance_beta
    )
    return log_probs, group_idx >= 0


def _compute_chain_log_probs_neutral(x, xp, mels, fermion_size, balance_beta):
    valid = jnp.abs(mels) > 0
    mask_ss, mask_ff, mask_sf, group_idx = _classify_move_types(
        x, xp, valid, fermion_size
    )
    log_bias = jnp.where(valid, 0.0, -jnp.inf)
    log_probs = _compute_log_candidate_probabilities(
        log_bias, mask_ss, mask_ff, mask_sf, balance_beta
    )
    return log_probs, group_idx >= 0


@struct.dataclass
class HamiltonianRuleWithProposalJax(HamiltonianRuleBase):
    r"""
    Hamiltonian-connected proposal with fermion occupation bias and
    move-type balancing between spin-spin, fermion-fermion, and joint moves.
    """

    operator: DiscreteJaxOperator = struct.field(pytree_node=True)
    occupations: jax.Array | None = struct.field(pytree_node=True)
    fermion_size: int = struct.field(pytree_node=False)
    spin_size: int = struct.field(pytree_node=False)
    balance_beta: float = struct.field(pytree_node=False)
    noise_strength: float = struct.field(pytree_node=False)
    mixing: float = struct.field(pytree_node=False)
    aggregate_duplicate_entries: bool = struct.field(pytree_node=False)
    eps: float = struct.field(pytree_node=False, default=1.0e-12)

    def __init__(
        self,
        operator: DiscreteJaxOperator,
        *,
        occupations: np.ndarray | list[float] | None = None,
        balance_beta: float = 0.5,
        noise_strength: float = 100.0,
        mixing: float = 0.05,
        aggregate_duplicate_entries: bool = False,
        eps: float = 1.0e-12,
    ):
        if not isinstance(operator, DiscreteJaxOperator):
            raise TypeError(
                "HamiltonianRuleWithProposal currently requires a DiscreteJaxOperator, "
                f"but got {type(operator)}."
            )
        if balance_beta < 0.0 or balance_beta > 1.0:
            raise ValueError("balance_beta must lie in the interval [0, 1].")
        if noise_strength <= 0.0:
            raise ValueError("noise_strength must be positive.")
        if mixing < 0.0 or mixing >= 1.0:
            raise ValueError("mixing must lie in the interval [0, 1).")
        if not isinstance(aggregate_duplicate_entries, (bool, np.bool_)):
            raise TypeError("aggregate_duplicate_entries must be a boolean flag.")

        fermion_hi, fermion_size, spin_size = _infer_kondo_layout(operator.hilbert)

        self.operator = operator
        self.occupations = _build_mode_occupations_or_none(fermion_hi, occupations)
        self.fermion_size = fermion_size
        self.spin_size = spin_size
        self.balance_beta = balance_beta
        self.noise_strength = noise_strength
        self.mixing = mixing
        self.aggregate_duplicate_entries = bool(aggregate_duplicate_entries)
        self.eps = eps

    def transition(self, _0, _1, _2, _3, key, x):
        # Neutral-limit equivalence: with occupations=None and balance_beta=0,
        # the general weighted path would assign identical logits to every valid
        # connected state and -inf to padded entries. categorical(log_probs)
        # therefore samples uniformly on the same support as HamiltonianRule.
        # We still special-case this branch to reuse the original randint-based
        # implementation exactly, which is cheaper and reproduces the same
        # seed-by-seed trajectory as the baseline sampler.
        if (
            self.occupations is None
            and self.balance_beta == 0.0
            and not self.aggregate_duplicate_entries
        ):
            xp, mels = self.operator.get_conn_padded(x)
            n_conn = self.operator.n_conn(x)
            rand_i = jax.random.randint(
                key, shape=(x.shape[0],), minval=0, maxval=n_conn
            )

            nonzeros = jnp.abs(mels) > 0
            nonzero_i_plus1 = (jnp.cumsum(nonzeros, axis=-1)) * nonzeros
            rand_i_mask = nonzero_i_plus1 == jnp.expand_dims(rand_i + 1, -1)
            x_proposed = (xp * jnp.expand_dims(rand_i_mask, -1)).sum(
                axis=1, dtype=xp.dtype
            )
            n_conn_proposed = self.operator.n_conn(x_proposed)
            log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
            return x_proposed.astype(x.dtype), log_prob_corr

        # Enumerate the exact Hamiltonian-connected support, just like the base
        # HamiltonianRule. All proposal biasing happens only inside this support.
        xp, mels = self.operator.get_conn_padded(x)
        n_chains = x.shape[0]
        batch_i = jnp.arange(n_chains)

        key_occ, key_select = jax.random.split(key)
        select_keys = jax.random.split(key_select, n_chains)

        if self.occupations is None:
            log_probs_fwd, valid_fwd = jax.vmap(
                _compute_chain_log_probs_neutral, in_axes=(0, 0, 0, None, None)
            )(
                x,
                xp,
                mels,
                self.fermion_size,
                self.balance_beta,
            )
            occ = None
        else:
            # Sample one noisy occupation profile per chain and reuse it for both
            # the forward and backward proposal probabilities.
            occ_keys = jax.random.split(key_occ, n_chains)
            occupations = jnp.broadcast_to(
                self.occupations, (n_chains, self.fermion_size)
            )
            occ = jax.vmap(
                _sample_noisy_occupations, in_axes=(0, 0, None, None, None)
            )(
                occ_keys,
                occupations,
                self.noise_strength,
                self.mixing,
                self.eps,
            )
            log_probs_fwd, valid_fwd = jax.vmap(
                _compute_chain_log_probs, in_axes=(0, 0, 0, 0, None, None, None)
            )(
                x,
                xp,
                mels,
                occ,
                self.fermion_size,
                self.balance_beta,
                self.eps,
            )

        selected = jax.vmap(jax.random.categorical)(select_keys, log_probs_fwd)
        x_proposed = xp[batch_i, selected]
        if self.aggregate_duplicate_entries:
            log_q_fwd = jax.vmap(_log_state_probability)(
                xp, log_probs_fwd, x_proposed, valid_fwd
            )
        else:
            log_q_fwd = jnp.take_along_axis(
                log_probs_fwd, selected[:, None], axis=1
            )[:, 0]

        # Recompute the reverse proposal on the proposed states using the same
        # noisy occupations. This is the expensive part compared with the base
        # HamiltonianRule, but it is required for the non-uniform proposal.
        xp_bwd, mels_bwd = self.operator.get_conn_padded(x_proposed)
        if occ is None:
            log_probs_bwd, valid_bwd = jax.vmap(
                _compute_chain_log_probs_neutral, in_axes=(0, 0, 0, None, None)
            )(
                x_proposed,
                xp_bwd,
                mels_bwd,
                self.fermion_size,
                self.balance_beta,
            )
        else:
            log_probs_bwd, valid_bwd = jax.vmap(
                _compute_chain_log_probs, in_axes=(0, 0, 0, 0, None, None, None)
            )(
                x_proposed,
                xp_bwd,
                mels_bwd,
                occ,
                self.fermion_size,
                self.balance_beta,
                self.eps,
            )

        if self.aggregate_duplicate_entries:
            log_q_bwd = jax.vmap(_log_state_probability)(
                xp_bwd, log_probs_bwd, x, valid_bwd
            )
        else:
            log_q_bwd = jax.vmap(_log_representative_entry_probability)(
                xp_bwd, log_probs_bwd, x, valid_bwd
            )
        return x_proposed.astype(x.dtype), log_q_bwd - log_q_fwd

    def __repr__(self):
        n_occ = None if self.occupations is None else len(self.occupations)
        return (
            "HamiltonianRuleWithProposalJax("
            f"operator={self.operator}, "
            f"# of occupations={n_occ}, "
            f"balance_beta={self.balance_beta}, "
            f"noise_strength={self.noise_strength}, "
            f"mixing={self.mixing}, "
            f"aggregate_duplicate_entries={self.aggregate_duplicate_entries})"
        )


def HamiltonianRuleWithProposal(
    operator,
    *,
    occupations=None,
    balance_beta=0.5,
    noise_strength=100.0,
    mixing=0.05,
    aggregate_duplicate_entries=False,
):
    r"""
    Rule proposing Hamiltonian-connected moves with fermion occupation bias and
    move-type balancing.
    """
    if isinstance(operator, DiscreteJaxOperator):
        return HamiltonianRuleWithProposalJax(
            operator,
            occupations=occupations,
            balance_beta=balance_beta,
            noise_strength=noise_strength,
            mixing=mixing,
            aggregate_duplicate_entries=aggregate_duplicate_entries,
        )
    raise TypeError(
        "HamiltonianRuleWithProposal currently supports only DiscreteJaxOperator instances."
    )
