# Copyright 2021 The NetKet Authors - All rights reserved.
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

import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_REPO_ROOT, _REPO_ROOT / "Hamiltonian"):
    _extra_str = str(_extra)
    if _extra_str not in sys.path:
        sys.path.insert(0, _extra_str)

import netket as nk
from netket.sampler.metropolis import MetropolisHamiltonianWithProposal
from netket.sampler.rules.hamiltonian import HamiltonianRule
from netket.sampler.rules._hamiltonian_with_proposal_local_descriptor import (
    build_transition_signature,
    extract_local_descriptor_from_dense_targets,
    log_representative_probability_from_signature,
    log_state_probability_from_signature,
)
from netket.sampler.rules.hamiltonian_with_proposal import HamiltonianRuleWithProposal

from Hamiltonian.KondoHeisenbergChainSpinFermion import KondoHeisenbergChainSpinFermion

RULE_SEED = 123
STATE_SEED = 321
SAMPLER_SEED = 15324
N_CHAINS = 4
CHAIN_LENGTH = 5
N_STEPS = 6


class ConstantModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.zeros(x.shape[:-1], dtype=jnp.float32)


def _build_small_kondo_system():
    return KondoHeisenbergChainSpinFermion(
        Lx=2,
        t=1.0,
        J_K=1.0,
        J1=1.0,
        J2=0.0,
        delta_z=1.0,
        n_fermions=2,
    )


def _build_constant_model_and_variables(hilbert):
    model = ConstantModel()
    x = jnp.zeros((1, hilbert.size), dtype=jnp.int8)
    variables = model.init(jax.random.PRNGKey(0), x)
    return model, variables


def test_hamiltonian_with_proposal_rule_matches_hamiltonian_rule():
    system = _build_small_kondo_system()
    hilbert = system.joint_hilbert
    hamiltonian = system.hamiltonian

    states = hilbert.random_state(
        jax.random.PRNGKey(STATE_SEED), size=N_CHAINS, dtype=jnp.int8
    )
    reference_rule = HamiltonianRule(operator=hamiltonian)
    proposal_rule = HamiltonianRuleWithProposal(
        operator=hamiltonian,
        occupations=None,
        balance_beta=0.0,
    )

    reference_states = states
    proposal_states = states
    keys = jax.random.split(jax.random.PRNGKey(RULE_SEED), N_STEPS)

    for key in keys:
        reference_states, reference_corr = reference_rule.transition(
            None, None, None, None, key, reference_states
        )
        proposal_states, proposal_corr = proposal_rule.transition(
            None, None, None, None, key, proposal_states
        )

        np.testing.assert_array_equal(
            jax.device_get(proposal_states), jax.device_get(reference_states)
        )
        np.testing.assert_allclose(
            jax.device_get(proposal_corr), jax.device_get(reference_corr)
        )


def test_metropolis_hamiltonian_with_proposal_matches_baseline_sampler():
    system = _build_small_kondo_system()
    hilbert = system.joint_hilbert
    hamiltonian = system.hamiltonian
    model, variables = _build_constant_model_and_variables(hilbert)

    reference_sampler = nk.sampler.MetropolisHamiltonian(
        hilbert,
        hamiltonian=hamiltonian,
        n_chains=N_CHAINS,
        sweep_size=1,
        reset_chains=False,
    )
    proposal_sampler = MetropolisHamiltonianWithProposal(
        hilbert,
        hamiltonian=hamiltonian,
        occupations=None,
        balance_beta=0.0,
        n_chains=N_CHAINS,
        sweep_size=1,
        reset_chains=False,
    )

    reference_state = reference_sampler.init_state(model, variables, seed=SAMPLER_SEED)
    proposal_state = proposal_sampler.init_state(model, variables, seed=SAMPLER_SEED)

    reference_state = reference_sampler.reset(model, variables, state=reference_state)
    proposal_state = proposal_sampler.reset(model, variables, state=proposal_state)

    reference_samples, reference_state = reference_sampler.sample(
        model,
        variables,
        state=reference_state,
        chain_length=CHAIN_LENGTH,
    )
    proposal_samples, proposal_state = proposal_sampler.sample(
        model,
        variables,
        state=proposal_state,
        chain_length=CHAIN_LENGTH,
    )

    np.testing.assert_array_equal(
        jax.device_get(proposal_samples), jax.device_get(reference_samples)
    )
    np.testing.assert_array_equal(
        jax.device_get(getattr(proposal_state, "σ")),
        jax.device_get(getattr(reference_state, "σ")),
    )
    np.testing.assert_allclose(
        jax.device_get(proposal_state.log_prob),
        jax.device_get(reference_state.log_prob),
    )
    np.testing.assert_allclose(
        jax.device_get(proposal_state.n_steps),
        jax.device_get(reference_state.n_steps),
    )
    np.testing.assert_allclose(
        jax.device_get(proposal_state.n_accepted),
        jax.device_get(reference_state.n_accepted),
    )


def test_local_descriptor_extracts_kondo_move_data():
    x = jnp.array([1, 0, 0, 1, 1, -1], dtype=jnp.int8)
    xp = jnp.array(
        [
            [0, 1, 0, 1, 1, -1],
            [1, 0, 0, 1, -1, 1],
            [1, 1, 0, 0, -1, -1],
            [1, 0, 0, 1, 1, -1],
        ],
        dtype=jnp.int8,
    )
    valid = jnp.array([True, True, True, True])

    descriptor = extract_local_descriptor_from_dense_targets(
        x, xp, valid, fermion_size=4, spin_size=2
    )

    np.testing.assert_array_equal(
        jax.device_get(descriptor.move_type), np.array([1, 0, 2, -1])
    )
    np.testing.assert_array_equal(
        jax.device_get(descriptor.fermion_add_indices),
        np.array([[1, -1], [-1, -1], [1, -1], [-1, -1]]),
    )
    np.testing.assert_array_equal(
        jax.device_get(descriptor.fermion_remove_indices),
        np.array([[0, -1], [-1, -1], [3, -1], [-1, -1]]),
    )
    np.testing.assert_array_equal(
        jax.device_get(descriptor.spin_change_indices),
        np.array([[-1, -1], [0, 1], [0, -1], [-1, -1]]),
    )
    np.testing.assert_array_equal(
        jax.device_get(descriptor.duplicate_signature[2]),
        jax.device_get(build_transition_signature(x, xp[2], 4, 2)),
    )


def test_signature_based_duplicate_probability_matches_expected():
    x = jnp.array([1, 0, 0, 1, 1, -1], dtype=jnp.int8)
    xp = jnp.array(
        [
            [0, 1, 0, 1, 1, -1],
            [0, 1, 0, 1, 1, -1],
            [1, 0, 0, 1, -1, 1],
        ],
        dtype=jnp.int8,
    )
    valid = jnp.array([True, True, True])
    log_candidate_probs = jnp.log(jnp.array([0.2, 0.3, 0.5]))

    descriptor = extract_local_descriptor_from_dense_targets(
        x, xp, valid, fermion_size=4, spin_size=2
    )
    target_signature = build_transition_signature(x, xp[0], 4, 2)

    aggregated = log_state_probability_from_signature(
        descriptor.duplicate_signature, log_candidate_probs, target_signature, valid
    )
    representative = log_representative_probability_from_signature(
        descriptor.duplicate_signature, log_candidate_probs, target_signature, valid
    )

    np.testing.assert_allclose(
        jax.device_get(aggregated), jax.device_get(logsumexp(log_candidate_probs[:2]))
    )
    np.testing.assert_allclose(
        jax.device_get(representative), jax.device_get(jnp.max(log_candidate_probs[:2]))
    )


def test_hamiltonian_with_proposal_rule_runs_descriptor_path():
    system = _build_small_kondo_system()
    hilbert = system.joint_hilbert
    hamiltonian = system.hamiltonian

    states = hilbert.random_state(
        jax.random.PRNGKey(STATE_SEED), size=N_CHAINS, dtype=jnp.int8
    )
    proposal_rule = HamiltonianRuleWithProposal(
        operator=hamiltonian,
        occupations=np.array([0.35, 0.65]),
        balance_beta=0.5,
        aggregate_duplicate_entries=True,
    )

    proposed_states, log_prob_corr = proposal_rule.transition(
        None, None, None, None, jax.random.PRNGKey(RULE_SEED), states
    )

    assert proposed_states.shape == states.shape
    assert log_prob_corr.shape == (states.shape[0],)
    assert np.isfinite(jax.device_get(log_prob_corr)).all()
