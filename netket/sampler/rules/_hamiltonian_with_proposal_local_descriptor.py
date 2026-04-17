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
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from netket.utils import struct

MOVE_TYPE_INVALID = -1
MOVE_TYPE_SS = 0
MOVE_TYPE_FF = 1
MOVE_TYPE_SF = 2
MAX_CHANGED_INDICES = 4
MAX_FERMION_CHANGES = 2
MAX_SPIN_CHANGES = 2


@struct.dataclass
class LocalMoveDescriptor:
    move_type: jax.Array
    changed_count: jax.Array
    changed_indices: jax.Array
    fermion_add_count: jax.Array
    fermion_add_indices: jax.Array
    fermion_remove_count: jax.Array
    fermion_remove_indices: jax.Array
    spin_change_count: jax.Array
    spin_change_indices: jax.Array
    duplicate_signature: jax.Array


def _gather_sorted_padded_indices(mask, width):
    size = mask.shape[-1]
    indices = jnp.broadcast_to(jnp.arange(size, dtype=jnp.int32), mask.shape)
    fill_value = jnp.full(mask.shape, size, dtype=jnp.int32)
    sorted_indices = jnp.sort(jnp.where(mask, indices, fill_value), axis=-1)
    padded = sorted_indices[:, :width]
    return jnp.where(padded < size, padded, -jnp.ones_like(padded))


def extract_local_descriptor_from_dense_targets(x, xp, valid, fermion_size, spin_size):
    del spin_size

    valid_mask = valid[:, None]
    diff_mask = valid_mask & (xp != x[None, :])
    fermion_diff = diff_mask[:, :fermion_size]
    spin_diff = diff_mask[:, fermion_size:]

    fermion_add_mask = valid_mask & (xp[:, :fermion_size] > x[None, :fermion_size])
    fermion_remove_mask = valid_mask & (x[None, :fermion_size] > xp[:, :fermion_size])

    fermion_changed = jnp.any(fermion_diff, axis=-1)
    spin_changed = jnp.any(spin_diff, axis=-1)

    move_type = jnp.full(valid.shape, MOVE_TYPE_INVALID, dtype=jnp.int32)
    move_type = jnp.where((~fermion_changed) & spin_changed, MOVE_TYPE_SS, move_type)
    move_type = jnp.where(fermion_changed & (~spin_changed), MOVE_TYPE_FF, move_type)
    move_type = jnp.where(fermion_changed & spin_changed, MOVE_TYPE_SF, move_type)

    changed_count = jnp.sum(diff_mask, axis=-1, dtype=jnp.int32)
    changed_indices = _gather_sorted_padded_indices(diff_mask, MAX_CHANGED_INDICES)
    fermion_add_count = jnp.sum(fermion_add_mask, axis=-1, dtype=jnp.int32)
    fermion_add_indices = _gather_sorted_padded_indices(
        fermion_add_mask, MAX_FERMION_CHANGES
    )
    fermion_remove_count = jnp.sum(fermion_remove_mask, axis=-1, dtype=jnp.int32)
    fermion_remove_indices = _gather_sorted_padded_indices(
        fermion_remove_mask, MAX_FERMION_CHANGES
    )
    spin_change_count = jnp.sum(spin_diff, axis=-1, dtype=jnp.int32)
    spin_change_indices = _gather_sorted_padded_indices(spin_diff, MAX_SPIN_CHANGES)

    duplicate_signature = jnp.concatenate(
        [move_type[:, None], changed_count[:, None], changed_indices], axis=-1
    )
    invalid_signature = -jnp.ones_like(duplicate_signature)
    duplicate_signature = jnp.where(
        (move_type >= 0)[:, None], duplicate_signature, invalid_signature
    )

    return LocalMoveDescriptor(
        move_type=move_type,
        changed_count=changed_count,
        changed_indices=changed_indices,
        fermion_add_count=fermion_add_count,
        fermion_add_indices=fermion_add_indices,
        fermion_remove_count=fermion_remove_count,
        fermion_remove_indices=fermion_remove_indices,
        spin_change_count=spin_change_count,
        spin_change_indices=spin_change_indices,
        duplicate_signature=duplicate_signature,
    )


def build_transition_signature(source, target, fermion_size, spin_size):
    descriptor = extract_local_descriptor_from_dense_targets(
        source,
        target[None, :],
        jnp.ones((1,), dtype=jnp.bool_),
        fermion_size,
        spin_size,
    )
    return descriptor.duplicate_signature[0]


def compute_move_masks_from_descriptor(descriptor):
    mask_ss = descriptor.move_type == MOVE_TYPE_SS
    mask_ff = descriptor.move_type == MOVE_TYPE_FF
    mask_sf = descriptor.move_type == MOVE_TYPE_SF
    return mask_ss, mask_ff, mask_sf


def _sum_log_values_at_indices(values, indices):
    valid_idx = indices >= 0
    safe_idx = jnp.where(valid_idx, indices, 0)
    gathered = values[safe_idx]
    return jnp.sum(jnp.where(valid_idx, gathered, 0.0), axis=-1)


def compute_log_bias_from_descriptor(descriptor, occ, eps):
    valid = descriptor.move_type >= 0
    if occ is None:
        return jnp.where(valid, 0.0, -jnp.inf)

    occ = jnp.clip(occ, eps, 1.0 - eps)
    log_occ = jnp.log(occ)
    log_one_minus_occ = jnp.log1p(-occ)
    log_bias = _sum_log_values_at_indices(log_occ, descriptor.fermion_add_indices)
    log_bias = log_bias + _sum_log_values_at_indices(
        log_one_minus_occ, descriptor.fermion_remove_indices
    )
    return jnp.where(valid, log_bias, -jnp.inf)


def log_state_probability_from_signature(
    signatures, log_candidate_probs, target_signature, valid
):
    same_signature = jnp.all(signatures == target_signature[None, :], axis=-1)
    return logsumexp(
        jnp.where(valid & same_signature, log_candidate_probs, -jnp.inf)
    )


def log_representative_probability_from_signature(
    signatures, log_candidate_probs, target_signature, valid
):
    same_signature = jnp.all(signatures == target_signature[None, :], axis=-1)
    return jnp.max(jnp.where(valid & same_signature, log_candidate_probs, -jnp.inf))
