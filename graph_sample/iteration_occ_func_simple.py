"""VMC with periodic occupation updates for the proposal distribution."""

from __future__ import annotations

from itertools import product

import netket as nk
import numpy as np
import optax

from netket.logging import RuntimeLog
from netket.operator._fermion2nd.base import FermionOperator2ndBase

try:
    from .hamiltonian_graph import graph_from_hamiltonian
except ImportError:
    from hamiltonian_graph import graph_from_hamiltonian


def estimate_occupations_from_samples(
    samples,
    hilbert,
    *,
    spin_symmetric: bool = True,
):
    """Estimate mean occupations from sampled configurations."""
    samples = np.asarray(samples)
    # Merge chain/sample axes and keep only the mode axis.
    flat_samples = samples.reshape(-1, hilbert.size)

    if spin_symmetric and hilbert.n_spin_subsectors > 1:
        # Average over both samples and spin sectors -> one occupation per orbital.
        flat_samples = flat_samples.reshape(
            -1, hilbert.n_spin_subsectors, hilbert.n_orbitals
        )
        return flat_samples.mean(axis=(0, 1))

    return flat_samples.mean(axis=0)


def _mode_index(hi, spin_idx: int, orbital_idx: int) -> int:
    return spin_idx * hi.n_orbitals + orbital_idx


def _build_one_body_operator(hi, mode_i: int, mode_j: int):
    return nk.operator.FermionOperator2nd(
        hi,
        terms=[((int(mode_i), 1), (int(mode_j), 0))],
        weights=[1.0],
        dtype=np.complex128,
    )


def build_one_body_operator_cache(
    hi,
    *,
    spin_symmetric: bool = True,
):
    """
    Prebuild the ``c_i^dag c_j`` operators used to estimate the full 1-RDM.

    For spin-symmetric systems, this returns one orbital-space block per spin sector.
    Otherwise it returns a full mode-space cache.
    """
    if spin_symmetric and hi.n_spin_subsectors > 1:
        # Build one orbital-space block per spin sector.
        return [
            [[_build_one_body_operator(hi, _mode_index(hi, s, i), _mode_index(hi, s, j))
              for j in range(hi.n_orbitals)]
             for i in range(hi.n_orbitals)]
            for s in range(hi.n_spin_subsectors)
        ]

    # Build the full mode-space cache.
    return [
        [_build_one_body_operator(hi, i, j) for j in range(hi.size)]
        for i in range(hi.size)
    ]


def estimate_one_body_rdm(
    vstate,
    hi,
    *,
    spin_symmetric: bool = True,
    operator_cache=None,
):
    """
    Estimate the full one-body reduced density matrix.

    If ``spin_symmetric=True`` and the Hilbert space has multiple spin subsectors,
    the returned matrix is the orbital-space average over spin blocks. This matches
    the occupation convention used by ``MetropolisFermionHopWithProposal`` when one
    supplies one occupation per orbital.
    """
    if operator_cache is None:
        operator_cache = build_one_body_operator_cache(
            hi, spin_symmetric=spin_symmetric
        )

    if spin_symmetric and hi.n_spin_subsectors > 1:
        spin_blocks = []
        for block_ops in operator_cache:
            block = np.zeros((hi.n_orbitals, hi.n_orbitals), dtype=np.complex128)
            for i in range(hi.n_orbitals):
                for j in range(hi.n_orbitals):
                    block[i, j] = np.asarray(vstate.expect(block_ops[i][j]).mean)
            spin_blocks.append(block)

        # Average spin blocks and remove small non-Hermitian MC noise.
        rdm = sum(spin_blocks) / float(len(spin_blocks))
        rdm = 0.5 * (rdm + rdm.conj().T)
        return rdm

    rdm = np.zeros((hi.size, hi.size), dtype=np.complex128)
    for i in range(hi.size):
        for j in range(hi.size):
            rdm[i, j] = np.asarray(vstate.expect(operator_cache[i][j]).mean)
    return 0.5 * (rdm + rdm.conj().T)


def _canonicalize_eigenvectors(vecs: np.ndarray) -> np.ndarray:
    """
    Fix the arbitrary phase of each eigenvector for reproducible rotations.
    """
    vecs = np.array(vecs, copy=True)
    for col in range(vecs.shape[1]):
        column = vecs[:, col]
        # Use the largest-magnitude entry to set a stable global phase.
        pivot = int(np.argmax(np.abs(column)))
        amp = column[pivot]
        if np.abs(amp) > 0:
            vecs[:, col] *= np.exp(-1j * np.angle(amp))
    return vecs


def natural_orbitals_from_rdm(rdm: np.ndarray):
    """
    Diagonalize a Hermitian 1-RDM and return occupations and natural orbitals.
    """
    rdm = 0.5 * (np.asarray(rdm) + np.asarray(rdm).conj().T)
    occ, vecs = np.linalg.eigh(rdm)
    order = np.argsort(occ.real)[::-1]
    occ = np.asarray(occ[order].real)
    vecs = _canonicalize_eigenvectors(np.asarray(vecs[:, order]))
    return occ, vecs


def _expand_orbital_rotation_to_modes(hi, orbital_rotation: np.ndarray) -> np.ndarray:
    orbital_rotation = np.asarray(orbital_rotation, dtype=np.complex128)
    if orbital_rotation.shape != (hi.n_orbitals, hi.n_orbitals):
        raise ValueError(
            "orbital_rotation must have shape "
            f"({hi.n_orbitals}, {hi.n_orbitals}), got {orbital_rotation.shape}."
        )

    if hi.n_spin_subsectors == 1:
        return orbital_rotation

    # Apply the same orbital rotation independently to each spin sector.
    return np.kron(np.eye(hi.n_spin_subsectors, dtype=np.complex128), orbital_rotation)


def rotate_fermion_hamiltonian(
    H,
    mode_rotation: np.ndarray,
    *,
    cutoff: float | None = None,
):
    """
    Rotate a ``FermionOperator2nd`` Hamiltonian into a new one-particle basis.

    The matrix ``mode_rotation`` is interpreted as the column-stacked orbital
    coefficients of the new basis in the current basis, i.e. ``gamma_new =
    U^dag gamma U``.
    """
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "rotate_fermion_hamiltonian currently supports "
            f"FermionOperator2nd-compatible operators, got {type(H)}."
        )

    hi = H.hilbert
    mode_rotation = np.asarray(mode_rotation, dtype=np.complex128)
    if mode_rotation.shape != (hi.size, hi.size):
        raise ValueError(
            f"mode_rotation must have shape ({hi.size}, {hi.size}), "
            f"got {mode_rotation.shape}."
        )

    if cutoff is None:
        cutoff = H.cutoff

    op = H.to_normal_order()
    constant = 0.0 + 0.0j
    new_terms = []
    new_weights = []

    for term, weight in op.operators.items():
        if len(term) == 0:
            constant += weight
            continue

        choices = []
        for mode, dagger in term:
            # Rewrite each operator in the rotated basis and keep only nonzero entries.
            coeffs = mode_rotation[int(mode), :].conj() if dagger else mode_rotation[int(mode), :]
            nz = np.flatnonzero(np.abs(coeffs) > cutoff)
            if nz.size == 0:
                choices = []
                break
            choices.append([(int(idx), int(dagger), coeffs[idx]) for idx in nz])

        if not choices:
            continue

        # Enumerate all products generated by the basis rotation.
        for expanded in product(*choices):
            create_indices = [idx for idx, dagger, _ in expanded if dagger == 1]
            destroy_indices = [idx for idx, dagger, _ in expanded if dagger == 0]
            if len(set(create_indices)) != len(create_indices):
                continue
            if len(set(destroy_indices)) != len(destroy_indices):
                continue

            coeff = weight
            for _, _, value in expanded:
                coeff *= value

            if abs(coeff) <= cutoff:
                continue

            new_terms.append(tuple((idx, dagger) for idx, dagger, _ in expanded))
            new_weights.append(coeff)

    rotated = type(H)(
        hi,
        terms=new_terms,
        weights=new_weights,
        constant=constant,
        cutoff=cutoff,
        dtype=np.result_type(H.dtype, mode_rotation.dtype, np.complex128),
    )
    return rotated.reduce(order=True, inplace=True, cutoff=cutoff)


def get_optimizer_and_sr(steps: int = 300):
    """Create default optimizer (clipped Adam + warmup cosine decay) and SR preconditioner."""
    warmup_steps = max(1, int(steps * 0.05))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.005,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=5e-5,
    )
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    sr_shift = optax.linear_schedule(
        init_value=0.05, end_value=0.01,
        transition_steps=max(1, int(steps * 0.6)),
    )
    sr = nk.optimizer.SR(
        diag_shift=sr_shift, solver=nk.optimizer.solver.pinv,
        holomorphic=False, qgt=nk.optimizer.qgt.QGTJacobianDense,
    )
    return op, sr


def _build_schedule(N_ITER: int, occ_update_every: int, schedule):
    """Convert user-provided schedule or update interval into a list of segment lengths."""
    if schedule is not None:
        segs = [int(s) for s in schedule]
        if sum(segs) != N_ITER:
            raise ValueError(f"sum(schedule)={sum(segs)} != N_ITER={N_ITER}")
        return segs
    # Split the full run into equal-size update segments.
    segs, left = [], N_ITER
    while left > 0:
        s = min(occ_update_every, left)
        segs.append(s)
        left -= s
    return segs


def run_vmc_with_iteration_occ(
    *,
    hi,
    H,
    model,
    graph_sampler=None,
    occ_init=None,
    op=None,
    sr=None,
    N_ITER: int = 40,
    occ_update_every: int = 10,
    schedule: list[int] | None = None,
    n_samples: int = 256,
    n_discard_per_chain: int = 16,
    d_max: int = 1,
    spin_symmetric: bool = True,
    sampler_cls=nk.sampler.MetropolisFermionHopWithProposal,
    sampler_kwargs=None,
    estimate_fn=estimate_occupations_from_samples,
    out=None,
    obs=None,
    show_progress: bool = True,
    callback=lambda *a: True,
):
    """Run VMC and periodically reestimate occupations to update the proposal.

    Core loop:
        for each segment:
            1. driver.run(segment_steps)
            2. occ = estimate_fn(samples)
            3. rebuild sampler with new occ

    Each segment's occ is logged to ``out`` under key ``"occ_segment"`` with
    fields ``id``, ``step_start``, ``step_end``, ``occ``, so the occ used at
    any step can be recovered from the log afterwards.

    Returns a dict with driver, variational_state, log, energy_history,
    occ_history, final_occ.
    """
    # --- defaults ---
    if graph_sampler is None:
        graph_sampler = graph_from_hamiltonian(H)
    if occ_init is None:
        occ_init = np.full(hi.n_orbitals, 0.5)
    if op is None or sr is None:
        _op, _sr = get_optimizer_and_sr(N_ITER)
        op, sr = op or _op, sr or _sr
    if out is None:
        out = RuntimeLog()

    segment_lengths = _build_schedule(N_ITER, occ_update_every, schedule)
    sampler_kwargs = dict(sampler_kwargs or {})
    occ = np.asarray(occ_init, dtype=float)

    # --- build sampler, state, driver ---
    # The sampler proposal is parameterized by the current occupation estimate.
    sampler = sampler_cls(
        hi, graph=graph_sampler, occupations=occ,
        d_max=d_max, spin_symmetric=spin_symmetric, **sampler_kwargs,
    )
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model,
        n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
    )
    driver = nk.VMC(
        hamiltonian=H, optimizer=op, preconditioner=sr,
        variational_state=vstate,
    )

    # --- main loop ---
    occ_history = [occ.copy()]
    energy_history = []

    for seg_id, seg_steps in enumerate(segment_lengths):
        step_start = driver.step_count

        # Log which occ is used for this segment (recoverable per-step).
        if isinstance(out, RuntimeLog):
            out(step_start, {"occ_segment": {
                "id": seg_id,
                "step_start": step_start,
                "step_end": step_start + seg_steps,
                "occ": occ.copy(),
            }}, vstate)

        driver.run(
            n_iter=seg_steps, out=out, obs=obs,
            show_progress=show_progress, callback=callback,
        )

        # Refit the proposal from the samples generated by this segment.
        samples = np.asarray(vstate.samples)
        occ = np.asarray(
            estimate_fn(samples, hi, spin_symmetric=spin_symmetric), dtype=float,
        )
        occ_history.append(occ.copy())
        energy_history.append(float(np.real(driver.energy.mean)))

        # Rebuild sampler with updated occ for the next segment.
        if seg_id + 1 < len(segment_lengths):
            vstate.sampler = sampler_cls(
                hi, graph=graph_sampler, occupations=occ,
                d_max=d_max, spin_symmetric=spin_symmetric, **sampler_kwargs,
            )

    return {
        "driver": driver,
        "variational_state": vstate,
        "log": out,
        "occ_history": occ_history,
        "energy_history": energy_history,
        "final_occ": occ.copy(),
    }


def run_vmc_with_iteration_natural_orbitals(
    *,
    hi,
    H,
    model,
    graph_sampler=None,
    occ_init=None,
    op=None,
    sr=None,
    N_ITER: int = 40,
    orbital_update_every: int = 20,
    schedule: list[int] | None = None,
    n_samples: int = 256,
    n_discard_per_chain: int = 16,
    d_max: int = 1,
    spin_symmetric: bool = True,
    sampler_cls=nk.sampler.MetropolisFermionHopWithProposal,
    sampler_kwargs=None,
    out=None,
    obs=None,
    show_progress: bool = True,
    callback=lambda *a: True,
    hamiltonian_cutoff: float | None = None,
):
    """
    Run VMC and periodically update the proposal and Hamiltonian using natural orbitals.

    Compared with ``run_vmc_with_iteration_occ``, this routine estimates the full
    1-RDM after each segment, diagonalizes it to obtain natural occupations and
    natural orbitals, rotates the Hamiltonian into the new basis, and uses the
    natural occupations as the proposal target for the next segment.

    This update is much more expensive than the diagonal-occupation-only variant,
    so ``orbital_update_every`` should typically be chosen noticeably larger than
    the occupation-only update interval.
    """
    if not isinstance(H, FermionOperator2ndBase):
        raise TypeError(
            "run_vmc_with_iteration_natural_orbitals requires a "
            f"FermionOperator2nd-compatible Hamiltonian, got {type(H)}."
        )

    if graph_sampler is None:
        graph_sampler = graph_from_hamiltonian(H)
        update_graph_sampler = True
    else:
        update_graph_sampler = False

    if occ_init is None:
        occ_init = np.full(hi.n_orbitals if spin_symmetric else hi.size, 0.5)
    if op is None or sr is None:
        _op, _sr = get_optimizer_and_sr(N_ITER)
        op, sr = op or _op, sr or _sr
    if out is None:
        out = RuntimeLog()

    segment_lengths = _build_schedule(N_ITER, orbital_update_every, schedule)
    sampler_kwargs = dict(sampler_kwargs or {})
    occ = np.asarray(occ_init, dtype=float)
    current_H = H
    operator_cache = build_one_body_operator_cache(hi, spin_symmetric=spin_symmetric)
    # Tracks the cumulative basis change across all segments.
    orbital_rotation_total = np.eye(
        hi.n_orbitals if spin_symmetric else hi.size, dtype=np.complex128
    )

    sampler = sampler_cls(
        hi, graph=graph_sampler, occupations=occ,
        d_max=d_max, spin_symmetric=spin_symmetric, **sampler_kwargs,
    )
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model,
        n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
    )
    driver = nk.VMC(
        hamiltonian=current_H, optimizer=op, preconditioner=sr,
        variational_state=vstate,
    )

    occ_history = [occ.copy()]
    energy_history = []
    rdm_history = []
    natural_occupations_history = []
    natural_orbitals_history = []
    hamiltonian_history = [current_H]

    for seg_id, seg_steps in enumerate(segment_lengths):
        step_start = driver.step_count

        if isinstance(out, RuntimeLog):
            out(step_start, {"occ_segment": {
                "id": seg_id,
                "step_start": step_start,
                "step_end": step_start + seg_steps,
                "occ": occ.copy(),
            }}, vstate)

        driver.run(
            n_iter=seg_steps, out=out, obs=obs,
            show_progress=show_progress, callback=callback,
        )

        # Estimate the full 1-RDM, then diagonalize it into natural orbitals.
        rdm = estimate_one_body_rdm(
            vstate,
            hi,
            spin_symmetric=spin_symmetric,
            operator_cache=operator_cache,
        )
        natural_occ, natural_orbitals = natural_orbitals_from_rdm(rdm)
        occ = np.clip(natural_occ.real, 0.0, 1.0)

        mode_rotation = (
            _expand_orbital_rotation_to_modes(hi, natural_orbitals)
            if spin_symmetric else natural_orbitals
        )
        orbital_rotation_total = orbital_rotation_total @ natural_orbitals

        # Rotate the Hamiltonian into the new one-particle basis.
        current_H = rotate_fermion_hamiltonian(
            current_H,
            mode_rotation,
            cutoff=hamiltonian_cutoff,
        )
        driver._ham = current_H.collect()

        if update_graph_sampler:
            # Keep the proposal graph in sync with the rotated Hamiltonian.
            graph_sampler = graph_from_hamiltonian(current_H)

        rdm_history.append(rdm.copy())
        natural_occupations_history.append(natural_occ.copy())
        natural_orbitals_history.append(natural_orbitals.copy())
        occ_history.append(occ.copy())
        energy_history.append(float(np.real(driver.energy.mean)))
        hamiltonian_history.append(current_H)

        if isinstance(out, RuntimeLog):
            out(driver.step_count, {"natural_orbital_segment": {
                "id": seg_id,
                "step_start": step_start,
                "step_end": driver.step_count,
                "natural_occupations": natural_occ.copy(),
            }}, vstate)

        if seg_id + 1 < len(segment_lengths):
            # Use the natural occupations as the next proposal target.
            vstate.sampler = sampler_cls(
                hi, graph=graph_sampler, occupations=occ,
                d_max=d_max, spin_symmetric=spin_symmetric, **sampler_kwargs,
            )

    return {
        "driver": driver,
        "variational_state": vstate,
        "log": out,
        "occ_history": occ_history,
        "rdm_history": rdm_history,
        "energy_history": energy_history,
        "natural_occupations_history": natural_occupations_history,
        "natural_orbitals_history": natural_orbitals_history,
        "orbital_rotation_total": orbital_rotation_total,
        "hamiltonian_history": hamiltonian_history,
        "final_occ": occ.copy(),
        "final_rdm": rdm_history[-1].copy(),
        "final_natural_occupations": natural_occupations_history[-1].copy(),
        "final_natural_orbitals": natural_orbitals_history[-1].copy(),
        "final_H": current_H,
    }
