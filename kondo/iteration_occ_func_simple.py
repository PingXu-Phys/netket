"""VMC with periodic occupation updates for the proposal distribution."""

from __future__ import annotations

import netket as nk
import numpy as np
import optax

from hamiltonian_graph import graph_from_hamiltonian
from iteration_occ import estimate_occupations_from_samples
from netket.logging import RuntimeLog


def get_optimizer_and_sr(steps: int = 300):
    """Create default optimizer (clipped Adam + cosine decay) and SR preconditioner."""
    lr_schedule = optax.cosine_decay_schedule(
        init_value=0.01, decay_steps=steps, alpha=0.1,
    )
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    sr_shift = optax.linear_schedule(
        init_value=0.05, end_value=0.001,
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

        # Estimate new occupations from current samples.
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
