"""
Notebook-style helper function for VMC with periodic occupation updates.
"""

from __future__ import annotations

from collections.abc import Iterable

import netket as nk
import numpy as np
import optax

from hamiltonian_graph import graph_from_hamiltonian
from iteration_occ import estimate_occupations_from_samples
from netket.logging import AbstractLog, RuntimeLog


def get_optimizer_and_sr(steps: int = 300):
    """
    Notebook-style helper.

    This follows the naming style used in the notebook:
    - `op` is the optimizer
    - `sr` is the stochastic reconfiguration preconditioner
    """
    lr_schedule = optax.cosine_decay_schedule(
        init_value=0.01,
        decay_steps=steps,
        alpha=0.1,
    )
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )

    sr_shift = optax.linear_schedule(
        init_value=0.05,
        end_value=0.001,
        transition_steps=max(1, int(steps * 0.6)),
    )
    sr = nk.optimizer.SR(
        diag_shift=sr_shift,
        solver=nk.optimizer.solver.pinv,
        holomorphic=False,
        qgt=nk.optimizer.qgt.QGTJacobianDense,
    )
    return op, sr


def _manual_loggers(out) -> tuple[AbstractLog, ...]:
    """
    Convert `out` to a tuple of logger objects that can be called manually.

    Strings are ignored here because they are only meaningful inside
    `driver.run(...)`, where NetKet creates the corresponding file logger.
    """
    if out is None:
        return ()

    if isinstance(out, AbstractLog):
        return (out,)

    if isinstance(out, str):
        return ()

    if isinstance(out, Iterable):
        return tuple(logger for logger in out if isinstance(logger, AbstractLog))

    return ()


def _primary_runtime_log(out, manual_loggers: tuple[AbstractLog, ...]) -> RuntimeLog | None:
    """
    Return the RuntimeLog that should be inspected by the caller.

    If `out` itself is a RuntimeLog, prefer it so that all information ends up in
    the same logger object used by `driver.run(...)`.
    """
    if isinstance(out, RuntimeLog):
        return out

    for logger in manual_loggers:
        if isinstance(logger, RuntimeLog):
            return logger

    return None


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
    occ_update_every: int | None = 10,
    schedule: list[int] | tuple[int, ...] | None = None,
    n_samples: int = 256,
    n_discard_per_chain: int = 16,
    d_max: int = 1,
    spin_symmetric: bool = True,
    sampler_cls=nk.sampler.MetropolisFermionHopWithProposal,
    mc_state_cls=nk.vqs.MCState,
    sampler_kwargs=None,
    mc_state_kwargs=None,
    estimate_fn=estimate_occupations_from_samples,
    out=None,
    obs=None,
    step_size: int = 1,
    show_progress: bool = True,
    callback=lambda *args: True,
    timeit: bool = False,
):
    """
    Run VMC continuously and periodically recompute occupations.

    Main workflow:
    - build one MCState;
    - build one VMC driver;
    - run several segments continuously;
    - after each segment, estimate occupations from samples;
    - rebuild only the sampler, then continue training.

    Parameters
    ----------
    hi:
        Hilbert space. Usually this is the same `hi` already used to build `H`
        and the model in the notebook.

    H:
        Hamiltonian used by `nk.VMC`.

    model:
        Variational ansatz, for example `Slater2nd`, `DenseAGPJastrow`,
        `SafeAGPJastrow`, `RBM`, and so on.

    graph_sampler:
        Graph passed to the hopping sampler. If not provided, it is built from
        `H` with `graph_from_hamiltonian(H)`.

    occ_init:
        Initial occupations used by `MetropolisFermionHopWithProposal`.
        If not provided, a flat initial guess `0.5` is used on every orbital.

    op:
        Optimizer. If not provided, a default clipped Adam optimizer is created.

    sr:
        Stochastic reconfiguration preconditioner. If not provided, a default
        `nk.optimizer.SR(...)` is created.

    N_ITER:
        Total number of optimization steps across the whole run.

    occ_update_every:
        Recompute occupations every how many optimization steps.
        Used only when `schedule` is not provided.

    schedule:
        Explicit segment lengths. Example: `[50, 50, 100]`.
        If this is provided, it overrides `occ_update_every`.

    n_samples:
        Number of Monte Carlo samples used inside `MCState`.

    n_discard_per_chain:
        Number of discarded samples per chain before measurements.

    d_max:
        Maximum graph distance used by the fermion-hop sampler.

    spin_symmetric:
        Whether the sampler uses an orbital-level spin-symmetric graph.

    sampler_cls:
        Sampler constructor. Default is
        `nk.sampler.MetropolisFermionHopWithProposal`.

    mc_state_cls:
        MCState constructor. Default is `nk.vqs.MCState`.

    sampler_kwargs:
        Extra keyword arguments forwarded to `sampler_cls(...)`.

    mc_state_kwargs:
        Extra keyword arguments forwarded to `mc_state_cls(...)`.

    estimate_fn:
        Function used to estimate occupations from samples.

    out:
        Logger passed to `driver.run(...)`. If omitted, `RuntimeLog()` is used.

    obs:
        Optional observables forwarded to `driver.run(...)`.

    step_size:
        Logging step size passed to `driver.run(...)`.

    show_progress:
        Whether to display the NetKet progress bar.

    callback:
        Callback passed to `driver.run(...)`.

    timeit:
        Whether NetKet timing information should be enabled.

    Returns
    -------
    dict:
        A dictionary containing:
        - `occupations_history`: the occupations estimated after each segment
        - `occ_used_history`: the occupations actually used for each segment
        - `occ_used_by_step`: the occupations expanded to every optimization step
        - `energy_history`: the last energy statistic already available from the
          driver after each segment
        - `history_log`: the RuntimeLog containing the outer-loop occupation data.
          When `out` is a RuntimeLog, this is the same object.
        - plus the driver, logger, and final occupations
    """
    if graph_sampler is None:
        graph_sampler = graph_from_hamiltonian(H)

    if occ_init is None:
        occ_init = np.full(hi.n_orbitals, 0.5)

    if schedule is not None:
        segment_lengths = [int(step) for step in schedule]
        if len(segment_lengths) == 0:
            raise ValueError("schedule must not be empty.")
        if any(step <= 0 for step in segment_lengths):
            raise ValueError(
                f"all schedule entries must be positive, got {segment_lengths}."
            )
        if sum(segment_lengths) != N_ITER:
            raise ValueError(
                "sum(schedule) must equal N_ITER, "
                f"got sum(schedule)={sum(segment_lengths)} and N_ITER={N_ITER}."
            )
    else:
        if occ_update_every is None or occ_update_every <= 0:
            raise ValueError(
                "occ_update_every must be a positive integer when schedule is not provided."
            )

        segment_lengths = []
        steps_left = int(N_ITER)
        while steps_left > 0:
            step = min(int(occ_update_every), steps_left)
            segment_lengths.append(step)
            steps_left -= step

    if op is None or sr is None:
        default_op, default_sr = get_optimizer_and_sr(N_ITER)
        if op is None:
            op = default_op
        if sr is None:
            sr = default_sr

    if out is None:
        out = RuntimeLog()

    sampler_kwargs = dict(sampler_kwargs or {})
    mc_state_kwargs = dict(mc_state_kwargs or {})
    manual_loggers = _manual_loggers(out)
    history_log = _primary_runtime_log(out, manual_loggers)
    if history_log is None:
        history_log = RuntimeLog()

    occ = np.asarray(occ_init, dtype=float)

    sampler = sampler_cls(
        hi,
        graph=graph_sampler,
        occupations=occ,
        d_max=d_max,
        spin_symmetric=spin_symmetric,
        **sampler_kwargs,
    )

    vstate = mc_state_cls(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        **mc_state_kwargs,
    )

    driver = nk.VMC(
        hamiltonian=H,
        optimizer=op,
        preconditioner=sr,
        variational_state=vstate,
    )

    occupations_history = [occ.copy()]
    occ_used_history = []
    occ_used_by_step = []
    energy_history = []
    step_history = []

    for segment_id, segment_steps in enumerate(segment_lengths):
        step_start = driver.step_count
        occ_used_history.append(occ.copy())
        occ_used_by_step.extend([occ.copy() for _ in range(segment_steps)])

        per_step_item = {
            "iteration_occ": {
                "segment_id": int(segment_id),
                "segment_steps": int(segment_steps),
                "occ_used": occ.copy(),
            }
        }
        for step in range(step_start, step_start + segment_steps):
            history_log(step, per_step_item, vstate)
            for logger in manual_loggers:
                if logger is not history_log:
                    logger(step, per_step_item, vstate)

        driver.run(
            n_iter=segment_steps,
            out=out,
            obs=obs,
            step_size=step_size,
            show_progress=show_progress,
            callback=callback,
            timeit=timeit,
        )

        samples = np.asarray(vstate.samples)
        occ = np.asarray(
            estimate_fn(samples, hi, spin_symmetric=spin_symmetric),
            dtype=float,
        )

        occupations_history.append(occ.copy())
        energy_stats = driver.energy
        energy_history.append(float(np.real(energy_stats.mean)))
        step_history.append(driver.step_count)

        segment_summary_item = {
            "iteration_occ_segment": {
                "segment_id": int(segment_id),
                "segment_steps": int(segment_steps),
                "step_end": int(driver.step_count),
                "occ_used": occ_used_history[-1].copy(),
                "occ_next": occ.copy(),
                "energy_mean": float(np.real(energy_stats.mean)),
            }
        }
        history_log(driver.step_count, segment_summary_item, vstate)
        for logger in manual_loggers:
            if logger is not history_log:
                logger(driver.step_count, segment_summary_item, vstate)

        if segment_id + 1 < len(segment_lengths):
            vstate.sampler = sampler_cls(
                hi,
                graph=graph_sampler,
                occupations=occ,
                d_max=d_max,
                spin_symmetric=spin_symmetric,
                **sampler_kwargs,
            )

    return {
        "driver": driver,
        "variational_state": vstate,
        "log": out,
        "graph_sampler": graph_sampler,
        "schedule": segment_lengths,
        "step_history": step_history,
        "occupations_history": occupations_history,
        "occ_used_history": occ_used_history,
        "occ_used_by_step": occ_used_by_step,
        "energy_history": energy_history,
        "history_log": history_log,
        "final_occupations": occ.copy(),
    }
