# Copyright 2026 Ping Xu - All rights reserved.

"""
Stage-wise occupation updates for MetropolisFermionHopWithProposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import netket as nk
import numpy as np

from netket.hilbert import SpinOrbitalFermions
from netket.vqs import MCState


@dataclass
class OccupationIterationResult:
    occupations_history: list[np.ndarray]
    states: list[Any]


def estimate_occupations_from_samples(
    samples,
    hilbert: SpinOrbitalFermions,
    *,
    spin_symmetric: bool = True,
) -> np.ndarray:
    """
    Estimate mean occupations from a batch of sampled configurations.

    If ``spin_symmetric=True``, returns one mean occupation per orbital.
    Otherwise, returns one mean occupation per Hilbert mode.
    """
    samples = np.asarray(samples)
    flat_samples = samples.reshape(-1, hilbert.size)

    if spin_symmetric and hilbert.n_spin_subsectors > 1:
        flat_samples = flat_samples.reshape(
            -1, hilbert.n_spin_subsectors, hilbert.n_orbitals
        )
        return flat_samples.mean(axis=(0, 1))

    return flat_samples.mean(axis=0)


def iterate_occupations(
    hilbert: SpinOrbitalFermions,
    graph,
    occ_init,
    *,
    model,
    n_stages: int = 5,
    n_samples: int = 1024,
    n_discard_per_chain: int = 16,
    d_max: int = 1,
    spin_symmetric: bool = True,
    sampler_cls=nk.sampler.MetropolisFermionHopWithProposal,
    mc_state_cls=nk.vqs.MCState,
    **mc_state_kwargs,
) -> OccupationIterationResult:
    """
    Run stage-wise occupation updates.

    At each stage:
    1. build a sampler using the current occupations;
    2. construct an MCState;
    3. draw samples;
    4. estimate updated occupations from those samples.
    """
    if not isinstance(hilbert, SpinOrbitalFermions):
        raise TypeError(
            "hilbert must be SpinOrbitalFermions, "
            f"got {type(hilbert)}."
        )
    if n_stages <= 0:
        raise ValueError(f"n_stages must be positive, got {n_stages}.")

    occ = np.asarray(occ_init, dtype=float)
    occupations_history: list[np.ndarray] = [occ.copy()]
    states: list[Any] = []

    for _ in range(n_stages):
        sampler = sampler_cls(
            hilbert,
            graph=graph,
            occupations=occ,
            d_max=d_max,
            spin_symmetric=spin_symmetric,
        )

        vstate = mc_state_cls(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            **mc_state_kwargs,
        )

        samples = np.asarray(vstate.samples)
        occ = estimate_occupations_from_samples(
            samples, hilbert, spin_symmetric=spin_symmetric
        )

        occupations_history.append(occ.copy())
        states.append(vstate)

    return OccupationIterationResult(
        occupations_history=occupations_history,
        states=states,
    )


def _normalise_schedule(
    *,
    n_iter: int | None,
    occ_update_every: int | None,
    schedule: Sequence[int] | None,
) -> list[int]:
    """
    Build the segment lengths used by occupation updates.

    You can choose one of the two interfaces:
    1. `n_iter` + `occ_update_every`
    2. explicit `schedule=[..., ..., ...]`
    """
    if schedule is not None:
        segment_lengths = [int(step) for step in schedule]
        if len(segment_lengths) == 0:
            raise ValueError("schedule must not be empty.")
        if any(step <= 0 for step in segment_lengths):
            raise ValueError(
                f"all schedule entries must be positive, got {segment_lengths}."
            )
        if n_iter is not None and sum(segment_lengths) != n_iter:
            raise ValueError(
                "sum(schedule) must match n_iter when both are provided, "
                f"got sum(schedule)={sum(segment_lengths)} and n_iter={n_iter}."
            )
        return segment_lengths

    if n_iter is None:
        raise ValueError("n_iter must be provided when schedule is not specified.")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}.")
    if occ_update_every is None:
        raise ValueError(
            "occ_update_every must be provided when schedule is not specified."
        )
    if occ_update_every <= 0:
        raise ValueError(
            f"occ_update_every must be positive, got {occ_update_every}."
        )

    segment_lengths: list[int] = []
    steps_left = int(n_iter)
    while steps_left > 0:
        step = min(int(occ_update_every), steps_left)
        segment_lengths.append(step)
        steps_left -= step

    return segment_lengths


def run_vmc_with_occupation_schedule(
    hamiltonian,
    variational_state: MCState,
    *,
    optimizer,
    preconditioner=None,
    graph,
    occ_init,
    n_iter: int | None = None,
    occ_update_every: int | None = None,
    schedule: Sequence[int] | None = None,
    d_max: int = 1,
    spin_symmetric: bool = True,
    sampler_cls=nk.sampler.MetropolisFermionHopWithProposal,
    sampler_kwargs: dict[str, Any] | None = None,
    estimate_fn=estimate_occupations_from_samples,
    out=None,
    obs=None,
    step_size: int = 1,
    show_progress: bool = True,
    callback=lambda *args: True,
    timeit: bool = False,
):
    """
    Run VMC while periodically recomputing occupations and rebuilding the sampler.

    This function is designed for the workflow used in the notebook:
    1. build a model;
    2. build an `MCState`;
    3. build `optimizer` and `SR`;
    4. train for a few steps;
    5. recompute occupations from samples;
    6. update the sampler with the new occupations;
    7. continue training with the same `MCState` parameters.

    Important detail:
    - the `MCState` object is reused across segments;
    - only the sampler is replaced between segments;
    - therefore the model parameters are kept automatically.

    Returns a plain dictionary so it is easy to inspect in scripts and notebooks.
    """
    if not isinstance(variational_state, MCState):
        raise TypeError(
            "variational_state must be an instance of netket.vqs.MCState, "
            f"got {type(variational_state)}."
        )

    hilbert = variational_state.hilbert
    if not isinstance(hilbert, SpinOrbitalFermions):
        raise TypeError(
            "variational_state.hilbert must be SpinOrbitalFermions, "
            f"got {type(hilbert)}."
        )

    segment_lengths = _normalise_schedule(
        n_iter=n_iter,
        occ_update_every=occ_update_every,
        schedule=schedule,
    )

    if out is None:
        out = nk.logging.RuntimeLog()

    sampler_kwargs = dict(sampler_kwargs or {})
    occ = np.asarray(occ_init, dtype=float)
    occupations_history: list[np.ndarray] = [occ.copy()]
    energy_history: list[float] = []
    step_history: list[int] = []

    variational_state.sampler = sampler_cls(
        hilbert,
        graph=graph,
        occupations=occ,
        d_max=d_max,
        spin_symmetric=spin_symmetric,
        **sampler_kwargs,
    )

    driver = nk.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=preconditioner,
        variational_state=variational_state,
    )

    for segment_id, segment_steps in enumerate(segment_lengths):
        driver.run(
            n_iter=segment_steps,
            out=out,
            obs=obs,
            step_size=step_size,
            show_progress=show_progress,
            callback=callback,
            timeit=timeit,
        )

        samples = np.asarray(variational_state.samples)
        occ = np.asarray(
            estimate_fn(samples, hilbert, spin_symmetric=spin_symmetric),
            dtype=float,
        )

        occupations_history.append(occ.copy())
        energy_history.append(
            float(np.real(variational_state.expect(hamiltonian).mean))
        )
        step_history.append(driver.step_count)

        if segment_id + 1 < len(segment_lengths):
            variational_state.sampler = sampler_cls(
                hilbert,
                graph=graph,
                occupations=occ,
                d_max=d_max,
                spin_symmetric=spin_symmetric,
                **sampler_kwargs,
            )

    return {
        "driver": driver,
        "variational_state": variational_state,
        "log": out,
        "schedule": segment_lengths,
        "step_history": step_history,
        "occupations_history": occupations_history,
        "energy_history": energy_history,
        "final_occupations": occ.copy(),
    }
