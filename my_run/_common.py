"""Shared helpers for `run/*.py`.

This module has two layers.

1. Generic runner utilities reused by every entry point:
   - parser construction and shared CLI groups;
   - argument packaging for console / JSON summaries;
   - `MCState` construction, generic VMC/VMC_SR optimizer-driver helpers, driver execution, and artifact writing.

2. SIAM-specific helpers reused by the three SIAM runners:
   - Hamiltonian builder resolution;
   - Slater / AGP model registry and instantiation;
   - outer optimizer + SR construction.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

import numpy as np

# Keep local imports working no matter which runner under `run/` is executed.
REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "Hamiltonian", REPO_ROOT / "nqs_state"):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

import netket as nk
import optax
from flax import serialization

from graph_sample.hamiltonian_graph import graph_from_hamiltonian
from nqs_state.AGP import (
    AGPBackflow,
    AGPBackflowDeep,
    SmoothAGPJastrow,
    SmoothAGPNeuralJastrow,
    SmoothAGPNeuralJastrowDeep,
)
from nqs_state.slater import (
    LogNeuralBackflow,
    LogNeuralBackflowDeep,
    LogNeuralJastrowBackflow,
    LogNeuralJastrowBackflowDeep,
)


class _ArgumentParserFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Help formatter that preserves examples and still shows defaults."""


_UNSET = object()
DEFAULT_HAMILTONIAN_BUILDER = "SIAM:SIAM"
HAMILTONIAN_ROOT = REPO_ROOT / "Hamiltonian"

# ---------------------------------------------------------------------------
# SIAM model registries
# ---------------------------------------------------------------------------
SLATER_MODELS = {
    "backflow": LogNeuralBackflow,
    "backflow-deep": LogNeuralBackflowDeep,
    "jastrowbackflow": LogNeuralJastrowBackflow,
    "jastrowbackflow-deep": LogNeuralJastrowBackflowDeep,
}

AGP_MODELS = {
    "agp": SmoothAGPJastrow,
    "agp-jastrow": SmoothAGPNeuralJastrow,
    "agp-jastrow-deep": SmoothAGPNeuralJastrowDeep,
    "agp-backflow": AGPBackflow,
    "agp-backflow-deep": AGPBackflowDeep,
}

ALL_MODELS = {**SLATER_MODELS, **AGP_MODELS}

_MODEL_LAYER_VARIANTS = {
    "backflow": {1: "backflow", 2: "backflow-deep"},
    "jastrowbackflow": {1: "jastrowbackflow", 2: "jastrowbackflow-deep"},
    "agp-jastrow": {1: "agp-jastrow", 2: "agp-jastrow-deep"},
    "agp-backflow": {1: "agp-backflow", 2: "agp-backflow-deep"},
}
_MODEL_FAMILY_BY_NAME = {
    "backflow": "backflow",
    "backflow-deep": "backflow",
    "jastrowbackflow": "jastrowbackflow",
    "jastrowbackflow-deep": "jastrowbackflow",
    "agp-jastrow": "agp-jastrow",
    "agp-jastrow-deep": "agp-jastrow",
    "agp-backflow": "agp-backflow",
    "agp-backflow-deep": "agp-backflow",
}
_MODELS_WITH_HIDDEN_UNITS = set(ALL_MODELS) - {"agp"}
_MODELS_WITH_INIT_BACKFLOW_SCALE = {"agp-backflow", "agp-backflow-deep"}

# Shared linear solvers exposed by both `nk.optimizer.SR` and `nk.driver.VMC_SR`.
LINEAR_SOLVER_BUILDERS = {
    "LU": nk.optimizer.solver.LU,
    "cholesky": nk.optimizer.solver.cholesky,
    "pinv": nk.optimizer.solver.pinv,
    "pinv_smooth": nk.optimizer.solver.pinv_smooth,
    "solve": nk.optimizer.solver.solve,
    "svd": nk.optimizer.solver.svd,
}

_SR_SOLVER_BUILDERS = LINEAR_SOLVER_BUILDERS

_SR_QGT_BUILDERS = {
    "QGTAuto": nk.optimizer.qgt.QGTAuto,
    "QGTJacobianDense": nk.optimizer.qgt.QGTJacobianDense,
    "QGTJacobianPyTree": nk.optimizer.qgt.QGTJacobianPyTree,
    "QGTOnTheFly": nk.optimizer.qgt.QGTOnTheFly,
}


def _discover_hamiltonian_modules() -> tuple[str, ...]:
    """List builder modules under `Hamiltonian/` for CLI help text."""
    if not HAMILTONIAN_ROOT.exists():
        return ()
    return tuple(
        sorted(
            path.stem
            for path in HAMILTONIAN_ROOT.glob("*.py")
            if path.is_file() and path.stem != "__init__" and not path.stem.startswith("_")
        )
    )


DISCOVERED_HAMILTONIAN_MODULES = _discover_hamiltonian_modules()


# ---------------------------------------------------------------------------
# Generic parser and runtime helpers
# ---------------------------------------------------------------------------
def create_argument_parser(
    description: str,
    *,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Create a parser with the same formatting style across every runner."""
    return argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=_ArgumentParserFormatter,
    )



def add_progress_arguments(container) -> None:
    """Add the common progress-bar toggles used by every runner."""
    container.add_argument("--show-progress", action="store_true", default=True)
    container.add_argument("--no-progress", action="store_false", dest="show_progress")



def add_runtime_sampling_arguments(
    container,
    *,
    n_iter_default: int,
    n_samples_default: int,
    n_discard_per_chain_default: int,
    d_max_default=_UNSET,
    n_chains_per_rank_default=_UNSET,
    sweep_size_default=_UNSET,
    seed_default=_UNSET,
) -> None:
    """Add the shared arguments that later feed samplers, `MCState`, and drivers."""
    container.add_argument("--n-iter", type=int, default=int(n_iter_default))
    container.add_argument("--n-samples", type=int, default=int(n_samples_default))
    container.add_argument(
        "--n-discard-per-chain",
        type=int,
        default=int(n_discard_per_chain_default),
    )

    if d_max_default is not _UNSET:
        container.add_argument("--d-max", type=int, default=int(d_max_default))
    if n_chains_per_rank_default is not _UNSET:
        container.add_argument(
            "--n-chains-per-rank",
            type=int,
            default=int(n_chains_per_rank_default),
        )
    if sweep_size_default is not _UNSET:
        container.add_argument("--sweep-size", type=int, default=sweep_size_default)
    if seed_default is not _UNSET:
        container.add_argument("--seed", type=int, default=int(seed_default))



def add_output_argument(container, *, default_out_dir: str) -> None:
    """Add a shared `--out-dir` flag for runners that persist artifacts."""
    container.add_argument("--out-dir", type=str, default=str(default_out_dir))



def list_or_none(values):
    """Convert an optional tuple/list from argparse into a JSON-friendly list."""
    return None if values is None else list(values)



def tuple_or_none(values):
    """Convert an optional tuple/list from argparse into the tuple expected by builders."""
    return None if values is None else tuple(values)


def bool_or_auto(value):
    """Render optional bool toggles so `None` preserves NetKet's internal auto mode."""
    return "auto" if value is None else bool(value)



def namespace_snapshot(
    args,
    *fields: str,
    transforms: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect selected argparse fields into a plain dict for printing or JSON."""
    transforms = transforms or {}
    snapshot: dict[str, object] = {}
    for field in fields:
        value = getattr(args, field)
        transform = transforms.get(field)
        if transform is not None:
            value = transform(value)
        snapshot[field] = value
    return snapshot



def print_mapping(title: str, values: dict[str, object]) -> None:
    """Print a mapping in a stable `key=value` block."""
    print(title)
    for key, value in values.items():
        print(f"  {key}={value}")



def build_mcstate(sampler, model, args, *, seed=_UNSET, **extra_kwargs):
    """Create `nk.vqs.MCState` from a sampler/model plus the shared runtime args."""
    mcstate_kwargs = dict(
        sampler=sampler,
        model=model,
        n_samples=int(args.n_samples),
        n_discard_per_chain=int(args.n_discard_per_chain),
    )

    if seed is _UNSET:
        if hasattr(args, "seed") and getattr(args, "seed") is not None:
            mcstate_kwargs["seed"] = int(args.seed)
    elif seed is not None:
        mcstate_kwargs["seed"] = int(seed)

    mcstate_kwargs.update(extra_kwargs)
    return nk.vqs.MCState(**mcstate_kwargs)



def run_driver(driver, args):
    """Run a NetKet driver using the common `n_iter` and progress arguments."""
    log = nk.logging.RuntimeLog()
    driver.run(
        n_iter=int(args.n_iter),
        out=log,
        show_progress=bool(args.show_progress),
    )
    return log



def energy_summary(hamiltonian, vstate, driver) -> dict[str, float | int | None]:
    """Collect post-run energy and acceptance metrics in a JSON-friendly form."""
    acceptance = getattr(vstate.sampler_state, "acceptance", None)
    energy = vstate.expect(hamiltonian)
    return {
        "step_count": int(driver.step_count),
        "energy_mean_real": float(np.real(energy.mean)),
        "energy_error_real": float(np.real(energy.error_of_mean)),
        "variance_real": float(np.real(energy.variance)),
        "acceptance": (None if acceptance is None else float(acceptance)),
    }



def write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable formatting and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def save_vstate(vstate, destination: Path) -> None:
    """Serialize the final `MCState` so it can be reloaded later."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialization.to_bytes(vstate))



def save_model_artifacts(base_out_dir: Path | str, model_name: str, *, summary: dict, vstate) -> Path:
    """Persist the standard per-model artifacts and return the model output directory."""
    model_dir = Path(base_out_dir) / model_name
    save_vstate(vstate, model_dir / "final_vstate.msgpack")
    write_json(model_dir / "summary.json", summary)
    return model_dir


# ---------------------------------------------------------------------------
# Shared VMC / VMC_SR helpers
# ---------------------------------------------------------------------------
def add_vmc_driver_arguments(
    container,
    *,
    driver_choices=("vmc", "vmc-sr"),
    driver_default="vmc-sr",
    optimizer_choices=("sgd", "adam"),
    optimizer_default="adam",
    learning_rate_default: float,
    diag_shift_default: float,
    linear_solver_default="cholesky",
    mode_default="real",
) -> None:
    """Add a shared VMC/VMC_SR argument block for lightweight runners."""
    container.add_argument("--driver", choices=tuple(driver_choices), default=driver_default)
    container.add_argument("--optimizer", choices=tuple(optimizer_choices), default=optimizer_default)
    container.add_argument("--learning-rate", type=float, default=float(learning_rate_default))
    container.add_argument("--diag-shift", type=float, default=float(diag_shift_default))
    container.add_argument("--proj-reg", type=float, default=None)
    container.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="SPRING-style momentum used by nk.driver.VMC_SR.",
    )
    container.add_argument(
        "--linear-solver",
        choices=tuple(sorted(LINEAR_SOLVER_BUILDERS)),
        default=linear_solver_default,
    )
    container.add_argument("--mode", choices=("real", "complex"), default=mode_default)
    container.add_argument("--use-ntk", action="store_true", dest="use_ntk")
    container.add_argument("--no-use-ntk", action="store_false", dest="use_ntk")
    container.add_argument("--on-the-fly", action="store_true", dest="on_the_fly")
    container.add_argument("--no-on-the-fly", action="store_false", dest="on_the_fly")
    # Leave these toggles unset unless the user explicitly overrides them, so
    # VMC_SR can fall back to its internal heuristics based on parameter count.
    container.set_defaults(use_ntk=None, on_the_fly=None)


def validate_vmc_driver_args(args) -> None:
    """Reject VMC_SR-only arguments when the runner is configured for plain VMC."""
    if getattr(args, "driver", None) == "vmc-sr":
        return

    if getattr(args, "proj_reg", None) is not None:
        raise ValueError("--proj-reg only applies to --driver vmc-sr.")
    if getattr(args, "momentum", None) is not None:
        raise ValueError("--momentum only applies to --driver vmc-sr.")
    if getattr(args, "linear_solver", "cholesky") != "cholesky":
        raise ValueError("--linear-solver only applies to --driver vmc-sr.")
    if getattr(args, "use_ntk", None) is not None:
        raise ValueError("--use-ntk/--no-use-ntk only apply to --driver vmc-sr.")
    if getattr(args, "on_the_fly", None) is not None:
        raise ValueError("--on-the-fly/--no-on-the-fly only apply to --driver vmc-sr.")


def collect_vmc_driver_config(args) -> dict[str, object]:
    """Package the shared VMC/VMC_SR knobs into a summary dict."""
    config = namespace_snapshot(
        args,
        "driver",
        "optimizer",
        "learning_rate",
        "diag_shift",
    )
    config["preconditioner"] = None

    if getattr(args, "driver", None) == "vmc-sr":
        config.update(
            namespace_snapshot(
                args,
                "proj_reg",
                "momentum",
                "linear_solver",
                "mode",
                "use_ntk",
                "on_the_fly",
                transforms={
                    "use_ntk": bool_or_auto,
                    "on_the_fly": bool_or_auto,
                },
            )
        )
    else:
        config.update(
            {
                "proj_reg": None,
                "momentum": None,
                "linear_solver": None,
                "mode": None,
                "use_ntk": None,
                "on_the_fly": None,
            }
        )
        config["preconditioner"] = "nk.optimizer.SR"

    return config


def build_basic_optimizer(args):
    """Build a simple outer optimizer shared by lightweight VMC runners."""
    if args.optimizer == "sgd":
        return optax.sgd(float(args.learning_rate))
    if args.optimizer == "adam":
        return nk.optimizer.Adam(learning_rate=float(args.learning_rate))
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_vmc_or_vmc_sr_driver(hamiltonian, vstate, optimizer, args):
    """Build either plain `nk.VMC` or `nk.driver.VMC_SR` from shared CLI args."""
    if args.driver == "vmc-sr":
        # `None` keeps VMC_SR's internal heuristics for these toggles.
        return nk.driver.VMC_SR(
            hamiltonian,
            optimizer,
            variational_state=vstate,
            diag_shift=float(args.diag_shift),
            proj_reg=args.proj_reg,
            momentum=args.momentum,
            linear_solver=LINEAR_SOLVER_BUILDERS[args.linear_solver],
            mode=args.mode,
            use_ntk=args.use_ntk,
            on_the_fly=args.on_the_fly,
        )

    preconditioner = nk.optimizer.SR(diag_shift=float(args.diag_shift))
    return nk.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=preconditioner,
        variational_state=vstate,
    )


# ---------------------------------------------------------------------------
# SIAM parser helpers
# ---------------------------------------------------------------------------
def add_siam_hamiltonian_arguments(container) -> None:
    """Add the Hamiltonian-builder arguments shared by the SIAM runners."""
    discovered = ", ".join(DISCOVERED_HAMILTONIAN_MODULES) or "none"
    container.add_argument(
        "--hamiltonian-builder",
        default=DEFAULT_HAMILTONIAN_BUILDER,
        help=(
            "Hamiltonian builder in 'Module:Callable' or 'Module.Callable' form. "
            f"Default: {DEFAULT_HAMILTONIAN_BUILDER}. Discovered modules: {discovered}."
        ),
    )
    container.add_argument(
        "--hamiltonian-kw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword arguments passed to the Hamiltonian builder. Repeat as "
            "needed. VALUE is parsed with Python literal syntax when possible."
        ),
    )
    container.add_argument("--L", type=int, default=19)
    container.add_argument("--U", type=float, default=4.0)
    container.add_argument("--V", type=float, default=0.15)
    container.add_argument("--ti", type=float, default=1.0)
    container.add_argument(
        "--n-fermions-per-spin",
        type=int,
        nargs=2,
        default=None,
        metavar=("N_UP", "N_DN"),
    )
    container.add_argument("--pbc", action="store_true")
    container.add_argument("--penalty-strength", type=float, default=0.0)
    container.add_argument("--penalty-target-spin", type=float, default=None)



def add_siam_model_arguments(container) -> None:
    """Add the ansatz-selection arguments shared by the SIAM runners."""
    container.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ALL_MODELS),
        default=["backflow"],
        help="Model registry names. Deep variants can also be selected via --n-layers when supported.",
    )
    container.add_argument(
        "--hidden-units",
        "--n-hidden",
        dest="hidden_units",
        type=int,
        default=32,
        help="Hidden width used by ansatzes that expose a neural hidden dimension.",
    )
    container.add_argument("--symmetric", action="store_true")



def base_parser(
    description: str,
    *,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Create the shared parser used by the SIAM runner family."""
    parser = create_argument_parser(description, epilog=epilog)

    system_group = parser.add_argument_group("Hamiltonian / system")
    add_siam_hamiltonian_arguments(system_group)

    model_group = parser.add_argument_group("Model")
    add_siam_model_arguments(model_group)

    runtime_group = parser.add_argument_group("Sampling / runtime")
    add_runtime_sampling_arguments(
        runtime_group,
        n_iter_default=300,
        n_samples_default=4096,
        n_discard_per_chain_default=16,
        d_max_default=1,
    )
    add_progress_arguments(runtime_group)
    return parser



def add_model_optimizer_sr_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add SIAM model-variant controls plus outer optimizer / SR knobs."""
    model_group = parser.add_argument_group("Model variants")
    model_group.add_argument(
        "--n-layers",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help=(
            "Override the model hidden-layer count when the selected ansatz has "
            "both shallow and deep variants. Use 0 only with 'agp'."
        ),
    )
    model_group.add_argument(
        "--init-backflow-scale",
        type=float,
        default=1.0e-2,
        help="Initial backflow_scale parameter for AGP backflow models.",
    )

    optimizer_group = parser.add_argument_group("Optimizer / SR")
    optimizer_group.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd", "rmsprop"),
        default="adam",
        help="Outer optimizer used by nk.VMC after SR preconditioning.",
    )
    optimizer_group.add_argument(
        "--optimizer-schedule",
        choices=("warmup-cosine", "constant"),
        default="warmup-cosine",
        help="Learning-rate schedule for the outer optimizer.",
    )
    optimizer_group.add_argument("--learning-rate", type=float, default=5.0e-3)
    optimizer_group.add_argument("--learning-rate-end", type=float, default=5.0e-5)
    optimizer_group.add_argument(
        "--optimizer-warmup-fraction",
        type=float,
        default=0.05,
        help="Warmup fraction used by the warmup-cosine learning-rate schedule.",
    )
    optimizer_group.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm applied before the outer optimizer. <= 0 disables clipping.",
    )
    optimizer_group.add_argument(
        "--optimizer-weight-decay",
        type=float,
        default=1.0e-4,
        help="Weight decay used by adamw.",
    )
    optimizer_group.add_argument(
        "--optimizer-momentum",
        type=float,
        default=0.0,
        help="Momentum used by optax.sgd.",
    )
    optimizer_group.add_argument("--adam-b1", type=float, default=0.9)
    optimizer_group.add_argument("--adam-b2", type=float, default=0.999)
    optimizer_group.add_argument("--adam-eps", type=float, default=1.0e-8)
    optimizer_group.add_argument("--rmsprop-decay", type=float, default=0.9)
    optimizer_group.add_argument("--rmsprop-eps", type=float, default=1.0e-7)
    optimizer_group.add_argument("--diag-shift", type=float, default=5.0e-2)
    optimizer_group.add_argument("--diag-shift-end", type=float, default=1.0e-2)
    optimizer_group.add_argument(
        "--diag-shift-schedule",
        choices=("linear", "constant"),
        default="linear",
        help="Schedule used for the SR diagonal shift.",
    )
    optimizer_group.add_argument(
        "--diag-shift-transition-fraction",
        type=float,
        default=0.6,
        help="Transition fraction used by the linear diag-shift schedule.",
    )
    optimizer_group.add_argument(
        "--sr-solver",
        choices=tuple(sorted(LINEAR_SOLVER_BUILDERS)),
        default="pinv",
        help="Linear solver used by nk.optimizer.SR.",
    )
    optimizer_group.add_argument(
        "--sr-qgt",
        choices=tuple(sorted(_SR_QGT_BUILDERS)),
        default="QGTJacobianDense",
        help="QGT implementation used by nk.optimizer.SR.",
    )
    optimizer_group.add_argument(
        "--sr-holomorphic",
        action="store_true",
        default=False,
        help="Pass holomorphic=True to nk.optimizer.SR.",
    )
    return parser


# ---------------------------------------------------------------------------
# SIAM model helpers
# ---------------------------------------------------------------------------
def model_hidden_layers(name: str) -> int:
    """Infer the effective hidden-layer count from a resolved registry name."""
    resolved_name = name.strip()
    if resolved_name == "agp":
        return 0
    if resolved_name.endswith("-deep"):
        return 2
    return 1



def model_uses_hidden_units(name: str) -> bool:
    """Return whether the resolved model consumes `--hidden-units`."""
    return name in _MODELS_WITH_HIDDEN_UNITS



def model_uses_init_backflow_scale(name: str) -> bool:
    """Return whether the resolved model consumes `--init-backflow-scale`."""
    return name in _MODELS_WITH_INIT_BACKFLOW_SCALE



def resolve_model_name(name: str, hidden_layers: int | None) -> str:
    """Resolve a requested model name plus optional `--n-layers` override."""
    if hidden_layers is None:
        if name not in ALL_MODELS:
            raise ValueError(f"Unknown model name {name!r}.")
        return name

    if name == "agp":
        if hidden_layers != 0:
            raise ValueError("Model 'agp' only supports --n-layers 0 (or omitting it).")
        return name

    family = _MODEL_FAMILY_BY_NAME.get(name)
    if family is None:
        raise ValueError(f"Unknown model name {name!r}.")

    try:
        return _MODEL_LAYER_VARIANTS[family][hidden_layers]
    except KeyError as exc:
        allowed = ", ".join(str(level) for level in sorted(_MODEL_LAYER_VARIANTS[family]))
        raise ValueError(
            f"Model {name!r} only supports --n-layers in {{{allowed}}}, got {hidden_layers}."
        ) from exc



def summarize_model_config(name: str, args) -> dict[str, object]:
    """Package the resolved SIAM ansatz choice into a print/JSON-friendly dict."""
    resolved_name = resolve_model_name(name, getattr(args, "n_layers", None))
    cls = ALL_MODELS[resolved_name]
    summary: dict[str, object] = {
        "requested_model": name,
        "resolved_model": resolved_name,
        "model_module": cls.__module__,
        "model_class": cls.__name__,
        "hidden_layers": model_hidden_layers(resolved_name),
        "symmetric": bool(getattr(args, "symmetric", False)),
    }
    if model_uses_hidden_units(resolved_name):
        summary["hidden_units"] = int(getattr(args, "hidden_units", 32))
    if model_uses_init_backflow_scale(resolved_name):
        summary["init_backflow_scale"] = float(getattr(args, "init_backflow_scale", 1.0e-2))
    return summary



def build_model(name: str, hi, args):
    """Instantiate a SIAM ansatz by registry name using the shared CLI namespace."""
    missing = [attr for attr in ("n_orbitals", "n_fermions_per_spin") if not hasattr(hi, attr)]
    if missing:
        raise TypeError(
            "The current runner model registry expects a SpinOrbitalFermions-like "
            f"Hilbert space; missing attributes {missing} on {type(hi)}."
        )

    resolved_name = resolve_model_name(name, getattr(args, "n_layers", None))
    cls = ALL_MODELS[resolved_name]
    hidden_units = int(getattr(args, "hidden_units", 32))
    symmetric = bool(getattr(args, "symmetric", False))
    init_backflow_scale = float(getattr(args, "init_backflow_scale", 1.0e-2))

    if resolved_name in SLATER_MODELS:
        return cls(hilbert=hi, hidden_units=hidden_units)

    n_up, n_dn = hi.n_fermions_per_spin
    kwargs = dict(
        hilbert=hi,
        n_sites=hi.n_orbitals,
        n_up=n_up,
        n_down=n_dn,
        symmetric=symmetric,
    )
    if model_uses_hidden_units(resolved_name):
        kwargs["hidden_units"] = hidden_units
    if model_uses_init_backflow_scale(resolved_name):
        kwargs["init_backflow_scale"] = init_backflow_scale
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Hamiltonian helpers
# ---------------------------------------------------------------------------
def _parse_builder_ref(builder_ref: str) -> tuple[str, str]:
    """Normalize `Module:Callable` / `Module.Callable` Hamiltonian references."""
    builder_ref = builder_ref.strip()
    if not builder_ref:
        raise ValueError("hamiltonian_builder must not be empty.")

    if ":" in builder_ref:
        module_ref, builder_name = builder_ref.split(":", maxsplit=1)
    elif "." in builder_ref:
        module_ref, builder_name = builder_ref.rsplit(".", maxsplit=1)
    else:
        module_ref = builder_ref
        builder_name = builder_ref

    module_ref = module_ref.strip()
    builder_name = builder_name.strip()
    if not module_ref or not builder_name:
        raise ValueError(
            "hamiltonian_builder must be in 'Module:Callable' or 'Module.Callable' form."
        )

    module_path = (
        module_ref if module_ref.startswith("Hamiltonian.") else f"Hamiltonian.{module_ref}"
    )
    return module_path, builder_name



def _is_siam_builder(builder_ref: str) -> bool:
    """Return whether the selected builder is the legacy default SIAM builder."""
    return _parse_builder_ref(builder_ref) == ("Hamiltonian.SIAM", "SIAM")



def _load_hamiltonian_builder(builder_ref: str):
    """Import and validate the requested Hamiltonian builder callable."""
    module_path, builder_name = _parse_builder_ref(builder_ref)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Could not import Hamiltonian builder module {module_path!r} "
            f"from builder ref {builder_ref!r}."
        ) from exc

    try:
        builder = getattr(module, builder_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_path!r} does not define builder {builder_name!r}."
        ) from exc

    if not callable(builder):
        raise TypeError(f"Hamiltonian builder {module_path}:{builder_name} is not callable.")

    return builder, f"{module_path}:{builder_name}"



def _parse_builder_kwargs(raw_kwargs: list[str]) -> dict[str, object]:
    """Parse repeated `--hamiltonian-kw KEY=VALUE` entries into keyword args."""
    parsed: dict[str, object] = {}
    for entry in raw_kwargs:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --hamiltonian-kw entry {entry!r}; expected KEY=VALUE."
            )
        key, raw_value = entry.split("=", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(
                f"Invalid --hamiltonian-kw entry {entry!r}; KEY must not be empty."
            )

        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        parsed[key] = value
    return parsed



def _default_siam_kwargs(args) -> dict[str, object]:
    """Translate the default SIAM CLI fields into builder keyword arguments."""
    return {
        "L": args.L,
        "U": args.U,
        "V": args.V,
        "ti": args.ti,
        "pbc": args.pbc,
        "n_fermions_per_spin": tuple_or_none(args.n_fermions_per_spin),
        "penalty_strength": args.penalty_strength,
        "penalty_target_spin": args.penalty_target_spin,
    }



def _summarize_hilbert(hi) -> str:
    """Render a compact human-readable Hilbert-space summary."""
    fields: list[str] = [f"type={type(hi).__name__}"]
    if hasattr(hi, "n_orbitals"):
        fields.append(f"n_orbitals={hi.n_orbitals}")
    if hasattr(hi, "size"):
        fields.append(f"size={hi.size}")
    if hasattr(hi, "n_fermions_per_spin"):
        fields.append(f"n_fermions_per_spin={hi.n_fermions_per_spin}")
    elif hasattr(hi, "n_fermions"):
        fields.append(f"n_fermions={hi.n_fermions}")
    return ", ".join(fields)



def build_hamiltonian(args):
    """Build the initial Hamiltonian + Hilbert space from the selected builder."""
    builder, builder_label = _load_hamiltonian_builder(args.hamiltonian_builder)

    builder_kwargs: dict[str, object] = {}
    if _is_siam_builder(args.hamiltonian_builder):
        builder_kwargs.update(_default_siam_kwargs(args))
    builder_kwargs.update(_parse_builder_kwargs(args.hamiltonian_kw))

    result = builder(**builder_kwargs)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError(
            f"Hamiltonian builder {builder_label} must return (H, hi), got {type(result)}."
        )

    H, hi = result
    print(f"Hamiltonian builder: {builder_label}")
    if builder_kwargs:
        print(f"  kwargs: {builder_kwargs}")
    print(f"  hilbert: {_summarize_hilbert(hi)}")
    return H, hi


# ---------------------------------------------------------------------------
# Optimizer / SR helpers
# ---------------------------------------------------------------------------
def _legacy_get_optimizer_and_sr(steps: int = 300):
    """Reproduce the historical hard-coded optimizer/SR configuration."""
    warmup_steps = max(1, int(steps * 0.05))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.005,
        warmup_steps=warmup_steps,
        decay_steps=max(1, steps),
        end_value=5e-5,
    )
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    sr_shift = optax.linear_schedule(
        init_value=0.05,
        end_value=0.01,
        transition_steps=max(1, int(steps * 0.6)),
    )
    sr = nk.optimizer.SR(
        diag_shift=sr_shift,
        solver=nk.optimizer.solver.pinv,
        holomorphic=False,
        qgt=nk.optimizer.qgt.QGTJacobianDense,
    )
    return op, sr



def _optimizer_learning_rate(steps: int, args):
    """Build the learning-rate object consumed by the selected outer optimizer."""
    if args.optimizer_schedule == "constant":
        return float(args.learning_rate)

    warmup_steps = max(1, int(max(1, steps) * float(args.optimizer_warmup_fraction)))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(args.learning_rate),
        warmup_steps=warmup_steps,
        decay_steps=max(1, steps),
        end_value=float(args.learning_rate_end),
    )



def _sr_diag_shift(steps: int, args):
    """Build the diagonal-shift object consumed by `nk.optimizer.SR`."""
    if args.diag_shift_schedule == "constant":
        return float(args.diag_shift)

    transition_steps = max(
        1,
        int(max(1, steps) * float(args.diag_shift_transition_fraction)),
    )
    return optax.linear_schedule(
        init_value=float(args.diag_shift),
        end_value=float(args.diag_shift_end),
        transition_steps=transition_steps,
    )



def summarize_optimizer_and_sr_config(args) -> dict[str, object]:
    """Package the CLI-controlled SIAM optimizer/SR settings into a summary dict."""
    return {
        "driver": "nk.VMC + nk.optimizer.SR",
        "optimizer": args.optimizer,
        "optimizer_schedule": args.optimizer_schedule,
        "learning_rate": float(args.learning_rate),
        "learning_rate_end": float(args.learning_rate_end),
        "optimizer_warmup_fraction": float(args.optimizer_warmup_fraction),
        "clip_grad_norm": float(args.clip_grad_norm),
        "optimizer_weight_decay": float(args.optimizer_weight_decay),
        "optimizer_momentum": float(args.optimizer_momentum),
        "adam_b1": float(args.adam_b1),
        "adam_b2": float(args.adam_b2),
        "adam_eps": float(args.adam_eps),
        "rmsprop_decay": float(args.rmsprop_decay),
        "rmsprop_eps": float(args.rmsprop_eps),
        "diag_shift": float(args.diag_shift),
        "diag_shift_end": float(args.diag_shift_end),
        "diag_shift_schedule": args.diag_shift_schedule,
        "diag_shift_transition_fraction": float(args.diag_shift_transition_fraction),
        "sr_solver": args.sr_solver,
        "sr_qgt": args.sr_qgt,
        "sr_holomorphic": bool(args.sr_holomorphic),
    }



def get_optimizer_and_sr(steps_or_args=300, args=None):
    """Create the outer optimizer and SR preconditioner.

    Backward compatibility:
    - `get_optimizer_and_sr(steps)` keeps the historical hard-coded behavior.
    - `get_optimizer_and_sr(args)` or `get_optimizer_and_sr(steps, args=args)`
      uses the CLI-controlled optimizer/SR settings.
    """
    if args is None and hasattr(steps_or_args, "n_iter"):
        args = steps_or_args
        steps = int(args.n_iter)
    elif args is None:
        steps = int(steps_or_args)
    else:
        steps = int(steps_or_args)

    if args is None or not hasattr(args, "optimizer"):
        return _legacy_get_optimizer_and_sr(steps)

    learning_rate = _optimizer_learning_rate(steps, args)

    if args.optimizer == "adam":
        op = optax.adam(
            learning_rate=learning_rate,
            b1=float(args.adam_b1),
            b2=float(args.adam_b2),
            eps=float(args.adam_eps),
        )
    elif args.optimizer == "adamw":
        op = optax.adamw(
            learning_rate=learning_rate,
            b1=float(args.adam_b1),
            b2=float(args.adam_b2),
            eps=float(args.adam_eps),
            weight_decay=float(args.optimizer_weight_decay),
        )
    elif args.optimizer == "sgd":
        op = optax.sgd(
            learning_rate=learning_rate,
            momentum=float(args.optimizer_momentum),
        )
    elif args.optimizer == "rmsprop":
        op = optax.rmsprop(
            learning_rate=learning_rate,
            decay=float(args.rmsprop_decay),
            eps=float(args.rmsprop_eps),
        )
    else:
        raise ValueError(f"Unsupported optimizer {args.optimizer!r}.")

    clip_grad_norm = float(args.clip_grad_norm)
    if clip_grad_norm > 0.0:
        op = optax.chain(optax.clip_by_global_norm(clip_grad_norm), op)

    sr = nk.optimizer.SR(
        diag_shift=_sr_diag_shift(steps, args),
        solver=LINEAR_SOLVER_BUILDERS[args.sr_solver],
        holomorphic=bool(args.sr_holomorphic),
        qgt=_SR_QGT_BUILDERS[args.sr_qgt],
    )
    return op, sr


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt(values) -> str:
    """Format a numeric array for compact terminal printing."""
    return np.array2string(
        np.asarray(values),
        precision=4,
        suppress_small=False,
        max_line_width=120,
    )