"""Minimal benchmark wrapper for run_kondoheisenbergchain_transformers.py.

This benchmark keeps the simplest path:
- site-token transformer only;
- full Hamiltonian sampler without proposal bias;
- small open chain;
- VMC_SR + Adam runner defaults, with every optimizer knob written explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path


# System: small open-chain benchmark.
Lx = 4  # Chain length, i.e. number of lattice sites.
t = 1.0  # Fermion hopping amplitude.
J_K = 1.0  # Kondo coupling between fermions and local spins.
J1 = 1.0  # Nearest-neighbor Heisenberg coupling.
J2 = 0.0  # Next-nearest-neighbor Heisenberg coupling.
delta_z = 1.0  # Spin anisotropy factor in the z direction.
pbc = False  # False means open boundary conditions.
n_fermions = 4  # Total number of fermions in the chain.
joint_two_sz = 0  # Target joint sector written as 2*Sz_tot

# Model: use only the site-token transformer.
models = ['site-token']  # Use the site-token transformer only.
d_model = 16  # Transformer embedding width.
n_heads = 2  # Number of attention heads per layer.
n_layers = 2  # Number of transformer layers.
mlp_ratio = 4  # Transformer FFN hidden width = d_model * mlp_ratio

# Sampler: keep the neutral full-Hamiltonian connectivity by default.
joint_sampler = 'hamiltonian'  # proposal_* settings below only affect 'hamiltonian-with-proposal'.
proposal_occupations_file = None  # Optional occupation prior file; accepts length L or 2L = [dn block | up block].
proposal_occupation_value = None  # Optional scalar occupation prior replicated over L sites before spin-block expansion.
proposal_noise_strength = 0.0  # Beta concentration nu; larger nu means less noise. Relevant only with 'hamiltonian-with-proposal' plus an occupation prior.
proposal_mixing = 0.05  # Uniform mixing lambda in occ_safe = (1 - lambda) * occ + lambda * 0.5 before Beta resampling.
proposal_balance_beta = 0.5  # Soft balancing beta in [0, 1] for ss / ff / sf move classes; beta=0 with no occupation prior recovers the neutral limit.

# Driver / optimizer: keep the runner defaults for VMC_SR + Adam, but spell them out.
driver = 'vmc-sr'  # Use the VMC_SR driver rather than plain VMC.
optimizer = 'adam'  # Outer optimizer for parameter updates.
optimizer_schedule = 'constant'  # Constant schedule preserves the historical runner behavior.
learning_rate = 1.0e-2  # Base learning rate, or the peak value for warmup-cosine schedules.
learning_rate_end = 5.0e-5  # End value used only by the warmup-cosine learning-rate schedule.
optimizer_warmup_fraction = 0.05  # Warmup fraction used only by the warmup-cosine learning-rate schedule.
clip_grad_norm = 0.0  # Global gradient clipping norm; <= 0 keeps clipping disabled.
optimizer_weight_decay = 1.0e-4  # Weight decay used only by adamw.
optimizer_momentum = 0.0  # Momentum used only by sgd; this is not the VMC_SR SPRING momentum below.
adam_b1 = 0.9  # Adam first-moment decay.
adam_b2 = 0.999  # Adam second-moment decay.
adam_eps = 1.0e-8  # Adam epsilon for numerical stability.
rmsprop_decay = 0.9  # RMSProp decay parameter.
rmsprop_eps = 1.0e-7  # RMSProp epsilon for numerical stability.
diag_shift = 5.0e-2  # SR diagonal shift for numerical stability.
momentum = None  # Optional VMC_SR SPRING momentum, distinct from SGD / Adam optimizer momentum.
proj_reg = None  # Optional projection regularization; only implemented on the NTK/minSR path.
linear_solver = 'cholesky'  # Linear solver used inside VMC_SR.
mode = 'real'  # Optimize a real-valued variational state.
use_ntk = None  # Let VMC_SR choose NTK mode automatically from parameter count vs samples.
on_the_fly = None  # Let VMC_SR choose on-the-fly mode automatically; it defaults with the NTK path.

# Runtime: small but nontrivial benchmark, not a converged production run.
n_iter = 20  # Number of optimization iterations.
n_samples = 256  # Monte Carlo samples per iteration.
n_discard_per_chain = 16  # Burn-in samples discarded for each chain.
n_chains_per_rank = 4  # Number of Markov chains per rank / process.
d_max = 1  # Unused for joint Hamiltonian samplers; only relevant for fermion-hop style proposals.
sweep_size = None  # Internal Metropolis updates between saved samples; None means use hilbert.size.
seed = 1234  # Random seed for reproducibility.
show_progress = False  # Disable the progress bar for cleaner logs.

THIS_DIR = Path(__file__).resolve().parent  # Directory containing this wrapper file.
REPO_ROOT = THIS_DIR.parent  # Repository root inferred from the wrapper location.
out_dir = str(REPO_ROOT / 'run_outputs' / 'khc_site_token_small_benchmark')  # Output directory for summaries and final state.

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_kondoheisenbergchain_transformers as runner


def build_argv() -> list[str]:
    argv = [
        'run_kondoheisenbergchain_transformers.py',
        '--Lx', str(Lx),
        '--t', str(t),
        '--J_K', str(J_K),
        '--J1', str(J1),
        '--J2', str(J2),
        '--delta-z', str(delta_z),
        '--n-fermions', str(n_fermions),
        '--joint-two-sz', str(joint_two_sz),
        '--joint-sampler', joint_sampler,
        '--models', *models,
        '--d-model', str(d_model),
        '--n-heads', str(n_heads),
        '--n-layers', str(n_layers),
        '--mlp-ratio', str(mlp_ratio),
        '--driver', driver,
        '--optimizer', optimizer,
        '--optimizer-schedule', optimizer_schedule,
        '--learning-rate', str(learning_rate),
        '--learning-rate-end', str(learning_rate_end),
        '--optimizer-warmup-fraction', str(optimizer_warmup_fraction),
        '--clip-grad-norm', str(clip_grad_norm),
        '--optimizer-weight-decay', str(optimizer_weight_decay),
        '--optimizer-momentum', str(optimizer_momentum),
        '--adam-b1', str(adam_b1),
        '--adam-b2', str(adam_b2),
        '--adam-eps', str(adam_eps),
        '--rmsprop-decay', str(rmsprop_decay),
        '--rmsprop-eps', str(rmsprop_eps),
        '--diag-shift', str(diag_shift),
        '--n-iter', str(n_iter),
        '--n-samples', str(n_samples),
        '--n-discard-per-chain', str(n_discard_per_chain),
        '--n-chains-per-rank', str(n_chains_per_rank),
        '--d-max', str(d_max),
        '--seed', str(seed),
        '--out-dir', out_dir,
    ]

    if pbc:
        argv.append('--pbc')
    if proposal_occupations_file is not None:
        argv.extend(['--proposal-occupations-file', str(proposal_occupations_file)])
    if proposal_occupation_value is not None:
        argv.extend(['--proposal-occupation-value', str(proposal_occupation_value)])
    if joint_sampler == 'hamiltonian-with-proposal':
        argv.extend([
            '--proposal-noise-strength', str(proposal_noise_strength),
            '--proposal-mixing', str(proposal_mixing),
            '--proposal-balance-beta', str(proposal_balance_beta),
        ])
    if proj_reg is not None:
        argv.extend(['--proj-reg', str(proj_reg)])
    if momentum is not None:
        argv.extend(['--momentum', str(momentum)])
    if linear_solver is not None:
        argv.extend(['--linear-solver', linear_solver])
    if mode is not None:
        argv.extend(['--mode', mode])
    if use_ntk is True:
        argv.append('--use-ntk')
    elif use_ntk is False:
        argv.append('--no-use-ntk')
    if on_the_fly is True:
        argv.append('--on-the-fly')
    elif on_the_fly is False:
        argv.append('--no-on-the-fly')
    if sweep_size is not None:
        argv.extend(['--sweep-size', str(sweep_size)])
    if show_progress:
        argv.append('--show-progress')
    else:
        argv.append('--no-progress')
    return argv


def main() -> int:
    argv = build_argv()
    saved_argv = sys.argv[:]
    try:
        sys.argv = argv
        return runner.main()
    finally:
        sys.argv = saved_argv


if __name__ == '__main__':
    raise SystemExit(main())
