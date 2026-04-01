from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from _transformer import TransformerEncoder


def _shared_n_fermions_per_spin(hilbert: nk.hilbert.SpinOrbitalFermions) -> int:
    nonzero_counts = [nf for nf in hilbert.n_fermions_per_spin if nf > 0]
    if not nonzero_counts:
        raise ValueError("Expected at least one occupied spin sector.")

    nfs = nonzero_counts[0]
    if any(nf != nfs for nf in nonzero_counts):
        raise ValueError(
            "These shared-orbital Slater/backflow ansatzes assume the same number of "
            "fermions in each occupied spin sector. "
            f"Got n_fermions_per_spin={hilbert.n_fermions_per_spin}."
        )
    return nfs


def _single_log_backflow(
    n_one: jax.Array,
    correction_one: jax.Array,
    base_orbitals: jax.Array,
    hilbert: nk.hilbert.SpinOrbitalFermions,
) -> jax.Array:
    n_orb = hilbert.n_orbitals
    occupied = n_one.nonzero(size=hilbert.n_fermions)[0]
    log_det_sum = 0.0 + 0.0j
    i_start = 0

    for i, nf_i in enumerate(hilbert.n_fermions_per_spin):
        if nf_i == 0:
            continue
        occ_i = occupied[i_start : i_start + nf_i] - i * n_orb
        orbitals_i = (base_orbitals + correction_one[i])[occ_i]
        log_det_sum = log_det_sum + nk.jax.logdet_cmplx(orbitals_i)
        i_start += nf_i

    return log_det_sum


def _log_backflow(
    n: jax.Array,
    correction: jax.Array,
    base_orbitals: jax.Array,
    hilbert: nk.hilbert.SpinOrbitalFermions,
) -> jax.Array:
    if n.ndim == 1:
        return _single_log_backflow(n, correction, base_orbitals, hilbert)
    return jax.vmap(
        lambda n_one, corr_one: _single_log_backflow(
            n_one, corr_one, base_orbitals, hilbert
        )
    )(n, correction)


def _log_sum_dets(
    log_dets: jax.Array,
    log_coeffs: jax.Array,
) -> jax.Array:
    """Numerically stable log(Σ_k exp(log_coeffs_k + log_dets_k)) for complex log_dets.

    Args:
        log_dets: (..., n_dets) complex log-determinant values.
        log_coeffs: (n_dets,) real log-coefficients.
    """
    z = log_coeffs + log_dets  # (..., n_dets) complex
    max_real = jnp.max(z.real, axis=-1, keepdims=True)
    return max_real.squeeze(-1) + jnp.log(
        jnp.sum(jnp.exp(z - max_real), axis=-1).astype(jnp.complex128)
    )


class LogSlaterDeterminant(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    restricted: bool = False
    generalized: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        return nk.models.Slater2nd(
            self.hilbert,
            generalized=self.generalized,
            restricted=self.restricted,
            param_dtype=dtype,
        )(n)


class LogNeuralJastrowSlater(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int = 32
    restricted: bool = False
    generalized: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64

        log_slater = nk.models.Slater2nd(
            self.hilbert,
            generalized=self.generalized,
            restricted=self.restricted,
            param_dtype=dtype,
        )(n)

        x = n.astype(dtype)
        j = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="jastrow_dense_0",
        )(x)
        j = jax.nn.tanh(j)
        j = nn.Dense(1, param_dtype=dtype, name="jastrow_out")(j).squeeze(-1)

        return log_slater + j


class LogNeuralBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int = 32

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)

        M = self.param(
            "M",
            nn.initializers.lecun_normal(),
            (n_orb, nfs),
            dtype,
        )

        x = n.astype(dtype)
        F = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_0",
        )(x)
        F = jax.nn.silu(F)
        F = nn.Dense(
            n_spin * n_orb * nfs,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(F)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        return _log_backflow(n, F, M, self.hilbert)


class LogNeuralBackflowDeep(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int = 64

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)

        M = self.param(
            "M",
            nn.initializers.lecun_normal(),
            (n_orb, nfs),
            dtype,
        )

        x = n.astype(dtype)
        F = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_0",
        )(x)
        F = jax.nn.silu(F)
        F = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_1",
        )(F)
        F = jax.nn.silu(F)
        F = nn.Dense(
            n_spin * n_orb * nfs,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(F)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        return _log_backflow(n, F, M, self.hilbert)


class LogNeuralJastrowBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int = 32

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)

        M = self.param(
            "M",
            nn.initializers.normal(stddev=0.1),
            (n_orb, nfs),
            dtype,
        )

        x = n.astype(dtype)

        h_bf = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_0",
        )(x)
        h_bf = jax.nn.silu(h_bf)
        F = nn.Dense(
            n_spin * n_orb * nfs,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(h_bf)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        h_j = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="jastrow_dense_0",
        )(x)
        h_j = jax.nn.silu(h_j)
        J = nn.Dense(1, param_dtype=dtype, name="jastrow_out")(h_j).squeeze(-1)

        return _log_backflow(n, F, M, self.hilbert) + J


class LogNeuralJastrowBackflowDeep(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int = 64

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)

        M = self.param(
            "M",
            nn.initializers.normal(stddev=0.1),
            (n_orb, nfs),
            dtype,
        )

        x = n.astype(dtype)

        h_bf = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_0",
        )(x)
        h_bf = jax.nn.silu(h_bf)
        h_bf = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="backflow_dense_1",
        )(h_bf)
        h_bf = jax.nn.silu(h_bf)
        F = nn.Dense(
            n_spin * n_orb * nfs,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(h_bf)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        h_j = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="jastrow_dense_0",
        )(x)
        h_j = jax.nn.silu(h_j)
        h_j = nn.Dense(
            self.hidden_units,
            param_dtype=dtype,
            name="jastrow_dense_1",
        )(h_j)
        h_j = jax.nn.silu(h_j)
        J = nn.Dense(1, param_dtype=dtype, name="jastrow_out")(h_j).squeeze(-1)

        return _log_backflow(n, F, M, self.hilbert) + J


# ---------------------------------------------------------------------------
# Transformer-based ansatzes
# ---------------------------------------------------------------------------

class LogTransformerBackflow(nn.Module):
    """Slater determinant with Transformer-generated backflow correction.

    log Ψ(n) = Σ_spin log det [M + F_θ(n)]_{occupied rows}

    where F_θ is produced by a Transformer acting on the occupation sequence.
    The output layer is zero-initialized so the initial state is a pure Slater det.
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)
        n_modes = self.hilbert.size

        M = self.param("M", nn.initializers.lecun_normal(), (n_orb, nfs), dtype)

        h = TransformerEncoder(
            n_tokens=2, seq_len=n_modes,
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, mlp_ratio=self.mlp_ratio,
            param_dtype=dtype, name="encoder",
        )(n.astype(jnp.int32))  # (..., n_modes, d_model)

        F = nn.Dense(
            n_spin * n_orb * nfs, param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(h)  # (..., n_modes, n_spin*n_orb*nfs)

        F = jnp.mean(F, axis=-2)  # (..., n_spin*n_orb*nfs)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        return _log_backflow(n, F, M, self.hilbert)


class LogTransformerJastrowBackflow(nn.Module):
    """Slater determinant with Transformer backflow + Jastrow factor.

    log Ψ(n) = Σ_spin log det [M + F_θ(n)]_{occupied} + J_θ(n)

    Both F and J are derived from the same Transformer encoder features.
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        n_orb = self.hilbert.n_orbitals
        n_spin = self.hilbert.n_spin_subsectors
        nfs = _shared_n_fermions_per_spin(self.hilbert)
        n_modes = self.hilbert.size

        M = self.param("M", nn.initializers.normal(stddev=0.1), (n_orb, nfs), dtype)

        h = TransformerEncoder(
            n_tokens=2, seq_len=n_modes,
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, mlp_ratio=self.mlp_ratio,
            param_dtype=dtype, name="encoder",
        )(n.astype(jnp.int32))  # (..., n_modes, d_model)

        # Backflow branch
        F = nn.Dense(
            n_spin * n_orb * nfs, param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(h)
        F = jnp.mean(F, axis=-2)
        F = F.reshape(F.shape[:-1] + (n_spin, n_orb, nfs))

        # Jastrow branch (reuse encoder features)
        J = nn.Dense(1, param_dtype=dtype, name="jastrow_out")(h)  # (..., n_modes, 1)
        J = jnp.sum(J, axis=-2).squeeze(-1)  # (...,)

        return _log_backflow(n, F, M, self.hilbert) + J


__all__ = [
    "LogNeuralBackflow",
    "LogNeuralBackflowDeep",
    "LogNeuralJastrowBackflow",
    "LogNeuralJastrowBackflowDeep",
    "LogNeuralJastrowSlater",
    "LogSlaterDeterminant",
    "LogTransformerBackflow",
    "LogTransformerJastrowBackflow",
]
