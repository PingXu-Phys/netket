from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from _transformer import TransformerEncoder


def _validate_pairing_sector(n_up: int, n_down: int) -> None:
    if n_up != n_down:
        raise ValueError(
            "This AGP implementation builds a square up-down pairing determinant, "
            f"so it requires n_up == n_down. Got n_up={n_up}, n_down={n_down}."
        )


def _signature_vector(n_sites: int, n_up: int, dtype) -> jax.Array:
    return jnp.concatenate(
        [jnp.ones(n_up, dtype=dtype), -jnp.ones(n_sites - n_up, dtype=dtype)]
    )


def _pair_matrix_from_factor(factor_matrix: jax.Array, n_up: int, dtype) -> jax.Array:
    sig = _signature_vector(factor_matrix.shape[-1], n_up, dtype)
    return (factor_matrix * sig) @ jnp.swapaxes(factor_matrix, -1, -2)


def _occupied_indices(occ_array: jax.Array, num_particles: int) -> jax.Array:
    return jnp.argsort(-occ_array, axis=-1)[..., :num_particles]


def _single_log_agp(
    pair_matrix_one: jax.Array, idx_up_one: jax.Array, idx_down_one: jax.Array
) -> jax.Array:
    sub_matrix = pair_matrix_one[idx_up_one[:, None], idx_down_one]
    sign, log_det = jnp.linalg.slogdet(sub_matrix)
    return log_det + jnp.log(sign.astype(jnp.complex128))


def _log_agp_from_pair_matrix(
    pair_matrix: jax.Array,
    x: jax.Array,
    n_sites: int,
    n_up: int,
    n_down: int,
) -> jax.Array:
    _validate_pairing_sector(n_up, n_down)

    up_occ = x[..., :n_sites]
    down_occ = x[..., n_sites:]
    idx_up = _occupied_indices(up_occ, n_up)
    idx_down = _occupied_indices(down_occ, n_down)

    if x.ndim == 1:
        return _single_log_agp(pair_matrix, idx_up, idx_down)

    if pair_matrix.ndim == 2:
        pair_matrix = jnp.broadcast_to(pair_matrix, x.shape[:-1] + pair_matrix.shape)

    return jax.vmap(_single_log_agp)(pair_matrix, idx_up, idx_down)


def _svd_factor_matrix(
    W_U: jax.Array, log_sigma: jax.Array, W_V: jax.Array | None = None,
) -> jax.Array:
    """从无约束参数重建 A = U @ diag(exp(log_sigma)) @ V^T。
    U, V 通过 QR 分解保证正交性。
    若 W_V 为 None（对称模式），则 V = U，即 A = U @ diag(exp(log_sigma)) @ U^T。"""
    U, _ = jnp.linalg.qr(W_U)
    V = U if W_V is None else jnp.linalg.qr(W_V)[0]
    sigma = jnp.exp(log_sigma)
    return U * sigma @ V.T


class LogSlaterDeterminant(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions

    @nn.compact
    def __call__(self, n):
        return nk.models.Slater2nd(
            self.hilbert, generalized=False, restricted=True, param_dtype=float
        )(n)


class SmoothAGPJastrow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        W_U = self.param(
            "W_U",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.constant(-2.0),
            (self.n_sites,),
            dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)
        pair_matrix = _pair_matrix_from_factor(A, self.n_up, dtype)
        log_agp = _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down
        )

        g = self.param("impurity_penalty", nn.initializers.constant(5.0), ())
        is_double_occupied = n[..., 0] * n[..., self.n_sites]
        log_jastrow = -g * is_double_occupied.astype(dtype)

        return log_agp + log_jastrow


class SmoothAGPNeuralJastrow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    hidden_units: int = 32
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        W_U = self.param(
            "W_U",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.constant(-2.0),
            (self.n_sites,),
            dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)
        pair_matrix = _pair_matrix_from_factor(A, self.n_up, dtype)
        log_agp = _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down
        )

        g = self.param("impurity_penalty", nn.initializers.constant(5.0), ())
        is_double_occupied = n[..., 0] * n[..., self.n_sites]

        y = x
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="jastrow_dense_0",
        )(y)
        y = jax.nn.silu(y)
        log_neural_jastrow = nn.Dense(
            1, param_dtype=dtype, name="jastrow_out",
        )(y)
        log_neural_jastrow = (
            jnp.squeeze(log_neural_jastrow, axis=-1)
            - g * is_double_occupied.astype(dtype)
        )

        return log_agp + log_neural_jastrow


class SmoothAGPNeuralJastrowDeep(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    hidden_units: int = 64
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        W_U = self.param(
            "W_U",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.constant(-2.0),
            (self.n_sites,),
            dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)
        pair_matrix = _pair_matrix_from_factor(A, self.n_up, dtype)
        log_agp = _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down
        )

        g = self.param("impurity_penalty", nn.initializers.constant(5.0), ())
        is_double_occupied = n[..., 0] * n[..., self.n_sites]

        y = x
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="jastrow_dense_0",
        )(y)
        y = jax.nn.silu(y)
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="jastrow_dense_1",
        )(y)
        y = jax.nn.silu(y)
        log_neural_jastrow = nn.Dense(
            1, param_dtype=dtype, name="jastrow_out",
        )(y)
        log_neural_jastrow = (
            jnp.squeeze(log_neural_jastrow, axis=-1)
            - g * is_double_occupied.astype(dtype)
        )

        return log_agp + log_neural_jastrow


class AGPBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    hidden_units: int = 32
    init_backflow_scale: float = 1.0e-2
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        # 基础 A 矩阵：SVD 参数化，log 域学习奇异值
        W_U = self.param(
            "W_U",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.constant(-2.0),
            (self.n_sites,),
            dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)

        # Backflow 网络：1 隐藏层
        y = x
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="backflow_dense_0",
        )(y)
        y = jax.nn.silu(y)

        # 输出层零初始化，初始时 delta_A = 0，等价于纯 AGP
        delta_A_flat = nn.Dense(
            self.n_sites * self.n_sites,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(y)
        delta_A = delta_A_flat.reshape(
            delta_A_flat.shape[:-1] + (self.n_sites, self.n_sites)
        )

        backflow_scale = self.param(
            "backflow_scale",
            nn.initializers.constant(self.init_backflow_scale),
            (),
        )

        A_eff = A + backflow_scale * delta_A
        pair_matrix = _pair_matrix_from_factor(A_eff, self.n_up, dtype)

        return _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down
        )


class AGPBackflowDeep(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    hidden_units: int = 64
    init_backflow_scale: float = 1.0e-2
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        # 基础 A 矩阵：SVD 参数化
        W_U = self.param(
            "W_U",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.constant(-2.0),
            (self.n_sites,),
            dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V",
            nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites),
            dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)

        # Backflow 网络：2 隐藏层
        y = x
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="backflow_dense_0",
        )(y)
        y = jax.nn.silu(y)
        y = nn.Dense(
            self.hidden_units, param_dtype=dtype, name="backflow_dense_1",
        )(y)
        y = jax.nn.silu(y)

        delta_A_flat = nn.Dense(
            self.n_sites * self.n_sites,
            param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(y)
        delta_A = delta_A_flat.reshape(
            delta_A_flat.shape[:-1] + (self.n_sites, self.n_sites)
        )

        backflow_scale = self.param(
            "backflow_scale",
            nn.initializers.constant(self.init_backflow_scale),
            (),
        )

        A_eff = A + backflow_scale * delta_A
        pair_matrix = _pair_matrix_from_factor(A_eff, self.n_up, dtype)

        return _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down
        )


class AGPTransformerBackflow(nn.Module):
    """AGP with Transformer-generated backflow correction on the factor matrix A.

    A_eff(n) = A_base + scale * ΔA_θ(n)

    where ΔA_θ is produced by a Transformer acting on the occupation sequence.
    The output layer is zero-initialized so the initial state is a pure AGP.
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_sites: int
    n_up: int
    n_down: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_backflow_scale: float = 1.0e-2
    symmetric: bool = False

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        x = n.astype(dtype)

        # Base A matrix (SVD parameterization)
        W_U = self.param(
            "W_U", nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites), dtype,
        )
        log_sigma = self.param(
            "log_sigma", nn.initializers.constant(-2.0),
            (self.n_sites,), dtype,
        )
        W_V = None if self.symmetric else self.param(
            "W_V", nn.initializers.orthogonal(),
            (self.n_sites, self.n_sites), dtype,
        )
        A = _svd_factor_matrix(W_U, log_sigma, W_V)

        # Transformer backflow
        n_modes = self.hilbert.size
        h = TransformerEncoder(
            n_tokens=2, seq_len=n_modes,
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, mlp_ratio=self.mlp_ratio,
            param_dtype=dtype, name="encoder",
        )(n.astype(jnp.int32))  # (..., n_modes, d_model)

        # Pool over sequence and project to delta_A
        delta_A_flat = nn.Dense(
            self.n_sites * self.n_sites, param_dtype=dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="backflow_out",
        )(h)  # (..., n_modes, n_sites^2)

        delta_A_flat = jnp.mean(delta_A_flat, axis=-2)  # (..., n_sites^2)
        delta_A = delta_A_flat.reshape(
            delta_A_flat.shape[:-1] + (self.n_sites, self.n_sites)
        )

        backflow_scale = self.param(
            "backflow_scale",
            nn.initializers.constant(self.init_backflow_scale), (),
        )

        A_eff = A + backflow_scale * delta_A
        pair_matrix = _pair_matrix_from_factor(A_eff, self.n_up, dtype)

        return _log_agp_from_pair_matrix(
            pair_matrix, x, self.n_sites, self.n_up, self.n_down,
        )


__all__ = [
    "AGPBackflow",
    "AGPBackflowDeep",
    "AGPTransformerBackflow",
    "LogSlaterDeterminant",
    "SmoothAGPJastrow",
    "SmoothAGPNeuralJastrow",
    "SmoothAGPNeuralJastrowDeep",
]
