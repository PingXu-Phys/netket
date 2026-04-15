from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

try:
    from ._transformer_official import TransformerBlock
except ImportError:
    from _transformer_official import TransformerBlock


@dataclass(frozen=True)
class SpinFermionLayout:
    """Small, hashable descriptor of a joint spin-fermion Hilbert layout.

    NetKet raw fermion samples are interpreted here in the same order used by
    `KondoHeisenbergChain.py`:
        [down-spin occupations | up-spin occupations]
    followed by any site-aligned spin blocks.
    """

    n_sites: int
    fermion_size: int
    spin_block_sizes: tuple[int, ...]
    n_fermions: int
    n_up: int | None = None
    n_dn: int | None = None

    @property
    def n_spin_blocks(self) -> int:
        return len(self.spin_block_sizes)

    @property
    def n_channels(self) -> int:
        return 2 + self.n_spin_blocks

    @property
    def uses_block_determinant(self) -> bool:
        return self.n_up is not None and self.n_dn is not None


def _resolve_raw_hilbert(system_or_hilbert: object) -> object:
    return getattr(system_or_hilbert, 'joint_hilbert', system_or_hilbert)


def resolve_spin_fermion_layout(system_or_hilbert: object) -> SpinFermionLayout:
    """Resolve the static layout needed by the spin-fermion Transformer ansatzes."""
    raw_hilbert = _resolve_raw_hilbert(system_or_hilbert)

    if isinstance(raw_hilbert, nk.hilbert.SpinOrbitalFermions):
        fermion_hilbert = raw_hilbert
        spin_block_sizes: tuple[int, ...] = ()
    elif hasattr(raw_hilbert, 'subspaces'):
        subspaces = tuple(raw_hilbert.subspaces)
        if not subspaces or not isinstance(subspaces[0], nk.hilbert.SpinOrbitalFermions):
            raise ValueError(
                'Expected a TensorHilbert whose first subspace is SpinOrbitalFermions.'
            )
        fermion_hilbert = subspaces[0]
        spin_block_sizes = tuple(int(sub_hi.size) for sub_hi in subspaces[1:])
    else:
        raise TypeError(
            'Expected a SpinOrbitalFermions, a TensorHilbert, or a system exposing joint_hilbert.'
        )

    if fermion_hilbert.n_spin_subsectors != 2:
        raise ValueError(
            'These Transformer+NNBF interfaces currently assume exactly two fermionic spin subsectors.'
        )

    n_sites = int(fermion_hilbert.n_orbitals)
    for idx, size in enumerate(spin_block_sizes):
        if size != n_sites:
            raise ValueError(
                'Every non-fermion subspace must be site-aligned with the fermion chain. '
                f'Got size={size} for subspace {idx + 1}, expected {n_sites}.'
            )

    raw_per_spin = tuple(getattr(fermion_hilbert, 'n_fermions_per_spin', (None, None)))
    if len(raw_per_spin) == 2 and raw_per_spin[0] is not None and raw_per_spin[1] is not None:
        n_dn = int(raw_per_spin[0])
        n_up = int(raw_per_spin[1])
        n_fermions = n_dn + n_up
    elif getattr(fermion_hilbert, 'n_fermions', None) is not None:
        n_fermions = int(fermion_hilbert.n_fermions)
        n_dn = None
        n_up = None
    else:
        raise ValueError(
            'These Transformer+NNBF interfaces require either a fixed total fermion number '
            'or fixed fermion numbers in both spin subsectors.'
        )

    return SpinFermionLayout(
        n_sites=n_sites,
        fermion_size=int(fermion_hilbert.size),
        spin_block_sizes=spin_block_sizes,
        n_fermions=n_fermions,
        n_up=n_up,
        n_dn=n_dn,
    )


def _split_joint_configuration(n: jax.Array, layout: SpinFermionLayout):
    n_ferm = n[..., : layout.fermion_size]
    spin_blocks = []
    offset = layout.fermion_size
    for size in layout.spin_block_sizes:
        spin_blocks.append(n[..., offset : offset + size])
        offset += size
    return n_ferm, tuple(spin_blocks)


def _fermion_site_tokens(n_ferm: jax.Array, n_sites: int):
    occ_dn = n_ferm[..., :n_sites].astype(jnp.int32)
    occ_up = n_ferm[..., n_sites : 2 * n_sites].astype(jnp.int32)
    site_token = occ_up + 2 * occ_dn
    return site_token, occ_up, occ_dn


def _binary_spin_tokens(x: jax.Array) -> jax.Array:
    return (x > 0).astype(jnp.int32)


def _base_orbitals_param(module: nn.Module, name: str, n_rows: int, n_cols: int, dtype):
    if n_cols == 0:
        return jnp.zeros((n_rows, 0), dtype=dtype)
    return module.param(name, nn.initializers.lecun_normal(), (n_rows, n_cols), dtype)


def _dense_zero_init(h: jax.Array, features: int, *, dtype, name: str):
    if features == 0:
        return jnp.zeros(h.shape[:-1] + (0,), dtype=dtype)
    return nn.Dense(
        features,
        dtype=dtype,
        param_dtype=dtype,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        name=name,
    )(h)


def _run_transformer_stack(
    h: jax.Array,
    *,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mlp_ratio: int,
    dtype,
):
    for idx in range(n_layers):
        h = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            param_dtype=dtype,
            name=f'block_{idx}',
        )(h)
    return nn.LayerNorm(dtype=dtype, param_dtype=dtype, name='ln_final')(h)


def _single_log_block_backflow(
    n_ferm_one: jax.Array,
    delta_up_one: jax.Array,
    delta_dn_one: jax.Array,
    base_up: jax.Array,
    base_dn: jax.Array,
    layout: SpinFermionLayout,
) -> jax.Array:
    n_sites = layout.n_sites
    log_value = jnp.array(0.0 + 0.0j, dtype=jnp.complex128)

    if layout.n_dn:
        occ_dn = jnp.nonzero(n_ferm_one[:n_sites], size=layout.n_dn)[0]
        log_value = log_value + nk.jax.logdet_cmplx((base_dn + delta_dn_one)[occ_dn])

    if layout.n_up:
        occ_up = jnp.nonzero(n_ferm_one[n_sites : 2 * n_sites], size=layout.n_up)[0]
        log_value = log_value + nk.jax.logdet_cmplx((base_up + delta_up_one)[occ_up])

    return log_value


def _log_block_backflow(
    n_ferm: jax.Array,
    delta_up: jax.Array,
    delta_dn: jax.Array,
    base_up: jax.Array,
    base_dn: jax.Array,
    layout: SpinFermionLayout,
) -> jax.Array:
    if n_ferm.ndim == 1:
        return _single_log_block_backflow(n_ferm, delta_up, delta_dn, base_up, base_dn, layout)

    batch_shape = n_ferm.shape[:-1]
    n_flat = n_ferm.reshape((-1, n_ferm.shape[-1]))
    up_flat = delta_up.reshape((-1,) + delta_up.shape[-2:])
    dn_flat = delta_dn.reshape((-1,) + delta_dn.shape[-2:])

    log_flat = jax.vmap(
        lambda n_one, up_one, dn_one: _single_log_block_backflow(
            n_one, up_one, dn_one, base_up, base_dn, layout
        )
    )(n_flat, up_flat, dn_flat)

    return log_flat.reshape(batch_shape)


def _single_log_generalized_backflow(
    n_ferm_one: jax.Array,
    delta_up_one: jax.Array,
    delta_dn_one: jax.Array,
    base_all: jax.Array,
    layout: SpinFermionLayout,
) -> jax.Array:
    phi_all = base_all + jnp.concatenate([delta_dn_one, delta_up_one], axis=0)
    occupied = jnp.nonzero(n_ferm_one, size=layout.n_fermions)[0]
    return nk.jax.logdet_cmplx(phi_all[occupied])


def _log_generalized_backflow(
    n_ferm: jax.Array,
    delta_up: jax.Array,
    delta_dn: jax.Array,
    base_all: jax.Array,
    layout: SpinFermionLayout,
) -> jax.Array:
    if n_ferm.ndim == 1:
        return _single_log_generalized_backflow(n_ferm, delta_up, delta_dn, base_all, layout)

    batch_shape = n_ferm.shape[:-1]
    n_flat = n_ferm.reshape((-1, n_ferm.shape[-1]))
    up_flat = delta_up.reshape((-1,) + delta_up.shape[-2:])
    dn_flat = delta_dn.reshape((-1,) + delta_dn.shape[-2:])

    log_flat = jax.vmap(
        lambda n_one, up_one, dn_one: _single_log_generalized_backflow(
            n_one, up_one, dn_one, base_all, layout
        )
    )(n_flat, up_flat, dn_flat)

    return log_flat.reshape(batch_shape)


def _backflow_feature_counts(layout: SpinFermionLayout) -> tuple[int, int]:
    if layout.uses_block_determinant:
        return int(layout.n_up), int(layout.n_dn)

    return layout.n_fermions, layout.n_fermions


def _evaluate_logdet(
    *,
    n_ferm: jax.Array,
    delta_up: jax.Array,
    delta_dn: jax.Array,
    layout: SpinFermionLayout,
    module: nn.Module,
    dtype,
):
    if layout.uses_block_determinant:
        base_up = _base_orbitals_param(module, 'base_up', layout.n_sites, int(layout.n_up), dtype)
        base_dn = _base_orbitals_param(module, 'base_dn', layout.n_sites, int(layout.n_dn), dtype)
        return _log_block_backflow(n_ferm, delta_up, delta_dn, base_up, base_dn, layout)

    base_all = _base_orbitals_param(module, 'base_all', 2 * layout.n_sites, layout.n_fermions, dtype)
    return _log_generalized_backflow(n_ferm, delta_up, delta_dn, base_all, layout)


class LogSiteTokenTransformerNNBF(nn.Module):
    """Scheme A: site token embedding with adaptive determinant backend.

    Input path:
    - split the joint sample into `(fermion block, spin blocks)`;
    - compress `(n_up[i], n_dn[i])` into one four-state site token;
    - add site embeddings and local-spin embeddings;
    - run a site-level Transformer of length `N`.

    Output path:
    - if the Hilbert fixes `n_up` and `n_dn`, emit a blocked determinant with
      independently sized up/down backflow matrices;
    - if the Hilbert fixes only the total fermion number, emit one generalized
      determinant over the full spin-orbital set, compatible with Kondo spin flips.
    """

    layout: SpinFermionLayout
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        layout = self.layout
        up_features, dn_features = _backflow_feature_counts(layout)
        n_ferm, spin_blocks = _split_joint_configuration(n, layout)
        site_token, _, _ = _fermion_site_tokens(n_ferm, layout.n_sites)

        fermion_embedding = self.param(
            'fermion_embedding',
            nn.initializers.normal(stddev=0.02),
            (4, self.d_model),
            dtype,
        )
        site_embedding = self.param(
            'site_embedding',
            nn.initializers.normal(stddev=0.02),
            (layout.n_sites, self.d_model),
            dtype,
        )

        h = fermion_embedding[site_token] + site_embedding

        for block_id, spin_block in enumerate(spin_blocks):
            spin_embedding = self.param(
                f'spin_embedding_{block_id}',
                nn.initializers.normal(stddev=0.02),
                (2, self.d_model),
                dtype,
            )
            h = h + spin_embedding[_binary_spin_tokens(spin_block)]

        h = _run_transformer_stack(
            h,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_ratio=self.mlp_ratio,
            dtype=dtype,
        )

        delta_up = _dense_zero_init(h, up_features, dtype=dtype, name='backflow_up')
        delta_dn = _dense_zero_init(h, dn_features, dtype=dtype, name='backflow_dn')
        return _evaluate_logdet(
            n_ferm=n_ferm,
            delta_up=delta_up,
            delta_dn=delta_dn,
            layout=layout,
            module=self,
            dtype=dtype,
        )


class LogSpinChannelTransformerNNBF(nn.Module):
    """Scheme B: channel-split embedding with adaptive determinant backend.

    Input path:
    - keep `down`, `up`, and every local-spin chain as separate site-aligned channels;
    - add one shared site embedding and one channel embedding to every token;
    - run a Transformer over the flattened `(site, channel)` sequence;
    - fuse channels back into one site feature.

    Output path matches `LogSiteTokenTransformerNNBF`: blocked determinant in fixed
    spin sectors with independent up/down widths, generalized determinant in
    fixed-total-fermion sectors.
    """

    layout: SpinFermionLayout
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, n):
        dtype = jnp.float64
        layout = self.layout
        up_features, dn_features = _backflow_feature_counts(layout)
        n_ferm, spin_blocks = _split_joint_configuration(n, layout)
        _, occ_up, occ_dn = _fermion_site_tokens(n_ferm, layout.n_sites)

        site_embedding = self.param(
            'site_embedding',
            nn.initializers.normal(stddev=0.02),
            (layout.n_sites, self.d_model),
            dtype,
        )
        channel_embedding = self.param(
            'channel_embedding',
            nn.initializers.normal(stddev=0.02),
            (layout.n_channels, self.d_model),
            dtype,
        )
        fermion_value_embedding = self.param(
            'fermion_value_embedding',
            nn.initializers.normal(stddev=0.02),
            (2, self.d_model),
            dtype,
        )

        channel_features = [
            fermion_value_embedding[occ_dn] + site_embedding + channel_embedding[0],
            fermion_value_embedding[occ_up] + site_embedding + channel_embedding[1],
        ]

        for block_id, spin_block in enumerate(spin_blocks):
            spin_value_embedding = self.param(
                f'spin_value_embedding_{block_id}',
                nn.initializers.normal(stddev=0.02),
                (2, self.d_model),
                dtype,
            )
            spin_token = _binary_spin_tokens(spin_block)
            channel_features.append(
                spin_value_embedding[spin_token] + site_embedding + channel_embedding[2 + block_id]
            )

        h = jnp.stack(channel_features, axis=-2)
        batch_shape = h.shape[:-3]
        h = h.reshape(batch_shape + (layout.n_sites * layout.n_channels, self.d_model))

        h = _run_transformer_stack(
            h,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_ratio=self.mlp_ratio,
            dtype=dtype,
        )

        h = h.reshape(batch_shape + (layout.n_sites, layout.n_channels, self.d_model))
        site_features = h.reshape(batch_shape + (layout.n_sites, layout.n_channels * self.d_model))
        site_features = nn.Dense(
            self.d_model,
            dtype=dtype,
            param_dtype=dtype,
            name='site_project',
        )(site_features)
        site_features = jax.nn.gelu(site_features)
        site_features = nn.LayerNorm(
            dtype=dtype,
            param_dtype=dtype,
            name='site_project_ln',
        )(site_features)

        delta_up = _dense_zero_init(site_features, up_features, dtype=dtype, name='backflow_up')
        delta_dn = _dense_zero_init(site_features, dn_features, dtype=dtype, name='backflow_dn')
        return _evaluate_logdet(
            n_ferm=n_ferm,
            delta_up=delta_up,
            delta_dn=delta_dn,
            layout=layout,
            module=self,
            dtype=dtype,
        )


__all__ = [
    'LogSiteTokenTransformerNNBF',
    'LogSpinChannelTransformerNNBF',
    'SpinFermionLayout',
    'resolve_spin_fermion_layout',
]