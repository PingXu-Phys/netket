"""Transformer building blocks using flax.linen for NQS ansatzes.

All modules support arbitrary leading batch dimensions:
input (..., L, d_model) -> output (..., L, d_model).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN -> SelfAttention -> residual -> LN -> FFN -> residual."""

    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=self.param_dtype, name="ln1")(x)
        y = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
            param_dtype=self.param_dtype,
            name="attn",
        )(y)
        x = x + y

        y = nn.LayerNorm(param_dtype=self.param_dtype, name="ln2")(x)
        y = nn.Dense(self.d_model * self.mlp_ratio, param_dtype=self.param_dtype, name="ffn_up")(y)
        y = jax.nn.gelu(y)
        y = nn.Dense(self.d_model, param_dtype=self.param_dtype, name="ffn_down")(y)
        return x + y


class TransformerEncoder(nn.Module):
    """Token embed -> positional embed -> N transformer blocks -> LayerNorm.

    Input:  integer tokens of shape (..., seq_len)
    Output: feature vectors of shape (..., seq_len, d_model)
    """

    n_tokens: int       # number of distinct token types (2 for binary occupation)
    seq_len: int        # sequence length (= hilbert.size)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x_int):
        tok_emb = self.param(
            "token_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.n_tokens, self.d_model),
            self.param_dtype,
        )
        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.seq_len, self.d_model),
            self.param_dtype,
        )

        h = tok_emb[x_int] + pos_emb

        for i in range(self.n_layers):
            h = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                name=f"block_{i}",
            )(h)

        return nn.LayerNorm(param_dtype=self.param_dtype, name="ln_final")(h)
