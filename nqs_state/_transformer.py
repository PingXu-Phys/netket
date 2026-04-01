"""Transformer building blocks in flax.linen for NQS ansatzes.

All modules support arbitrary leading batch dimensions:
input (..., L, d_model) → output (..., L, d_model).
"""
from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with batch-aware reshape."""
    d_model: int
    n_heads: int
    param_dtype: type = jnp.float64

    @nn.compact
    def __call__(self, x):
        # x: (..., L, d_model)
        head_dim = self.d_model // self.n_heads

        qkv = nn.Dense(
            3 * self.d_model, param_dtype=self.param_dtype, name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (..., L, d_model)

        # (..., L, d_model) → (..., L, n_heads, head_dim) → (..., n_heads, L, head_dim)
        def split_heads(t):
            return t.reshape(t.shape[:-1] + (self.n_heads, head_dim)).swapaxes(-3, -2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn = jnp.einsum("...qd,...kd->...qk", q, k) / math.sqrt(head_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("...qk,...kd->...qd", attn, v)

        # (..., n_heads, L, head_dim) → (..., L, d_model)
        out = out.swapaxes(-3, -2)
        out = out.reshape(out.shape[:-2] + (self.d_model,))

        return nn.Dense(
            self.d_model, param_dtype=self.param_dtype, name="o_proj",
        )(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → Attention → residual → LN → FFN → residual."""
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    param_dtype: type = jnp.float64

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=self.param_dtype, name="ln1")(x)
        y = MultiHeadSelfAttention(
            d_model=self.d_model, n_heads=self.n_heads,
            param_dtype=self.param_dtype, name="attn",
        )(y)
        x = x + y

        y = nn.LayerNorm(param_dtype=self.param_dtype, name="ln2")(x)
        y = nn.Dense(
            self.d_model * self.mlp_ratio,
            param_dtype=self.param_dtype, name="ffn_up",
        )(y)
        y = jax.nn.gelu(y)
        y = nn.Dense(
            self.d_model, param_dtype=self.param_dtype, name="ffn_down",
        )(y)
        return x + y


class TransformerEncoder(nn.Module):
    """Token embed → positional embed → N transformer blocks → LayerNorm.

    Input:  integer tokens of shape (..., seq_len)
    Output: feature vectors of shape (..., seq_len, d_model)
    """
    n_tokens: int       # number of distinct token types (2 for binary occupation)
    seq_len: int        # sequence length (= hilbert.size)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    param_dtype: type = jnp.float64

    @nn.compact
    def __call__(self, x_int):
        tok_emb = self.param(
            "token_embedding", nn.initializers.normal(stddev=0.02),
            (self.n_tokens, self.d_model), self.param_dtype,
        )
        pos_emb = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02),
            (self.seq_len, self.d_model), self.param_dtype,
        )

        h = tok_emb[x_int] + pos_emb  # (..., seq_len, d_model)

        for i in range(self.n_layers):
            h = TransformerBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio, param_dtype=self.param_dtype,
                name=f"block_{i}",
            )(h)

        return nn.LayerNorm(param_dtype=self.param_dtype, name="ln_final")(h)
