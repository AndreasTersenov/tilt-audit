"""Small UNet score network (noise prediction) for 64x64 GRFs, flax.linen.

Deliberately generic architecture — spatial convs, no Fourier-diagonal
inductive bias baked in: the point of T2 is realistic score error, not a
network that trivially nails the analytic answer. ~1.5M params.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def time_embedding(t, dim=64):
    """Sinusoidal embedding of log-time (t spans ~4 decades on the VP path)."""
    logt = jnp.log(t + 1e-3)
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, half))
    ang = logt[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class ConvBlock(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x, temb):
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3))(h)
        # FiLM-style time conditioning
        scale_shift = nn.Dense(2 * self.ch)(nn.silu(temb))
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        h = h * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3))(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1))(x)
        return x + h


class UNet(nn.Module):
    chs: tuple = (32, 64, 128)

    @nn.compact
    def __call__(self, x, t):
        # x: (B, n, n) -> (B, n, n, 1)
        h = x[..., None]
        temb = nn.Dense(128)(time_embedding(t))
        h = nn.Conv(self.chs[0], (3, 3))(h)
        skips = []
        for ch in self.chs:
            h = ConvBlock(ch)(h, temb)
            skips.append(h)
            h = nn.avg_pool(h, (2, 2), strides=(2, 2))
        h = ConvBlock(self.chs[-1])(h, temb)
        for ch, skip in zip(reversed(self.chs), reversed(skips)):
            B, H, W, C = h.shape
            h = jax.image.resize(h, (B, H * 2, W * 2, C), "nearest")
            h = jnp.concatenate([h, skip], axis=-1)
            h = ConvBlock(ch)(h, temb)
        eps = nn.Conv(1, (3, 3))(h)
        return eps[..., 0]  # (B, n, n) noise prediction
