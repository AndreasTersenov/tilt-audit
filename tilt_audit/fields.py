"""GRF prior N(0, C) on an n x n grid, C diagonal in Fourier space.

The canonical state representation is the *real diagonal basis* z: the unitary
2-D DFT coefficients repacked into n^2 real coordinates that are independent
Gaussians under the prior, z_k ~ N(0, Pz_k). Self-conjugate modes contribute
their (real) coefficient directly; each conjugate pair contributes
sqrt(2)*Re and sqrt(2)*Im of one representative, so variances stay P(k).

Every operator we care about tonight (prior covariance, Fourier-diagonal
smoothing A, OU dynamics) is diagonal in this basis, so samplers and metrics
decompose per coordinate while SMC weights/resampling stay global.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Basis:
    """Static index arrays mapping n x n complex DFT <-> d = n^2 real coords."""
    n: int
    sc: np.ndarray   # flat indices of self-conjugate modes
    pa: np.ndarray   # flat indices of pair representatives
    pb: np.ndarray   # flat indices of the conjugate partners
    kmag: np.ndarray # (n, n) |k| in fundamental units


def make_basis(n: int) -> Basis:
    idx = np.arange(n * n).reshape(n, n)
    conj_rows = (-np.arange(n)) % n
    conj_map = idx[np.ix_(conj_rows, conj_rows)].ravel()
    flat = idx.ravel()
    sc = np.where(flat == conj_map)[0]
    pa = np.where(flat < conj_map)[0]
    pb = conj_map[pa]
    freqs = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers
    kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)
    return Basis(n=n, sc=sc, pa=pa, pb=pb, kmag=kmag)


def make_pk(basis: Basis, ns: float = 0.96, kcut_frac: float = 0.7) -> np.ndarray:
    """Power spectrum on the (n, n) frequency grid, unit pixel variance.

    P(k) ~ max(k,1)^(ns-4) * exp(-(k/kcut)^2): a steep red cosmology-flavored
    slope with a UV cutoff; the DC mode gets the k=1 power so the field mean
    is a live (finite-variance) functional.
    """
    k = basis.kmag
    keff = np.maximum(k, 1.0)
    kcut = kcut_frac * (basis.n / 2)
    pk = keff ** (ns - 4.0) * np.exp(-((k / kcut) ** 2))
    pk = pk / pk.mean()  # unitary basis: pixel variance = mean_k P(k)
    return pk


def grid_to_z(field2d: np.ndarray, basis: Basis) -> np.ndarray:
    """Map a spectrum-like symmetric (n, n) grid quantity to its z-basis vector.

    Valid for any real function of |k| (P(k), a(k)): paired modes share values.
    """
    flat = field2d.ravel()
    return np.concatenate([flat[basis.sc], flat[basis.pa], flat[basis.pa]])


def pack(x: jnp.ndarray, basis: Basis) -> jnp.ndarray:
    """Pixel field(s) (..., n, n) -> real diagonal coords (..., d)."""
    n = basis.n
    X = jnp.fft.fft2(x) / n  # unitary
    Xf = X.reshape(*x.shape[:-2], n * n)
    sqrt2 = np.sqrt(2.0)
    return jnp.concatenate(
        [Xf[..., basis.sc].real,
         sqrt2 * Xf[..., basis.pa].real,
         sqrt2 * Xf[..., basis.pa].imag], axis=-1)


def unpack(z: jnp.ndarray, basis: Basis) -> jnp.ndarray:
    """Real diagonal coords (..., d) -> pixel field(s) (..., n, n)."""
    n = basis.n
    nsc, npa = len(basis.sc), len(basis.pa)
    zsc = z[..., :nsc]
    zre = z[..., nsc:nsc + npa]
    zim = z[..., nsc + npa:]
    Xf = jnp.zeros((*z.shape[:-1], n * n), dtype=jnp.complex128)
    Xf = Xf.at[..., basis.sc].set(zsc.astype(jnp.complex128))
    pair = (zre + 1j * zim) / np.sqrt(2.0)
    Xf = Xf.at[..., basis.pa].set(pair)
    Xf = Xf.at[..., basis.pb].set(jnp.conj(pair))
    return jnp.fft.ifft2(Xf.reshape(*z.shape[:-1], n, n) * n).real


def sample_prior_z(key: jax.Array, Pz: jnp.ndarray, shape=()) -> jnp.ndarray:
    """Draw z ~ N(0, diag(Pz)); shape prepends batch dims."""
    return jnp.sqrt(Pz) * jax.random.normal(key, (*shape, Pz.shape[0]))


def smoothing_operator(basis: Basis, ks_frac: float = 0.25) -> np.ndarray:
    """Fourier-diagonal Gaussian smoothing a(k) = exp(-(k/ks)^2 / 2), z-basis."""
    ks = ks_frac * basis.n  # e.g. 0.25*n = half the Nyquist frequency n/2
    a2d = np.exp(-0.5 * (basis.kmag / ks) ** 2)
    return grid_to_z(a2d, basis)


def band_masks(basis: Basis, n_bands: int = 3) -> list[np.ndarray]:
    """Boolean z-basis masks for log-spaced |k| bands (band-power functionals)."""
    kz = grid_to_z(basis.kmag, basis)
    edges = np.geomspace(1.0, basis.n / 2, n_bands + 1)
    masks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        masks.append((kz >= lo) & (kz < hi))
    return masks
