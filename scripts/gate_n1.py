#!/usr/bin/env python
"""Gate T-N1: the conditional net is actually USING y (wiring check).

After the 2k-step warmup, the net's denoiser at an uninformative x_t must
reconstruct the posterior mean from y alone. We evaluate at t=2.0 — there
x_t's signal fraction alpha^2 P/(sig^2) is ~2% (x_t nearly pure noise) while
x0hat = (x_t - sig*eps)/alpha stays numerically sane (alpha=0.135; at t=tf
the 1/alpha = 8000x amplification would drown the check). Criterion:
MSE(x0hat_net, mu*) <= MSE(prior mean 0, mu*) / 5 on held-out pairs, mu* the
exact Wiener posterior mean at b = s^2 (the trained Bayes point).

Note: the plan says "at low noise"; at low t x_t contains x0 itself and the
gate would pass without any y-wiring — high-t is where the test is pure.
Interpretation logged; the intent ("it is actually using y") is what's tested.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from tilt_audit import diffusion, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, unpack, pack, smoothing_operator
from tilt_audit.scorenet import UNetCond

T_EVAL = 2.0
M_HELDOUT = 64
S_OBS = 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/cond_warmup.pkl")
    p.add_argument("--params", default="params", choices=["params", "ema"])
    args = p.parse_args()

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    n = ck["n"]
    model = UNetCond(chs=tuple(ck["chs"]))
    params = ck[args.params]

    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis), dtype=jnp.float32)
    az = jnp.asarray(smoothing_operator(basis), dtype=jnp.float32)
    b = S_OBS**2

    key = jax.random.PRNGKey(31337)  # held out from training stream (42)
    kz, ky, ke = jax.random.split(key, 3)
    z0 = jnp.sqrt(Pz) * jax.random.normal(kz, (M_HELDOUT, Pz.shape[0]))
    y = az * z0 + S_OBS * jax.random.normal(ky, z0.shape)
    mu_star, _ = tilt.posterior_params(Pz, az, y, b)  # (M, d) per-y mean

    t = jnp.full(M_HELDOUT, T_EVAL, dtype=jnp.float32)
    alpha = jnp.exp(-T_EVAL)
    sig = jnp.sqrt(1.0 - jnp.exp(-2.0 * T_EVAL))
    x0 = unpack(z0, basis)
    xt = alpha * x0 + sig * jax.random.normal(ke, x0.shape)
    y_map = unpack(y, basis)
    eps = model.apply(params, xt, y_map, t)
    x0hat_z = pack((xt - sig * eps) / alpha, basis)

    mse_net = float(jnp.mean(jnp.sum((x0hat_z - mu_star) ** 2, axis=-1)))
    mse_prior = float(jnp.mean(jnp.sum(mu_star ** 2, axis=-1)))
    ratio = mse_prior / mse_net
    verdict = "PASS" if ratio >= 5.0 else "FAIL"
    print(f"T-N1 {verdict}: MSE(prior-mean baseline)/MSE(net) = {ratio:.2f} "
          f"(need >=5); mse_net={mse_net:.4g} mse_prior={mse_prior:.4g} "
          f"ckpt={args.ckpt} step={ck['step']}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
