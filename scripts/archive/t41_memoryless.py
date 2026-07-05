#!/usr/bin/env python
"""T4.1 — memoryless-schedule theorem (2409.08861) tested exactly on the oracle.

Their generative family, mapped to our VP-OU reverse time tau (t = tf - tau):
    dX = [X + (sig2/2 + eta) s(X,t)] dtau + sig dW,   eta == 1 here,
so sig2 = 2 is THE memoryless schedule (= our standard reverse SDE), and any
sig2 < 2 keeps the base marginals identical but leaves X_1 correlated with
the initialization X_0.

Reward fine-tuning as SOC clamps the initial marginal, so the fine-tuned
model's sample law is (their eq. 24, per mode, everything Gaussian):

    p*(x1) = int p(x0) p_sig(x1|x0) e^{r(x1)/beta} / Z(x0) dx0,
    Z(x0)  = E[e^{r/beta} | x0].

We integrate the per-mode transition (c, v) of the sig-schedule SDE, assemble
p* in closed form, and report exact W2/KL to the true tilt sigma* per
(sig2, tilt strength). Theorem's prediction: bias = 0 at sig2 = 2 (built-in
control), growing as sig2 -> 0. No SGD, no MC — the theorem's quantitative
content on the oracle.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import jax
import jax.numpy as jnp

from tilt_audit import diffusion, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def transition_cv(Pz, sig2, T=4096, tf=TF):
    """Per-mode (c, v): x_1 | x_0 ~ N(c x_0, v) under the sig2-schedule base
    SDE with exact score, integrated on the log grid (midpoint exponential)."""
    ts = diffusion.time_grid(T, tf)
    c = np.ones_like(Pz)
    v = np.zeros_like(Pz)
    for i in range(T):
        t, t_next = ts[i], ts[i + 1]
        dt = t - t_next
        tm = t - 0.5 * dt
        Vt = np.asarray(diffusion.marginal_var(tm, Pz))
        a = 1.0 - (sig2 / 2.0 + 1.0) / Vt  # drift coef: x + (sig2/2+eta) s
        phi = np.exp(a * dt)
        v = phi**2 * v + np.where(np.abs(a) > 1e-12,
                                  (phi**2 - 1.0) / (2.0 * a), dt) * sig2
        c = phi * c
    return c, v


def soc_fixed_point(c, v, az, y, b, V0):
    """Per-mode Gaussian p*(x1) of SOC fine-tuning with clamped init.

    The base process initializes from its true tf-marginal N(0, V0) (V0 -> 1
    only as tf -> inf; at short horizons the difference is large). Exponent:
      -x0^2/(2 V0) - (x1 - c x0)^2/(2v) - (a x1 - y)^2/(2b)
      + (a c x0 - y)^2 / (2(b + a^2 v))          [ = +log Z(x0) removed ]
    Marginalize x0 analytically (2x2 quadratic form).
    """
    D = b + az**2 * v
    # x0 quadratic coefficients
    A00 = 1.0 / V0 + c**2 / v - (az * c) ** 2 / D
    A01 = -c / v                     # cross term x0 x1 coefficient (times 1)
    lin0 = -(az * c) * y / D         # linear term in x0: (acx0 - y)^2 -> -2acy x0
    # note signs: exponent = -1/2 A00 x0^2 - A01 x0 x1 - lin0 x0 - 1/2 A11 x1^2 - lin1 x1
    A11 = 1.0 / v + az**2 / b
    lin1 = -az * y / b
    # integrate x0: x1 precision gets -A01^2/A00, linear gets +A01 lin0 / A00
    prec = A11 - A01**2 / A00
    linr = lin1 - A01 * lin0 / A00
    var = 1.0 / prec
    mean = -linr * var
    return mean, var


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sig2s", default="2.0,1.0,0.5,0.1")
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--tf", type=float, default=TF)
    p.add_argument("--out", default="results/t41_memoryless.jsonl")
    args = p.parse_args()

    basis = make_basis(args.n)
    Pz = np.asarray(grid_to_z(make_pk(basis), basis))
    az = np.asarray(smoothing_operator(basis))
    y = np.asarray(tilt.make_observation(jax.random.PRNGKey(Y_KEY),
                                         jnp.asarray(Pz), jnp.asarray(az),
                                         S_OBS)[0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for shift in [float(s) for s in args.shifts.split(",")]:
        b = float(tilt.calibrate_b(jnp.asarray(Pz), jnp.asarray(az),
                                   jnp.asarray(y), target_shift=shift))
        mu, Sig = tilt.posterior_params(jnp.asarray(Pz), jnp.asarray(az),
                                        jnp.asarray(y), b)
        mu, Sig = np.asarray(mu), np.asarray(Sig)
        for sig2 in [float(s) for s in args.sig2s.split(",")]:
            t0 = time.time()
            c, v = transition_cv(Pz, sig2, tf=args.tf)
            V0 = np.asarray(diffusion.marginal_var(args.tf, jnp.asarray(Pz)))
            marg_err = float(np.abs(c**2 * V0 + v - Pz).max() / Pz.max())
            mean, var = soc_fixed_point(c, v, az, y, b, V0)
            w2 = float(np.sqrt(np.sum((mean - mu) ** 2
                                      + (np.sqrt(var) - np.sqrt(Sig)) ** 2)))
            kl = float(0.5 * np.sum(var / Sig + (mean - mu) ** 2 / Sig
                                    - 1.0 + np.log(Sig / var)))
            row = dict(dim=args.n, shift=shift, b=b, sig2=sig2, tf=args.tf, w2=w2, kl=kl,
                       marg_err=marg_err,
                       c_max=float(np.abs(c).max()),
                       tag="exploratory", tier="T4.1",
                       wall=round(time.time() - t0, 2),
                       ts=time.strftime("%H:%M:%S"))
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[t41] tf={args.tf} shift={shift} sig2={sig2}: W2={w2:.4g} "
                  f"KL={kl:.4g} c_max={row['c_max']:.3g} marg_err={marg_err:.2e}",
                  flush=True)


if __name__ == "__main__":
    main()
