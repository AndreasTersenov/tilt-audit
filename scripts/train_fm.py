#!/usr/bin/env python
"""Conditional FLOW-MATCHING net on the kappa substrate (the JADE-class
column: amortized, observation-conditioned, sampled by deterministic ODE).

Construction (P-20260705d):
  pairs   g ~ GRF prior -> kappa(g);  y = az * pack(kappa) + S_OBS * xi
          (y fed to the net as a pixel map, exactly the A3 conditioning)
  path    x_t = (1 - t) * kappa + t * eps,  t ~ U(0,1)   (linear interpolant)
  target  v = eps - kappa;  loss ||v_hat(x_t, y, t) - v||^2
Sampling integrates dx/dt = v from t=1 (noise) to t=0 with a fixed-step Euler
ODE — NFE is the budget knob under test. Reuses UNetCond unchanged.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TILT_AUDIT_X64", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import optax

from tilt_audit import lognormal
from tilt_audit.fields import (grid_to_z, make_basis, make_pk,
                               smoothing_operator, pack, unpack)
from tilt_audit.scorenet import UNetCond

S_OBS = 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--chs", default="32,64,128")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lam = lognormal.default_lambda() if args.lam is None else args.lam
    basis = make_basis(args.n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis), dtype=jnp.float32)
    az = jnp.asarray(smoothing_operator(basis), dtype=jnp.float32)

    chs = tuple(int(c) for c in args.chs.split(","))
    model = UNetCond(chs=chs)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((2, args.n, args.n)),
                        jnp.zeros((2, args.n, args.n)), jnp.ones(2))
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"UNetCond params: {n_params/1e6:.2f}M, lam={lam:.4f} (flow matching)",
          flush=True)

    sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1)
    opt = optax.adam(sched)
    opt_state = opt.init(params)
    ema = params

    @jax.jit
    def train_step(params, opt_state, ema, key):
        kz, ky, kt, ke = jax.random.split(key, 4)
        z = jnp.sqrt(Pz) * jax.random.normal(kz, (args.batch, Pz.shape[0]))
        x0 = lognormal.kappa(unpack(z, basis), lam)          # kappa maps
        yz = az * pack(x0, basis) + S_OBS * jax.random.normal(
            ky, (args.batch, Pz.shape[0]))
        y_map = unpack(yz, basis)
        t = jax.random.uniform(kt, (args.batch,))
        eps = jax.random.normal(ke, x0.shape)
        xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * eps
        v_target = eps - x0

        def loss_fn(p):
            v = model.apply(p, xt, y_map, t)
            return jnp.mean((v - v_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema = jax.tree.map(lambda e, q: 0.999 * e + 0.001 * q, ema, params)
        return params, opt_state, ema, loss

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    t_last_ckpt = time.time()
    t0 = time.time()
    for step in range(args.steps):
        key, sub = jax.random.split(key)
        params, opt_state, ema, loss = train_step(params, opt_state, ema, sub)
        if step % 500 == 0:
            losses.append({"step": step, "loss": float(loss)})
            print(f"step {step} loss {float(loss):.5f} "
                  f"({(time.time()-t0):.0f}s)", flush=True)
        if time.time() - t_last_ckpt > 1800:
            with out.open("wb") as f:
                pickle.dump({"params": params, "ema": ema, "lam": lam,
                             "n": args.n, "chs": chs, "kind": "fm_cond",
                             "step": step, "losses": losses}, f)
            t_last_ckpt = time.time()
    with out.open("wb") as f:
        pickle.dump({"params": params, "ema": ema, "lam": lam, "n": args.n,
                     "chs": chs, "kind": "fm_cond", "step": args.steps,
                     "losses": losses}, f)
    print(f"done in {(time.time()-t0)/60:.1f} min; final loss "
          f"{float(loss):.5f}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
