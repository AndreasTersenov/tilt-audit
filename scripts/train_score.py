#!/usr/bin/env python
"""Train a VP score net (noise prediction) on 64^2 GRFs — S-clean or S-mis(eps).

Fresh GRF batches every step (infinite data). t sampled log-uniform in (t+c)
to match the sampling grid's resolution profile. EMA params saved. Checkpoints
every 30 min and at the end (pickle: {params, ema, config, losses}).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import optax

from tilt_audit import misspec
from tilt_audit.fields import grid_to_z, make_basis, make_pk, unpack
from tilt_audit.scorenet import UNet

TF = 9.0
T_MIN = 1e-3
C_GRID = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.0,
                   help="spectral-slope contamination of the training prior")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    basis = make_basis(args.n)
    pk = make_pk(basis)
    Pz = jnp.asarray(misspec.contaminated_pz(basis, pk, args.eps)
                     if args.eps != 0.0 else grid_to_z(pk, basis))

    model = UNet()
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((2, args.n, args.n)), jnp.ones(2))
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"UNet params: {n_params/1e6:.2f}M, eps={args.eps}", flush=True)

    sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1)
    opt = optax.adam(sched)
    opt_state = opt.init(params)
    ema = params

    @jax.jit
    def train_step(params, opt_state, ema, key):
        kz, kt, ke = jax.random.split(key, 3)
        z = jnp.sqrt(Pz) * jax.random.normal(kz, (args.batch, Pz.shape[0]))
        x0 = unpack(z, basis)
        # log-uniform t in (T_MIN, TF) via the grid's change of variables
        u = jax.random.uniform(kt, (args.batch,))
        t = C_GRID * ((1.0 + TF / C_GRID) ** u - 1.0) + T_MIN
        alpha = jnp.exp(-t)[:, None, None]
        sig = jnp.sqrt(1.0 - jnp.exp(-2.0 * t))[:, None, None]
        noise = jax.random.normal(ke, x0.shape)
        xt = alpha * x0 + sig * noise

        def loss_fn(p):
            pred = model.apply(p, xt, t)
            return jnp.mean((pred - noise) ** 2)

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
                pickle.dump({"params": params, "ema": ema, "eps": args.eps,
                             "n": args.n, "step": step, "losses": losses}, f)
            t_last_ckpt = time.time()
    with out.open("wb") as f:
        pickle.dump({"params": params, "ema": ema, "eps": args.eps,
                     "n": args.n, "step": args.steps, "losses": losses}, f)
    print(f"done in {(time.time()-t0)/60:.1f} min; final loss "
          f"{float(loss):.5f}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
