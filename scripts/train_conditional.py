#!/usr/bin/env python
"""Train the CONDITIONAL VP score net for the amortized arm (A3).

Pairs are free: z0 ~ prior, y = a z0 + s eps per sample (infinite paired
data), y fed to the net as a pixel map (unpack(y)). Same DSM loss, EMA,
30-min checkpoints as train_score.py. fp32 (defaults TILT_AUDIT_X64=0 before
the package import — training needs no fp64 and fp64 convs are ~30x slower).
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

from tilt_audit.fields import grid_to_z, make_basis, make_pk, unpack, smoothing_operator
from tilt_audit.scorenet import UNetCond

TF = 9.0
T_MIN = 1e-3
C_GRID = 0.05
S_OBS = 0.5  # matches run_t1.py: the true generative noise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--chs", default="32,64,128",
                   help="UNet channel widths (capacity knob)")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

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
    print(f"UNetCond params: {n_params/1e6:.2f}M, chs={chs}", flush=True)

    sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1)
    opt = optax.adam(sched)
    opt_state = opt.init(params)
    ema = params

    @jax.jit
    def train_step(params, opt_state, ema, key):
        kz, ky, kt, ke = jax.random.split(key, 4)
        z = jnp.sqrt(Pz) * jax.random.normal(kz, (args.batch, Pz.shape[0]))
        y = az * z + S_OBS * jax.random.normal(ky, z.shape)
        x0 = unpack(z, basis)
        y_map = unpack(y, basis)
        u = jax.random.uniform(kt, (args.batch,))
        t = C_GRID * ((1.0 + TF / C_GRID) ** u - 1.0) + T_MIN
        alpha = jnp.exp(-t)[:, None, None]
        sig = jnp.sqrt(1.0 - jnp.exp(-2.0 * t))[:, None, None]
        noise = jax.random.normal(ke, x0.shape)
        xt = alpha * x0 + sig * noise

        def loss_fn(p):
            pred = model.apply(p, xt, y_map, t)
            return jnp.mean((pred - noise) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema = jax.tree.map(lambda e, q: 0.999 * e + 0.001 * q, ema, params)
        return params, opt_state, ema, loss

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(step, losses):
        with out.open("wb") as f:
            pickle.dump({"params": params, "ema": ema, "chs": chs,
                         "n": args.n, "s_obs": S_OBS, "step": step,
                         "steps_total": args.steps, "losses": losses}, f)

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
            save(step, losses)
            t_last_ckpt = time.time()
    save(args.steps, losses)
    print(f"done in {(time.time()-t0)/60:.1f} min; final loss "
          f"{float(loss):.5f}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
