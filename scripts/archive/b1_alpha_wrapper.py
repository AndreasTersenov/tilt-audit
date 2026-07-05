#!/usr/bin/env python
"""Run the predecessor project' run_reliability.py with DEFENSIVE_IID_FRACTION overridden.

The alpha knob is a module constant there, not a CLI flag; this wrapper imports the
script as a module, patches the constant from $DEFENSIVE_ALPHA, and calls its main()
with the remaining CLI args — the predecessor project itself stays read-only.

Note: at N=16 the effective alpha is round(alpha*16)/16, so 0.10 and 0.15 both give
n_iid=2. Logged, not "fixed" — the sweep list is the pre-registered one.
"""
import importlib.util
import os
import sys

REPO = "<predecessor-project>"
spec = importlib.util.spec_from_file_location(
    "run_reliability", os.path.join(REPO, "experiments", "run_reliability.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["run_reliability"] = mod
spec.loader.exec_module(mod)

alpha = float(os.environ["DEFENSIVE_ALPHA"])
mod.DEFENSIVE_IID_FRACTION = alpha
print(f"[b1_alpha_wrapper] DEFENSIVE_IID_FRACTION={alpha} "
      f"(n_iid at N=16: {max(1, round(alpha * 16))})", flush=True)
mod.main()
