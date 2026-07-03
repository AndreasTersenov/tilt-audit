"""tilt-audit: oracle-verified audit of inference-time-steered diffusion sampling.

Samplers and metrics run in float64: the audit's value is exactness and those
workloads are small. Score-net TRAINING must opt out (TILT_AUDIT_X64=0):
fp64 convolutions are ~30x slower on A100 and training needs no fp64.
"""
import os

import jax

if os.environ.get("TILT_AUDIT_X64", "1") != "0":
    jax.config.update("jax_enable_x64", True)
