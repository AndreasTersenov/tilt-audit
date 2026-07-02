"""tilt-audit: oracle-verified audit of inference-time-steered diffusion sampling.

Everything runs in float64: the audit's value is exactness, and the workloads
are small enough that A100 fp64 throughput is a non-issue.
"""
import jax

jax.config.update("jax_enable_x64", True)
