"""
Pytest session setup.

Two things happen here, both aimed at the same problem: **test wall-time is
dominated by XLA compilation, not by simulation execution.** Measured on this
machine for one representative test config (500 particles), comparing a single
simulation_step compile against 200 executed steps:

    backend   compile (cold)   compile (cached)   200 steps
    GPU            13.8 s            3.1 s          0.53 s
    CPU             5.0 s            2.1 s          1.30 s

Compilation costs ~26x more than the simulation it enables, so everything
below optimizes for compiling less, and compiling cheaper.


1. Backend choice (left to the caller — GPU default)
----------------------------------------------------
Neither backend wins outright, so nothing is forced here. CPU compiles ~2.7x
faster cold; GPU executes ~7x faster at these sizes. Which dominates depends
on the file: test_hash / test_spatial are compile-bound (CPU wins), while
test_chemistry is execution-bound (GPU wins — a serial CPU run of it exceeded
10 minutes against 7:48 on GPU).

Set JAX_PLATFORMS=cpu on the command line for the compile-bound files.

Do NOT pair the CPU backend with a high pytest-xdist worker count: each
worker's XLA spawns threads across all cores, and oversubscription makes the
suite slower than fewer workers. `-n 4` with OMP_NUM_THREADS=4 is a reasonable
pairing on a 16-core box.


The single biggest cost is neither of the above
-----------------------------------------------
It is calling a chemistry kernel WITHOUT jax.jit. Eager mode dispatches every
primitive individually, and these kernels contain fori_loop sweeps of
fission_label_iters=64 plus BFS/subtree passes — thousands of primitives. One
eager apply_composite_decay call measured 29 s; a 40-call loop measured 206 s,
44% of the entire chemistry suite. Always jit kernels under test (see the
_JIT_* wrappers at the top of test_chemistry.py) and share the SimConfig so the
compile is paid once.


2. Persistent compilation cache
-------------------------------
Without it every pytest process recompiles every config variant from scratch.
With it, only genuinely new (config, code) combinations pay full price.

The cache is not a silver bullet: it skips XLA's *backend* compile but not
JAX's Python -> jaxpr -> HLO tracing, which must rerun in every process just to
compute the cache key. That residue is the 2-3 s "cached" column above, and it
is paid per distinct SimConfig per worker process. The way to go faster is to
construct fewer distinct SimConfigs — see the shared-config constants at the
top of test_chemistry.py.

The cache key covers backend/jaxlib/XLA-flag changes, so editing simulation
code (or anything that changes array shapes, e.g. num_species) naturally
invalidates affected entries. Delete ~/.cache/halflife-jax to force a cold
rebuild.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from halflife.utils import enable_persistent_compilation_cache

enable_persistent_compilation_cache()
