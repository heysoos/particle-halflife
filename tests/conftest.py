"""
Pytest session setup.

Enables JAX's persistent compilation cache before any test imports trigger
compilation. Test wall-time is dominated by XLA *compilation* (CPU-bound,
~10-30s per distinct SimConfig on the GPU backend), not by simulation
execution — without the cache every pytest process recompiles every config
variant it touches from scratch. With it, reruns load cached executables in
milliseconds; only genuinely new (config, code) combinations compile.

The cache key covers backend/jaxlib/XLA-flag changes, so editing simulation
code naturally invalidates affected entries. Delete ~/.cache/halflife-jax to
force a cold rebuild.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from halflife.utils import enable_persistent_compilation_cache

enable_persistent_compilation_cache()
