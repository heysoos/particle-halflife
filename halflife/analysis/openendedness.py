"""Open-endedness & temporal-evolution metrics for the diagnostic report.

Pure host-side post-processing over a cached RunResult — no JAX, no I/O. Two
type axes are reported: composition (the species multiset, via species_hash)
and structure (the bond-graph topology, via a Weisfeiler-Lehman hash). Every
metric consumes a list of per-snapshot "type id arrays" (one id per alive
composite, repeats allowed) so it is agnostic to which axis it is fed.

All time-resolved metrics are at SNAPSHOT cadence: the flat event stream in
RunResult loses per-step alignment, and structure ids need the edge arrays
that only snapshots carry. See the design spec for the trade-off.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Windowing ───────────────────────────────────────────────────────────────

def slice_windows(n_steps: int,
                  windows: Optional[int] = None,
                  window_width: Optional[int] = None) -> List[Tuple[int, int]]:
    """Slice [0, n_steps) into (start, end) ranges.

    windows=N        → N equal ranges; the last absorbs any remainder.
    window_width=W   → ceil(n_steps/W) ranges of width W (last may be shorter).
    both None        → default 5 windows.
    both set         → ValueError (mutually exclusive).
    """
    if windows is not None and window_width is not None:
        raise ValueError("slice_windows: pass windows OR window_width, not both")
    if windows is None and window_width is None:
        windows = 5

    if windows is not None:
        windows = max(1, min(int(windows), n_steps))
        base = n_steps // windows
        bounds = []
        start = 0
        for i in range(windows):
            end = n_steps if i == windows - 1 else start + base
            bounds.append((start, end))
            start = end
        return bounds

    w = int(window_width)
    bounds = []
    start = 0
    while start < n_steps:
        bounds.append((start, min(start + w, n_steps)))
        start += w
    return bounds


def _window_index(step: int, windows: List[Tuple[int, int]]) -> Optional[int]:
    """Index of the window containing `step`; final window's end is inclusive."""
    for i, (start, end) in enumerate(windows):
        if start <= step < end:
            return i
    if windows and step == windows[-1][1]:
        return len(windows) - 1
    return None
