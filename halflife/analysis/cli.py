"""CLI entry point for the diagnostic pipeline.

Usage:
  python -m halflife.analysis --scenario baseline --steps 10000
  python -m halflife.analysis --scenario current_experiment --steps 5000 \
      --override num_species=3,half_life_max=80

  # Re-render the report from the previous run without re-simulating:
  python -m halflife.analysis --scenario current_experiment --steps 5000 --from-cache
"""

import argparse
import dataclasses
import hashlib
import os
import time

from halflife.config import SimConfig
from halflife.analysis.runner import (
    run_diagnostic, save_run_result, load_run_result,
)
from halflife.analysis.report import render_html


# Canonical "factory defaults" — the SimConfig values the project was tuned
# against before the user's running experiments. Layered FIRST, then per-preset
# overrides, then any --override flags. Without this baseline-like presets
# would silently inherit whatever the user happens to be currently editing in
# halflife/config.py (e.g. num_species=3 for current_experiment), making
# 'baseline' identical to 'current_experiment' until config.py is reverted.
_FACTORY_DEFAULTS = {
    'num_species':    12,
    'half_life_max':  15.0,
}

# Each preset is a dict of {field: value} layered on top of _FACTORY_DEFAULTS.
PRESETS = {
    'baseline':           {},
    'current_experiment': {'num_species': 3, 'half_life_max': 100.0},
    'valence_off':        {'use_valence': False},
    'polymer_world':      {'max_valence': 2, 'num_species': 2},
    'branching_world':    {'max_valence': 3, 'num_species': 3},
    'old_star_spring':    {'bond_mode': 'star_spring'},
}


def _parse_overrides(s: str) -> dict:
    """Parse 'k1=v1,k2=v2' into a dict, with crude int/float/bool coercion."""
    if not s:
        return {}
    out = {}
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            raise ValueError(f"override missing '=': {chunk!r}")
        k, v = chunk.split('=', 1)
        k = k.strip(); v = v.strip()
        # Coerce.
        if v.lower() in ('true', 'false'):
            out[k] = v.lower() == 'true'
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v   # leave as string (e.g. 'edges')
    return out


def build_config(scenario: str, overrides: dict) -> SimConfig:
    """SimConfig defaults + preset + overrides + always-on emit_events."""
    if scenario not in PRESETS:
        raise SystemExit(
            f"unknown scenario {scenario!r}. Known: {sorted(PRESETS)}"
        )
    fields = {**_FACTORY_DEFAULTS, **PRESETS[scenario], **overrides, 'emit_events': True}
    config = SimConfig(**{**dataclasses.asdict(SimConfig()), **fields})
    # Stash the scenario name for the report header (not a SimConfig field
    # so we just attach as a private attribute via __dict__ since SimConfig
    # is frozen — use a wrapper dataclass instead).
    object.__setattr__(config, '_scenario_name', scenario)
    return config


def _default_cache_path(scenario: str, n_steps: int, seed: int,
                        sample_every: int, overrides: dict) -> str:
    """Cache filename derived from the run args.

    Same args → same cache slot, so a follow-up --from-cache invocation
    with the same flags as the original run finds the right file. Overrides
    are hashed into a short suffix so non-default overrides get their own
    slot without bloating the filename.
    """
    if overrides:
        # Sort for deterministic order; the user's argparse `--override`
        # dict ordering shouldn't matter for cache identity.
        ovr_str = ','.join(f"{k}={v}" for k, v in sorted(overrides.items()))
        ovr_suffix = '_ovr' + hashlib.md5(ovr_str.encode()).hexdigest()[:8]
    else:
        ovr_suffix = ''
    fname = f"{scenario}_n{n_steps}_seed{seed}_every{sample_every}{ovr_suffix}.pkl.gz"
    return os.path.join('tests', 'reports', 'cache', fname)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run a single composite diagnostic simulation.")
    p.add_argument('--scenario',     default='baseline', choices=sorted(PRESETS))
    p.add_argument('--steps',        type=int, default=10_000)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--sample-every', type=int, default=100,
                   help="Full-snapshot interval (compact metrics every step regardless).")
    p.add_argument('--top-k',        type=int, default=30,
                   help="K for the top-K transition / compatibility matrices.")
    p.add_argument('--windows',      type=int, default=None,
                   help="Number of equal time windows for Tier 5 (default 5; "
                        "mutually exclusive with --window-width).")
    p.add_argument('--window-width', type=int, default=None,
                   help="Fixed window width in steps for Tier 5 (mutually "
                        "exclusive with --windows).")
    p.add_argument('--override',     type=str, default='',
                   help="Comma-separated config overrides: k1=v1,k2=v2")
    p.add_argument('--out',          type=str, default='',
                   help="Output HTML path (default: tests/reports/diag_<scenario>_<ts>.html)")
    p.add_argument('--platform',     type=str, default='', choices=['', 'cpu', 'gpu'],
                   help="Force JAX platform (default: auto)")
    p.add_argument('--from-cache',   action='store_true',
                   help="Skip simulation; load the cached RunResult matching the other "
                        "args and re-render the HTML. Useful when iterating on report "
                        "presentation code without burning GPU time.")
    p.add_argument('--cache-path',   type=str, default='',
                   help="Override cache file path (default: derived from "
                        "scenario+steps+seed+sample-every+overrides)")
    p.add_argument('--no-cache',     action='store_true',
                   help="Don't save the run to cache (default: save, overwriting).")
    args = p.parse_args(argv)

    if args.windows is not None and args.window_width is not None:
        raise SystemExit("--windows and --window-width are mutually exclusive")

    if args.platform:
        os.environ['JAX_PLATFORMS'] = args.platform

    # On-disk XLA cache: repeat invocations with the same scenario/overrides
    # skip the ~10-30s simulation_step compile. Imported lazily so it runs
    # AFTER the JAX_PLATFORMS env override above takes effect.
    from halflife.utils import enable_persistent_compilation_cache
    enable_persistent_compilation_cache()

    overrides = _parse_overrides(args.override)
    cache_path = args.cache_path or _default_cache_path(
        args.scenario, args.steps, args.seed, args.sample_every, overrides,
    )

    print(f"[diag] scenario={args.scenario} steps={args.steps} seed={args.seed}")
    print(f"[diag] sample_every={args.sample_every} top_k={args.top_k}")
    if overrides:
        print(f"[diag] overrides: {overrides}")

    if args.from_cache:
        if not os.path.exists(cache_path):
            raise SystemExit(
                f"--from-cache: no cached run at {cache_path!r}. "
                f"Run without --from-cache first to populate the cache."
            )
        print(f"[diag] loading cached run from {cache_path}")
        t0 = time.time()
        result = load_run_result(cache_path)
        t1 = time.time()
        print(f"[diag] loaded in {t1 - t0:.1f}s  ({result.n_steps} steps)")
    else:
        config = build_config(args.scenario, overrides)
        t0 = time.time()
        result = run_diagnostic(
            config, n_steps=args.steps, seed=args.seed,
            sample_every=args.sample_every,
        )
        t1 = time.time()
        print(f"[diag] run finished in {t1 - t0:.1f}s  ({result.n_steps / (t1 - t0):.1f} steps/sec)")
        if not args.no_cache:
            save_run_result(result, cache_path)
            cache_kb = os.path.getsize(cache_path) / 1024
            print(f"[diag] cached run to {cache_path}  ({cache_kb:.0f} KB)")

    html = render_html(result, top_k=args.top_k,
                       windows=args.windows, window_width=args.window_width)

    out = args.out or _default_out(args.scenario)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[diag] wrote {out}  ({len(html) / 1024:.0f} KB)")
    return out


def _default_out(scenario: str) -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join('tests', 'reports', f'diag_{scenario}_{ts}.html')


if __name__ == '__main__':
    main()
