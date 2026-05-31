"""CLI entry point for the diagnostic pipeline.

Usage:
  python -m halflife.analysis --scenario baseline --steps 10000
  python -m halflife.analysis --scenario current_experiment --steps 5000 \
      --override num_species=3,half_life_max=80
"""

import argparse
import dataclasses
import os
import time

from halflife.config import SimConfig
from halflife.analysis.runner import run_diagnostic
from halflife.analysis.report import render_html


# Each preset is a dict of {field: value} layered on top of SimConfig defaults.
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
    fields = {**PRESETS[scenario], **overrides, 'emit_events': True}
    config = SimConfig(**{**dataclasses.asdict(SimConfig()), **fields})
    # Stash the scenario name for the report header (not a SimConfig field
    # so we just attach as a private attribute via __dict__ since SimConfig
    # is frozen — use a wrapper dataclass instead).
    object.__setattr__(config, '_scenario_name', scenario)
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description="Run a single composite diagnostic simulation.")
    p.add_argument('--scenario',     default='baseline', choices=sorted(PRESETS))
    p.add_argument('--steps',        type=int, default=10_000)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--sample-every', type=int, default=100,
                   help="Full-snapshot interval (compact metrics every step regardless).")
    p.add_argument('--top-k',        type=int, default=30,
                   help="K for the top-K transition / compatibility matrices.")
    p.add_argument('--override',     type=str, default='',
                   help="Comma-separated config overrides: k1=v1,k2=v2")
    p.add_argument('--out',          type=str, default='',
                   help="Output HTML path (default: tests/reports/diag_<scenario>_<ts>.html)")
    p.add_argument('--platform',     type=str, default='', choices=['', 'cpu', 'gpu'],
                   help="Force JAX platform (default: auto)")
    args = p.parse_args(argv)

    if args.platform:
        os.environ['JAX_PLATFORMS'] = args.platform

    overrides = _parse_overrides(args.override)
    config = build_config(args.scenario, overrides)

    print(f"[diag] scenario={args.scenario} steps={args.steps} seed={args.seed}")
    print(f"[diag] sample_every={args.sample_every} top_k={args.top_k}")
    if overrides:
        print(f"[diag] overrides: {overrides}")

    t0 = time.time()
    result = run_diagnostic(
        config, n_steps=args.steps, seed=args.seed, sample_every=args.sample_every,
    )
    t1 = time.time()
    print(f"[diag] run finished in {t1 - t0:.1f}s  ({result.n_steps / (t1 - t0):.1f} steps/sec)")

    html = render_html(result)

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
