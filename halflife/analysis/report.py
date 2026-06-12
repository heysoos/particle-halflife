"""HTML report assembly.

Single function: render_html(run_result) → str. Embeds all plots as base64
PNGs so the output file is fully self-contained (no external assets).

Layout philosophy:
  - Tier 1 and Tier 2 plots are arranged in a 2-column CSS grid so multiple
    fit on screen at once.
  - Tier 3 (empirical transitions) uses ONE matrix display with a radio-button
    toggle to switch between three views: size × size (default), composite ×
    composite (top-K), and composite × composite (all observed). Pure CSS,
    no JavaScript — uses the :checked sibling-combinator pattern.
  - Tier 4 uses the same toggle UI for the observed-composite compatibility
    matrix (top-K vs all).
"""

import dataclasses
from typing import Dict, List

import numpy as np

from halflife.analysis.runner import RunResult
from halflife.analysis import plots, compatibility, transitions
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Composite Diagnostic — {scenario}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    max-width: 1280px;
    margin: 1.5em auto;
    padding: 0 1.5em;
    color: #222;
    line-height: 1.4;
  }}
  h1 {{
    font-size: 1.5em;
    border-bottom: 2px solid #444;
    padding-bottom: 0.3em;
    margin-bottom: 0.6em;
  }}
  h2 {{
    font-size: 1.15em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.2em;
    margin-top: 2em;
    color: #333;
  }}
  h3 {{
    font-size: 0.95em;
    color: #555;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    font-weight: 600;
  }}
  p {{ font-size: 0.9em; }}
  p.note {{ color: #666; margin-top: -0.2em; font-style: italic; }}

  .meta {{
    background: #f7f7f9;
    padding: 0.6em 0.9em;
    border-radius: 4px;
    border-left: 3px solid #888;
    font-family: SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.8em;
    line-height: 1.55;
  }}

  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6em;
    margin: 1em 0 1.4em 0;
  }}
  .stat {{
    background: #f0f1f5;
    padding: 0.6em;
    border-radius: 4px;
    text-align: center;
  }}
  .stat .num {{
    font-size: 1.4em;
    font-weight: 700;
    color: #2c5b8f;
    line-height: 1.1;
  }}
  .stat .label {{
    font-size: 0.75em;
    color: #666;
    margin-top: 0.15em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}

  /* Two-column grid for time-series plots (Tier 1 / Tier 2). */
  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6em 1em;
    margin-top: 0.4em;
  }}
  .plot-grid img {{
    width: 100%;
    max-width: 100%;
    height: auto;
    border: 1px solid #eee;
    border-radius: 3px;
  }}

  /* Matrix wrapper — always shrinks the image to fit; never scrolls.
     The user explicitly wants the whole matrix visible at once even if
     individual cells get tiny for big matrices. */
  .matrix-wrap {{
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 0.5em;
    text-align: center;
    background: #fcfcfd;
  }}
  .matrix-wrap img {{
    max-width: 100%;
    height: auto;
    display: inline-block;
  }}

  /* Pure-CSS radio toggle. Hide the inputs themselves; style the labels
     as a button-strip; use :checked + sibling-combinator to reveal the
     matching view. */
  .toggle {{
    margin: 0.4em 0 0.6em 0;
  }}
  .toggle input[type="radio"] {{
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }}
  .toggle .toggle-labels {{
    display: inline-flex;
    border: 1px solid #bbb;
    border-radius: 4px;
    overflow: hidden;
    font-size: 0.8em;
  }}
  .toggle label {{
    padding: 0.35em 0.85em;
    cursor: pointer;
    background: #f5f5f7;
    color: #444;
    border-right: 1px solid #ddd;
    user-select: none;
    transition: background 0.1s;
  }}
  .toggle label:last-of-type {{ border-right: none; }}
  .toggle label:hover {{ background: #e8e8ec; }}

  .toggle .view {{ display: none; }}

  /* Per-toggle-group :checked selectors — one rule per radio id. */
  #t3_size:checked ~ .view-t3-size,
  #t3_topk:checked ~ .view-t3-topk,
  #t3_full:checked ~ .view-t3-full,
  #t4_size:checked ~ .view-t4-size,
  #t4_topk:checked ~ .view-t4-topk,
  #t4_full:checked ~ .view-t4-full {{
    display: block;
  }}

  /* "Active" highlight for the selected label. */
  #t3_size:checked ~ .toggle-labels label[for="t3_size"],
  #t3_topk:checked ~ .toggle-labels label[for="t3_topk"],
  #t3_full:checked ~ .toggle-labels label[for="t3_full"],
  #t4_size:checked ~ .toggle-labels label[for="t4_size"],
  #t4_topk:checked ~ .toggle-labels label[for="t4_topk"],
  #t4_full:checked ~ .toggle-labels label[for="t4_full"] {{
    background: #2c5b8f;
    color: white;
  }}

  footer {{
    font-size: 0.8em;
    color: #888;
    margin-top: 2.5em;
    padding-top: 0.8em;
    border-top: 1px solid #eee;
  }}
</style>
</head><body>
<h1>Composite Diagnostic Report</h1>
<div class="meta">
  <strong>Scenario:</strong> {scenario} &nbsp; <strong>Seed:</strong> {seed} &nbsp; <strong>Steps:</strong> {n_steps} &nbsp; <strong>Sample every:</strong> {sample_every}<br>
  <strong>Wall time:</strong> {wall:.1f}s &nbsp; <strong>Git SHA:</strong> {git_sha}<br>
  <strong>Config:</strong> num_particles={num_particles}, num_species={num_species}, max_composite_size={max_composite_size}, max_valence={max_valence}, use_valence={use_valence}, bond_mode={bond_mode}, fusion_threshold={fusion_threshold}, half_life_min={half_life_min}, half_life_max={half_life_max}<br>
  <strong>Per-species valences:</strong> {valences}
</div>

<div class="stat-grid">
  <div class="stat"><div class="num">{peak_max_size}</div><div class="label">peak max size</div></div>
  <div class="stat"><div class="num">{final_max_size}</div><div class="label">final max size</div></div>
  <div class="stat"><div class="num">{mean_alive:.1f}</div><div class="label">mean alive count</div></div>
  <div class="stat"><div class="num">{degree_sat:.0%}</div><div class="label">mean degree sat.</div></div>
</div>

<h2>Tier 1 — Macroscopic time series</h2>
<div class="plot-grid">
  <img src="data:image/png;base64,{img_size_trajectory}">
  <img src="data:image/png;base64,{img_size_dist}">
  <img src="data:image/png;base64,{img_free_particle}">
</div>

<h2>Tier 2 — Valence / edge structure</h2>
<div class="plot-grid">
  <img src="data:image/png;base64,{img_degree_sat}">
  <img src="data:image/png;base64,{img_free_bonds}">
  <img src="data:image/png;base64,{img_edges_rings}">
</div>

<h2>Tier 3 — Chemical network (empirical)</h2>
<p class="note">Built from {n_fusion} fusion events + {n_fission} fission events. Each event contributes 2 cells (fusion: A→C, B→C; fission: C→A, C→B). Toggle the binning below.</p>
<div class="toggle">
  <input type="radio" name="t3" id="t3_size" checked>
  <input type="radio" name="t3" id="t3_topk">
  <input type="radio" name="t3" id="t3_full">
  <div class="toggle-labels">
    <label for="t3_size">Size × size</label>
    <label for="t3_topk">Composite × composite (top {top_k})</label>
    <label for="t3_full">Composite × composite (all {n_full_hashes})</label>
  </div>
  <div class="view view-t3-size">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_size_bin_matrix}"></div>
  </div>
  <div class="view view-t3-topk">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_top_k_matrix}"></div>
  </div>
  <div class="view view-t3-full">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_full_matrix}"></div>
  </div>
</div>

<h2>Tier 4 — Fusion compatibility (theoretical)</h2>
<p class="note">Pure chemistry — what <strong>could</strong> happen if these pairs met. Greyed cells fail the BE threshold; small × markers mark pairs that fail the valence gate even at their structural maximum.</p>
<p class="note">Compare Tier 4 against Tier 3: bright cells in Tier 4 that are cold in Tier 3 are pairs that <em>could</em> fuse but never did — kinetic limitation, or valence-saturated by pre-existing edges.</p>

<h3>Matrix 4a: Species-pair compatibility</h3>
<div class="matrix-wrap"><img src="data:image/png;base64,{img_compat_species}"></div>

<h3>Matrix 4b: Observed-composite compatibility</h3>
<div class="toggle">
  <input type="radio" name="t4" id="t4_size" checked>
  <input type="radio" name="t4" id="t4_topk">
  <input type="radio" name="t4" id="t4_full">
  <div class="toggle-labels">
    <label for="t4_size">Size × size</label>
    <label for="t4_topk">Composite × composite (top {top_k})</label>
    <label for="t4_full">Composite × composite (all {n_full_hashes})</label>
  </div>
  <div class="view view-t4-size">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_compat_observed_size}"></div>
  </div>
  <div class="view view-t4-topk">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_compat_observed_topk}"></div>
  </div>
  <div class="view view-t4-full">
    <div class="matrix-wrap"><img src="data:image/png;base64,{img_compat_observed_full}"></div>
  </div>
</div>

<h2>Tier 5 — Open-endedness &amp; temporal evolution</h2>
<p class="note">Two type axes: <strong>composition</strong> (species multiset) and <strong>structure</strong> (bond-graph topology, Weisfeiler-Lehman hash). Time-resolved at snapshot cadence ({sample_every} steps); {n_windows} windows.</p>
<p class="note">A discovery curve that keeps climbing and sustained per-window novelty/turnover indicate ongoing open-endedness; a plateau means the chemistry has closed. Structure metrics are only meaningful in <code>bond_mode="edges"</code> runs.</p>

<h3>5a–c, 5e: discovery, novelty, diversity, size by window</h3>
<div class="plot-grid">
  <img src="data:image/png;base64,{img_oe_discovery}">
  <img src="data:image/png;base64,{img_oe_novelty}">
  <img src="data:image/png;base64,{img_oe_diversity}">
  <img src="data:image/png;base64,{img_oe_size_facets}">
</div>
<h3>5d: Window-to-window turnover</h3>
<div class="matrix-wrap"><img src="data:image/png;base64,{img_oe_turnover}"></div>

<footer>
  Generated by halflife.analysis on {timestamp}. JAX platform: {jax_platform}.
</footer>
</body></html>
"""


def _species_letter(idx: int) -> str:
    """Excel-style species letters: 0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …

    For num_species ≤ 26 (the common case) this is just 'A'..'Z'.
    Beyond Z we go to AA/AB/... so we never run out of distinct labels.
    """
    if idx < 26:
        return chr(ord('A') + idx)
    return _species_letter(idx // 26 - 1) + chr(ord('A') + idx % 26)


def _multiset_to_formula(multiset) -> str:
    """Render a sorted species multiset as a chemistry-style formula.

    Examples (assuming species 0,1,2 → A,B,C):
      (0,)            → 'A_1'
      (0, 1, 1, 1, 2) → 'A_1 B_3 C_1'
      (0, 0)          → 'A_2'

    Counts are always shown explicitly (no implicit "_1") so cells with
    the same species composition but different multiplicities line up
    visually in the matrix tick labels.
    """
    from collections import Counter
    counts = Counter(multiset)
    return ' '.join(
        f"{_species_letter(s)}_{counts[s]}" for s in sorted(counts)
    )


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "(unknown)"


def _unique_multisets_from_snapshots(snapshots, particles_species: np.ndarray = None):
    """Walk snapshots and return (hashes, multisets, total_incidence) for all unique
    composite types observed."""
    seen = {}
    incidence = {}
    for snap in snapshots:
        alive_idx = np.where(snap.alive)[0]
        for c in alive_idx:
            h = int(snap.species_hash[c])
            incidence[h] = incidence.get(h, 0) + 1
            if h not in seen:
                # Recover the multiset: snap.members[c, :n] → species via particles
                # We don't have particles species here per-snapshot. The caller passes
                # it in via the top-level config (initial state — species don't change).
                if particles_species is not None:
                    n = int(snap.member_count[c])
                    member_ids = snap.members[c, :n]
                    member_ids = member_ids[member_ids >= 0]
                    sp = tuple(sorted(int(particles_species[m]) for m in member_ids))
                    seen[h] = sp
                else:
                    seen[h] = ()
    return seen, incidence


def render_html(result: RunResult, top_k: int = 30,
                windows: int = None, window_width: int = None) -> str:
    """Build a single self-contained HTML string from a RunResult."""
    from halflife.state import (
        initialize_world,
        initialize_physics_params,
    )
    import datetime
    import jax

    # We need particle species for multiset reconstruction; species are constant
    # over the run, so just re-init a world with the same seed.
    init_world = initialize_world(result.config, seed=result.seed)
    particles_species = np.asarray(init_world.particles.species)
    physics = initialize_physics_params(result.config)

    # Tier 1, 2 plots.
    img_size_trajectory = plots.plot_size_trajectory(result.per_step_metrics)
    img_size_dist       = plots.plot_size_distribution_heatmap(result.per_step_metrics)
    img_free_particle   = plots.plot_free_particle_fraction(result.per_step_metrics)
    img_degree_sat      = plots.plot_degree_saturation(result.per_step_metrics)
    img_free_bonds      = plots.plot_free_bonds_heatmap(result.per_step_metrics)
    img_edges_rings     = plots.plot_edge_and_ring_counts(result.per_step_metrics)

    # Tier 3 matrices — all three views always rendered, the report's
    # CSS toggle picks which one is displayed.
    M_size_bin = transitions.size_bin_transition_matrix(
        result.events, result.config.max_composite_size
    )
    M_top_k, top_k_hashes = transitions.top_k_transition_matrix(result.events, K=top_k)
    M_full, full_hashes = transitions.full_transition_matrix(result.events)

    # We need the multiset for each hash to render chemical-formula labels.
    # _unique_multisets_from_snapshots gives us hash → sorted-species-tuple.
    seen, incidence = _unique_multisets_from_snapshots(result.snapshots, particles_species)

    def _formula_label(h):
        """Format a hash as 'A_1 B_3 C_1'; fall back to hex if multiset unknown."""
        if h is None:
            return 'other'
        m = seen.get(h)
        return _multiset_to_formula(m) if m else f"0x{h:08x}"

    # Size-binned: explicit size labels '0', '1', …, 'M' so axis ticks
    # mean what they look like (the row/col IS the size class).
    size_labels = [str(i) for i in range(M_size_bin.shape[0])]
    img_size_bin_matrix = plots.plot_transition_matrix(
        M_size_bin, labels=size_labels,
        title='Size-class transitions (rows = source size, cols = product size)',
    )
    img_top_k_matrix = plots.plot_transition_matrix(
        M_top_k, labels=[_formula_label(h) for h in top_k_hashes],
        title=f'Top-{top_k} composite-type transitions',
    )
    # All-observed: composite-indexed, NOT size-indexed. Hide ticks so
    # the integer positions don't get mistaken for size values; the
    # axis label communicates the sort order instead.
    img_full_matrix = plots.plot_transition_matrix(
        M_full, labels=None, hide_dense_ticks=True,
        title=f'All {len(full_hashes)} observed composite types',
    )

    # Tier 4 — Matrix 4a is species-pair (always small). Matrix 4b now has
    # both a top-K and an all-observed view, mirroring Tier 3's structure.
    be_a, pbe_a, pval_a = compatibility.species_pair_compat_matrix(result.config, physics)
    img_compat_species = plots.plot_compatibility_matrix(
        be_a, pbe_a, pval_a,
        title='Species-pair compatibility',
        labels=[f's{i}' for i in range(result.config.num_species)],
    )

    # seen/incidence were already computed up in the Tier 3 section.
    if seen:
        from collections import Counter
        # Top-K view: K most-trafficked hashes by incidence, sorted by size then hash.
        topk_hashes = [h for h, _ in Counter(incidence).most_common(top_k)]
        topk_hashes.sort(key=lambda h: (len(seen.get(h, ())), h))
        topk_multisets = [seen.get(h, ()) for h in topk_hashes]
        be_b_topk, pbe_b_topk, pval_b_topk = compatibility.observed_pair_compat_matrix(
            np.array(topk_hashes, dtype=np.uint32), topk_multisets, result.config, physics,
        )
        img_compat_observed_topk = plots.plot_compatibility_matrix(
            be_b_topk, pbe_b_topk, pval_b_topk,
            title=f'Top-{top_k} observed-composite compatibility',
            labels=[_multiset_to_formula(m) for m in topk_multisets],
        )

        # All-observed view: every unique hash seen across snapshots, sorted by size.
        all_hashes = sorted(seen.keys(), key=lambda h: (len(seen[h]), h))
        all_multisets = [seen[h] for h in all_hashes]
        be_b_full, pbe_b_full, pval_b_full = compatibility.observed_pair_compat_matrix(
            np.array(all_hashes, dtype=np.uint32), all_multisets, result.config, physics,
        )
        # Don't pass labels for the full view — too many to draw legibly.
        # hide_dense_ticks: the integer indices would otherwise look like
        # sizes; they're actually positions in a sorted-by-size list.
        img_compat_observed_full = plots.plot_compatibility_matrix(
            be_b_full, pbe_b_full, pval_b_full,
            title=f'All {len(all_hashes)} observed-composite compatibility',
            labels=None, hide_dense_ticks=True,
        )
        n_full_hashes = len(all_hashes)

        # Size-binned view: aggregate the all-observed pairwise matrix into
        # (max_composite_size+1)² cells by source/product size. Cell = mean
        # merged BE over all observed type-pairs of that size combination.
        # Pass explicit size labels so the ticks read as sizes, not indices.
        be_b_size, pbe_b_size, pval_b_size = compatibility.size_pair_compat_matrix(
            np.array(all_hashes, dtype=np.uint32), all_multisets,
            result.config, physics,
        )
        size_compat_labels = [str(i) for i in range(be_b_size.shape[0])]
        img_compat_observed_size = plots.plot_compatibility_matrix(
            be_b_size, pbe_b_size, pval_b_size,
            title='Size-pair compatibility (mean BE over observed types)',
            labels=size_compat_labels,
        )
    else:
        # No composites ever formed — render 1×1 placeholders for both views.
        img_compat_observed_topk = plots.plot_compatibility_matrix(
            np.zeros((1, 1)), np.zeros((1, 1), bool), np.ones((1, 1), bool),
            title='(no observed composites)',
        )
        img_compat_observed_full = img_compat_observed_topk
        img_compat_observed_size = img_compat_observed_topk
        n_full_hashes = 0

    # Headline derived numbers.
    peak_max_size  = int(result.per_step_metrics['max_size'].max())
    final_max_size = int(result.per_step_metrics['max_size'][-1])
    mean_alive     = float(result.per_step_metrics['alive_count'].mean())
    degree_sat     = float(result.per_step_metrics['degree_saturation_pct'].mean())

    n_fusion  = int((result.events.kind == KIND_FUSION).sum())
    n_fission = int((result.events.kind == KIND_FISSION).sum())

    # ── Tier 5: open-endedness ───────────────────────────────────────────────
    from halflife.analysis import openendedness as oe
    win = oe.slice_windows(result.n_steps, windows=windows, window_width=window_width)
    win_labels = [f"W{i+1}\n{s}-{e}" for i, (s, e) in enumerate(win)]
    snap_steps = [s.step for s in result.snapshots]

    comp_sets = [oe.composition_type_ids(s) for s in result.snapshots]
    struct_sets = [oe.structure_type_ids(s, particles_species) for s in result.snapshots]

    d_steps, comp_cum = oe.discovery_curve(comp_sets, snap_steps)
    _, struct_cum = oe.discovery_curve(struct_sets, snap_steps)
    total_comp_ev = oe.total_composition_types_from_events(result.events)

    img_oe_discovery = plots.plot_discovery_curves(d_steps, comp_cum, struct_cum, total_comp_ev)
    img_oe_novelty = plots.plot_novelty_rate(
        win_labels,
        oe.novelty_rate(comp_sets, snap_steps, win),
        oe.novelty_rate(struct_sets, snap_steps, win),
    )
    img_oe_diversity = plots.plot_hill_diversity(
        d_steps, oe.hill_diversity(comp_sets), oe.hill_diversity(struct_sets))
    img_oe_turnover = plots.plot_turnover_grid(
        oe.window_turnover(comp_sets, snap_steps, win),
        oe.window_turnover(struct_sets, snap_steps, win),
        win_labels,
    )
    img_oe_size_facets = plots.plot_window_size_facets(
        oe.per_window_size_hist(result.per_step_metrics, win), win_labels)

    return _HTML_TEMPLATE.format(
        scenario=getattr(result.config, '_scenario_name', '(custom)'),
        seed=result.seed,
        n_steps=result.n_steps,
        sample_every=result.sample_every,
        wall=result.wall_seconds,
        git_sha=_git_sha(),
        num_particles=result.config.num_particles,
        num_species=result.config.num_species,
        max_composite_size=result.config.max_composite_size,
        max_valence=result.config.max_valence,
        use_valence=result.config.use_valence,
        bond_mode=getattr(result.config, 'bond_mode', '(n/a)'),
        fusion_threshold=physics.fusion_threshold,
        half_life_min=result.config.half_life_min,
        half_life_max=result.config.half_life_max,
        valences=result.species_values.tolist(),
        peak_max_size=peak_max_size,
        final_max_size=final_max_size,
        mean_alive=mean_alive,
        degree_sat=degree_sat,
        img_size_trajectory=img_size_trajectory,
        img_size_dist=img_size_dist,
        img_free_particle=img_free_particle,
        img_degree_sat=img_degree_sat,
        img_free_bonds=img_free_bonds,
        img_edges_rings=img_edges_rings,
        img_size_bin_matrix=img_size_bin_matrix,
        img_top_k_matrix=img_top_k_matrix,
        img_full_matrix=img_full_matrix,
        img_compat_species=img_compat_species,
        img_compat_observed_size=img_compat_observed_size,
        img_compat_observed_topk=img_compat_observed_topk,
        img_compat_observed_full=img_compat_observed_full,
        n_fusion=n_fusion, n_fission=n_fission,
        top_k=top_k,
        n_full_hashes=n_full_hashes,
        n_windows=len(win),
        img_oe_discovery=img_oe_discovery,
        img_oe_novelty=img_oe_novelty,
        img_oe_diversity=img_oe_diversity,
        img_oe_turnover=img_oe_turnover,
        img_oe_size_facets=img_oe_size_facets,
        jax_platform=jax.default_backend(),
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
    )
