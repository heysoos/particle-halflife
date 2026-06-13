"""Matplotlib plot helpers for the diagnostic report.

Each function takes the slice of RunResult it needs (so plots can be regenerated
in isolation) and returns a base64-encoded PNG string suitable for inline use
as <img src="data:image/png;base64,...">.

Plot sizing convention:
  - Time-series and per-step plots are 5.5 × 2.6 in (~660×312 px @ 120 dpi),
    designed to fit two-per-row in a CSS grid.
  - Square heatmaps/matrices are auto-sized in the range 3.5–7.5 in per side,
    scaled by matrix dimensions but capped so they never dominate the page.
  - All figures use constrained_layout so labels and colorbars never overflow.
"""

import io
import base64
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; no display required.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm


# ── Style configuration ─────────────────────────────────────────────────────
# Set once at module import. Aims for a quiet, scientific look:
#   - small serifed-ish sans-serif type
#   - faint gridlines as backdrop
#   - thinner axis spines, more whitespace
#   - consistent label sizes so plots compose well in a grid
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 120,
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.titleweight': 'semibold',
    'axes.labelsize': 8.5,
    'axes.labelcolor': '#333',
    'axes.edgecolor': '#666',
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.color': '#555',
    'ytick.color': '#555',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'grid.color': '#dcdcdc',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Color palette tuned for the page background (white) and accessibility.
_C_PRIMARY   = '#2c5b8f'   # muted blue
_C_SECONDARY = '#c95227'   # warm orange
_C_TERTIARY  = '#3d8b54'   # quiet green
_C_ACCENT    = '#8e5fa1'   # purple
_C_DANGER    = '#b03a3a'   # red — saturation/failures


def _fig_to_base64(fig: Figure) -> str:
    """Render a Matplotlib Figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    # constrained_layout already handles padding; bbox_inches='tight' would
    # otherwise re-pad and slightly clip colorbars.
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _square_matrix_size(n_cells: int) -> float:
    """Square figure side (inches) for an n×n matrix.

    Caps at 4.6in so even a 500-cell matrix fits comfortably on screen
    without a scroll wrapper — the user explicitly wants to see the
    whole matrix in one go, even if individual cells get tiny.
    """
    return float(np.clip(2.6 + 0.06 * n_cells, 3.2, 4.6))


def _style_matrix_axes(ax, matrix: np.ndarray, labels: List[str] = None,
                       hide_dense_ticks: bool = False):
    """Apply consistent border + tick styling for heatmap axes.

    Matrix plots use a full 4-sided border (the global rcParams strip
    top/right for line plots, but for heatmaps the box looks right).

    Tick policy:
      - labels provided AND short enough (≤32) → show all labels
      - labels provided BUT too long → fall back to coarse numeric ticks
        (the numbers are sensible — they're indices into the label list,
        i.e. row/col positions, which is what the labels would name)
      - labels is None AND hide_dense_ticks=True → hide ticks completely
        (used for composite-indexed dense matrices where integer ticks
        would falsely suggest "size" semantics)
      - labels is None AND hide_dense_ticks=False → numeric ticks
        (used for size-indexed matrices where ticks ARE the size class)
    """
    # Full 4-sided box, slightly heavier than the line-plot spines.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color('#666')

    n_rows, n_cols = matrix.shape
    if labels is not None and len(labels) <= 32:
        # Few enough labels to render legibly — show them all.
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
    elif hide_dense_ticks:
        # Composite-indexed dense matrix — integer ticks would mislead.
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Size-indexed (or otherwise meaningful) dense matrix: show ~10
        # evenly spaced numeric ticks per axis so the user knows the scale.
        step_x = max(1, n_cols // 10)
        step_y = max(1, n_rows // 10)
        ax.set_xticks(range(0, n_cols, step_x))
        ax.set_yticks(range(0, n_rows, step_y))
        ax.tick_params(axis='both', labelsize=6.5)


# ── Tier 1: macroscopic time-series ─────────────────────────────────────────

def plot_size_trajectory(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: max + mean size on top, alive count on bottom."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.5, 3.2), sharex=True, constrained_layout=True
    )
    steps = np.arange(per_step['max_size'].shape[0])
    ax1.plot(steps, per_step['max_size'], color=_C_PRIMARY, lw=1.1, label='max')
    ax1.plot(steps, per_step['mean_size'], color=_C_SECONDARY, lw=1.0, alpha=0.75, label='mean')
    ax1.set_ylabel('composite size')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, axis='y')
    ax1.set_title('Composite size trajectory')

    ax2.plot(steps, per_step['alive_count'], color=_C_TERTIARY, lw=1.1)
    ax2.set_xlabel('step')
    ax2.set_ylabel('# alive')
    ax2.grid(True, axis='y')
    return _fig_to_base64(fig)


def plot_size_distribution_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: size × time heatmap of composite-count distribution."""
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    hist = per_step['size_histogram']            # (steps, max_size)
    im = ax.imshow(
        hist.T, aspect='auto', origin='lower',
        cmap='cividis', interpolation='nearest',
    )
    ax.set_xlabel('step')
    ax.set_ylabel('size')
    ax.set_title('Size distribution over time')
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('count', rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6.5)
    return _fig_to_base64(fig)


def plot_free_particle_fraction(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: fraction of particles not in any composite, over time."""
    fig, ax = plt.subplots(figsize=(5.5, 2.4), constrained_layout=True)
    steps = np.arange(per_step['free_particle_fraction'].shape[0])
    ax.plot(steps, per_step['free_particle_fraction'], color=_C_DANGER, lw=1.1)
    ax.fill_between(steps, 0, per_step['free_particle_fraction'],
                    color=_C_DANGER, alpha=0.08)
    ax.set_xlabel('step')
    ax.set_ylabel('free / total')
    ax.set_ylim(0, 1)
    ax.set_title('Free-particle fraction')
    ax.grid(True, axis='y')
    return _fig_to_base64(fig)


# ── Tier 2: valence / edge structure ────────────────────────────────────────

def plot_degree_saturation(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: fraction of particles with degree == valence."""
    fig, ax = plt.subplots(figsize=(5.5, 2.4), constrained_layout=True)
    steps = np.arange(per_step['degree_saturation_pct'].shape[0])
    ax.plot(steps, per_step['degree_saturation_pct'], color=_C_ACCENT, lw=1.1)
    ax.fill_between(steps, 0, per_step['degree_saturation_pct'],
                    color=_C_ACCENT, alpha=0.08)
    ax.set_xlabel('step')
    ax.set_ylabel('saturated fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Degree saturation (degree == valence)')
    ax.grid(True, axis='y')
    return _fig_to_base64(fig)


def plot_free_bonds_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: free_bonds distribution per timestep."""
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    hist = per_step['free_bonds_histogram']
    im = ax.imshow(hist.T, aspect='auto', origin='lower',
                   cmap='cividis', interpolation='nearest')
    ax.set_xlabel('step')
    ax.set_ylabel('free_bonds')
    ax.set_title('Composite free_bonds distribution')
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('count', rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6.5)
    return _fig_to_base64(fig)


def plot_edge_and_ring_counts(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: edge count + ring count over time."""
    fig, ax = plt.subplots(figsize=(5.5, 2.4), constrained_layout=True)
    steps = np.arange(per_step['edge_count_total'].shape[0])
    ax.plot(steps, per_step['edge_count_total'], color=_C_PRIMARY, lw=1.1, label='edges')
    ax.plot(steps, per_step['ring_count_total'], color=_C_SECONDARY, lw=1.1, label='rings')
    ax.set_xlabel('step')
    ax.set_ylabel('count')
    ax.set_title('Total edges and rings (alive composites)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, axis='y')
    return _fig_to_base64(fig)


# ── Tier 3 & 4: matrices ────────────────────────────────────────────────────

def plot_transition_matrix(matrix: np.ndarray, labels: List[str] = None,
                           title: str = '', cmap: str = 'rocket_r',
                           log_color: bool = True,
                           hide_dense_ticks: bool = False) -> str:
    """Render a transition matrix (any size) as a heatmap.

    The `rocket_r` colormap (matplotlib via seaborn-style palette) reads
    cleanly on a white background, with high-traffic cells dark; if the
    name isn't registered, falls back to 'magma_r' which has the same look.
    """
    # Degenerate runs (no events observed) hand us a (0, 0) matrix —
    # matplotlib + matrix.max() both raise on zero-size arrays. Render a
    # one-cell placeholder so the report still assembles cleanly.
    if matrix.size == 0:
        matrix = np.zeros((1, 1), dtype=np.int64)
        labels = ['(no events)']

    n = max(matrix.shape)
    side = _square_matrix_size(n)
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)

    # Fall back if the user's matplotlib doesn't have rocket_r registered.
    if cmap not in plt.colormaps():
        cmap = 'magma_r'

    if log_color and matrix.max() > 0:
        # +1 to avoid log(0); colorbar then reads as count.
        im = ax.imshow(matrix + 1, cmap=cmap,
                       norm=LogNorm(vmin=1, vmax=matrix.max() + 1),
                       interpolation='nearest')
    else:
        im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')

    _style_matrix_axes(ax, matrix, labels, hide_dense_ticks=hide_dense_ticks)
    ax.set_xlabel('product' + (' (sorted by size →)' if hide_dense_ticks else ''))
    ax.set_ylabel('source' + (' (sorted by size →)' if hide_dense_ticks else ''))
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('count', rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6.5)
    return _fig_to_base64(fig)


def plot_compatibility_matrix(
    be: np.ndarray, passes_be: np.ndarray, passes_val: np.ndarray,
    title: str = '', labels: List[str] = None,
    hide_dense_ticks: bool = False,
) -> str:
    """Tier 4: merged BE colormap with grey-out for failed BE and × for failed valence."""
    if be.size == 0:
        be = np.zeros((1, 1), dtype=np.float32)
        passes_be = np.zeros((1, 1), dtype=bool)
        passes_val = np.ones((1, 1), dtype=bool)
        labels = ['(no data)']

    n = max(be.shape)
    side = _square_matrix_size(n)
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)

    # Base layer: BE as heatmap. Cells failing BE render as NaN → light grey.
    display = np.where(passes_be, be, np.nan)
    cmap = plt.cm.cividis.copy()
    cmap.set_bad('#e8e8ec')
    im = ax.imshow(display, cmap=cmap, interpolation='nearest')

    # Mark valence-blocked cells with a subtle × marker. For very large
    # matrices we skip the markers (they'd be unreadable noise anyway).
    if n <= 60:
        fail_val_y, fail_val_x = np.where(~passes_val)
        if fail_val_x.size > 0:
            marker_size = max(3.0, 18.0 / max(1.0, n / 10.0))
            ax.scatter(fail_val_x, fail_val_y,
                       marker='x', color='#333', s=marker_size, alpha=0.6, linewidths=0.5)

    _style_matrix_axes(ax, be, labels, hide_dense_ticks=hide_dense_ticks)
    ax.set_xlabel('partner B' + (' (sorted by size →)' if hide_dense_ticks else ''))
    ax.set_ylabel('partner A' + (' (sorted by size →)' if hide_dense_ticks else ''))
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('merged BE', rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6.5)
    return _fig_to_base64(fig)


# ── Tier 5: Open-endedness & temporal evolution ─────────────────────────────

def plot_discovery_curves(steps, comp_cum, struct_cum, total_comp_events: int) -> str:
    """Cumulative distinct-types vs step, composition + structure overlaid."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(steps, comp_cum, color=_C_PRIMARY, lw=1.6, label='composition')
    ax.plot(steps, struct_cum, color=_C_ACCENT, lw=1.6, label='structure')
    ax.axhline(total_comp_events, color=_C_SECONDARY, lw=0.9, ls='--',
               label=f'all composition types (events): {total_comp_events}')
    ax.set_xlabel('step'); ax.set_ylabel('distinct types (cumulative)')
    ax.set_title('Type discovery curve')
    ax.grid(True); ax.legend(loc='upper left')
    return _fig_to_base64(fig)


def plot_novelty_rate(window_labels, comp_counts, struct_counts) -> str:
    """Grouped bars: new types first seen per window, both axes."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    x = np.arange(len(window_labels))
    ax.bar(x - 0.2, comp_counts, width=0.4, color=_C_PRIMARY, label='composition')
    ax.bar(x + 0.2, struct_counts, width=0.4, color=_C_ACCENT, label='structure')
    ax.set_xticks(x); ax.set_xticklabels(window_labels, fontsize=6.5)
    ax.set_ylabel('new types'); ax.set_title('Novelty rate per window')
    ax.grid(True, axis='y'); ax.legend()
    return _fig_to_base64(fig)


def _plot_hill_panel(ax, steps, hill, title):
    ax.plot(steps, hill['q0'], color=_C_SECONDARY, lw=1.4, label='q=0 richness')
    ax.plot(steps, hill['q1'], color=_C_PRIMARY, lw=1.4, label='q=1 Shannon')
    ax.plot(steps, hill['q2'], color=_C_TERTIARY, lw=1.4, label='q=2 Simpson')
    ax.set_xlabel('step'); ax.set_ylabel('effective # types')
    ax.set_title(title); ax.grid(True); ax.legend(fontsize=6.5)


def plot_hill_diversity(steps, comp_hill, struct_hill) -> str:
    """Two side-by-side panels: alive-type diversity for each axis."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax1, ax2 = fig.subplots(1, 2)
    _plot_hill_panel(ax1, steps, comp_hill, 'Composition diversity')
    _plot_hill_panel(ax2, steps, struct_hill, 'Structure diversity')
    return _fig_to_base64(fig)


def plot_turnover_grid(comp_turnover, struct_turnover, window_labels) -> str:
    """2×2 heatmaps: {Jaccard, Bray-Curtis} × {composition, structure}."""
    fig = Figure(figsize=(5.5, 5.2), constrained_layout=True)
    axes = fig.subplots(2, 2)
    panels = [
        (axes[0, 0], comp_turnover['jaccard'],      'Composition · Jaccard'),
        (axes[0, 1], struct_turnover['jaccard'],    'Structure · Jaccard'),
        (axes[1, 0], comp_turnover['bray_curtis'],  'Composition · Bray-Curtis'),
        (axes[1, 1], struct_turnover['bray_curtis'],'Structure · Bray-Curtis'),
    ]
    short = [l.split('\n')[0] for l in window_labels]
    for ax, M, title in panels:
        im = ax.imshow(np.nan_to_num(M, nan=0.0), cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(window_labels)))
        ax.set_yticks(range(len(window_labels)))
        ax.set_xticklabels(short, fontsize=6.5)
        ax.set_yticklabels(short, fontsize=6.5)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64(fig)


def plot_window_size_facets(per_window_hist, window_labels) -> str:
    """Overlaid composite-size distributions, one line per window."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    cmap = plt.get_cmap('plasma')
    n = len(per_window_hist)
    for i, (h, label) in enumerate(zip(per_window_hist, window_labels)):
        sizes = np.arange(1, len(h) + 1)
        ax.plot(sizes, h, lw=1.3, color=cmap(i / max(n - 1, 1)),
                label=label.split('\n')[0])
    ax.set_xlabel('composite size'); ax.set_ylabel('mean count')
    ax.set_yscale('symlog'); ax.set_title('Size distribution by window')
    ax.grid(True); ax.legend(fontsize=6.5)
    return _fig_to_base64(fig)


def plot_degree_distribution(window_labels, deg_frac) -> str:
    """Stacked bars per window: bonded-particle degree distribution (1/2/3/4+)."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    x = np.arange(len(window_labels))
    colors = [_C_TERTIARY, _C_PRIMARY, _C_SECONDARY, _C_ACCENT]
    labels = ['tip (deg 1)', 'chain (deg 2)', 'branch (deg 3)', 'branch (deg 4+)']
    bottom = np.zeros(len(window_labels))
    for k in range(4):
        ax.bar(x, deg_frac[:, k], bottom=bottom, width=0.7,
               color=colors[k], label=labels[k])
        bottom += deg_frac[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('\n')[0] for l in window_labels], fontsize=6.5)
    ax.set_ylabel('fraction of bonded particles'); ax.set_ylim(0, 1)
    ax.set_title('Degree distribution by window')
    ax.legend(fontsize=6, ncol=2, loc='upper center')
    return _fig_to_base64(fig)


def _plot_topo_panel(ax, window_labels, frac, title):
    x = np.arange(len(window_labels))
    colors = [_C_PRIMARY, _C_TERTIARY, _C_SECONDARY]
    labels = ['chain', 'tree-branch', 'cyclic']
    bottom = np.zeros(len(window_labels))
    for k in range(3):
        ax.bar(x, frac[:, k], bottom=bottom, width=0.7, color=colors[k], label=labels[k])
        bottom += frac[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('\n')[0] for l in window_labels], fontsize=6.5)
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=8)


def plot_topology_split(window_labels, topo_count, topo_mass) -> str:
    """Two stacked-bar panels: topology class by composite COUNT vs particle MASS.

    The count-vs-mass contrast is the point: many tiny chains can dominate the
    left panel while a few large cyclic networks dominate the right.
    """
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax1, ax2 = fig.subplots(1, 2)
    _plot_topo_panel(ax1, window_labels, topo_count, 'By composite count')
    _plot_topo_panel(ax2, window_labels, topo_mass, 'By particle mass')
    ax1.set_ylabel('fraction')
    ax2.legend(fontsize=6.5, loc='upper right')
    return _fig_to_base64(fig)
