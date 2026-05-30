"""Matplotlib plot helpers for the diagnostic report.

Each function takes the slice of RunResult it needs (so plots can be regenerated
in isolation) and returns a base64-encoded PNG string suitable for inline use
as <img src="data:image/png;base64,...">.
"""

import io
import base64
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; no display required.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _fig_to_base64(fig: Figure) -> str:
    """Render a Matplotlib Figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def plot_size_trajectory(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: max size + alive count over time, 2-row subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    steps = np.arange(per_step['max_size'].shape[0])
    ax1.plot(steps, per_step['max_size'], color='#1f77b4', lw=1.2, label='max')
    ax1.plot(steps, per_step['mean_size'], color='#ff7f0e', lw=1.0, alpha=0.6, label='mean')
    ax1.set_ylabel('composite size')
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)
    ax1.set_title('Composite size trajectory')

    ax2.plot(steps, per_step['alive_count'], color='#2ca02c', lw=1.2)
    ax2.set_xlabel('step'); ax2.set_ylabel('alive composites')
    ax2.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_size_distribution_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: size × time heatmap of composite-count distribution."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = per_step['size_histogram']            # (steps, max_size)
    # Transpose so x=step, y=size.
    im = ax.imshow(
        hist.T, aspect='auto', origin='lower',
        cmap='magma', interpolation='nearest',
    )
    ax.set_xlabel('step'); ax.set_ylabel('size')
    ax.set_title('Composite size distribution over time')
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_free_particle_fraction(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: fraction of particles not in any composite, over time."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['free_particle_fraction'].shape[0])
    ax.plot(steps, per_step['free_particle_fraction'], color='#d62728', lw=1.2)
    ax.set_xlabel('step'); ax.set_ylabel('free / total')
    ax.set_ylim(0, 1)
    ax.set_title('Free-particle fraction')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_degree_saturation(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: percent of particles with degree == valence (saturated)."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['degree_saturation_pct'].shape[0])
    ax.plot(steps, per_step['degree_saturation_pct'], color='#9467bd', lw=1.2)
    ax.set_xlabel('step'); ax.set_ylabel('saturated fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Per-particle degree saturation (degree == valence)')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_free_bonds_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: free_bonds distribution per timestep."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = per_step['free_bonds_histogram']
    im = ax.imshow(hist.T, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('step'); ax.set_ylabel('free_bonds')
    ax.set_title('Composite free_bonds distribution')
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_edge_and_ring_counts(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: edge count + ring count over time."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['edge_count_total'].shape[0])
    ax.plot(steps, per_step['edge_count_total'], color='#1f77b4', lw=1.2, label='edges')
    ax.plot(steps, per_step['ring_count_total'], color='#ff7f0e', lw=1.2, label='rings')
    ax.set_xlabel('step'); ax.set_ylabel('count')
    ax.set_title('Total edges and rings across all alive composites')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_transition_matrix(matrix: np.ndarray, labels: List[str] = None,
                           title: str = '', cmap: str = 'Reds',
                           log_color: bool = True) -> str:
    """Render a transition matrix (any size) as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, min(20, matrix.shape[1] * 0.3)),
                                     max(6, min(20, matrix.shape[0] * 0.3))))
    if log_color and matrix.max() > 0:
        from matplotlib.colors import LogNorm
        # +1 to avoid log(0); colorbar then reads as count.
        im = ax.imshow(matrix + 1, cmap=cmap, norm=LogNorm(vmin=1, vmax=matrix.max() + 1))
    else:
        im = ax.imshow(matrix, cmap=cmap)
    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('product'); ax.set_ylabel('source')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_compatibility_matrix(
    be: np.ndarray, passes_be: np.ndarray, passes_val: np.ndarray,
    title: str = '', labels: List[str] = None,
) -> str:
    """Tier 4: merged BE colormap with grey-out for failed BE and hatch for failed valence."""
    fig, ax = plt.subplots(figsize=(max(6, min(20, be.shape[1] * 0.3)),
                                     max(6, min(20, be.shape[0] * 0.3))))
    # Base layer: BE as heatmap.
    display = np.where(passes_be, be, np.nan)
    im = ax.imshow(display, cmap='viridis')
    fig.colorbar(im, ax=ax, label='merged BE')

    # Hatch the cells failing valence (overlay translucent diagonal lines).
    fail_val_y, fail_val_x = np.where(~passes_val)
    ax.scatter(fail_val_x, fail_val_y, marker='x', color='black', s=8, alpha=0.7)

    # Grey-out for failed BE happens implicitly (NaN displays as the colormap's bad color).
    # Force the bad color to be light grey.
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#dddddd')
    im.set_cmap(cmap)

    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('partner B'); ax.set_ylabel('partner A')
    ax.set_title(title)
    return _fig_to_base64(fig)
