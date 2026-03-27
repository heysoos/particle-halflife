"""
tests/test_composite_statistics.py — Analyze composite formation, growth, and evolution.

This test sweeps key parameters that control composite dynamics and generates
a detailed HTML report with plots showing:
  - How fusion threshold affects composite size and formation rate
  - How interaction radius affects composite formation
  - Composite size distributions and lifetimes
  - Growth trajectories under different conditions

Run standalone:  python tests/test_composite_statistics.py
Run under pytest: pytest tests/test_composite_statistics.py -v -s

Output: tests/reports/composite_statistics_<timestamp>.html
"""

import sys
import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params, initialize_physics_params
from halflife.step import simulation_step

matplotlib.use('Agg')  # Non-interactive backend


@dataclass
class CompositeSnapshot:
    """Snapshot of composite state at a single step."""
    step: int
    num_alive: int
    max_size: int
    mean_size: float
    median_size: float
    min_size: int
    size_distribution: list  # histogram bins
    total_members: int
    alive_composites: list  # List of (id, size) tuples for composites


@dataclass
class SimulationRun:
    """Complete simulation run with parameters and collected metrics."""
    fusion_threshold: float
    interaction_radius: float
    composite_size_decay_scale: float
    num_particles: int
    num_species: int
    num_steps: int
    dt: float
    duration_sec: float

    snapshots: list  # List of CompositeSnapshot

    # Derived metrics
    peak_composites: int
    peak_composite_size: int
    mean_max_size: float
    final_composites: int
    final_max_size: int
    formation_rate: float  # new composites per step


def fig_to_base64(fig):
    """Convert matplotlib figure to base64-encoded PNG for embedding in HTML."""
    canvas = FigureCanvasAgg(fig)
    img_data = io.BytesIO()
    canvas.print_png(img_data)
    img_data.seek(0)
    return base64.b64encode(img_data.getvalue()).decode()


def run_simulation_with_stats(config: SimConfig, num_steps: int = 500) -> SimulationRun:
    """
    Run a single simulation and collect detailed composite statistics.

    Args:
        config: SimConfig with target fusion_threshold and interaction_radius
        num_steps: Number of steps to simulate

    Returns:
        SimulationRun with all collected metrics
    """
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    state = initialize_world(config, seed=0)

    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    snapshots = []
    prev_composites = set()
    new_composite_count = 0

    t_start = time.time()
    for step in range(num_steps):
        state = step_fn(state, params, config, physics)
        state.particles.position.block_until_ready()

        # Collect snapshot
        composites = state.composites
        alive_mask = np.asarray(composites.alive)
        member_counts = np.asarray(composites.member_count)

        alive_indices = np.where(alive_mask)[0]
        sizes = member_counts[alive_indices]

        if len(sizes) > 0:
            max_size = int(np.max(sizes))
            mean_size = float(np.mean(sizes))
            median_size = float(np.median(sizes))
            min_size = int(np.min(sizes))
            total_members = int(np.sum(sizes))

            # Size distribution (histogram)
            hist, _ = np.histogram(sizes, bins=range(1, config.max_composite_size + 2))
            size_dist = hist.tolist()

            # Track formation: new composites = current alive - previous alive + deaths
            current_set = set(alive_indices)
            newly_formed = len(current_set - prev_composites)
            new_composite_count += newly_formed
            prev_composites = current_set

            alive_comps = [(int(i), int(member_counts[i])) for i in alive_indices]
        else:
            max_size = 0
            mean_size = 0.0
            median_size = 0.0
            min_size = 0
            total_members = 0
            size_dist = [0] * config.max_composite_size
            alive_comps = []

        snapshot = CompositeSnapshot(
            step=step,
            num_alive=len(alive_indices),
            max_size=max_size,
            mean_size=mean_size,
            median_size=median_size,
            min_size=min_size,
            size_distribution=size_dist,
            total_members=total_members,
            alive_composites=alive_comps,
        )
        snapshots.append(snapshot)

    t_end = time.time()
    duration = t_end - t_start

    # Compute derived metrics
    max_sizes = [s.max_size for s in snapshots]
    num_alive_list = [s.num_alive for s in snapshots]
    mean_max_sizes = [s.max_size for s in snapshots if s.max_size > 0]

    run = SimulationRun(
        fusion_threshold=config.fusion_threshold,
        interaction_radius=config.interaction_radius,
        composite_size_decay_scale=config.composite_size_decay_scale,
        num_particles=config.num_particles,
        num_species=config.num_species,
        num_steps=num_steps,
        dt=config.dt,
        duration_sec=duration,
        snapshots=snapshots,
        peak_composites=int(np.max(num_alive_list)) if num_alive_list else 0,
        peak_composite_size=int(np.max(max_sizes)) if max_sizes else 0,
        mean_max_size=float(np.mean(mean_max_sizes)) if mean_max_sizes else 0.0,
        final_composites=snapshots[-1].num_alive if snapshots else 0,
        final_max_size=snapshots[-1].max_size if snapshots else 0,
        formation_rate=float(new_composite_count / num_steps) if num_steps > 0 else 0.0,
    )
    return run


def create_trajectory_plot(runs: list) -> str:
    """Create plot of max composite size trajectory with compact labeling."""
    # Group by decay scale for separate subplots
    by_decay = {}
    for run in runs:
        if run.snapshots[0].step == 0:  # Access via snapshot to get decay info
            decay_key = getattr(run, '_decay_scale', 0.05)  # Fallback
            if decay_key not in by_decay:
                by_decay[decay_key] = []
            by_decay[decay_key].append(run)

    # Just use all runs with color/style encoding
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_idx = 0

    for run in runs:
        steps = [s.step for s in run.snapshots]
        max_sizes = [s.max_size for s in run.snapshots]
        num_alive = [s.num_alive for s in run.snapshots]

        # Compact label: use abbreviated parameters
        label = f"T{run.fusion_threshold:.1f}R{run.interaction_radius:.1f}D{run.composite_size_decay_scale:.2f}"
        color = colors[color_idx % len(colors)]

        ax1.plot(steps, max_sizes, color=color, linewidth=1.5, alpha=0.6, label=label)
        ax2.plot(steps, num_alive, color=color, linewidth=1.5, alpha=0.6, label=label)
        color_idx += 1

    ax1.set_xlabel("Simulation Step", fontsize=11)
    ax1.set_ylabel("Max Composite Size (members)", fontsize=11)
    ax1.set_title("Growth Trajectory (T=threshold, R=radius, D=decay_scale)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Simulation Step", fontsize=11)
    ax2.set_ylabel("Number of Alive Composites", fontsize=11)
    ax2.set_title("Population Dynamics", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_size_distribution_plots(runs: list) -> dict:
    """Create individual histograms for each run, return as base64-encoded dict."""
    plots_dict = {}

    for i, run in enumerate(runs):
        fig, ax = plt.subplots(figsize=(10, 5))

        final_snapshot = run.snapshots[-1]
        sizes = final_snapshot.size_distribution
        bins = range(1, len(sizes) + 1)

        ax.bar(bins, sizes, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Composite Size (members)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlim(0.5, max(bins) + 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        title = f"Threshold={run.fusion_threshold:.2f} | Radius={run.interaction_radius:.1f} | Decay Scale={run.composite_size_decay_scale:.2f}"
        ax.set_title(title, fontsize=13, fontweight='bold')

        key = f"run_{i}_T{run.fusion_threshold:.2f}_R{run.interaction_radius:.1f}_D{run.composite_size_decay_scale:.2f}"
        plots_dict[key] = {
            'base64': fig_to_base64(fig),
            'threshold': run.fusion_threshold,
            'radius': run.interaction_radius,
            'decay_scale': run.composite_size_decay_scale,
            'label': f"T={run.fusion_threshold:.2f}, R={run.interaction_radius:.1f}, D={run.composite_size_decay_scale:.2f}"
        }
        plt.close(fig)

    return plots_dict


def create_metrics_comparison_plot(runs: list) -> str:
    """Create comparison plots of key metrics across parameter sweeps, including decay scale."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    thresholds_set = sorted(set(r.fusion_threshold for r in runs))
    radii_set = sorted(set(r.interaction_radius for r in runs))
    decay_set = sorted(set(r.composite_size_decay_scale for r in runs))

    colors_decay = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Peak size vs decay (for fixed threshold and radius)
    if len(decay_set) > 1 and len(thresholds_set) > 0 and len(radii_set) > 0:
        threshold = thresholds_set[0]
        radius = radii_set[0]
        decay_vals = []
        peak_sizes = []
        for d in decay_set:
            matching = [r for r in runs if r.fusion_threshold == threshold and
                       r.interaction_radius == radius and r.composite_size_decay_scale == d]
            if matching:
                decay_vals.append(d)
                peak_sizes.append(matching[0].peak_composite_size)
        if decay_vals:
            axes[0, 0].plot(decay_vals, peak_sizes, 'o-', linewidth=2.5, markersize=10, color='#9467bd')
            axes[0, 0].set_xlabel("Size Decay Scale", fontsize=11)
            axes[0, 0].set_ylabel("Peak Composite Size", fontsize=11)
            axes[0, 0].set_title(f"Effect of Size Penalty\n(T={threshold:.2f}, R={radius:.1f})", fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)

    # Peak size vs threshold (for fixed radius and decay)
    if len(thresholds_set) > 1 and len(radii_set) > 0 and len(decay_set) > 0:
        radius = radii_set[0]
        decay = decay_set[len(decay_set)//2]  # Middle decay value
        threshold_vals = []
        peak_sizes_list = []
        for t in thresholds_set:
            matching = [r for r in runs if r.fusion_threshold == t and
                       r.interaction_radius == radius and r.composite_size_decay_scale == decay]
            if matching:
                threshold_vals.append(t)
                peak_sizes_list.append(matching[0].peak_composite_size)
        if threshold_vals:
            axes[0, 1].plot(threshold_vals, peak_sizes_list, 's-', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel("Fusion Threshold", fontsize=11)
            axes[0, 1].set_ylabel("Peak Composite Size", fontsize=11)
            axes[0, 1].set_title(f"Threshold Sensitivity\n(R={radius:.1f}, D={decay:.2f})", fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)

    # Formation rate vs decay
    if len(decay_set) > 1 and len(thresholds_set) > 0 and len(radii_set) > 0:
        threshold = thresholds_set[0]
        radius = radii_set[0]
        decay_vals = []
        rates = []
        for d in decay_set:
            matching = [r for r in runs if r.fusion_threshold == threshold and
                       r.interaction_radius == radius and r.composite_size_decay_scale == d]
            if matching:
                decay_vals.append(d)
                rates.append(matching[0].formation_rate)
        if decay_vals:
            axes[1, 0].plot(decay_vals, rates, 'd-', linewidth=2.5, markersize=10, color='#17becf')
            axes[1, 0].set_xlabel("Size Decay Scale", fontsize=11)
            axes[1, 0].set_ylabel("Formation Rate (composites/step)", fontsize=11)
            axes[1, 0].set_title(f"Formation Rate vs Penalty\n(T={threshold:.2f}, R={radius:.1f})", fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)

    # Peak size vs radius (for fixed threshold and decay)
    if len(radii_set) > 1 and len(thresholds_set) > 0 and len(decay_set) > 0:
        threshold = thresholds_set[0]
        decay = decay_set[len(decay_set)//2]
        radius_vals = []
        peak_sizes_list = []
        for r in radii_set:
            matching = [run for run in runs if run.fusion_threshold == threshold and
                       run.interaction_radius == r and run.composite_size_decay_scale == decay]
            if matching:
                radius_vals.append(r)
                peak_sizes_list.append(matching[0].peak_composite_size)
        if radius_vals:
            axes[1, 1].plot(radius_vals, peak_sizes_list, '^-', linewidth=2, markersize=8, color='red')
            axes[1, 1].set_xlabel("Interaction Radius", fontsize=11)
            axes[1, 1].set_ylabel("Peak Composite Size", fontsize=11)
            axes[1, 1].set_title(f"Radius Sensitivity\n(T={threshold:.2f}, D={decay:.2f})", fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_html_report(runs: list, output_path: str):
    """Generate a professional HTML report with embedded plots, sortable tables, and interactive elements."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate plots
    trajectory_plot = create_trajectory_plot(runs)
    size_dist_plots_dict = create_size_distribution_plots(runs)
    metrics_plot = create_metrics_comparison_plot(runs)

    # Build histogram selector HTML
    histogram_options = ""
    histogram_images = ""
    for i, (key, plot_info) in enumerate(size_dist_plots_dict.items()):
        selected = "selected" if i == 0 else ""
        histogram_options += f'<option value="hist_{i}" {selected}>{plot_info["label"]}</option>\n'
        display = "block" if i == 0 else "none"
        histogram_images += f'<div id="hist_{i}" class="histogram-display" style="display: {display};"><img src="data:image/png;base64,{plot_info["base64"]}" alt="Histogram"></div>\n'

    # Build metrics table rows with data attributes for sorting
    metrics_rows = ""
    for i, run in enumerate(runs):
        metrics_rows += f"""
        <tr class="{'row-highlight' if i % 2 == 0 else ''}" data-threshold="{run.fusion_threshold:.3f}" data-radius="{run.interaction_radius:.2f}" data-decay="{run.composite_size_decay_scale:.2f}" data-peak-size="{run.peak_composite_size}" data-formation="{run.formation_rate:.3f}">
            <td>{run.fusion_threshold:.3f}</td>
            <td>{run.interaction_radius:.2f}</td>
            <td>{run.composite_size_decay_scale:.3f}</td>
            <td>{run.peak_composite_size}</td>
            <td>{run.peak_composites}</td>
            <td>{run.mean_max_size:.2f}</td>
            <td>{run.formation_rate:.3f}</td>
            <td>{run.duration_sec:.1f}s</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Composite Statistics Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                padding: 40px;
            }}

            header {{
                border-bottom: 3px solid #2c3e50;
                margin-bottom: 30px;
                padding-bottom: 20px;
            }}

            h1 {{
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}

            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                font-style: italic;
            }}

            h2 {{
                color: #34495e;
                font-size: 1.8em;
                margin-top: 40px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}

            h3 {{
                color: #555;
                font-size: 1.2em;
                margin-top: 25px;
                margin-bottom: 15px;
            }}

            .intro-text {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                line-height: 1.8;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
            }}

            table th {{
                background: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                cursor: pointer;
                user-select: none;
            }}

            table th:hover {{
                background: #2c3e50;
            }}

            table th::after {{
                content: ' ⇅';
                opacity: 0.5;
            }}

            table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}

            table tr:hover {{
                background: #f8f9fa;
            }}

            table tr.row-highlight {{
                background: #f0f4f8;
            }}

            .plot-container {{
                margin: 30px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 5px;
                text-align: center;
            }}

            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}

            .caption {{
                color: #7f8c8d;
                font-size: 0.9em;
                font-style: italic;
                margin-top: 10px;
                line-height: 1.6;
            }}

            .findings {{
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}

            .findings strong {{
                color: #2c3e50;
            }}

            .metric {{
                display: inline-block;
                background: white;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 15px 20px;
                margin: 10px;
                text-align: center;
                min-width: 150px;
            }}

            .metric-value {{
                font-size: 1.8em;
                font-weight: bold;
                color: #2980b9;
            }}

            .metric-label {{
                font-size: 0.85em;
                color: #7f8c8d;
                margin-top: 5px;
            }}

            .selector-container {{
                margin: 20px 0;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 5px;
            }}

            select {{
                padding: 10px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 1em;
                cursor: pointer;
            }}

            .histogram-display {{
                display: none;
                margin-top: 20px;
            }}

            .histogram-display.active {{
                display: block;
            }}

            .size-penalty-info {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}

            .size-penalty-info code {{
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}

            footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
        <script>
            function sortTable(th) {{
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const columnIndex = Array.from(th.parentNode.children).indexOf(th);
                const isNumeric = !isNaN(parseFloat(rows[0].children[columnIndex].textContent));

                rows.sort((a, b) => {{
                    const aVal = a.children[columnIndex].textContent.trim();
                    const bVal = b.children[columnIndex].textContent.trim();

                    if (isNumeric) {{
                        return parseFloat(aVal) - parseFloat(bVal);
                    }} else {{
                        return aVal.localeCompare(bVal);
                    }}
                }});

                rows.forEach(row => tbody.appendChild(row));
            }}

            function switchHistogram(value) {{
                document.querySelectorAll('.histogram-display').forEach(el => el.style.display = 'none');
                document.getElementById(value).style.display = 'block';
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                document.querySelectorAll('table th').forEach(th => {{
                    th.addEventListener('click', function() {{ sortTable(this); }});
                }});
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🧬 Composite Statistics Report</h1>
                <div class="timestamp">Generated: {timestamp}</div>
            </header>

            <section class="intro-text">
                <p><strong>Objective:</strong> Understand how fusion threshold, interaction radius, and
                <strong>size decay penalty</strong> affect composite formation, growth, and final size distribution.
                This analysis sweeps key parameters to identify the conditions that maximize open-ended complexity.</p>
            </section>

            <h2>📊 Summary Metrics</h2>
            <div>
                <div class="metric">
                    <div class="metric-value">{len(runs)}</div>
                    <div class="metric-label">Simulations Run</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{max(r.peak_composite_size for r in runs)}</div>
                    <div class="metric-label">Max Observed Size</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{max(r.peak_composites for r in runs)}</div>
                    <div class="metric-label">Max Alive Count</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{max(r.formation_rate for r in runs):.3f}</div>
                    <div class="metric-label">Peak Formation Rate</div>
                </div>
            </div>

            <h2>⚠️ The Size Decay Penalty</h2>
            <div class="size-penalty-info">
                <p><strong>Critical Finding:</strong> Larger composites have <strong>shorter half-lives</strong>,
                making them decay faster. This penalty is controlled by <code>composite_size_decay_scale</code> (default: 0.05).</p>

                <p style="margin-top: 10px;"><strong>How it works:</strong> For a composite with N members,</p>
                <code style="display: block; margin: 10px 0; padding: 10px; background: white;">
                    half_life = base_hl / (1.0 + decay_scale × max(0, N - 2))
                </code>

                <p style="margin-top: 10px;"><strong>Example:</strong> With decay_scale=0.05:
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>2-member composite: half_life = base_hl / 1.0 = <strong>100%</strong></li>
                    <li>5-member composite: half_life = base_hl / 1.15 = <strong>87% stability</strong></li>
                    <li>10-member composite: half_life = base_hl / 1.4 = <strong>71% stability</strong></li>
                </ul></p>

                <p style="margin-top: 10px;"><strong>This sweep tests:</strong> Does disabling (decay_scale=0.0) or increasing
                (decay_scale=0.10) this penalty allow composites to grow larger?</p>
            </div>

            <h2>📈 Trajectory Analysis</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{trajectory_plot}" alt="Trajectory Plot">
                <p class="caption">
                    <strong>Top:</strong> Maximum composite size trajectory (T=threshold, R=radius, D=decay_scale).
                    Steeper curves = faster growth; plateaus = saturation. Click column headers below to sort by any metric.
                    <strong>Bottom:</strong> Population dynamics showing alive composite counts over time.
                </p>
            </div>

            <h2>📋 Size Distributions (Interactive)</h2>
            <div class="selector-container">
                <label for="hist-select"><strong>Select a simulation:</strong></label>
                <select id="hist-select" onchange="switchHistogram(this.value)">
                    {histogram_options}
                </select>
            </div>
            {histogram_images}
            <p class="caption" style="margin-top: 10px;">
                Use the dropdown to explore final size distributions across different parameter combinations.
                A broad distribution indicates diverse composite sizes; a narrow one suggests composites are
                locked at specific sizes due to binding energy constraints.
            </p>

            <h2>🔍 Parameter Sensitivity</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{metrics_plot}" alt="Metrics Comparison Plot">
                <p class="caption">
                    Sensitivity analysis: how peak size and formation rate respond to parameter changes.
                    Steep curves = strong sensitivity; flat lines = weak response.
                </p>
            </div>

            <h2>📑 Detailed Metrics Table (Click headers to sort)</h2>
            <table id="metrics-table">
                <thead>
                    <tr>
                        <th>Fusion Threshold</th>
                        <th>Interaction Radius</th>
                        <th>Size Decay Scale</th>
                        <th>Peak Size</th>
                        <th>Peak Composites</th>
                        <th>Mean Max Size</th>
                        <th>Formation Rate</th>
                        <th>Runtime</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>

            <h2>💡 Key Insights</h2>
            <div class="findings">
                <h3>Size Decay Penalty (NEW)</h3>
                <p>This sweep directly tests the hypothesis: <strong>Is the size penalty the growth limiter?</strong></p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>decay_scale = 0.0 (disabled):</strong> Composites not penalized by size.
                        If they grow larger than with penalty, then size penalty is the bottleneck.</li>
                    <li><strong>decay_scale = 0.05 (default):</strong> Standard configuration. Larger composites decay 30% faster at 10 members.</li>
                    <li><strong>decay_scale = 0.10 (severe):</strong> Extra aggressive penalty. Expect smaller composites, faster turnover.</li>
                </ul>

                <h3>Fusion Threshold + Interaction Radius</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Fusion Threshold:</strong> Controls binding energy barrier. Lower = more fusions, higher formation rate.</li>
                    <li><strong>Interaction Radius:</strong> Larger radius = more neighbors in range = more fusion opportunities.</li>
                </ul>
            </div>

            <h2>🎯 What to Look For</h2>
            <ol style="margin-left: 20px; margin-top: 15px; line-height: 2;">
                <li><strong>Decay scale effect:</strong> Do composites grow larger when decay_scale = 0.0?
                    If yes, size penalty is a major bottleneck. If no, other factors (binding energy) dominate.</li>
                <li><strong>Peak composite size:</strong> Does it exceed ~8 members under any condition?
                    This would indicate hash chemistry permits larger stable structures.</li>
                <li><strong>Population dynamics:</strong> Are populations stable or oscillatory?
                    Stability suggests sustainable formation-decay cycles.</li>
                <li><strong>Formation rate trends:</strong> Does decay_scale affect formation rate?
                    If composites live longer, they may accumulate more, increasing apparent formation rate.</li>
            </ol>

            <footer>
                <p>Half-Life Particle Simulator | Composite Reaction Network Evolution | Phase 2 Analysis</p>
                <p>Report generated by test_composite_statistics.py</p>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Report saved to {output_path}")


def test_composite_statistics():
    """Main test: run parameter sweep and generate report."""
    print("\n" + "="*70)
    print("COMPOSITE STATISTICS TEST — Parameter Sensitivity Analysis")
    print("="*70)

    # Define parameter sweep — now including size penalty!
    fusion_thresholds = [0.10, 0.20]
    interaction_radii = [3.5, 4.5]
    size_decay_scales = [0.0, 0.05, 0.10]  # NEW: test how size penalty affects growth
    num_steps = 600

    print(f"\nParameter sweep:")
    print(f"  Fusion thresholds: {fusion_thresholds}")
    print(f"  Interaction radii: {interaction_radii}")
    print(f"  Size decay scales: {size_decay_scales}")
    print(f"  Steps per run: {num_steps}")
    print(f"  Total runs: {len(fusion_thresholds) * len(interaction_radii) * len(size_decay_scales)}")
    print()

    runs = []
    run_count = 0
    total_runs = len(fusion_thresholds) * len(interaction_radii) * len(size_decay_scales)

    for threshold in fusion_thresholds:
        for radius in interaction_radii:
            for decay_scale in size_decay_scales:
                run_count += 1
                print(f"[{run_count}/{total_runs}] Running: threshold={threshold:.2f}, radius={radius:.1f}, decay={decay_scale:.2f}... ", end="", flush=True)

                config = SimConfig(
                    fusion_threshold=threshold,
                    interaction_radius=radius,
                    composite_size_decay_scale=decay_scale,
                )

                t0 = time.time()
                run = run_simulation_with_stats(config, num_steps=num_steps)
                elapsed = time.time() - t0

                print(f"done ({elapsed:.1f}s) | peak_size={run.peak_composite_size}, "
                      f"peak_composites={run.peak_composites}, formation_rate={run.formation_rate:.3f}")

                runs.append(run)

    # Generate report
    print("\nGenerating HTML report... ", end="", flush=True)
    os_reports_dir = "tests/reports"
    os.makedirs(os_reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(os_reports_dir, f"composite_statistics_{timestamp}.html")

    generate_html_report(runs, report_path)
    print(f"done\n")

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    for run in runs:
        print(f"\nThreshold={run.fusion_threshold:.2f}, Radius={run.interaction_radius:.1f}")
        print(f"  Peak size: {run.peak_composite_size} members")
        print(f"  Max alive: {run.peak_composites} composites")
        print(f"  Mean max size: {run.mean_max_size:.2f}")
        print(f"  Formation rate: {run.formation_rate:.3f} composites/step")
        print(f"  Final state: {run.final_composites} alive, max size {run.final_max_size}")

    print("\n" + "="*70)
    print(f"✓ Report: {report_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_composite_statistics()
