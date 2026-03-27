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
    """Create plot of max composite size trajectory for all runs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for run in runs:
        steps = [s.step for s in run.snapshots]
        max_sizes = [s.max_size for s in run.snapshots]
        num_alive = [s.num_alive for s in run.snapshots]

        label = f"threshold={run.fusion_threshold:.2f}, radius={run.interaction_radius:.1f}"
        ax1.plot(steps, max_sizes, label=label, linewidth=2, alpha=0.7)
        ax2.plot(steps, num_alive, label=label, linewidth=2, alpha=0.7)

    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Max Composite Size (members)")
    ax1.set_title("Composite Growth Trajectory: Maximum Size Over Time")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Number of Alive Composites")
    ax2.set_title("Composite Population: Alive Composites Over Time")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_size_distribution_plot(runs: list) -> str:
    """Create plot of final size distributions for all runs."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5*len(runs), 4))
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        final_snapshot = run.snapshots[-1]
        sizes = final_snapshot.size_distribution
        bins = range(1, len(sizes) + 1)

        ax.bar(bins, sizes, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Composite Size (members)")
        ax.set_ylabel("Count")
        title = f"Threshold={run.fusion_threshold:.2f}\nRadius={run.interaction_radius:.1f}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_metrics_comparison_plot(runs: list) -> str:
    """Create comparison plots of key metrics across parameter sweeps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Group by parameter
    by_threshold = {}
    by_radius = {}

    for run in runs:
        if run.interaction_radius not in by_threshold:
            by_threshold[run.interaction_radius] = {}
        if run.fusion_threshold not in by_radius:
            by_radius[run.fusion_threshold] = {}

        by_threshold[run.interaction_radius][run.fusion_threshold] = run
        by_radius[run.fusion_threshold][run.interaction_radius] = run

    # Peak size vs threshold (for fixed radius)
    if len(by_threshold) > 0:
        radius = list(by_threshold.keys())[0]
        thresholds = sorted(by_threshold[radius].keys())
        peak_sizes = [by_threshold[radius][t].peak_composite_size for t in thresholds]
        axes[0, 0].plot(thresholds, peak_sizes, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Fusion Threshold")
        axes[0, 0].set_ylabel("Peak Composite Size")
        axes[0, 0].set_title(f"Peak Size vs Threshold (radius={radius})")
        axes[0, 0].grid(True, alpha=0.3)

    # Peak size vs radius (for fixed threshold)
    if len(by_radius) > 0:
        threshold = list(by_radius.keys())[0]
        radii = sorted(by_radius[threshold].keys())
        peak_sizes = [by_radius[threshold][r].peak_composite_size for r in radii]
        axes[0, 1].plot(radii, peak_sizes, 's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel("Interaction Radius")
        axes[0, 1].set_ylabel("Peak Composite Size")
        axes[0, 1].set_title(f"Peak Size vs Radius (threshold={threshold})")
        axes[0, 1].grid(True, alpha=0.3)

    # Formation rate
    thresholds_set = sorted(set(r.fusion_threshold for r in runs))
    radii_set = sorted(set(r.interaction_radius for r in runs))

    if len(thresholds_set) > 1:
        rates = [r.formation_rate for r in runs if r.interaction_radius == radii_set[0]]
        axes[1, 0].plot(thresholds_set, rates, 'd-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel("Fusion Threshold")
        axes[1, 0].set_ylabel("Formation Rate (composites/step)")
        axes[1, 0].set_title(f"Formation Rate vs Threshold (radius={radii_set[0]})")
        axes[1, 0].grid(True, alpha=0.3)

    if len(radii_set) > 1:
        rates = [r.formation_rate for r in runs if r.fusion_threshold == thresholds_set[0]]
        axes[1, 1].plot(radii_set, rates, '^-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_xlabel("Interaction Radius")
        axes[1, 1].set_ylabel("Formation Rate (composites/step)")
        axes[1, 1].set_title(f"Formation Rate vs Radius (threshold={thresholds_set[0]})")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_html_report(runs: list, output_path: str):
    """Generate a professional HTML report with embedded plots."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate plots
    trajectory_plot = create_trajectory_plot(runs)
    size_dist_plot = create_size_distribution_plot(runs)
    metrics_plot = create_metrics_comparison_plot(runs)

    # Build metrics table
    metrics_rows = ""
    for i, run in enumerate(runs):
        metrics_rows += f"""
        <tr class="{'row-highlight' if i % 2 == 0 else ''}">
            <td>{run.fusion_threshold:.3f}</td>
            <td>{run.interaction_radius:.2f}</td>
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

            footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🧬 Composite Statistics Report</h1>
                <div class="timestamp">Generated: {timestamp}</div>
            </header>

            <section class="intro-text">
                <p><strong>Objective:</strong> Understand how fusion threshold and interaction radius
                affect composite formation, growth, and final size distribution. This analysis sweeps
                key parameters to identify the conditions that maximize open-ended complexity in the
                particle dynamics.</p>
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

            <h2>📈 Trajectory Analysis</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{trajectory_plot}" alt="Trajectory Plot">
                <p class="caption">
                    <strong>Top:</strong> Maximum composite size over time for each parameter combination.
                    Steeper curves indicate faster growth; plateauing curves indicate saturation.
                    <strong>Bottom:</strong> Number of alive composites over time, showing population dynamics.
                </p>
            </div>

            <div class="findings">
                <strong>Key Observation:</strong> The trajectory plots reveal whether parameter changes
                enable composites to grow larger or form more consistently. Lower fusion thresholds typically
                accelerate formation but may lead to unstable, short-lived composites.
            </div>

            <h2>📋 Size Distributions</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{size_dist_plot}" alt="Size Distribution Plot">
                <p class="caption">
                    Distribution of composite sizes at the end of each simulation. A broad, right-skewed
                    distribution indicates diverse composite complexity. A narrow distribution suggests
                    composites are locked at a specific size.
                </p>
            </div>

            <h2>🔍 Parameter Sensitivity</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{metrics_plot}" alt="Metrics Comparison Plot">
                <p class="caption">
                    Sensitivity analysis: how peak size and formation rate respond to fusion threshold
                    and interaction radius. Steep curves indicate strong parameter sensitivity.
                </p>
            </div>

            <h2>📑 Detailed Metrics Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Fusion Threshold</th>
                        <th>Interaction Radius</th>
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

            <h2>💡 Interpretation Guide</h2>
            <div class="findings">
                <h3>Fusion Threshold Effects</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Lower threshold (0.05-0.1):</strong> More fusions, rapid composite formation,
                        but binding energy diversity may prevent large stable composites.</li>
                    <li><strong>Medium threshold (0.15-0.2):</strong> Balanced formation and stability.
                        Most composites will survive longer and grow more consistently.</li>
                    <li><strong>Higher threshold (0.25-0.3):</strong> Rare fusions, fewer composites,
                        but those that form tend to be more stable.</li>
                </ul>

                <h3>Interaction Radius Effects</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Smaller radius (2.0-2.5):</strong> Fewer neighbors per particle, fewer
                        opportunities for fusion, smaller composites.</li>
                    <li><strong>Larger radius (4.0-5.0):</strong> More neighbors, higher fusion rate,
                        potential for larger composites but more competition.</li>
                </ul>
            </div>

            <h2>🎯 What to Look For</h2>
            <ol style="margin-left: 20px; margin-top: 15px; line-height: 2;">
                <li><strong>Peak composite size:</strong> Does it grow beyond ~8 members? This indicates
                    the hash chemistry permits large stable structures.</li>
                <li><strong>Plateau behavior:</strong> Do curves flatten out? If so, the system has reached
                    an equilibrium size limit—likely the binding energy bottleneck.</li>
                <li><strong>Population stability:</strong> Do composite counts stabilize or oscillate?
                    Stability suggests sustainable autocatalytic cycles.</li>
                <li><strong>Formation rate:</strong> Does the rate change over time? Early bursts followed
                    by plateaus suggest transient complexity.</li>
            </ol>

            <footer>
                <p>Half-Life Particle Simulator | Composite Reaction Network Evolution</p>
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

    # Define parameter sweep
    fusion_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    interaction_radii = [2.5, 3.5, 4.5]
    num_steps = 600

    print(f"\nParameter sweep:")
    print(f"  Fusion thresholds: {fusion_thresholds}")
    print(f"  Interaction radii: {interaction_radii}")
    print(f"  Steps per run: {num_steps}")
    print(f"  Total runs: {len(fusion_thresholds) * len(interaction_radii)}")
    print()

    runs = []
    run_count = 0
    total_runs = len(fusion_thresholds) * len(interaction_radii)

    for threshold in fusion_thresholds:
        for radius in interaction_radii:
            run_count += 1
            print(f"[{run_count}/{total_runs}] Running: threshold={threshold:.2f}, radius={radius:.1f}... ", end="", flush=True)

            config = SimConfig(
                fusion_threshold=threshold,
                interaction_radius=radius,
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
