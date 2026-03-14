"""
Entry point for the Half-Life particle simulator.

Run with:
    python -m halflife.main
    python -m halflife.main --seed 42 --species 16 --particles 20000

The main loop:
  1. Initialize world and renderer
  2. JIT-warm-up the simulation step
  3. Enter the render loop:
     a. Handle pygame events (keyboard controls)
     b. Dispatch simulation_step() (async — returns immediately)
     c. Call renderer.update() — this triggers the GPU→CPU sync (np.asarray)
     d. Call renderer.render() — draws the frame
     e. Await JAX computation for next iteration

Async overlap: JAX dispatches the next simulation step before rendering
the current frame. While the GPU computes, the CPU prepares the render
from the previous step. This hides most of the simulation latency.

Keyboard controls:
  Space       — pause / resume
  + / =       — increase simulation steps per render frame
  - / _       — decrease simulation steps per render frame
  B           — toggle composite visualization (bonds ↔ merged)
  R           — reset to initial state
  S           — save screenshot (PNG)
  Q / Escape  — quit
"""

import argparse
import time
import os
import numpy as np
import jax
import jax.numpy as jnp

import pygame

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params
from halflife.step import simulation_step
from halflife.renderer import Renderer


def parse_args():
    p = argparse.ArgumentParser(description="Half-Life Particle Simulator")
    p.add_argument('--seed',      type=int,   default=0,      help='Random seed')
    p.add_argument('--species',   type=int,   default=None,   help='Number of species (overrides config)')
    p.add_argument('--particles', type=int,   default=None,   help='Number of initial particles (overrides config)')
    p.add_argument('--width',     type=float, default=None,   help='World width')
    p.add_argument('--height',    type=float, default=None,   help='World height')
    p.add_argument('--no-chemistry', action='store_true',     help='Disable fusion and decay (physics only)')
    return p.parse_args()


def build_config(args) -> SimConfig:
    """Build SimConfig, applying any command-line overrides."""
    # Start with defaults
    kwargs = {}
    if args.species   is not None: kwargs['num_species']        = args.species
    if args.particles is not None: kwargs['num_particles_init'] = args.particles
    if args.width     is not None: kwargs['world_width']        = args.width
    if args.height    is not None: kwargs['world_height']       = args.height
    return SimConfig(**kwargs)


def run(config: SimConfig = None, seed: int = 0, enable_chemistry: bool = True):
    """
    Main simulation and render loop.

    Args:
        config:           SimConfig — if None, uses defaults
        seed:             random seed for world initialization
        enable_chemistry: if False, skips fusion/decay (useful for physics debugging)
    """
    if config is None:
        config = SimConfig()

    # ── Initialize ────────────────────────────────────────────────────────────
    print(f"Initializing world: {config.num_particles_init:,} particles, "
          f"{config.num_species} species, world {config.world_width}x{config.world_height}")

    state  = initialize_world(config, seed=seed)
    params = initialize_interaction_params(config, seed=seed + 1)
    renderer = Renderer(config)

    # JIT-compile the simulation step (first call triggers compilation)
    print("JIT-compiling simulation step... (this takes ~10-30 seconds first time)")
    t0 = time.time()
    _step_fn = jax.jit(simulation_step, static_argnums=(2,))
    # Warm up
    _ = _step_fn(state, params, config)
    jax.block_until_ready(_)
    print(f"JIT compilation done in {time.time() - t0:.1f}s")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    running       = True
    paused        = False
    steps_per_frame = 1      # simulation steps per rendered frame
    clock         = pygame.time.Clock()
    frame_count   = 0
    fps_history   = []

    # Screenshot counter
    screenshot_dir = "screenshots"

    print("Running. Controls: Space=pause, +/-=speed, B=composite mode, R=reset, Q=quit")

    while running:
        t_frame_start = time.time()

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(steps_per_frame * 2, 64)
                    print(f"Steps per frame: {steps_per_frame}")

                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    steps_per_frame = max(steps_per_frame // 2, 1)
                    print(f"Steps per frame: {steps_per_frame}")

                elif event.key == pygame.K_b:
                    renderer.toggle_composite_mode()
                    print(f"Composite mode: {renderer.composite_mode}")

                elif event.key == pygame.K_r:
                    print("Resetting...")
                    state = initialize_world(config, seed=seed)
                    params = initialize_interaction_params(config, seed=seed + 1)

                elif event.key == pygame.K_s:
                    os.makedirs(screenshot_dir, exist_ok=True)
                    fname = os.path.join(screenshot_dir, f"halflife_{int(time.time())}.png")
                    pygame.image.save(pygame.display.get_surface(), fname)
                    print(f"Screenshot saved: {fname}")

        # ── Simulate ──────────────────────────────────────────────────────────
        if not paused:
            # Dispatch next frame's simulation before rendering current
            # (JAX async dispatch — computation runs in background on GPU)
            for _ in range(steps_per_frame):
                state = _step_fn(state, params, config)

        # ── Render ────────────────────────────────────────────────────────────
        # renderer.update() triggers the GPU→CPU transfer (np.asarray)
        # This also syncs with JAX's async dispatch from the previous step
        renderer.update(state)

        n_alive = int(np.sum(np.asarray(state.particles.alive)))
        step_count = int(np.asarray(state.step_count))
        fps = clock.get_fps()

        renderer.render(fps, step_count, n_alive)

        # ── Timing ────────────────────────────────────────────────────────────
        clock.tick(config.fps_target)
        frame_count += 1

        if frame_count % 60 == 0:
            fps_val = clock.get_fps()
            sim_time = float(np.asarray(state.time))
            print(f"FPS: {fps_val:.1f} | Sim time: {sim_time:.1f} | "
                  f"Alive: {n_alive:,} | Steps: {step_count:,}")

    renderer.close()
    print("Simulation ended.")


def main():
    args = parse_args()
    config = build_config(args)
    run(config=config, seed=args.seed, enable_chemistry=not args.no_chemistry)


if __name__ == '__main__':
    main()
