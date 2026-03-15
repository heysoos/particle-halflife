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
import collections
import time
import os
import numpy as np
import jax
import jax.numpy as jnp

import pygame

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params, initialize_physics_params
from halflife.step import make_run_n_steps
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

    state   = initialize_world(config, seed=seed)
    params  = initialize_interaction_params(config, seed=seed + 1)
    physics = initialize_physics_params(config)
    renderer = Renderer(config)

    # JIT-compile via make_run_n_steps (first call triggers compilation)
    print("JIT-compiling simulation step... (this takes ~10-30 seconds first time)")
    t0 = time.time()
    run_n = make_run_n_steps(config)
    # Warm up with a single step
    _ = run_n(state, params, physics, 1)
    jax.block_until_ready(_)
    print(f"JIT compilation done in {time.time() - t0:.1f}s")

    # Frame-time profiling deques (rolling 60-frame windows)
    _t_sim    = collections.deque(maxlen=60)
    _t_update = collections.deque(maxlen=60)
    _t_render = collections.deque(maxlen=60)

    # ── Main Loop ─────────────────────────────────────────────────────────────
    running         = True
    paused          = False
    steps_per_frame = 1      # simulation steps per rendered frame
    clock           = pygame.time.Clock()
    frame_count     = 0

    # Screenshot counter
    screenshot_dir = "screenshots"

    print("Running. Controls: Space=pause, +/-=speed, B=composite mode, R=reset, Q=quit")

    # Async pipeline: pre-dispatch first batch so GPU starts immediately
    pending_state = state
    if not paused:
        pending_state = run_n(pending_state, params, physics, steps_per_frame)

    while running:
        t_frame_start = time.time()

        # ── Events ────────────────────────────────────────────────────────────
        reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = renderer.handle_click(event.pos)
                if action == 'pause':
                    paused = not paused
                    renderer.set_paused(paused)
                    print("Paused" if paused else "Resumed")
                elif action == 'toggle_bonds':
                    renderer.toggle_composite_mode()
                    print(f"Composite mode: {renderer.composite_mode}")
                elif action == 'toggle_stats':
                    renderer.toggle_stats()
                elif action == 'toggle_events':
                    renderer.toggle_events()
                elif action == 'toggle_params':
                    renderer.toggle_params()
                elif action == 'reset':
                    reset_requested = True
                else:
                    renderer.handle_mousedown_slider(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                renderer.handle_mousemotion(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                renderer.handle_mouseup()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    renderer.set_paused(paused)
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
                    reset_requested = True

                elif event.key == pygame.K_s:
                    os.makedirs(screenshot_dir, exist_ok=True)
                    fname = os.path.join(screenshot_dir, f"halflife_{int(time.time())}.png")
                    pygame.image.save(pygame.display.get_surface(), fname)
                    print(f"Screenshot saved: {fname}")

        if reset_requested:
            print("Resetting...")
            state  = initialize_world(config, seed=seed)
            params = initialize_interaction_params(config, seed=seed + 1)
            # physics intentionally NOT reset — slider values persist across resets
            pending_state = state
            if not paused:
                pending_state = run_n(pending_state, params, physics, steps_per_frame)

        # ── Consume slider updates (before next dispatch) ─────────────────────
        updates = renderer.get_physics_updates()
        if updates:
            physics = physics._replace(**{k: jnp.float32(v) for k, v in updates.items()})

        # ── Async pipeline ────────────────────────────────────────────────────
        # Dispatch NEXT batch before blocking on current — GPU computes frame N+1
        # while CPU renders frame N. This hides simulation latency.
        t0_sim = time.perf_counter()
        if not paused and not reset_requested:
            next_pending = run_n(pending_state, params, physics, steps_per_frame)
        else:
            next_pending = pending_state
        _t_sim.append(time.perf_counter() - t0_sim)

        # ── Render ────────────────────────────────────────────────────────────
        # renderer.update() triggers the GPU→CPU transfer for pending_state.
        # Meanwhile the GPU is already working on next_pending.
        t0_update = time.perf_counter()
        renderer.update(pending_state)
        _t_update.append(time.perf_counter() - t0_update)

        n_alive    = int(np.sum(np.asarray(pending_state.particles.alive)))
        step_count = int(np.asarray(pending_state.step_count))
        fps        = clock.get_fps()

        t0_render = time.perf_counter()
        renderer.render(fps, step_count, n_alive)
        _t_render.append(time.perf_counter() - t0_render)

        # Advance pipeline
        if not paused:
            pending_state = next_pending

        # ── Timing ────────────────────────────────────────────────────────────
        clock.tick(config.fps_target)
        frame_count += 1

        if frame_count % 60 == 0:
            fps_val  = clock.get_fps()
            sim_time = float(np.asarray(pending_state.time))
            print(f"FPS: {fps_val:.1f} | Sim time: {sim_time:.1f} | "
                  f"Alive: {n_alive:,} | Steps: {step_count:,}")
            if _t_update:
                ms = lambda d: sum(d) / len(d) * 1000
                print(f"  frame ms: sim={ms(_t_sim):.1f}  update={ms(_t_update):.1f}  render={ms(_t_render):.1f}")

    renderer.close()
    print("Simulation ended.")


def main():
    args = parse_args()
    config = build_config(args)
    run(config=config, seed=args.seed, enable_chemistry=not args.no_chemistry)


if __name__ == '__main__':
    main()
