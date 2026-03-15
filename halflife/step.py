"""
Main simulation step — the central JIT-compiled function.

simulation_step() advances the WorldState by one dt. The entire function
is compiled to a single XLA computation graph:

  Phase 1: Build cell list (spatial indexing)
  Phase 2: Find all neighbors
  Phase 3: Compute pairwise forces
  Phase 4: Integrate (velocity Verlet / Euler)
  Phase 5: Apply boundary conditions
  Phase 6: Chemistry — fusion attempts
  Phase 7: Chemistry — decay and fission
  Phase 8: Energy accounting
  Phase 9: Update ages and global counters

Usage:
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config)

Multi-step (no Python overhead):
    run_n_steps = jax.jit(lambda s, p: jax.lax.scan(
        lambda s, _: (simulation_step(s, p, config), None),
        s, None, length=n
    )[0], static_argnums=())
"""

import functools
import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.state import WorldState, InteractionParams, PhysicsParams
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.interactions import compute_all_forces
from halflife.chemistry import attempt_fusion, apply_composite_decay
from halflife.energy import compute_total_energy, apply_soft_energy_conservation
from halflife.utils import apply_boundary


# ── Bond Forces (particles within composites) ─────────────────────────────────

def compute_bond_forces(state: WorldState, config: SimConfig,
                        physics: PhysicsParams) -> jnp.ndarray:
    """
    COM-spring forces keeping composite members together.
    Each member is attracted toward the composite center of mass.
    O(C*M) instead of O(C*M^2) — no all-pairs computation needed.

    Returns: (N, 2) float32 — additional forces from bonds
    """
    particles = state.particles
    composites = state.composites
    N = config.max_particles
    M = config.max_composite_size
    spring_k = physics.spring_k

    def bonds_for_composite(c):
        is_alive = composites.alive[c]
        n_members = composites.member_count[c]

        # Compute COM using minimum-image convention to handle periodic boundaries.
        # Use first member as reference; compute min-image displacements from it.
        safe_members = jnp.where(composites.members[c] >= 0, composites.members[c], 0)
        valid_mask = ((composites.members[c] >= 0) &
                      (jnp.arange(M) < n_members)).astype(jnp.float32)
        ref_pos = particles.position[safe_members[0]]  # (2,) reference position

        def _disp_from_ref(m_idx):
            mpos = particles.position[safe_members[m_idx]]
            d = mpos - ref_pos
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        member_disps = jax.vmap(_disp_from_ref)(jnp.arange(M))  # (M, 2)
        com_disp = jnp.sum(member_disps * valid_mask[:, None], axis=0) / (
            n_members.astype(jnp.float32) + 1e-8
        )
        com = ref_pos + com_disp  # may be outside [0,W) — min-image in spring calc corrects it

        def member_spring(m_idx):
            pid = composites.members[c, m_idx]
            valid = is_alive & (m_idx < n_members) & (pid >= 0)
            d = com - particles.position[pid]  # direction toward COM
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            f = spring_k * d
            return jnp.where(valid, pid, 0), valid, jnp.where(valid, f, jnp.zeros(2))

        pids, valids, forces = jax.vmap(member_spring)(jnp.arange(M, dtype=jnp.int32))
        return pids, valids, forces  # (M,), (M,), (M, 2)

    all_pids, all_valid, all_forces = jax.vmap(bonds_for_composite)(
        jnp.arange(config.max_composites, dtype=jnp.int32)
    )  # (C, M), (C, M), (C, M, 2)

    flat_pids   = all_pids.reshape(-1)
    flat_valid  = all_valid.reshape(-1)
    flat_forces = all_forces.reshape(-1, 2)
    safe_pids   = jnp.where(flat_valid, flat_pids, 0)

    bond_forces = jnp.zeros((N, 2), dtype=jnp.float32)
    bond_forces = bond_forces.at[safe_pids].add(
        jnp.where(flat_valid[:, None], flat_forces, 0.0)
    )
    return bond_forces


# ── Main Simulation Step ──────────────────────────────────────────────────────

def simulation_step(state: WorldState, params: InteractionParams,
                    config: SimConfig, physics: PhysicsParams) -> WorldState:
    """
    Advance WorldState by one timestep (dt).

    Args:
        state:   WorldState — current simulation state
        params:  InteractionParams — species-dependent force parameters
        config:  SimConfig — static simulation parameters (static_argnums=(2,))
        physics: PhysicsParams — runtime-tunable physics scalars

    Returns:
        WorldState — updated state after one dt
    """
    particles = state.particles

    # ── Phase 1: Spatial Indexing ─────────────────────────────────────────────
    cell_list = build_cell_list(particles.position, particles.alive, config)

    # ── Phase 2: Neighbor Finding ─────────────────────────────────────────────
    neighbors = find_all_neighbors(particles.position, particles.alive, cell_list, config)

    # ── Phase 3: Force Computation ────────────────────────────────────────────
    # Polarity modifier: composite members use their composite's net_polarity;
    # free particles use 1.0 (no scaling).
    is_composite = particles.composite_id >= 0
    safe_cid = jnp.clip(particles.composite_id, 0, config.max_composites - 1)
    net_pol = state.composites.net_polarity[safe_cid]   # (N,) float32
    attr_mod = jnp.where(is_composite, net_pol, 1.0)    # (N,) float32

    forces = compute_all_forces(
        particles.position, particles.species, particles.alive, neighbors, params, config,
        physics, attr_mod
    )

    # Bond forces (optional — expensive; enable via config.use_bond_forces)
    if config.use_bond_forces:
        bond_forces = compute_bond_forces(state, config, physics)
        forces = forces + bond_forces

    # ── Phase 4: Integration (Euler) ──────────────────────────────────────────
    new_vel = particles.velocity + (forces / particles.mass[:, None]) * config.dt
    new_vel = new_vel * physics.damping
    new_vel = jnp.clip(new_vel, -config.max_velocity, config.max_velocity)
    new_pos = particles.position + new_vel * config.dt

    # Dead particles stay put (no NaN propagation)
    alive_f = particles.alive[:, None].astype(jnp.float32)
    new_vel = new_vel * alive_f
    new_pos = new_pos * alive_f + particles.position * (1.0 - alive_f)

    # ── Phase 5: Boundary Conditions ─────────────────────────────────────────
    new_pos, new_vel = apply_boundary(new_pos, new_vel, config)

    state = state._replace(
        particles=particles._replace(position=new_pos, velocity=new_vel)
    )

    # ── Phase 6: Fusion ───────────────────────────────────────────────────────
    state = attempt_fusion(state, neighbors, params, config, physics)

    # ── Phase 7: Decay ────────────────────────────────────────────────────────
    state = apply_composite_decay(state, config)

    # ── Phase 8: Energy Accounting ────────────────────────────────────────────
    current_energy = compute_total_energy(state)
    state = apply_soft_energy_conservation(state, state.total_energy)

    # ── Phase 9: Increment Ages and Counters ──────────────────────────────────
    new_age = state.particles.age + config.dt * state.particles.alive.astype(jnp.float32)
    new_comp_age = state.composites.age + config.dt * state.composites.alive.astype(jnp.float32)

    return state._replace(
        particles=state.particles._replace(age=new_age),
        composites=state.composites._replace(age=new_comp_age),
        time=state.time + config.dt,
        total_energy=current_energy,
        step_count=state.step_count + 1,
    )


# ── Multi-step Helper ─────────────────────────────────────────────────────────

def make_run_n_steps(config: SimConfig):
    """
    Return a JIT-compiled function that runs n steps without returning to Python.
    Uses jax.lax.scan for maximum throughput — all N steps fuse into one XLA kernel.

    config is captured by closure (static/frozen).
    params and physics are regular JAX arguments (can change without recompile).
    n_steps is a concrete Python int; JAX retraces per unique value (7 values max).

    Usage:
        run_n = make_run_n_steps(config)
        state = run_n(state, params, physics, n_steps=1)
    """
    @functools.partial(jax.jit, static_argnums=(3,))
    def run_n_steps(state: WorldState, params: InteractionParams,
                    physics: PhysicsParams, n_steps: int) -> WorldState:
        def body(s, _):
            return simulation_step(s, params, config, physics), None
        final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
        return final_state
    return run_n_steps
