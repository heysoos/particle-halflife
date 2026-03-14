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

import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.state import WorldState, InteractionParams
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.interactions import compute_all_forces
from halflife.chemistry import attempt_fusion, apply_particle_decay, apply_composite_decay
from halflife.energy import compute_total_energy, apply_soft_energy_conservation
from halflife.utils import apply_boundary


# ── Bond Forces (particles within composites) ─────────────────────────────────

def compute_bond_forces(state: WorldState, config: SimConfig) -> jnp.ndarray:
    """
    Spring-like forces keeping composite members together.
    Each pair (i, j) within a composite attracts toward rest distance fusion_radius.

    Returns: (N, 2) float32 — additional forces from bonds
    """
    particles = state.particles
    composites = state.composites
    N = config.max_particles
    bond_forces = jnp.zeros((N, 2), dtype=jnp.float32)

    spring_k = 5.0          # spring constant
    rest_len = config.fusion_radius * 0.8

    def bonds_for_composite(c):
        """Compute bond forces for composite c, return (N, 2) contributions."""
        is_alive = composites.alive[c]
        n_members = composites.member_count[c]

        def pair_force(m_pair):
            m_a, m_b = m_pair[0], m_pair[1]
            i = composites.members[c, m_a]
            j = composites.members[c, m_b]
            valid = (
                is_alive &
                (m_a < n_members) & (m_b < n_members) &
                (m_a != m_b) & (i >= 0) & (j >= 0)
            )
            d = particles.position[i] - particles.position[j]
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            r = jnp.linalg.norm(d) + 1e-10
            d_hat = d / r
            f_mag = spring_k * (r - rest_len)
            f = -f_mag * d_hat  # toward equilibrium
            return jnp.where(valid, i, 0), jnp.where(valid, j, 0), jnp.where(valid, f, jnp.zeros(2)), valid

        M = config.max_composite_size
        pairs = jnp.array([[a, b] for a in range(M) for b in range(a + 1, M)], dtype=jnp.int32)
        idx_i, idx_j, f_pairs, valid_pairs = jax.vmap(pair_force)(pairs)

        # Accumulate forces into per-particle arrays (relative to this composite)
        # Return sparse updates: (n_pairs, 2) — applied to idx_i as +f, idx_j as -f
        return idx_i, idx_j, f_pairs, valid_pairs

    # For performance, skip bond forces in Phase 2 (no composites yet).
    # In Phase 4+, these are active. The vmap below is over composites.
    # This is expensive for large MAX_COMPOSITES — consider limiting to alive composites.
    all_i, all_j, all_f, all_valid = jax.vmap(bonds_for_composite)(
        jnp.arange(config.max_composites, dtype=jnp.int32)
    )
    # all_i: (C, n_pairs) int32 etc.

    flat_i = all_i.reshape(-1)
    flat_j = all_j.reshape(-1)
    flat_f = all_f.reshape(-1, 2)
    flat_valid = all_valid.reshape(-1)

    safe_i = jnp.where(flat_valid, flat_i, 0)
    safe_j = jnp.where(flat_valid, flat_j, 0)

    bond_forces = bond_forces.at[safe_i].add(
        jnp.where(flat_valid[:, None], flat_f, 0.0)
    )
    bond_forces = bond_forces.at[safe_j].add(
        jnp.where(flat_valid[:, None], -flat_f, 0.0)
    )
    return bond_forces


# ── Main Simulation Step ──────────────────────────────────────────────────────

def simulation_step(state: WorldState, params: InteractionParams,
                    config: SimConfig) -> WorldState:
    """
    Advance WorldState by one timestep (dt).

    Args:
        state:  WorldState — current simulation state
        params: InteractionParams — species-dependent force parameters
        config: SimConfig — static simulation parameters

    Returns:
        WorldState — updated state after one dt
    """
    particles = state.particles

    # ── Phase 1: Spatial Indexing ─────────────────────────────────────────────
    cell_list = build_cell_list(particles.position, particles.alive, config)

    # ── Phase 2: Neighbor Finding ─────────────────────────────────────────────
    neighbors = find_all_neighbors(particles.position, particles.alive, cell_list, config)

    # ── Phase 3: Force Computation ────────────────────────────────────────────
    forces = compute_all_forces(
        particles.position, particles.species, particles.alive, neighbors, params, config
    )

    # Bond forces (zero in Phase 2 since no composites; active in Phase 4+)
    bond_forces = compute_bond_forces(state, config)
    forces = forces + bond_forces

    # ── Phase 4: Integration (Euler) ──────────────────────────────────────────
    new_vel = particles.velocity + (forces / particles.mass[:, None]) * config.dt
    new_vel = new_vel * config.damping
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
    state = attempt_fusion(state, neighbors, config)

    # ── Phase 7: Decay ────────────────────────────────────────────────────────
    state = apply_composite_decay(state, config)
    state = apply_particle_decay(state, config)

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
    Uses jax.lax.scan for maximum throughput.

    Usage:
        run = make_run_n_steps(config)
        state = run(state, params, n_steps=100)
    """
    @jax.jit
    def run_n_steps(state: WorldState, params: InteractionParams,
                    n_steps: int) -> WorldState:
        def body(s, _):
            return simulation_step(s, params, config), None
        final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
        return final_state
    return run_n_steps
