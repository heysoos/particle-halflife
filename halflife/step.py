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
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import WorldState, InteractionParams, PhysicsParams
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.interactions import compute_all_forces
from halflife.chemistry import attempt_fusion, apply_composite_decay
from halflife.energy import compute_total_energy, apply_soft_energy_conservation
from halflife.utils import apply_boundary

if TYPE_CHECKING:
    from halflife.profiler import ProfileMetrics


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
    N = config.num_particles
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


def compute_edge_bond_forces(state: WorldState, params: InteractionParams,
                              config: SimConfig, physics: PhysicsParams) -> jnp.ndarray:
    """
    Per-edge harmonic spring forces (sparse covalent bonds).

    For each valid edge (i, j) in every alive composite:
        F_on_i = -k_bond * (r - r_rest[s_i, s_j]) * (pos_i - pos_j) / r
        F_on_j = -F_on_i
    Forces are scatter-added into a (N, 2) buffer. Min-image displacement is
    used so bonds across periodic boundaries don't snap.

    Cost: O(C · E_max) vmap cells. Per-cell work is one min-image distance,
    one species-pair gather (r_rest), one harmonic spring evaluation.

    Returns: (N, 2) float32 — additional forces from bonds
    """
    particles = state.particles
    composites = state.composites
    N = config.num_particles
    C = config.max_composites
    E = config.e_max
    # Read stiffness from PhysicsParams (runtime-tunable via slider) instead
    # of the SimConfig default. config.k_bond is still used to seed the
    # initial physics value in initialize_physics_params.
    k = physics.k_bond

    e_idx = jnp.arange(E, dtype=jnp.int32)

    def per_composite(c):
        is_alive = composites.alive[c]
        count    = composites.edge_count[c]
        valid_e  = is_alive & (e_idx < count)  # (E,)

        pid_a = composites.edges[c, :, 0]  # (E,)
        pid_b = composites.edges[c, :, 1]  # (E,)
        safe_a = jnp.where(pid_a >= 0, pid_a, 0)
        safe_b = jnp.where(pid_b >= 0, pid_b, 0)

        pa = particles.position[safe_a]  # (E, 2)
        pb = particles.position[safe_b]  # (E, 2)
        sa = particles.species[safe_a]   # (E,)
        sb = particles.species[safe_b]   # (E,)

        d = pa - pb
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
        r = jnp.linalg.norm(d, axis=-1) + 1e-10  # (E,)
        d_hat = d / r[:, None]                    # (E, 2)
        # Hash-derived per-species-pair rest length, uniformly scaled by the
        # physics-knob multiplier (r_rest_scale slider). Default 1.0 leaves
        # the hash-determined chemistry intact; <1 tightens, >1 loosens.
        r_rest = params.r_rest[sa, sb] * physics.r_rest_scale  # (E,)

        # F_on_i = -k * (r - r_rest) * d_hat   ; d_hat = (pos_i - pos_j) / r
        # When r > r_rest (stretched), force pulls i toward j (along -d_hat in actual position space).
        f_on_a = -k * (r - r_rest)[:, None] * d_hat  # (E, 2)
        f_on_b = -f_on_a                              # Newton's third law

        # Mask out invalid edges
        mask = valid_e[:, None].astype(jnp.float32)
        f_on_a = f_on_a * mask
        f_on_b = f_on_b * mask

        # Route invalid pids to OOB index N → mode='drop' silently discards.
        drop_a = jnp.where(valid_e, pid_a, N)
        drop_b = jnp.where(valid_e, pid_b, N)
        return drop_a, drop_b, f_on_a, f_on_b

    pid_a_all, pid_b_all, f_a_all, f_b_all = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )  # (C, E), (C, E), (C, E, 2), (C, E, 2)

    flat_pid_a = pid_a_all.reshape(-1)
    flat_pid_b = pid_b_all.reshape(-1)
    flat_f_a   = f_a_all.reshape(-1, 2)
    flat_f_b   = f_b_all.reshape(-1, 2)

    forces = jnp.zeros((N, 2), dtype=jnp.float32)
    forces = forces.at[flat_pid_a].add(flat_f_a, mode='drop')
    forces = forces.at[flat_pid_b].add(flat_f_b, mode='drop')
    return forces


# ── Composite Size Statistics ─────────────────────────────────────────────────

def compute_composite_size_stats(composites, config: SimConfig) -> tuple:
    """
    Compute composite size statistics from CompositeState.

    Returns:
        (max_size, mean_size, distribution_histogram)
        where distribution_histogram[i] = count of composites with i members
    """
    alive_composites = composites.alive.astype(jnp.int32)  # (max_composites,)
    counts = composites.member_count * alive_composites  # (max_composites,)

    # Convert to numpy for statistics (happens on CPU, not in JAX ops)
    counts_np = np.asarray(counts)
    alive_indices = np.where(counts_np > 0)[0]

    if len(alive_indices) == 0:
        return 0, 0.0, np.zeros(config.max_composite_size + 1, dtype=np.int32)

    alive_counts = counts_np[alive_indices]
    max_size = int(np.max(alive_counts))
    mean_size = float(np.mean(alive_counts))

    # Histogram: count of composites at each size
    histogram = np.zeros(config.max_composite_size + 1, dtype=np.int32)
    for count in alive_counts:
        histogram[int(count)] += 1

    return max_size, mean_size, histogram


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
    cell_list = build_cell_list(particles.position, config)

    # ── Phase 2: Neighbor Finding ─────────────────────────────────────────────
    neighbors = find_all_neighbors(particles.position, cell_list, config)

    # ── Phase 3: Force Computation ────────────────────────────────────────────
    forces = compute_all_forces(
        particles.position, particles.species, neighbors, params, config, physics
    )

    # Bond forces — dispatched on static config.bond_mode so XLA traces only
    # the live branch. Existing use_bond_forces flag is honored for backward
    # compat in star_spring mode (bond_mode='star_spring' + use_bond_forces=False
    # is equivalent to bond_mode='off').
    if config.bond_mode == "edges":
        bond_forces = compute_edge_bond_forces(state, params, config, physics)
        forces = forces + bond_forces
    elif config.bond_mode == "star_spring" and config.use_bond_forces:
        bond_forces = compute_bond_forces(state, config, physics)
        forces = forces + bond_forces
    # bond_mode == "off" → no bond force added

    # ── Phase 4: Integration (Euler) ──────────────────────────────────────────
    new_vel = particles.velocity + (forces / particles.mass[:, None]) * physics.dt
    new_vel = new_vel * physics.damping
    # Magnitude clamp: cap |v| at max_velocity. Per-component clip would let
    # diagonal motion reach |v| = sqrt(2) * max_velocity, which is unphysical.
    speed = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
    new_vel = new_vel * jnp.minimum(
        1.0, config.max_velocity / (speed + 1e-10)
    )
    new_pos = particles.position + new_vel * physics.dt

    # ── Phase 5: Boundary Conditions ─────────────────────────────────────────
    new_pos, new_vel = apply_boundary(new_pos, new_vel, config)

    state = state._replace(
        particles=particles._replace(position=new_pos, velocity=new_vel)
    )

    # ── Phase 5b: Per-particle degree cache ─────────────────────────────────────
    # degree[i] counts edges incident to particle i across all alive composites.
    # Used by the per-particle valence gate in Phase 6 and by Phase 6b ring
    # closure. Cheap (O(C · E_max) scatter-add). Recomputed once per step from
    # the pre-fusion edge state; phases 6 and 6b update it incrementally via
    # their scan carries.
    from halflife.chemistry import compute_degree, compute_composite_free_bonds, _species_valences
    degree = compute_degree(state.composites, config)
    species_valences = _species_valences(config)
    composite_free_bonds = compute_composite_free_bonds(
        state.particles, state.composites, degree, species_valences, config
    )

    # ── Phase 6: Fusion ───────────────────────────────────────────────────────
    state, degree = attempt_fusion(
        state, neighbors, params, config, physics,
        degree=degree, species_valences=species_valences,
    )

    # ── Phase 6b: Ring closure (intra-composite fusion) ───────────────────────
    from halflife.chemistry import attempt_ring_closure
    state, degree = attempt_ring_closure(
        state, neighbors, params, config, physics,
        degree=degree, species_valences=species_valences,
    )

    # ── Phase 7: Decay ────────────────────────────────────────────────────────
    state = apply_composite_decay(state, config, physics)

    # ── Phase 8: Energy Accounting ────────────────────────────────────────────
    current_energy = compute_total_energy(state)
    # state = apply_soft_energy_conservation(state, state.total_energy)

    # Re-clamp velocities. The post-integration clamp at phase 4 only catches
    # force-driven overshoots; phase 7 fission kicks (chemistry.py) and phase 8
    # soft energy rescale can both push speeds back over max_velocity. Without
    # this final clamp, soft conservation alone compounds 1% per step into
    # large violations within ~100 steps. Same magnitude semantics as phase 4.
    final_speed = jnp.linalg.norm(state.particles.velocity, axis=-1, keepdims=True)
    final_vel = state.particles.velocity * jnp.minimum(
        1.0, config.max_velocity / (final_speed + 1e-10)
    )
    state = state._replace(
        particles=state.particles._replace(velocity=final_vel)
    )

    # ── Phase 9: Increment Ages and Counters ──────────────────────────────────
    new_age = state.particles.age + physics.dt
    new_comp_age = state.composites.age + physics.dt * state.composites.alive.astype(jnp.float32)

    final_state = state._replace(
        particles=state.particles._replace(age=new_age),
        composites=state.composites._replace(age=new_comp_age),
        time=state.time + physics.dt,
        total_energy=current_energy,
        step_count=state.step_count + 1,
    )

    return final_state


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
