"""
Core data structures for the Half-Life simulator.

All simulation state is represented as fixed-size JAX arrays organized into
NamedTuples. Fixed sizes are mandatory for XLA JIT compilation (static shapes).
Variable-size structures (composites) use padding + boolean masks.

Key structures:
  ParticleState   — per-particle arrays (position, velocity, species, ...)
  CompositeState  — per-composite arrays (members, binding energy, ...)
  WorldState      — root container (particles + composites + global scalars)

Initialize with: initialize_world(config, seed=0)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from halflife.config import SimConfig
from halflife.utils import make_species_colors


# ── State NamedTuples ────────────────────────────────────────────────────────

class ParticleState(NamedTuple):
    """
    State of the full particle pool. Leading dimension is NUM_PARTICLES.
    All particles are always alive; no dead slots exist.
    """
    position:     jnp.ndarray  # (N, 2)  float32 — spatial position
    velocity:     jnp.ndarray  # (N, 2)  float32 — velocity
    species:      jnp.ndarray  # (N,)    int32   — species index [0, NUM_SPECIES)
    energy:       jnp.ndarray  # (N,)    float32 — kinetic + internal energy
    mass:         jnp.ndarray  # (N,)    float32 — particle mass
    age:          jnp.ndarray  # (N,)    float32 — time since creation/last spawn
    composite_id: jnp.ndarray  # (N,)    int32   — composite index, -1 = free particle


class CompositeState(NamedTuple):
    """
    State of the composite pool. Leading dimension is MAX_COMPOSITES.
    A composite is a group of bonded particles.

    members[c, :member_count[c]] are the particle indices belonging to composite c.
    slots beyond member_count are padded with -1.

    edges[c, :edge_count[c]] are particle-id pairs (i, j) for each bond.
    slots beyond edge_count are padded with -1. Bond force is a harmonic
    spring per edge in edge-mode; ignored in star_spring / off mode.
    """
    members:        jnp.ndarray  # (C, M) int32   — particle indices per composite
    member_count:   jnp.ndarray  # (C,)   int32   — number of valid members
    alive:          jnp.ndarray  # (C,)   bool    — active composite mask
    binding_energy: jnp.ndarray  # (C,)   float32 — energy released on formation
    half_life:      jnp.ndarray  # (C,)   float32 — composite decay half-life
    age:            jnp.ndarray  # (C,)   float32 — time since formation
    species_hash:   jnp.ndarray  # (C,)   uint32  — hash of sorted member species
    free_bonds:     jnp.ndarray  # (C,)   int32   — remaining bond capacity (Σ v_s − 2*(n−1))
    edges:          jnp.ndarray  # (C, E_max, 2) int32 — bond particle-id pairs; -1 = unused
    edge_count:     jnp.ndarray  # (C,)   int32   — number of valid edges


class WorldState(NamedTuple):
    """Root container for the complete simulation state."""
    particles:    ParticleState
    composites:   CompositeState
    time:         jnp.ndarray   # ()    float32 — current simulation time
    rng_key:      jnp.ndarray   # (2,)  uint32  — JAX PRNG key
    total_energy: jnp.ndarray   # ()    float32 — total energy (kinetic + binding)
    step_count:   jnp.ndarray   # ()    int32   — number of steps taken


# ── Initialization ───────────────────────────────────────────────────────────

def initialize_world(config: SimConfig, seed: int = 0) -> WorldState:
    """
    Create the initial WorldState with randomly placed particles.

    Particles are placed uniformly at random within the world bounds.
    Species are assigned uniformly at random.
    Speeds are sampled from a Maxwell-Boltzmann-like distribution.
    All particles start as free (composite_id = -1).

    Args:
        config: SimConfig (frozen dataclass with all parameters)
        seed:   random seed for reproducibility

    Returns:
        WorldState ready for simulation
    """
    key = jax.random.PRNGKey(seed)
    N = config.num_particles
    C = config.max_composites
    M = config.max_composite_size

    # ── Particles ────────────────────────────────────────────────────────────
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    # Positions: uniform in world
    pos = jax.random.uniform(k1, (N, 2),
                              minval=jnp.array([0.0, 0.0]),
                              maxval=jnp.array([config.world_width, config.world_height]))

    # Velocities: random direction, speed ~ config.init_speed
    angles = jax.random.uniform(k2, (N,), minval=0.0, maxval=2 * jnp.pi)
    speeds = jax.random.uniform(k3, (N,),
                                 minval=0.0,
                                 maxval=config.init_speed * 2.0)
    vel = speeds[:, None] * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    # Species: uniform random
    species = jax.random.randint(k4, (N,), 0, config.num_species)

    # Energy: 0.5 * mass * |v|^2
    mass = jnp.ones(N, dtype=jnp.float32)
    energy = 0.5 * mass * jnp.sum(vel ** 2, axis=-1)

    age = jnp.zeros(N, dtype=jnp.float32)
    composite_id = jnp.full(N, -1, dtype=jnp.int32)

    particles = ParticleState(
        position=pos,
        velocity=vel,
        species=species,
        energy=energy,
        mass=mass,
        age=age,
        composite_id=composite_id,
    )

    # ── Composites ───────────────────────────────────────────────────────────
    E = config.e_max
    composites = CompositeState(
        members=jnp.full((C, M), -1, dtype=jnp.int32),
        member_count=jnp.zeros(C, dtype=jnp.int32),
        alive=jnp.zeros(C, dtype=bool),
        binding_energy=jnp.zeros(C, dtype=jnp.float32),
        half_life=jnp.zeros(C, dtype=jnp.float32),
        age=jnp.zeros(C, dtype=jnp.float32),
        species_hash=jnp.zeros(C, dtype=jnp.uint32),
        free_bonds=jnp.zeros(C, dtype=jnp.int32),
        edges=jnp.full((C, E, 2), -1, dtype=jnp.int32),
        edge_count=jnp.zeros(C, dtype=jnp.int32),
    )

    # ── Global scalars ────────────────────────────────────────────────────────
    total_energy = jnp.sum(energy)

    return WorldState(
        particles=particles,
        composites=composites,
        time=jnp.array(0.0, dtype=jnp.float32),
        rng_key=key,
        total_energy=total_energy,
        step_count=jnp.array(0, dtype=jnp.int32),
    )


# ── Interaction Parameters (not part of WorldState, passed separately) ────────

class InteractionParams(NamedTuple):
    """
    Species-dependent pairwise force parameters. Not part of WorldState.
    Passed as a regular JAX array argument (not static), so these can be
    changed without recompiling the simulation step.

    Per-pair radii are stored as FRACTIONS of config.interaction_radius:
      r_peak[i,j]   = interaction_radius * peak_fraction[i,j]
      r_cutoff[i,j] = interaction_radius * cutoff_fraction[i,j]
    Repulsion is global (physics.repulsion_radius, scalar) — steric exclusion
    is universal across species pairs. Ordering invariant enforced at init:
      0 < repulsion_fraction < peak_fraction[i,j] < cutoff_fraction[i,j] <= 1
    """
    # All matrices: (num_species, num_species) float32
    attraction:        jnp.ndarray  # signed strength in [-1, 1]
    peak_fraction:     jnp.ndarray  # peak-attraction radius / interaction_radius
    cutoff_fraction:   jnp.ndarray  # zero-force radius / interaction_radius
    r_rest:            jnp.ndarray  # hash-derived bond rest length per species pair


def initialize_interaction_params(config: SimConfig,
                                   seed: int = 42) -> InteractionParams:
    """
    Initialize random interaction parameters.

    The attraction matrix is the primary knob for rich behaviour:
    - Asymmetric values (A attracts B but B repels A) produce chasing dynamics
    - Symmetric negative values produce clustering
    - Symmetric positive values produce avoidance

    peak_fraction and cutoff_fraction are sampled per-species-pair so each
    pair gets its own force-shape (peak position + range), not just amplitude.
    The fractions are sampled in [0.3, 0.95] then sorted per-pair to enforce
    peak < cutoff with at least a 0.1 gap.

    Args:
        config: SimConfig
        seed:   random seed for the interaction matrix
    Returns:
        InteractionParams
    """
    key = jax.random.PRNGKey(seed)
    S = config.num_species
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Random signed attraction: uniform in [-1, 1]
    attraction = jax.random.uniform(k1, (S, S), minval=-1.0, maxval=1.0)

    # Per-pair radii as fractions of interaction_radius. Sample two values
    # in [0.3, 0.95] and sort per-pair: smaller → peak, larger → cutoff.
    # Lower bound 0.3 keeps the peak well outside repulsion_fraction
    # (= repulsion_radius / interaction_radius, default 0.2). Enforce a
    # minimum 0.1 gap so the triangle has non-degenerate width.
    a = jax.random.uniform(k2, (S, S), minval=0.3, maxval=0.95)
    b = jax.random.uniform(k3, (S, S), minval=0.3, maxval=0.95)
    peak_fraction   = jnp.minimum(a, b)
    cutoff_fraction = jnp.maximum(a, b)
    cutoff_fraction = jnp.minimum(jnp.maximum(cutoff_fraction, peak_fraction + 0.1), 1.0)

    # Hash-derived per-species-pair bond rest length. Symmetric by construction
    # (uses the commutative species-pair hash). Independent of `seed` — this
    # is part of the "universe" determined by config.hash_modulus, like valence.
    from halflife.chemistry import _hash_to_rest_length  # local import: chemistry imports state
    species_idx = jnp.arange(S, dtype=jnp.int32)
    r_rest = jax.vmap(
        lambda i: jax.vmap(lambda j: _hash_to_rest_length(i, j, config))(species_idx)
    )(species_idx)  # (S, S)

    return InteractionParams(
        attraction=attraction,
        peak_fraction=peak_fraction,
        cutoff_fraction=cutoff_fraction,
        r_rest=r_rest,
    )


# ── Physics Parameters (runtime-tunable, dynamic JAX arg) ────────────────────

class PhysicsParams(NamedTuple):
    """
    Physics scalars that can be adjusted at runtime without recompiling.
    Passed as a regular JAX argument (not static), so slider changes take
    effect on the next simulation step without triggering recompilation.
    """
    damping:                  jnp.ndarray  # () float32 — velocity damping per step
    repulsion_strength:       jnp.ndarray  # () float32 — hard-core repulsion magnitude
    fusion_threshold:         jnp.ndarray  # () float32 — min binding energy to fuse [0,1]
    binding_energy_scale:     jnp.ndarray  # () float32 — energy released on fusion
    repulsion_radius:         jnp.ndarray  # () float32 — inner hard-core repulsion radius
    r_cutoff_scale:           jnp.ndarray  # () float32 — multiplier on per-species r_cutoff
    spring_k:                 jnp.ndarray  # () float32 — composite COM-spring stiffness
    attraction_scale:         jnp.ndarray  # () float32 — global attraction magnitude multiplier
    dt:                       jnp.ndarray  # () float32 — integration timestep


def initialize_physics_params(config: SimConfig) -> PhysicsParams:
    """Create PhysicsParams from SimConfig defaults."""
    return PhysicsParams(
        damping=jnp.float32(config.damping),
        repulsion_strength=jnp.float32(config.repulsion_strength),
        fusion_threshold=jnp.float32(config.fusion_threshold),
        binding_energy_scale=jnp.float32(config.binding_energy_scale),
        repulsion_radius=jnp.float32(config.repulsion_radius),
        r_cutoff_scale=jnp.float32(1.0),
        spring_k=jnp.float32(config.spring_k),
        attraction_scale=jnp.float32(1.0),
        dt=jnp.float32(config.dt),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_species_colors(config: SimConfig) -> np.ndarray:
    """Return (num_species, 3) float32 RGB color palette."""
    return make_species_colors(config.num_species)
