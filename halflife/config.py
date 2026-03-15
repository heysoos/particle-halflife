"""
SimConfig — the user's primary tuning surface.

A frozen dataclass that parameterizes every aspect of the simulation.
Passed as static_argnums to all JIT-compiled functions so XLA can see
array shapes at compile time.

Experiment by changing values here. Different `hash_modulus` values give
entirely different "universes" / chemistries. Different `num_species` and
interaction radii produce qualitatively different emergent behaviors.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # ── World ────────────────────────────────────────────────────────────────
    world_width: float = 200.0      # spatial extent in x
    world_height: float = 200.0     # spatial extent in y
    dt: float = 0.02                # simulation timestep

    # Boundary mode: "periodic" (torus) or "reflect" (bouncing walls)
    boundary_mode: str = "periodic"

    # ── Particles ────────────────────────────────────────────────────────────
    max_particles: int = 4_000      # fixed pool capacity (alive + dead slots)
    num_particles_init: int = 2_000 # how many to spawn at initialization
    num_species: int = 12           # number of distinct particle types
    state_dim: int = 8              # internal state vector size (NCA-style, future use)

    # ── Composites ───────────────────────────────────────────────────────────
    max_composites: int = 200       # fixed composite pool capacity
    max_composite_size: int = 8     # JAX buffer size — not a physics cap; chemistry determines stability

    # ── Spatial Indexing ─────────────────────────────────────────────────────
    # cell_size should equal interaction_radius for optimal neighbor queries
    cell_size: float = 4.0
    # max particles per cell in the cell list (4x expected density is safe)
    cell_capacity: int = 8

    # ── Interactions ─────────────────────────────────────────────────────────
    interaction_radius: float = 4.0   # pairwise force cutoff distance
    max_neighbors: int = 16           # max neighbors per particle (cap for fixed arrays)

    # Particle Life-style force shape:
    #   [0, repulsion_radius]       → strong repulsion (hard core)
    #   [repulsion_radius, r_peak]  → ramp up to peak attraction/repulsion
    #   [r_peak, interaction_radius]→ fall off to zero
    repulsion_radius: float = 0.8     # inner hard-core repulsion boundary
    repulsion_strength: float = 2.0   # magnitude of hard-core repulsion

    # ── Fusion ───────────────────────────────────────────────────────────────
    # Fusion occurs when two free particles are within fusion_radius AND
    # the hash-derived binding energy exceeds fusion_threshold
    fusion_radius: float = 1.0        # must be < interaction_radius
    fusion_threshold: float = 0.2     # minimum binding energy to trigger fusion [0,1]

    # ── Decay / Half-life ────────────────────────────────────────────────────
    # Particle half-lives are sampled uniformly from this range at init
    half_life_min: float = 50.0       # shortest particle half-life (sim time units)
    half_life_max: float = 500.0      # longest particle half-life

    # Composites get a half-life derived from their species hash; this scales it
    composite_half_life_scale: float = 3.0  # composites live ~3x longer than particles

    # ── Energy ───────────────────────────────────────────────────────────────
    # Kinetic energy scale at initialization (controls initial temperature)
    init_speed: float = 1.5
    # Binding energy scale — how much energy fusing releases
    binding_energy_scale: float = 1.0
    # Cost multiplier for fission (energy required = binding_energy * fission_cost)
    fission_cost: float = 0.5
    # Velocity damping per step (1.0 = no damping, 0.99 = slight damping)
    damping: float = 0.995
    max_velocity: float = 8.0         # velocity clamp

    # ── Hash Chemistry ───────────────────────────────────────────────────────
    # Change this to get a completely different "universe" / chemistry
    hash_modulus: int = 100_000_007   # large prime
    hash_prime_a: int = 1_000_003     # multiplier prime
    hash_prime_b: int = 7             # offset

    # Max products from a single decay event (padded to this length)
    max_decay_products: int = 3

    # ── Polarity Chemistry ────────────────────────────────────────────────────
    # Each species has a signed polarity charge p[s] ∈ [-1, 1].
    # Opposite polarities fuse more readily; neutral composites live longer.
    polarity_fusion_scale:      float = 0.3   # bonus/penalty to binding energy
    polarity_stability_scale:   float = 0.5   # neutrality boost to composite half-life
    composite_size_decay_scale: float = 0.3   # size penalty on composite half-life (larger → shorter hl)

    # ── Performance Caps ─────────────────────────────────────────────────────
    # Cap fusion scan to first K candidates per step (avoids O(N) sequential scan)
    max_fusions_per_step: int = 100
    # Cap decay spawns per step (keeps find_free_slots fast)
    max_decay_per_step: int = 200
    # Enable spring bond forces between composite members (expensive; off by default)
    use_bond_forces: bool = True

    # ── Rendering ────────────────────────────────────────────────────────────
    window_width: int = 1280
    window_height: int = 720
    fps_target: int = 60
    point_size_min: float = 2.0       # minimum particle render size (pixels)
    point_size_max: float = 14.0      # maximum (scales with mass)
    background_color: tuple = (0.05, 0.05, 0.08, 1.0)  # dark blue-black

    # ── Derived (computed from above) ────────────────────────────────────────
    # Not actual dataclass fields — computed as properties for convenience

    @property
    def world_size(self) -> tuple:
        return (self.world_width, self.world_height)

    @property
    def num_cells_x(self) -> int:
        return max(1, int(self.world_width / self.cell_size))

    @property
    def num_cells_y(self) -> int:
        return max(1, int(self.world_height / self.cell_size))

    @property
    def num_cells(self) -> int:
        return self.num_cells_x * self.num_cells_y
