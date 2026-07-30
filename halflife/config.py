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
    # World aspect ratio is matched to window aspect ratio (16:9) so the
    # vertex shaders' (in_position / u_world_size) → NDC mapping doesn't
    # stretch the world non-uniformly when it fills the window.
    # 200 × 720/1280 = 112.5
    world_width: float = 200.0      # spatial extent in x
    world_height: float = 112.5     # spatial extent in y
    dt: float = 0.06                # simulation timestep

    # Boundary mode: "periodic" (torus) or "reflect" (bouncing walls)
    boundary_mode: str = "periodic"

    # ── Particles ────────────────────────────────────────────────────────────
    num_particles: int = 5_000      # total particle count (fixed, all always alive)
    num_species: int = 4           # number of distinct particle types
    state_dim: int = 8              # internal state vector size (NCA-style, future use)

    # ── Composites ───────────────────────────────────────────────────────────
    max_composites: int = 3_000       # fixed composite pool capacity
    max_composite_size: int = 256     # JAX buffer size — not a physics cap; chemistry determines stability

    # ── Spatial Indexing ─────────────────────────────────────────────────────
    # cell_size should equal interaction_radius for optimal neighbor queries
    cell_size: float = 8.0
    # max particles per cell in the cell list (4x expected density is safe)
    cell_capacity: int = 64

    # ── Interactions ─────────────────────────────────────────────────────────
    interaction_radius: float = 8.0   # pairwise force cutoff distance
    max_neighbors: int = 256           # max neighbors per particle (cap for fixed arrays)

    # Particle Life-style force shape:
    #   [0, repulsion_radius]       → strong repulsion (hard core)
    #   [repulsion_radius, r_peak]  → ramp up to peak attraction/repulsion
    #   [r_peak, interaction_radius]→ fall off to zero
    repulsion_radius: float = 0.8     # inner hard-core repulsion boundary
    repulsion_strength: float = 2.0   # magnitude of hard-core repulsion

    # ── Fusion ───────────────────────────────────────────────────────────────
    # Fusion occurs when two entities (free particles or composites) have a
    # member-pair within fusion_radius AND the hash-derived binding energy
    # exceeds fusion_threshold. Proximity is the minimum member-member distance
    # between the two entities (nearest contacting pair), not rep-to-rep — so
    # this radius is the surface-contact gap, not a body-overlap distance.
    fusion_radius: float = 1.5        # must be < interaction_radius
    fusion_threshold: float = 0.6     # minimum binding energy to trigger fusion [0,1]

    # ── Decay / Half-life ────────────────────────────────────────────────────
    # Composite half-lives are derived from their species hash using this range
    half_life_min: float = 1.0       # shortest composite half-life (sim time units)
    half_life_max: float = 200.0      # longest composite half-life

    # ── Energy ───────────────────────────────────────────────────────────────
    # Kinetic energy scale at initialization (controls initial temperature)
    init_speed: float = 1.5
    # Binding energy scale — how much energy fusing releases
    binding_energy_scale: float = 1.0
    # Velocity damping per step (1.0 = no damping, 0.99 = slight damping)
    damping: float = 0.995
    max_velocity: float = 8.0         # velocity clamp

    # ── Hash Chemistry ───────────────────────────────────────────────────────
    # Change this to get a completely different "universe" / chemistry
    hash_modulus: int = 100_000_007   # large prime
    hash_prime_a: int = 1_000_003     # multiplier prime
    hash_prime_b: int = 7             # offset

    # ── Composite Stability ───────────────────────────────────────────────────
    # stability_mode picks how half-life is determined:
    #   "liquid_drop" (default) — live fissility law, recomputed every step:
    #       E_coh = Σ bond E_b − surface_energy_coeff · n^(2/3)
    #       x     = E_rep / (2 · E_coh)          (E_rep = internal hard-core PE)
    #       hl    = hl_min + (hl_max − hl_min) · t_coh · clip(1 − x, 0, 1)^fissility_exponent
    #       with t_coh = clip(E_coh / (cohesion_hl_scale · n), 0, 1).
    #     Big/crammed/weakly-bonded composites fission fast; the BE→hl values
    #     written at fusion/fission time become initial placeholders only.
    #   "legacy" — the original hash-BE → half-life formula with the size
    #     penalty below, fixed at creation time.
    # Static field — changing it retraces once.
    stability_mode: str = "liquid_drop"
    surface_energy_coeff: float = 0.5   # a_s — cohesion penalty × n^(2/3)
    cohesion_hl_scale: float = 5.0      # per-member cohesion needed for max stability (≈⟨E_b⟩ so t_coh is a real gradient, not saturated at 1)
    # Long-range Coulomb-analog disruption: E_coulomb = disruption_scale·n²/R_g.
    # Grows super-linearly with size (R_g ∝ √n for a compact blob), so fissility
    # x climbs with n and big/compact composites fission at a tunable critical
    # size. 0 → legacy hard-core-only fissility (backward-compatible). Runtime-
    # tunable via PhysicsParams ("disrupt k" slider). Final default tuned with
    # the diagnostic (see spec §6).
    disruption_scale: float = 0.5
    fissility_exponent: float = 1.0     # sharpness of the collapse as x → 1
    composite_size_decay_scale: float = 0.05   # size penalty on composite half-life (legacy mode + creation-time placeholder values)

    # ── Fission fracture (bond-cut, 2026-06-12) ──────────────────────────────
    # Fission no longer partitions members by hashing slot indices — it
    # fractures along the bond cut that maximizes total product binding
    # energy (the hash-BE landscape acting as the shell-structure analog).
    # Products keep the parent edges internal to them; crossing edges break.
    # The kick is the Q-value max(BE(p0) + BE(p1) − BE(parent), 0), replacing
    # the old binding_energy * (1 − fission_cost) release.
    #
    # Iteration cap for the graph sweeps (BFS / subtree sums / fragment
    # labeling) inside fission and bond scission. Correct for bond graphs of
    # diameter <= this value; members beyond the horizon stay with the root
    # fragment (graceful degradation for extreme chains). Cost is linear.
    fission_label_iters: int = 64
    # Barrier analog: when True, a decay roll whose best cut has Q < 0 is
    # suppressed entirely — hash-favored ("magic") composites become stable
    # against spontaneous fission and only break kinetically/thermally via
    # bond scission. When False, endothermic fission fires with zero kick.
    forbid_endothermic_fission: bool = True

    # ── Valence / Free Bonds (hash-encoded per-species bond capacity) ─────────
    # Each species has a fixed hash-derived valence v_s ∈ [1, max_valence] (the
    # number of "hands" a particle of that species can use to hold neighbors).
    # A composite of n members with total valence V has free_bonds = V - 2*(n-1)
    # (spanning-tree accounting: every fusion consumes one bond on each side).
    # Fusion is permitted iff both entities have free_bonds >= 1.
    # Fission products with free_bonds < 0 are structurally unsound and shatter
    # into free particles rather than forming a composite. Particle conservation
    # holds regardless. BE-threshold preference is unchanged and still drives
    # per-multiset specificity; valence layers physical saturation on top.
    use_valence: bool = True
    # Per-species valence drawn from [1, max_valence]. The choice of max_valence
    # sets the "chemistry regime" — different values give qualitatively different
    # reachable composite topologies:
    #   1 = stub world — every species v=1, no composite past size 2 (dimers only).
    #   2 = polymer world — species ∈ {1,2}, v=2 chains have free_bonds=2 at any N,
    #       so polymers of arbitrary length are structurally valid.
    #   3 = branching world — species ∈ {1,2,3}, v=3 scaffolds can host branches
    #       (free_bonds grows linearly with N).
    #   4 = carbon-like (default) — species ∈ {1..4}, full flexibility; v=4 acts
    #       like carbon, supporting both long chains and dense branching.
    max_valence: int = 4

    # ── Performance Caps ─────────────────────────────────────────────────────
    # Fusion scan length. Each unit costs ~45µs of sequential scan per step
    # (launch-bound), so this is the single biggest throughput knob. Measured
    # steady-state demand (2026-06-12, emit_events over 800 steps) is ~4.7
    # fusions/step with p99=15; only the first ~50 condensation steps burst
    # higher (max 186). Excess candidates simply retry next step, so a low cap
    # just spreads the initial condensation over a few more steps.
    max_fusions_per_step: int = 64
    # Chemistry conflict-resolution mode (static field; one-time JAX retrace
    # per mode). Gates BOTH fusion and ring closure:
    #   "matching" — parallel mutual-best matching + one batched apply.
    #                A pair fuses (or ring-closes) iff each side's best
    #                candidate is the other ("handshake"); the pair set is
    #                node-disjoint by construction so the whole batch applies
    #                in a single vmapped pass. Unmatched candidates retry next
    #                step. Replaces the sequential scans whose ~45µs/iteration
    #                of launch-bound kernels were the single biggest per-step
    #                cost (2026-06-12).
    #   "scan"     — legacy sequential greedy scans over shuffled candidates.
    fusion_mode: str = "matching"
    # Fission batch width. apply_composite_decay runs its heavy per-fission
    # work (hash-partition argsort over M, product COMs, the per-member kick
    # grid) over a compacted (K, ...) batch instead of all max_composites
    # slots. Measured steady-state demand is ~15 fissions/step at
    # num_species=3 (≤14 observed at 12 species), so 64 is conservative.
    # Fissions beyond the budget defer: the composite stays alive and
    # re-rolls its decay next step.
    max_fissions_per_step: int = 64
    # Enable spring bond forces between composite members (expensive; off by default)
    use_bond_forces: bool = True
    # Stiffness of the composite-member spring (used by step.py when
    # use_bond_forces is True). Runtime-tunable via the Params panel.
    spring_k: float = 5.0

    # ── Sparse covalent bonds (new bond model) ───────────────────────────────
    # bond_mode selects which kernel runs in Phase 3b:
    #   "edges"       — sparse covalent bonds (new)
    #   "star_spring" — current COM-spring (legacy; reads spring_k, use_bond_forces)
    #   "off"         — no bond force; pure pairwise dynamics
    # Static field — changes trigger one-time JAX retrace per mode.
    bond_mode: str = "edges"

    # Harmonic stiffness for edge-mode bonds. Much larger than spring_k because
    # each edge is a local constraint, not an aggregate COM tie — so a bonded
    # pair at displacement 1 from rest length should still feel a force well
    # above the species-pair attraction (~1).
    k_bond: float = 20.0

    # Hash-derived per-species-pair rest lengths are no longer configured as
    # absolute bounds — they span [repulsion_radius, fusion_radius] (floor pinned
    # to the hard core, ceiling at the fusion distance), so the band auto-rescales
    # with fusion_radius. See chemistry._hash_to_rest_length.

    # ── Angle-locking (covalent bond geometry, edges mode) ────────────────────
    # Angular force between a composite's bonds so geometry isn't floppy.
    #   "off"      → no angle force (default; bit-identical to pre-feature)
    #   "vsepr"    → bond directions repel & spread evenly (2π/degree); emergent
    #               rest angle, no frustration at degree ≥ 3. Fixes floppy chains.
    #   "harmonic" → bonds pulled toward a hash-derived target angle θ0 per central
    #               species (cosine form). Robust 2-D route to prescribed bent
    #               low-valence shapes (water-analog). Intended for degree ≤ 2.
    angle_mode: str = "off"
    k_angle: float = 10.0          # angle stiffness; seeds runtime PhysicsParams.k_angle
    theta_min_deg: float = 90.0    # harmonic θ0 band floor (degrees)
    theta_max_deg: float = 180.0   # harmonic θ0 band ceiling (degrees)

    # Ring closure: allow intra-composite fusion when both members still have
    # per-particle free bonds (degree[i] < v_{species[i]}).
    allow_ring_closure: bool = True
    # Same sequential-scan cost model as max_fusions_per_step (~45µs/unit).
    max_ring_closures_per_step: int = 16

    # ── Chemical bond scission (per-bond breaking channel, 2026-06-12) ───────
    # Every edge carries a hash-derived dissociation energy E_b (per species
    # pair, decorrelated from BE / valence / rest length). Two break modes:
    #   kinetic: stretch strain energy 0.5·k_bond·max(r − r_rest, 0)² >= E_b
    #            → the bond snaps deterministically (the harmonic well is no
    #            longer bottomless).
    #   thermal: below threshold, P = 1 − exp(−dt·ν0·exp(−(E_b − strain)/kT))
    #            (Arrhenius). kT = 0 disables thermal breaking entirely.
    # Compression never breaks a bond — only stretch counts.
    # At most one bond per composite breaks per step (the most-overstretched
    # breaking edge). A broken bridge splits the composite into its two
    # connected halves (no kick); a broken ring edge just removes the edge.
    # Requires bond_mode == "edges".
    enable_bond_scission: bool = True
    # bond_energy_scale sits ABOVE the natural equilibrium-bond strain band
    # (measured mean ~0.25, p99 ~2.3, max ~3.3 at k_bond=20), so the kinetic
    # channel snaps only genuinely overstretched bonds (a distance-4 bond has
    # strain ~90 ≫ 10) rather than normal equilibrium bonds — a scale of 2.0
    # capped every composite at a dimer (2026-06-12). E_b = hash_frac × this.
    bond_energy_scale: float = 10.0       # E_b = hash_frac × this
    bond_temperature: float = 1.0         # kT for the Arrhenius thermal channel
    bond_break_attempt_rate: float = 0.1  # ν0 — attempt frequency per sim-time
    max_scissions_per_step: int = 32      # budget; excess breaks defer a step

    # ── Profiling / Instrumentation ──────────────────────────────────────────
    enable_profiling: bool = False
    cc_fusion_event_logging: bool = False  # Log individual C+C fusion events to console

    # ── Diagnostic event log ─────────────────────────────────────────────────
    # When True, attempt_fusion and apply_composite_decay emit per-reaction
    # event tuples (source/product slot/hash/size) for the analysis pipeline.
    # Static-arg: when False, the emission code path is dead-code-eliminated
    # before JIT compilation — bit-for-bit the same kernel as before. The
    # halflife.analysis runner sets this True, and since 2026-06-12 the live
    # app (main.py build_config) also turns it on to drive the renderer's
    # event sprites from real events (+3.0% step cost measured at 5k
    # particles). Headless/test configs keep the False default.
    emit_events: bool = False

    # ── Rendering ────────────────────────────────────────────────────────────
    window_width: int = 1280
    window_height: int = 720
    fps_target: int = 120
    point_size_min: float = 2.0       # minimum particle render size (pixels)
    point_size_max: float = 14.0      # maximum (scales with mass)
    background_color: tuple = (0.004, 0.004, 0.007, 1.0)  # dark blue-black, in LINEAR sRGB (post-tonemap displays ≈ 5% gray-blue)

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

    @property
    def e_max(self) -> int:
        """Maximum edges per composite: enough for any all-bonds-used graph.

        Floored at max_composite_size - 1 so a path-spanning tree always fits
        even at max_valence=1 (where M*v/2 would be too small to hold the
        tree edges fusion creates, silently dropping bonds).
        """
        return max(self.max_composite_size - 1,
                   (self.max_composite_size * self.max_valence) // 2)
