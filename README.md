# Half-Life Particle Simulator

A GPU-accelerated 2D particle simulator where **everything decays**. Particles interact
via species-dependent forces, fuse into composite structures, and probabilistically
disintegrate — all governed by exponential half-life decay. The design goal is emergent
complexity: self-maintaining clusters, autocatalytic loops, and evolutionary dynamics
arising purely from physics and implicit chemistry.

---

## Quick Start

```bash
# Default run (2,000 particles)
python -m halflife.main

# Custom particle count and seed
python -m halflife.main --particles 4000 --seed 42

# Physics only, no fusion/decay
python -m halflife.main --no-chemistry
```

**Requirements:** Python 3.10 (WSL/Ubuntu), JAX + CUDA 12, ModernGL, pygame.
See `requirements.txt` for exact versions.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `+` / `-` | More / fewer simulation steps per frame |
| `B` | Toggle composite view: bonds (GL_LINES) ↔ merged (large point) |
| `R` | Reset to initial state |
| `S` | Save screenshot |
| `Q` / `Esc` | Quit |

---

## How It Works

### Simulation Loop

Each frame, `simulation_step()` runs a single JIT-compiled JAX function through 9 phases:

```
Phase 1 — Spatial indexing    build_cell_list()         O(N log N)
Phase 2 — Neighbor finding    find_all_neighbors()      O(N × K)
Phase 3 — Force computation   compute_all_forces()      O(N × K)
Phase 4 — Integration         Euler step + damping      O(N)
Phase 5 — Boundary            periodic wrap / reflect   O(N)
Phase 6 — Fusion              attempt_fusion()          O(N + F)
Phase 7 — Decay               apply_*_decay()           O(N + C)
Phase 8 — Energy accounting   soft velocity rescaling   O(N)
Phase 9 — Age / counters      increment ages            O(N)
```

`N` = max particles, `K` = max neighbors per particle, `F` = max fusions per step,
`C` = max composites. All sizes are compile-time constants (JAX/XLA requires static shapes).

---

### Particles

Each particle carries:

| Field | Type | Description |
|-------|------|-------------|
| `position` | (N, 2) f32 | World coordinates |
| `velocity` | (N, 2) f32 | Velocity |
| `species` | (N,) i32 | Type index [0, num_species) |
| `alive` | (N,) bool | Slot occupancy mask |
| `mass` | (N,) f32 | Mass (currently uniform = 1.0) |
| `half_life` | (N,) f32 | Individual decay half-life |
| `composite_id` | (N,) i32 | Which composite owns this particle (-1 = free) |
| `internal` | (N, 8) f32 | Hidden state vector (NCA-style, reserved for future use) |

Dead particles (`alive=False`) are empty pool slots. They contribute zero to all
sums and forces via masking — no compaction is needed.

---

### Forces (Particle Life Kernel)

Every particle pair (i, j) within `interaction_radius` experiences a **Particle
Life**-style triangle force kernel that is fully parameterized by species:

```
distance r:
  [0,          repulsion_radius]  →  strong repulsion (hard core)
  [repulsion_radius, r_attract]  →  ramp to peak attraction/repulsion
  [r_attract,  r_cutoff]         →  ramp back to zero
  [r_cutoff,   ∞]               →  no force
```

The `attraction[species_i, species_j]` matrix is a random (num_species × num_species)
array initialized at startup. Asymmetric entries produce chasing dynamics; symmetric
negative entries produce clustering. The matrix is a regular JAX array — it can be
changed without recompiling.

Forces are computed via double `vmap`: outer over all N particles, inner over each
particle's up to `max_neighbors=32` neighbors from the cell list.

---

### Spatial Indexing (Cell List)

The world is divided into a grid of cells of size `cell_size = interaction_radius`.
Each step:

1. Every alive particle is assigned to a cell (`floor(pos / cell_size)`).
2. Particles are sorted by cell index (`jnp.argsort`), then scattered into a
   `(num_cells, cell_capacity)` lookup table.
3. Neighbor queries scan only the 3×3 cell neighborhood (9 cells × 16 slots = 144
   candidates per particle), then filter by actual distance.

This gives O(N) average neighbor-finding cost instead of O(N²).

---

### Hash Chemistry

Reaction rules are **implicit** — there is no lookup table. Instead, a polynomial
rolling hash over the **sorted multiset of member species** determines all composite
properties:

```python
h = 1
for species in sorted(member_species):
    h = (h * PRIME_A + species + PRIME_B) % MODULUS
```

From `h`, three properties are derived deterministically:

| Property | Bits used |
|----------|-----------|
| `half_life` | `h % 1000` → scaled to [half_life_min, half_life_max] × composite_half_life_scale |
| `binding_energy` | `(h // 1000) % 1000` → [0, 1] |
| `decay_products` | `h >> 20`, `h * k % modulus` → product count and species |

The same species composition always produces the same hash → same chemistry. Changing
`config.hash_modulus` gives an entirely different "universe" with different reaction rules.

---

### Fusion

Each step, for every free particle i, its neighbors within `fusion_radius` are checked.
If both particles are free, their hash-derived `binding_energy > fusion_threshold`, they
fuse into a new composite.

**Conflict resolution:** a sequential `lax.scan` over the first `max_fusions_per_step`
candidates (pre-filtered from all N particles) ensures each particle is claimed at most
once per step.

A fused composite records:
- The indices of its member particles
- Its `binding_energy` and `half_life` (from the species hash)
- Its `species_hash` (for future multi-body reactions)

---

### Decay and Fission

Every step, each alive entity draws a uniform random number. It decays if:

```
rand < 1 - exp(-dt × ln2 / half_life)
```

**Particle decay:** the particle is killed; a product particle of a hash-derived species
is spawned into a nearby free slot (up to `max_decay_per_step` spawns per step).

**Composite fission:** the composite is marked dead; all member particles are released
back to free status and receive a radial velocity kick proportional to
`sqrt(2 × binding_energy_per_member × (1 - fission_cost))`.

---

### Energy Conservation

Total energy = kinetic energy of all particles + binding energy of all composites.
After each step, a soft correction rescales all velocities uniformly by at most 1% to
keep total energy near its initial value, preventing long-term drift without imposing
hard constraints that could destabilize dynamics.

---

### Rendering

The renderer uses **ModernGL + pygame** sharing a single OpenGL context:

- **Particles:** rendered as point sprites via a GLSL vertex+fragment shader.
  Color = HSV hue by species; size ∝ log(mass); brightness ∝ speed.
- **Bonds mode** (`B`): `GL_LINES` connecting every pair within each alive composite.
- **Merged mode** (`B`): single oversized point at composite center of mass.

Data flow per frame:
1. `np.asarray()` — JAX GPU array → CPU numpy (GPU→CPU sync point)
2. Pack vertex buffer: `(x, y, r, g, b, a, size)` per particle
3. `vbo.write()` — upload to GPU via OpenGL
4. `glDrawArrays(GL_POINTS)` + optional `GL_LINES`
5. `pygame.display.flip()`

---

## Configuration (`halflife/config.py`)

All parameters live in `SimConfig` (frozen dataclass). Key knobs:

```python
SimConfig(
    # Scale
    max_particles       = 4_000,   # pool size; increase for richer dynamics
    num_particles_init  = 2_000,   # particles spawned at t=0
    num_species         = 12,      # more species → richer chemistry

    # Force shape
    interaction_radius  = 4.0,     # force cutoff distance
    repulsion_radius    = 0.8,     # hard-core repulsion inner radius
    repulsion_strength  = 2.0,

    # Chemistry
    fusion_radius       = 1.0,     # must be < interaction_radius
    fusion_threshold    = 0.2,     # min binding energy to trigger fusion [0,1]
    half_life_min       = 50.0,    # particle half-life range (sim time units)
    half_life_max       = 500.0,

    # Universe identity
    hash_modulus        = 100_000_007,  # change this for a different chemistry
    hash_prime_a        = 1_000_003,
    hash_prime_b        = 7,

    # Performance caps
    max_fusions_per_step = 100,    # fusion scan length cap
    max_decay_per_step   = 200,    # spawn slots allocated per step
    use_bond_forces      = False,  # spring forces within composites (expensive)
)
```

---

## Project Structure

```
halflife-particle/
├── CLAUDE.md           — AI assistant project context
├── PLAN.md             — Progress tracking and known issues
├── README.md           — This file
├── requirements.txt
└── halflife/
    ├── config.py       — SimConfig frozen dataclass
    ├── state.py        — ParticleState, CompositeState, WorldState, initialize_world()
    ├── utils.py        — hash_multiset(), find_free_slots(), boundary helpers
    ├── spatial.py      — build_cell_list(), find_all_neighbors()
    ├── interactions.py — InteractionParams, particle_life_force(), compute_all_forces()
    ├── chemistry.py    — attempt_fusion(), apply_particle_decay(), apply_composite_decay()
    ├── energy.py       — compute_total_energy(), apply_soft_energy_conservation()
    ├── step.py         — simulation_step() — single @jax.jit orchestrator
    ├── renderer.py     — ModernGL + pygame visualization
    └── main.py         — Entry point, event loop, CLI args
```

---

## Bug Fixes Applied During Development

### 1. `chemistry.py` — Shape mismatch in center-of-mass calculation
**Symptom:** `TypeError: mul got incompatible shapes (1000, 2) and (8, 1)` on first run.

**Root cause:** `apply_composite_decay` computed composite center-of-mass by multiplying
`particles.position` (shape `(N, 2)`) by a member validity mask (shape `(M, 1)`). These
don't broadcast because `N ≠ M`.

**Fix:** Replaced with indexed gather — use the member particle IDs to index into
`particles.position`, producing an `(M, 2)` array, then mask and sum normally.

### 2. `renderer.py` — `KeyError: 'u_window_size'`
**Symptom:** Crash in renderer `__init__` on the line setting `u_window_size`.

**Root cause:** The vertex shader declared `uniform vec2 u_window_size` but never
referenced it in the shader body. GLSL compilers eliminate unused uniforms, so ModernGL
doesn't register it as a program member. The Python code then tried to set it by name
and got a `KeyError`.

**Fix:** Removed the dead uniform from both the shader source and the Python init code.

---

## Performance History

| State | Step time | Notes |
|-------|-----------|-------|
| Initial run (N=10K, no fixes) | ~500ms | 2 FPS observed |
| After perf fixes (N=2K) | ~3.9ms | ~258 FPS potential |

Four bottlenecks were identified and fixed:

### 1. `attempt_fusion` — O(N) sequential scan (critical)
`jax.lax.scan` over all N particles, carrying the full `CompositeState` (a
`(max_composites, max_composite_size)` members array) as scan carry state. Each of the
N sequential iterations modifies this large array. At N=20,000 with 5,000 composites,
this dominated total step time.

**Fix:** Pre-filter to the first `max_fusions_per_step` particles that have a valid
fusion candidate using `cumsum` + `sort`, then scan only over those.

### 2. `find_free_slots` — O(N²) slot search (critical)
The original implementation used `jax.vmap` over N ordinals, each running
`jnp.min(jnp.where(...))` over all N slots — O(N²) total.

**Fix:** Use `jnp.sort` on a candidate array where free slots keep their index and
occupied slots get sentinel `N`. Free slots sort to the front. O(N log N).

### 3. `compute_bond_forces` — unconditional iteration over all composites (high)
Spring bond forces between composite members were computed every step by vmapping over
all `max_composites=5,000` composites and all `M*(M-1)/2 = 28` pairs each — 140,000
pair evaluations per step even when nearly all composites were dead.

**Fix:** Gated behind `config.use_bond_forces` (default `False`). The normal Particle
Life attraction force already keeps composite members together.

### 4. Default config too large (high)
`max_particles=20,000` and `num_particles_init=10,000` made all O(N) operations 10×
more expensive than early benchmarks suggested (which used N=1,000).

**Fix:** Reduced defaults to `max_particles=4,000`, `num_particles_init=2,000`,
`max_composites=500`. Scale up in `config.py` once behavior is tuned.

---

## Design Inspirations

| Source | Contribution |
|--------|-------------|
| **Particle Life** | Species-dependent asymmetric attraction matrices |
| **Particle Lenia** | Energy-based formulation, ring-shaped force kernels |
| **Hash Chemistry** (Sayama) | Implicit reactions via hash of sorted species multiset |
| **Ising Model** | Criticality, phase transitions, composites as bonds |
| **Boids** | Cell-list O(N) spatial indexing |
| **Neural CAs** | `internal` hidden state vector per particle (future) |
| **FlowLenia** | Mass-conserving advection (planned Phase 6) |
