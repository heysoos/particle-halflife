# Half-Life Particle Simulator

A GPU-accelerated 2D particle simulator where **everything decays**. Particles interact
via species-dependent forces, fuse into composite structures, and probabilistically
disintegrate — all governed by exponential half-life decay. The design goal is emergent
complexity: self-maintaining clusters, autocatalytic loops, and evolutionary dynamics
arising purely from physics and implicit chemistry.

---

## Quick Start

```bash
# Default run (5,000 particles, 3 species — see config.py)
python -m halflife.main

# Override particle count, species count, world size, or seed
python -m halflife.main --particles 4000 --species 6 --seed 42
python -m halflife.main --width 300 --height 168.75

# Physics only, no fusion/decay
python -m halflife.main --no-chemistry
```

**Requirements:** Python 3.10 (WSL/Ubuntu), JAX + CUDA 12, ModernGL, pygame.
See `requirements.txt` for exact versions.

### Controls

**Keyboard:**

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `+` / `-` | More / fewer simulation steps per frame |
| `B` | Cycle composite view: **bonds → merged → none → bonds** |
| `R` | Reset to initial state (re-uses current interaction params) |
| `S` | Save screenshot |
| `Q` / `Esc` | Quit |

**Mouse (on-screen buttons, left edge of window):**

| Button | Action |
|--------|--------|
| Pause / Resume | Toggle pause |
| Bonds / Merged / None | Cycle composite visualization. Label shows the *current* mode. |
| Events | Toggle reaction-site event sprites (fusion/fission rings) |
| Reset | Re-initialize world (same interaction params, fresh particles) |
| Params | Toggle the live physics-slider panel (see [PhysicsParams](#physicsparams-runtime-tunable)) |
| Reroll All | New interaction matrix **and** new particle initial conditions |
| Reroll IC | New particle initial conditions only (keep current chemistry) |
| Reroll Chem | New interaction matrix only (keep particle positions/species) |

A separate **Stats** button sits in the top-right corner and toggles the live
stats panel — see [Stats Panel](#stats-panel) below.

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
| `mass` | (N,) f32 | Mass (currently uniform = 1.0) |
| `energy` | (N,) f32 | Kinetic + internal energy (book-keeping for stats) |
| `age` | (N,) f32 | Time since the particle last became free |
| `composite_id` | (N,) i32 | Which composite owns this particle (-1 = free) |

All particles are *always alive* — the pool is a fixed pre-allocation. There's no
`alive` mask on particles; species identity is conserved across the whole run
(fission and fusion never transmute particles, they only rearrange bonding).

### Composites

A composite is a bonded cluster of particles. Composite state lives in a fixed pool
of size `max_composites`:

| Field | Type | Description |
|-------|------|-------------|
| `members` | (C, M) i32 | Particle indices per composite, padded with `-1` |
| `member_count` | (C,) i32 | Number of valid members (densely packed from index 0) |
| `alive` | (C,) bool | Slot occupancy mask |
| `binding_energy` | (C,) f32 | Released on formation, paid back on fission |
| `half_life` | (C,) f32 | Hash-derived; modulated by size penalty |
| `age` | (C,) f32 | Time since formation |
| `species_hash` | (C,) u32 | Commutative hash of member species multiset |
| `free_bonds` | (C,) i32 | Remaining bond capacity (Σ valences − 2·(n−1)) — see [Valence](#valence--free-bonds) |

---

### Forces (Particle Life Kernel)

Every particle pair (i, j) within `interaction_radius` experiences a **Particle
Life**-style triangle force kernel. The shape is parameterized per *species pair*:

```
distance r (units of interaction_radius):
  [0,                   repulsion_radius] →  strong repulsion (hard core, global)
  [repulsion_radius,    r_peak[i,j]]      →  ramp toward signed peak
  [r_peak[i,j],         r_cutoff[i,j]]    →  ramp back to zero
  [r_cutoff[i,j],       ∞]                →  no force
```

`InteractionParams` carries three matrices, all `(num_species, num_species)` f32:

| Matrix | Range | What it tunes |
|--------|-------|---------------|
| `attraction[i,j]` | signed, `[-1, 1]` | sign + magnitude of the force peak |
| `peak_fraction[i,j]` | `(0, 1]` | radius of peak attraction, as fraction of `interaction_radius` |
| `cutoff_fraction[i,j]` | `(0, 1]` | zero-force radius, as fraction of `interaction_radius` |

Ordering invariant enforced at init: `0 < repulsion_fraction < peak_fraction < cutoff_fraction ≤ 1`.
Asymmetric entries (A attracts B but B repels A) produce chasing dynamics; symmetric
negative entries produce clustering.

Forces are computed via double `vmap`: outer over all N particles, inner over each
particle's up to `max_neighbors` neighbors from the cell list.

#### PhysicsParams (runtime-tunable)

A small set of scalars — `dt`, `damping`, `repulsion_strength`, `repulsion_radius`,
`fusion_threshold`, `binding_energy_scale`, `r_cutoff_scale`, `spring_k`,
`attraction_scale` — live in a separate `PhysicsParams` NamedTuple that's passed to
the simulation as a *regular* (non-static) JAX argument. Changing them does not
trigger XLA recompilation. The on-screen **Params** panel exposes them as live
sliders; defaults are sourced from `SimConfig` via `initialize_physics_params(config)`.

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

Reaction rules are **implicit** — there is no lookup table. Instead, a
commutative hash over the multiset of member species determines all composite
properties. The hash is built by summing a per-species "atom hash" (a Fibonacci-
mixed remix of the species index) over members, mod `hash_modulus`. The
multiset hash never depends on member order, so a permutation of the same
species always produces the same value:

```python
h = (sum(_entity_hash_val(s) for s in member_species)) % MODULUS
```

From `h`, three properties are derived deterministically via decorrelating
re-mixes (different multipliers and shifts so they aren't aliased):

| Property | How it's derived |
|----------|------------------|
| `binding_energy` | hash remix → fraction in `[0, 1]`, scaled by `physics.binding_energy_scale` |
| `half_life` | hash remix → `[half_life_min, half_life_max]`, then size-penalized via `composite_size_decay_scale` (larger composites decay faster) |
| `fission partition` | `_hash_to_partition` returns a per-member ∈ {0, 1} assignment plus a hash-derived split pivot in `[1, n−1]` — see [Fission](#decay-and-fission) |

The same species composition always produces the same hash → same chemistry.
Changing `config.hash_modulus`, `hash_prime_a`, or `hash_prime_b` gives an
entirely different "universe" with different reaction rules.

### Valence & Free Bonds

Each species has a fixed **valence** `v_s ∈ [1, max_valence]`, deterministically
hashed from the species index (decorrelated from the BE/half-life draws by a
separate re-mix). Valence represents the maximum number of "hands" a particle
of that species can use to hold neighbors — analogous to molecular valence
(H=1, O=2, C=4 …).

For any entity (free particle or composite), a running **free-bond count** is
maintained:

```
free_bonds(free particle s) = v_s
free_bonds(composite n)     = Σ v_s_i  −  2 × (n − 1)
```

The `2 × (n − 1)` term is spanning-tree accounting: an `n`-member composite has
`n − 1` internal edges, each consuming one bond on each of its endpoints.

Two gates use this:

* **Fusion** (`attempt_fusion`): a pair fuses only if both entities have
  `free_bonds ≥ 1`. The merged composite's new free-bond count is
  `free_bonds(i) + free_bonds(j) − 2`.
* **Fission** (`apply_composite_decay`): products whose own member multiset
  would give `free_bonds < 0` are *structurally unsound* and shatter into free
  particles rather than forming a sub-composite. The fission kick still fires
  (binding-energy release happens regardless of whether the pieces re-bind).

Both gates are statically toggled on `config.use_valence` (default `True`) so
turning the feature off is zero-cost. BE-threshold preference still drives
per-multiset specificity; valence layers a physical saturation cap on top.

---

### Fusion

Each step, for every entity (free particle or live composite) i, neighbors within
`fusion_radius` are scored against `physics.fusion_threshold` on the *merged*
multiset's hash-derived binding energy. The pair fuses iff:

1. Both i and j are in range
2. `binding_energy(merged hash) > physics.fusion_threshold`
3. Combined member count ≤ `max_composite_size`
4. (If `use_valence`) both entities have `free_bonds ≥ 1`

**Conflict resolution:** a sequential `lax.scan` over the first `max_fusions_per_step`
candidates (pre-filtered from all N entities) ensures each entity is claimed at most
once per step. Composite-composite fusion is supported — the result reuses one parent
slot, while the other is freed.

A fused composite records its members, `binding_energy`, `half_life` (with size
penalty), `species_hash`, and its new `free_bonds = free_bonds(i) + free_bonds(j) − 2`.

---

### Decay and Fission

Every step, each alive **composite** draws a uniform random number. It decays if:

```
rand < 1 - exp(-dt × ln2 / half_life)
```

Free particles do not decay — chemistry only acts on composite structure. Particle
species are conserved across the whole run.

**Binary fission** (`apply_composite_decay`): the multiset hash is fed through
`_hash_to_partition`, which assigns each member to product 0 or product 1 via a
deterministic per-slot key sort + hash-derived pivot. Both products are guaranteed
non-empty.

For each product:

* If it has ≥ 2 members and its own multiset is valence-feasible
  (`free_bonds ≥ 0`), it becomes a sub-composite. Product 0 reuses the parent
  composite slot; product 1 takes a fresh free slot.
* Otherwise (size-1 product, or `free_bonds < 0`), its members shatter into
  *free particles*.

In every case, members receive a momentum-conserving velocity kick along the
COM-vs-COM axis. The kick magnitude is set so the released kinetic energy
matches `binding_energy × (1 − fission_cost)`. Particle and per-species counts
are exactly conserved through every fission.

---

### Energy Tracking

Total energy = kinetic energy of all particles + binding energy of all composites.
The accumulator is updated each step and shown in the stats panel.

A soft global velocity rescale that nudged total energy back toward its initial
value used to run in Phase 8, but it was found to be unnecessary once velocity
damping (`physics.damping ≈ 0.995`) and the velocity clamp (`max_velocity`) were
tuned, and it sometimes hid genuine cooling/heating in chemistry experiments.
It is currently disabled — energy drifts naturally, and the trace in the stats
panel makes the drift directly visible.

---

### Rendering

The renderer uses **ModernGL + pygame** sharing a single OpenGL context:

- **Particles:** point sprites (GLSL). Color = HSV by species; size ∝ log(mass);
  brightness ∝ speed.
- **Composite view** (`B` cycles through three modes):
  - **Bonds:** `GL_LINES` between composite members. To prevent O(N²) line counts
    on large composites, only `MAX_BONDS_PER_PARTICLE = 3` forward-slot bonds are
    emitted per member (each member i connects to slots i+1..i+3). Bond set is
    deterministic across frames — no flicker. Line endpoints use the minimum-image
    convention so lines never cross the screen at periodic boundaries.
  - **Merged:** single oversized point at the composite's center of mass. COM is
    computed periodic-aware (anchor on member slot 0, minimum-image displacements,
    wrap result) so composites straddling a boundary don't teleport to mid-screen.
  - **None:** plain particles only — no overlay.
- **Event sprites:** a separate point-sprite pass draws expanding rings at reaction
  sites. Colors: fusion = gold, fission = cyan. Each ring expands over its sim-time
  lifetime then disappears.
- **HUD overlay:** a `pygame.Surface(RGBA)` is drawn each frame (buttons, optional
  stats/params panels, key-hint bar) and uploaded as a `moderngl.Texture` rendered
  on a fullscreen quad. The surface is only re-rendered + re-uploaded when something
  changed (dirty-flag) so static UI doesn't burn CPU each frame.

Data flow per frame:
1. Single batched `jax.device_get(...)` — JAX GPU → CPU numpy (one CUDA sync, one DMA)
2. Pack particle VBO: `(x, y, r, g, b, a, size)` per particle, written in-place
3. Pack bond VBO (if Bonds mode) or COM VBO (if Merged mode)
4. Draw particles (`GL_POINTS`), bonds (`GL_LINES`), event rings (`GL_POINTS`)
5. Render pygame HUD surface as fullscreen textured quad
6. `pygame.display.flip()`

### Stats Panel

Toggled from the **Stats** button. Top-right overlay with:

- **Scalars:** FPS, step count, sim time, total particles
- **Sparklines** (last ~150 frames each): Free particles, Composites, Unique types,
  Energy, Fusions/s, Decays/s
- **Composite-size histogram:** bar chart of population by member count. X-axis
  spans 1 .. `max_composite_size`; bins widen automatically if the cap exceeds
  the `MAX_BINS_HIST = 100` bar budget. Auto-zooms to the largest live composite
  (grows instantly, shrinks lazily every ~100 frames to suppress jitter).
- **Throttling:** the `np.histogram` recompute runs at most every 10 frames AND
  only while the panel is open — the bar chart is the only stat that's not
  needed for sparkline continuity, so it's skippable when hidden.

"Unique types" is the count of distinct `species_hash` values across alive
composites — the size of the chemical zoo currently present. Useful for spotting
autocatalytic monocultures: if Composites is high but Unique is low, one
multiset is dominating.

### Params Panel

Toggled from the **Params** button. Each slider is bound to one `PhysicsParams`
field and pushes updates into the running simulation without recompilation. Slider
defaults are sourced from `initialize_physics_params(config)`, so changing a
SimConfig value and restarting puts the knob at the right position.

| Group | Sliders |
|-------|---------|
| Force kernel shape | `repulsion_strength`, `repulsion_radius`, `attraction_scale`, `r_cutoff_scale` |
| Fusion chemistry | `fusion_threshold`, `binding_energy_scale` |
| Particle dynamics | `dt`, `damping`, `spring_k` |

Most sliders are log-scale (0.1× – 10× of the default); `dt` and `damping` are
linear within hardcoded ranges (`(0.001, 0.1)` and `(0.0, 1.0)` respectively).

---

## Configuration (`halflife/config.py`)

All parameters live in `SimConfig` (frozen dataclass — passed as `static_argnums`
to JIT-compiled functions, so changing a value forces XLA recompilation). The
*runtime-tunable* subset lives in `PhysicsParams` and goes through the Params
sliders without recompilation; everything else needs a restart.

Current key defaults (see `halflife/config.py` for the full list):

```python
SimConfig(
    # ── World ──
    world_width          = 200.0,
    world_height         = 112.5,           # 16:9 aspect matched to window
    dt                   = 0.06,
    boundary_mode        = "periodic",      # or "reflect"

    # ── Scale ──
    num_particles        = 5_000,           # fixed pool, always alive
    num_species          = 3,
    max_composites       = 3_000,
    max_composite_size   = 128,             # JAX buffer (not a physics cap)

    # ── Force shape ──
    interaction_radius   = 8.0,             # force cutoff
    repulsion_radius     = 0.8,
    repulsion_strength   = 2.0,

    # ── Fusion ──
    fusion_radius        = 4.0,             # must be < interaction_radius
    fusion_threshold     = 0.6,             # min binding energy to fuse [0,1]
    binding_energy_scale = 1.0,
    half_life_min        = 1.0,             # composite half-life range
    half_life_max        = 15.0,
    fission_cost         = 0.5,             # fraction of BE consumed by fission
    composite_size_decay_scale = 0.05,      # larger composites decay faster

    # ── Universe identity ──
    hash_modulus         = 100_000_007,
    hash_prime_a         = 1_000_003,
    hash_prime_b         = 7,

    # ── Valence / free bonds ──
    use_valence          = True,
    max_valence          = 4,               # per-species v_s drawn from [1, max_valence]

    # ── Performance / misc ──
    max_fusions_per_step = 200,
    use_bond_forces      = True,
    spring_k             = 5.0,             # composite-member spring stiffness

    # ── Rendering ──
    window_width         = 1280,
    window_height        = 720,
    fps_target           = 120,
)
```

**Changing the "universe":** tweak `hash_modulus`, `hash_prime_a`, or
`hash_prime_b` to get an entirely different reaction graph (same code, different
chemistry). Combined with a small `num_species`, the reaction network is small
enough to study by hand.

---

## Project Structure

```
halflife-particle/
├── CLAUDE.md           — AI assistant project context (run patterns, git, conventions)
├── PLAN.md             — Progress tracking and known issues
├── README.md           — This file
├── requirements.txt
├── halflife/
│   ├── config.py       — SimConfig frozen dataclass (static_argnums to JIT)
│   ├── state.py        — ParticleState, CompositeState, WorldState, PhysicsParams,
│   │                     InteractionParams, initialize_world() and friends
│   ├── utils.py        — find_free_slots(), boundary helpers, species color palette
│   ├── spatial.py      — build_cell_list(), find_all_neighbors()
│   ├── interactions.py — particle_life_force(), compute_all_forces()
│   ├── chemistry.py    — attempt_fusion(), apply_composite_decay(),
│   │                     _hash_to_partition(), _hash_to_valence(), free-bond math
│   ├── energy.py       — compute_total_energy() (soft conservation is currently off)
│   ├── step.py         — simulation_step() — single @jax.jit orchestrator
│   ├── profiler.py     — ProfileMetrics (max composite size tracking, etc.)
│   ├── renderer.py     — ModernGL + pygame visualization, sliders, stats panel
│   └── main.py         — Entry point, event loop, CLI args, async overlap
└── tests/
    ├── test_chemistry.py — Fusion / decay / fission / conservation / valence
    └── test_hash.py      — Hash helpers (BE, partition, valence)
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
| After renderer overhaul (N=5K) | render frame headroom restored at `fps_target=120` even with large composites | bond cap + COM fix + histogram throttle (see below) |

### Sim-side bottlenecks (early)

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

### Renderer-side wins (recent)

The simulation itself was already fast enough; the renderer was the bottleneck
once big composites started forming. Each line below is one merged commit:

* **Bond cap** — bonds view re-emitted `N·(N−1)/2` line segments per composite
  every frame. A single ~50-member clique was 1225 segments; with multiple mid-
  size composites alive, the CPU spent more time in the pair-gen than in physics.
  Replaced with a deterministic offset-pair scheme (`O(K·N)` per composite,
  `K=3`); pair set is stable across frames so bonds don't flicker.
* **Periodic-aware COM** — merged-mode COM was a naive position average and
  jumped to mid-screen whenever a composite straddled a wrap boundary. Now
  computed via minimum-image displacements from member slot 0.
* **Histogram throttling** — `np.histogram` on alive composite counts used to
  run every frame whether or not the stats panel was visible. Now runs at most
  every 10 frames AND only while the panel is open. The unique-types `np.unique`
  call stays at full rate (microseconds, feeds a sparkline).
* **Auto-zoom + damped histogram x-axis** — the bar chart x-axis grows
  instantly to fit a new largest composite, but only shrinks every ~100 frames
  to suppress jitter when fission/fusion churns the tail.
* **HUD dirty flag** — the pygame HUD surface is only re-rendered + re-uploaded
  to GL when a toggle/slider actually changed something (or while stats panel is
  open, since sparklines move every frame).
* **Async overlap** — `simulation_step(N+1)` is dispatched on the GPU before
  rendering frame N completes, hiding the JAX→OpenGL sync cost.

---

## Design Inspirations

| Source | Contribution |
|--------|-------------|
| **Particle Life** | Species-dependent asymmetric attraction matrices |
| **Particle Lenia** | Energy-based formulation, ring-shaped force kernels |
| **Hash Chemistry** (Sayama) | Implicit reactions via hash of sorted species multiset |
| **Ising Model** | Criticality, phase transitions, composites as bonds |
| **Molecular valence** | Per-species `v_s`, free-bond accounting, fusion saturation |
| **Boids** | Cell-list O(N) spatial indexing |
| **FlowLenia** | Mass-conserving advection (planned Phase 6) |
