# Half-Life Particle Simulator — Project Context

## Running Python in This Project

This project uses a **WSL-based Python interpreter** (Ubuntu under Windows) with the venv at `.venv/`.
The shell used by Claude Code is Git Bash (Windows), so activating the venv requires going through WSL.

**Standard command pattern — always use this:**
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && <your command here>"
```

**Run the simulator:**
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && python -m halflife.main"
```

**Run a script:**
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && python halflife/step.py"
```

**Install a package:**
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && pip install <package>"
```

**Python version:** 3.10.12 (WSL/Ubuntu)
**Venv path (WSL):** `/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv`
**Already installed:** jax 0.6.2 + CUDA 12, jaxlib, moderngl 5.12, pygame 2.6.1, numpy 2.2.6

---

## What This Is

> **For a full human-readable explanation of the implementation, see [`README.md`](README.md).**
> It covers: simulation loop phases, data structures, force kernel, hash chemistry,
> fusion/decay/fission, rendering pipeline, config knobs, bug fixes, and performance history.

A GPU-accelerated 2D particle simulator where **everything decays**. Every particle and
composite structure has a half-life — an exponential decay probability. Particles interact
via species-dependent force kernels, fuse into composites whose properties are determined by
a hash function applied to their member species, and constantly face probabilistic fission.

The design goal is emergent complexity: autocatalytic sets, self-maintaining organizations,
and evolutionary dynamics that arise purely from the interplay of forces, fusion, and decay.

## Key Inspirations

| Inspiration | What It Contributes |
|------------|---------------------|
| **Ising Model** | Criticality, phase transitions, bonds as edges |
| **Particle Life** | Pairwise species-dependent force kernels, asymmetric attraction matrices |
| **Particle Lenia** | Energy-based formulation, ring-shaped interaction kernels |
| **FlowLenia** | Mass-conserving advection (earmarked for Phase 6) |
| **Reintegration Tracking** | GPU gather-based mass transport (Phase 6) |
| **Neural CAs** | `internal` state vector per particle, future NCA-style dynamics |
| **Hash Chemistry (Sayama)** | Hash of sorted member-species multiset → implicit reaction rules |
| **Boids** | Spatial cell-list O(N) neighbor queries |
| **Evolutionary sims** | Compartmentalization, group-level selection, autocatalysis |

## Tech Stack

- **JAX** — GPU simulation via XLA (jit, vmap, scan, lax primitives)
- **ModernGL + pygame** — Real-time rendering (VBO point sprites, GL_LINES for bonds)
- **Python 3.10+**

## Project Structure

```
halflife-particle/
├── CLAUDE.md           ← You are here (AI context)
├── README.md           ← Full implementation documentation for humans
├── PLAN.md             ← Progress tracking, todo list, known issues
├── requirements.txt
├── halflife/
│   ├── config.py       ← SimConfig frozen dataclass (the user's tuning surface)
│   ├── state.py        ← ParticleState, CompositeState, WorldState (NamedTuples)
│   ├── utils.py        ← hash_multiset(), find_free_slots(), boundary helpers
│   ├── spatial.py      ← build_cell_list(), find_all_neighbors()
│   ├── interactions.py ← InteractionParams, pairwise_force(), compute_all_forces()
│   ├── chemistry.py    ← attempt_fusion(), apply_decay(), hash_to_products()
│   ├── energy.py       ← energy tracking and soft conservation
│   ├── step.py         ← simulation_step() — single @jax.jit orchestrator
│   ├── renderer.py     ← ModernGL + pygame visualization
│   └── main.py         ← Entry point, event loop, async overlap
└── tests/
    ├── test_hash.py
    ├── test_spatial.py
    ├── test_step.py
    └── test_chemistry.py
```

## Core Data Structures (state.py)

All state uses **fixed-size JAX arrays** with boolean masks. This is mandatory for XLA
JIT compilation (static shapes required).

```
WorldState
├── ParticleState  (MAX_PARTICLES,)
│   ├── position   (N, 2) float32
│   ├── velocity   (N, 2) float32
│   ├── species    (N,)   int32    — type index [0, NUM_SPECIES)
│   ├── alive      (N,)   bool     — active slot mask
│   ├── energy     (N,)   float32
│   ├── mass       (N,)   float32
│   ├── half_life  (N,)   float32
│   ├── age        (N,)   float32
│   ├── composite_id (N,) int32    — -1 = free particle
│   └── internal   (N, STATE_DIM) float32  — NCA-style hidden state
└── CompositeState (MAX_COMPOSITES,)
    ├── members      (C, MAX_COMPOSITE_SIZE) int32  — padded with -1
    ├── member_count (C,) int32
    ├── alive        (C,) bool
    ├── binding_energy (C,) float32
    ├── half_life    (C,) float32
    ├── age          (C,) float32
    └── species_hash (C,) uint32   — hash of sorted member species multiset
```

## JAX Conventions

- **No Python control flow inside JIT**: use `jax.lax.cond`, `jax.lax.fori_loop`,
  `jax.lax.scan`, `jax.lax.switch`
- **`config` is `static_argnums`** in all JIT-compiled functions — it determines shapes
- **`params` (InteractionParams) are regular JAX arrays** — can change without recompile
- **PRNG**: always thread the key through state as `state.rng_key`; split before each use
- **Immutable updates**: use `state._replace(particles=state.particles._replace(...))`
- **Masking**: dead particles contribute 0 to sums; use `* alive` or `jnp.where(alive, x, 0)`

## Hash Chemistry

The reaction rules are **implicit** — no lookup table. A polynomial rolling hash over the
sorted multiset of member species determines all composite properties:

```python
# h = hash of sorted [species_0, species_1, ..., species_k]
h = (h * PRIME_A + species[i] + PRIME_B) % MODULUS  # in lax.fori_loop

composite_half_life    = f(h)   # derived from hash bits
composite_binding_energy = g(h) # derived from hash bits
decay_products         = parse_products(h)  # number and species of products
```

Same species set → same hash → same properties every time. This creates a consistent
"chemistry" without storing any rules. Different hash constants give different universes.

## Visualization

- **ModernGL point sprites**: each alive particle is a circle, colored by species (HSV),
  sized by mass, brightness by energy
- **Composite modes** (toggle with `B` key):
  - **Bonds mode**: GL_LINES between composite members (shows internal structure)
  - **Merged mode**: single large point at center of mass
- **Async overlap**: `simulation_step(N+1)` dispatched before rendering frame N

## Keyboard Controls (main.py)

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| `+` / `-` | Speed up / slow down (steps per render frame) |
| `B` | Toggle composite visualization (bonds ↔ merged) |
| `R` | Reset to initial state |
| `S` | Save screenshot |
| `Q` / Esc | Quit |

## Configuration

All tunable parameters live in `halflife/config.py` as `SimConfig` (frozen dataclass).
Key experiment knobs:

```python
config = SimConfig(
    num_species=16,          # more species → richer chemistry
    max_particles=50_000,    # total particle pool
    base_half_life_range=(10.0, 1000.0),  # particle lifetimes
    interaction_radius=2.0,  # force cutoff
    fusion_radius=0.5,       # must be < interaction_radius
    hash_modulus=100_000_007, # changes the "universe" / chemistry
)
```

## Development Notes

- **Build order**: config → state → utils → spatial → interactions → step → renderer → main → chemistry → energy
- **Test each phase visually** before adding the next layer
- **JIT warm-up**: first call compiles; subsequent calls are fast. Don't profile the first call.
- **Cell list overflow**: if particles cluster too much, increase `cell_capacity` in spatial.py
- **Energy conservation**: expect small drift (~1% per 1000 steps); the soft correction in energy.py keeps it bounded
