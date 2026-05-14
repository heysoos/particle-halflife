# Half-Life Particle Simulator — Project Context

## ⚠️ Comment Preservation (READ THIS FIRST)

**Do NOT delete existing comments in this codebase when editing or refactoring.** Comments
in this project are written by the user, for the user. They are load-bearing context — not
clutter. This rule overrides the default system-prompt guidance about minimizing comments.

- When refactoring a function, **preserve every comment and docstring line that is not
  itself describing code being deleted**. If you remove a feature, you may remove the
  comments that explicitly describe that feature — nothing else.
- "Clean up while I'm in there" is not a reason to delete a comment. Be surgical.
- If you believe a comment is wrong, stale, or misleading, **ask the user first** instead
  of silently removing it.
- This applies to inline `#` comments, docstrings, module headers, and any commentary in
  `.py`, `.md`, shell scripts, or anywhere else.
- When *writing new code yourself*, the default rule about minimal comments still applies
  (only comment the non-obvious WHY). This section is about not destroying what already
  exists.

Background: commit `a33593b` (refactor: drop attr_mod machinery) collaterally deleted
distance/direction labels, sign-convention notes, vmap annotations, and full Args
docstrings in `halflife/interactions.py` — none of which were related to the polarity
removal. The user flagged this as wrong; this rule exists so it doesn't happen again.

## Running Python in This Project

This project uses a **WSL-based Python interpreter** (Ubuntu under Windows) with the venv at `.venv/`.
The user runs Claude Code from two environments — patterns differ:

- **Git Bash on Windows** (`MINGW*` / `MSYS*`): venv activation must be tunneled through `wsl bash -c`.
- **Claude Code running natively in WSL** (`Linux *microsoft-standard-WSL2*`): just activate the venv directly. No `wsl bash -c` wrapper — `wsl` isn't even on PATH inside WSL itself, and the wrapper would error with `command not found`.

**Detect at session start** with `uname -a` and pick the matching pattern below.

### Pattern A — Git Bash on Windows (needs `wsl bash -c`)

```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && <your command here>"
```

Run the simulator:
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && python -m halflife.main"
```

Install a package:
```bash
wsl bash -c "source '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv/bin/activate' && pip install <package>"
```

`cd` paths with spaces require single quotes inside the outer double-quoted `wsl bash -c "..."` string. Using double quotes inside will break the shell.

### Pattern B — Claude Code running natively in WSL (no wrapper)

```bash
source .venv/bin/activate && python -m halflife.main
```

Or one-shot without activating:
```bash
.venv/bin/python -m halflife.main
```

Working directory is already the project root; absolute paths still work too.

**Python version:** 3.10.12 (WSL/Ubuntu)
**Venv path (WSL):** `/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle/.venv`
**Already installed:** jax 0.6.2 + CUDA 12, jaxlib, moderngl 5.12, pygame 2.6.1, numpy 2.2.6

---

## Git Operations

**IMPORTANT: Git has no global identity configured in WSL.** Always supply `-c user.email` and
`-c user.name` inline on every `add`/`commit` — never `git config --global` (that would mutate shared WSL state). This applies to **both** shell environments below.

### From Git Bash on Windows
```bash
wsl bash -c "cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && git -c user.email='heysoos@local' -c user.name='Heysoos' add <files...> && git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m 'message'"
```

### From Claude Code running natively in WSL
No wrapper, no `cd` (already at project root):
```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/foo.py halflife/bar.py \
  && git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m 'your message here'
```

**Never** use `git add -A` or `git add .` — the repo contains `.idea/`, `__pycache__/`, `bash.exe.stackdump`, and `init_prompt.txt` that should not be committed.

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
│   ├── chemistry.py    ← attempt_fusion(), apply_composite_decay(), _hash_to_partition(),
│   │                     _hash_to_binding_energy(), _hash_to_capacity()
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
│   ├── energy     (N,)   float32
│   ├── mass       (N,)   float32
│   ├── age        (N,)   float32
│   └── composite_id (N,) int32    — -1 = free particle (all particles always alive)
└── CompositeState (MAX_COMPOSITES,)
    ├── members      (C, MAX_COMPOSITE_SIZE) int32  — padded with -1
    ├── member_count (C,) int32
    ├── alive        (C,) bool
    ├── binding_energy (C,) float32
    ├── half_life    (C,) float32
    ├── age          (C,) float32
    └── species_hash (C,) uint32   — commutative additive hash over member species

InteractionParams  (passed separately, not part of WorldState)
    ├── attraction       (S, S) float32 — signed attraction matrix
    ├── peak_fraction    (S, S) float32 — peak-attraction radius as fraction of interaction_radius
    └── cutoff_fraction  (S, S) float32 — zero-force cutoff radius as fraction of interaction_radius

PhysicsParams  (runtime-tunable scalars; passed as dynamic JAX arg, no recompile)
    damping, repulsion_strength, fusion_threshold, binding_energy_scale,
    repulsion_radius, r_cutoff_scale, spring_k, attraction_scale, dt
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
# Per-species value, mixed once:
f(s) = (s+1)^2 * PRIME_A + (s+1) * PRIME_B   # in _entity_hash_val

# Multiset hash is the *commutative sum* of per-species values:
H(multiset) = sum(f(s) for s in members) % MODULUS
# So H(i ∪ j) = (H(i) + H(j)) % MODULUS — no sort, no fori_loop needed.

# Composite properties derived from H:
binding_energy    = _hash_to_binding_energy(h)   # Fibonacci-mixed → [0, scale]
half_life         = f(BE, size)                  # high BE + small size → long HL
fission_partition = _hash_to_partition(h, n)     # binary split into two non-empty products
capacity_caps     = _hash_to_capacity(h, config) # optional (Option 5) per-species cap vector
```

Same species set → same hash → same properties every time. Different hash constants give
different universes.

## Hash Fission (binary partition)

Composite decay is **binary fission**: a decaying composite is partitioned by
`_hash_to_partition(h, n)` into two non-empty products. Slot assignment is hash-determined
(deterministic per multiset). Product 0 reuses the parent's composite slot; product 1
claims a fresh free slot. Products of size 1 become free particles; products of size ≥ 2
become new composites. Energy: `binding_energy * (1 - fission_cost)` is split equally
between products as a momentum-conserving COM-axis kick. Species are conserved — fission
never transmutes.

## Capacity Caps (Option 5: hash-encoded saturation)

Optional gate (`config.use_capacity_caps`). Every multiset hash also rolls a per-species
capacity vector via `_hash_to_capacity` (Fibonacci remix with per-species salt — scales
to any `num_species`, not bit-budget-limited). A would-be merged composite is rejected
if any species count would exceed its cap. At fission time, a product whose own multiset
violates its hash-rolled caps **shatters** — its members become free particles instead
of forming a composite. The kick still fires. Particle conservation holds.

Toggled via Python `if config.use_capacity_caps:` since config is `static_argnums`, so
XLA traces only the live branch — zero runtime cost when off.

## Visualization

- **Particles**: point sprites colored by species (HSV), sized by log(mass), brightness by speed
- **Composite modes** (toggle with `B` key or button):
  - **Bonds mode**: GL_LINES between composite members with periodic-boundary wrapping
  - **Merged mode**: single large point at center of mass
- **HUD overlay**: pygame RGBA surface uploaded each frame as an OpenGL texture on a
  fullscreen quad. Buttons on the left edge; key hints at the bottom.
- **Event sprites**: expanding ring point sprites at fusion (gold), fission (cyan),
  spawn (green), decay (red) sites. Age tracked in sim-time; capped at 200.
- **Stats panel** (toggle): FPS, step, sim time, alive, composites, energy,
  composite-size histogram.
- **Async overlap**: `simulation_step(N+1)` dispatched before rendering frame N

## Controls

### Keyboard (main.py)

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| `+` / `-` | More / fewer simulation steps per frame |
| `B` | Toggle composite visualization (bonds ↔ merged) |
| `R` | Reset to initial state |
| `S` | Save screenshot |
| `Q` / Esc | Quit |

### Mouse (on-screen buttons, left edge)

| Button | Action |
|--------|--------|
| Pause / Resume | Toggle pause |
| Bonds / Merged | Toggle composite visualization |
| Stats | Toggle live stats panel |
| Events | Toggle event sprites |
| Reset | Re-initialize world |

## Configuration

All tunable parameters live in `halflife/config.py` as `SimConfig` (frozen dataclass).
Key experiment knobs:

```python
config = SimConfig(
    num_species=12,            # more species → richer chemistry / bigger hash bucket space
    num_particles=5_000,       # total particle pool (fixed; all always alive)
    interaction_radius=8.0,    # force cutoff
    fusion_radius=4.0,         # must be < interaction_radius
    fusion_threshold=0.6,      # min binding energy to fuse [0,1]
    half_life_min=1.0,
    half_life_max=15.0,
    hash_modulus=100_000_007,  # changes the "universe" / chemistry
    composite_size_decay_scale=0.05,  # bigger composites decay faster

    # Capacity caps (Option 5: hash-encoded saturation, off by default)
    use_capacity_caps=False,
    capacity_max=16,           # per-species cap drawn from [1, capacity_max]
)
```

## Development Notes

- **Build order**: config → state → utils → spatial → interactions → step → renderer → main → chemistry → energy
- **Test each phase visually** before adding the next layer
- **JIT warm-up**: first call compiles; subsequent calls are fast. Don't profile the first call.
- **Cell list overflow**: if particles cluster too much, increase `cell_capacity` in spatial.py
- **Energy conservation**: expect small drift (~1% per 1000 steps); the soft correction in energy.py keeps it bounded
- **GPU contention with live sim**: integration tests in `test_chemistry.py` default to GPU. If
  the user has the live sim running, force CPU with `JAX_PLATFORMS=cpu pytest ...` — otherwise
  pytest can hang for an hour fighting for SM time.
- **Diagnostic scripts**: live in `/tmp/` as throwaways. They need explicit
  `sys.path.insert(0, "/mnt/.../halflife-particle")` (not `dirname(__file__)`) since they're
  outside the project root.
