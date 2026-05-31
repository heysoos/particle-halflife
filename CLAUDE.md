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
│   │                     _hash_to_binding_energy(), _hash_to_valence()
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

# Per-species (not per-multiset) valence — fixed once at config:
v_s               = _hash_to_valence(s, config)  # in [1, max_valence]; saturation gate
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

## Valence & Free Bonds (hash-encoded saturation)

Optional gate (`config.use_valence`, default True). Each species `s` gets a fixed
valence `v_s ∈ [1, max_valence]`, hashed from the species index via `_hash_to_valence`
(Fibonacci remix, decorrelated from BE). Valence is the number of "hands" a particle
of that species can use to hold neighbors — molecular-valence analog (H=1, O=2, C=4 …).

Composites carry a `free_bonds: (C,) int32` field tracking remaining bond capacity
under spanning-tree accounting:

```
free_bonds(free particle s) = v_s
free_bonds(composite n)     = Σ v_s_i  −  2 × (n − 1)
```

(An n-member composite has n−1 internal edges; each consumes one bond on each endpoint.)

Two gates:

- **Fusion**: a pair fuses iff both entities have `free_bonds ≥ 1`. The merged
  composite's bond count becomes `free_bonds(i) + free_bonds(j) − 2`.
- **Fission**: a product whose own member multiset gives `free_bonds < 0` is
  structurally unsound (more edges required than the members offer) and **shatters**
  into free particles rather than forming a sub-composite. The fission kick still
  fires; particle conservation holds.

Toggled via Python `if config.use_valence:` since config is `static_argnums`, so XLA
traces only the live branch — zero runtime cost when off. BE-threshold preference is
unaffected; per-multiset specificity rides on the BE check, valence layers physical
saturation on top.

This *replaces* the earlier capacity-cap mechanic (commit `7ccad71`, since reverted):
caps were a static per-multiset upper bound on each species count; valence is a
dynamic free-bond counter on each composite. Caps had the unphysical property that
a 2-particle composite could randomly roll a ceiling of [32, 32, 32]; valence makes
small composites of low-v species *immediately* saturated, as in real molecules.

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

    # Valence saturation (per-species free-bond gate, on by default)
    use_valence=True,
    max_valence=4,             # per-species valence drawn from [1, max_valence]
)
```

## Composite Diagnostic Reports (`halflife/analysis/`)

For diagnosing composite formation dynamics — *why* large composites are/aren't forming.
CLI-driven; runs one simulation and emits a single self-contained HTML report.

Spec: [`docs/superpowers/specs/2026-05-30-composite-diagnostic-design.md`](docs/superpowers/specs/2026-05-30-composite-diagnostic-design.md)

### Quick start (from WSL)

```bash
# Most common pattern: short-medium GPU run on a named scenario
.venv/bin/python -m halflife.analysis --scenario baseline --steps 3000 --sample-every 100 --platform gpu

# Output: tests/reports/diag_<scenario>_<timestamp>.html  (~600KB-1MB, opens in any browser)
```

3000 steps takes ~100 seconds on GPU. Step rate scales with `num_particles × num_species²`.

### Scenarios

| Preset name | What it changes from default `SimConfig` |
|---|---|
| `baseline` | (default config — `num_species=12, half_life_max=15.0`) |
| `current_experiment` | `num_species=3, half_life_max=100.0` (the user's running experiment) |
| `valence_off` | `use_valence=False` |
| `polymer_world` | `max_valence=2, num_species=2` |
| `branching_world` | `max_valence=3, num_species=3` |
| `old_star_spring` | `bond_mode='star_spring'` |

When the user's `current_experiment` knobs change, update the preset in `halflife/analysis/cli.py`.

### Common flags

| Flag | Default | Purpose |
|---|---|---|
| `--scenario <name>` | `baseline` | Preset name (see table above) |
| `--steps N` | `10000` | Total sim steps. Use 1000-3000 for "quick look", 10000+ for thorough diagnosis. |
| `--seed S` | `0` | RNG seed |
| `--sample-every K` | `100` | Full-snapshot interval. Compact metrics are emitted every step regardless. |
| `--top-k N` | `30` | K for the top-K transition / compatibility matrices |
| `--override "k1=v1,k2=v2"` | none | Per-run config overrides (e.g. `"num_species=5,fusion_radius=3.0"`) |
| `--out PATH` | auto | Output path (default: `tests/reports/diag_<scenario>_<ts>.html`) |
| `--platform cpu\|gpu` | auto | Force JAX platform. Use `cpu` for tiny test runs, `gpu` for real diagnosis |

### Comparing scenarios

The tool produces one HTML per run; there's no built-in comparison view. To diagnose
a regression: run the tool twice (e.g. `baseline` vs `current_experiment`) and open
both HTMLs in side-by-side browser windows.

The **Tier 4** fusion-compatibility matrices show what *could* happen chemistry-wise; the
**Tier 3** transition matrices show what *actually did* happen. Diffing the two visually
across the two scenarios is the core diagnostic move:
- High-BE cell in Tier 4b but cold in Tier 3 Matrix 2 → those composites *could* fuse
  but never met (kinetic problem) OR were valence-saturated in practice
- Hatched-out cells in Tier 4a/b → valence-blocked regardless of BE

### Cost

- ~30 steps/sec on GPU at default `num_particles=5000`
- Memory: ~5 MB compact metrics + ~100 MB events + ~65 MB snapshots for 10k steps (well within RAM)
- Sub-second post-processing (transition matrices + compatibility matrices)
- Output HTML is typically 500KB-2MB depending on `--top-k` and event count

### Live-app cost

**Zero.** The diagnostic kernel emission is gated by `SimConfig.emit_events` (default `False`,
`static_argnums`). When the live app runs with the default config, the emit branch is
dead-code-eliminated before JIT — the compiled kernel is bit-for-bit unchanged from the
pre-pipeline version.

### Testing the pipeline itself

The pipeline has its own pytest suite (~22 tests):
```bash
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py tests/test_analysis_metrics.py \
  tests/test_analysis_transitions.py tests/test_analysis_compatibility.py tests/test_analysis_pipeline.py -v
```
~2 minutes on CPU. Run after touching any kernel-emission code (`halflife/chemistry.py:attempt_fusion`,
`halflife/chemistry.py:apply_composite_decay`, `halflife/step.py:simulation_step`) or any
`halflife/analysis/` module.

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
