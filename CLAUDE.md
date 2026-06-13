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

**Identity is set in WSL's global config** (`/home/heysoos/.gitconfig`) — `user.email =
sina.abdollahi@gmail.com`, `user.name = Heysoos`. This is the email tied to the user's
GitHub account, so commits made with it appear on GitHub's contribution graph. **Do not
override it inline** (`-c user.email=...`) — earlier guidance in this file used to recommend
that, and it produced ~80 commits attributed to a fake `heysoos@local` address that GitHub
couldn't link to any account. Just let the global config flow through.

Both shell environments below ultimately invoke `git` inside WSL, so both pick up the same
WSL global identity.

### From Git Bash on Windows
```bash
wsl bash -c "cd '/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle' && git add <files...> && git commit -m 'message'"
```

### From Claude Code running natively in WSL
No wrapper, no `cd` (already at project root):
```bash
git add halflife/foo.py halflife/bar.py && git commit -m 'your message here'
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
│   ├── graph.py        ← bond-graph algorithms (bfs_tree, subtree_sums, descendant/reachable masks)
│   ├── interactions.py ← InteractionParams, pairwise_force(), compute_all_forces() (returns forces + repulsion PE)
│   ├── chemistry.py    ← attempt_fusion(), attempt_ring_closure(), apply_composite_decay() (bond-cut fission),
│   │                     apply_bond_scission(), compute_liquid_drop_half_life(), _apply_binary_splits(),
│   │                     compute_degree(), compute_composite_free_bonds(), _hash_to_bond_energy(),
│   │                     _hash_to_binding_energy(), _hash_to_valence(), _hash_to_rest_length()
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
    ├── species_hash (C,) uint32   — commutative additive hash over member species
    ├── free_bonds   (C,) int32    — remaining bond capacity (Σ v_s − 2*edge_count)
    ├── edges        (C, E_max, 2) int32 — bond particle-id pairs; -1 = unused
    └── edge_count   (C,) int32    — number of valid edges (graph, not just tree)

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
half_life         = f(BE, size)                  # high BE + small size → long HL (placeholder; see liquid drop)

# Per-species (not per-multiset) valence — fixed once at config:
v_s               = _hash_to_valence(s, config)  # in [1, max_valence]; saturation gate
```

Same species set → same hash → same properties every time. Different hash constants give
different universes.

## Hash Fission (bond-cut fracture)

Composite decay is **bond-cut binary fission** (2026-06-12, replacing the old
slot-order `_hash_to_partition` path that minted long bonds). A decaying composite is
fractured along the cut that maximizes total product binding energy — the hash-BE
landscape acting as the "shell structure / magic number" analog, so fission favors
hash-stable fragments and is naturally asymmetric. Mechanism:

1. Build a BFS spanning tree of the composite's bond graph (`halflife/graph.py:bfs_tree`,
   fixed-shape `lax.fori_loop` sweeps capped at `config.fission_label_iters`).
2. Score all `n−1` candidate tree-edge cuts in one `subtree_sums` pass: a cut at slot
   `v` gives fragment hashes `(subtree_hash, total − subtree_hash)` via the additive
   commutative hash (uint32 wraparound, then `% modulus`). Pick the cut maximizing
   `BE(frag0) + BE(frag1)`.
3. Each fragment keeps only the parent edges internal to it — **no edge is ever minted**,
   so fission can't create long bonds regardless of member slot order (the bug fix).

Product 0 reuses the parent slot; product 1 claims a fresh free slot. Products of size 1
become free particles; structurally unsound products (`free_bonds < 0` under valence)
shatter. Energy: the kick is the **Q-value** `max(BE(p0) + BE(p1) − BE(parent), 0)`,
split equally as a momentum-conserving COM-axis kick (replaces the removed `fission_cost`).
With `config.forbid_endothermic_fission` (default True), a decay roll whose best cut has
`Q < 0` is **suppressed** — hash-favored composites become a fission barrier and only break
kinetically/thermally via bond scission. Species are conserved — fission never transmutes.

The shared back half (member/edge compaction, fragment hashes, COM kick, slot writes,
event emission) is factored into `_apply_binary_splits`, reused by chemical bond scission.

## Chemical Bond Scission (per-bond breaking)

The **second** breaking channel (orthogonal to half-life fission). Half-life fission is
"nuclear" (whole-body, stochastic); scission is "chemical" (one bond at a time, driven
by local strain). It makes the harmonic edge spring a **finite** well: every edge carries
a hash-derived dissociation energy `E_b = _hash_to_bond_energy(s_i, s_j)` (per species
pair, in `[0, bond_energy_scale]`, decorrelated from BE/valence/rest-length via a fourth
Fibonacci stream). `apply_bond_scission` (step Phase 6c, between ring closure and decay)
runs every step in `bond_mode="edges"`:

- **Kinetic break:** stretch strain `0.5·k_bond·max(r − r_rest, 0)² ≥ E_b` → the bond
  snaps deterministically. Only *stretch* counts; compression never breaks a bond.
- **Thermal break:** below threshold, Arrhenius probability
  `P = 1 − exp(−dt · ν0 · exp(−(E_b − strain)/kT))` with `ν0 = bond_break_attempt_rate`,
  `kT = bond_temperature` (kT → 0 disables thermal cleanly).

**At most one bond per composite breaks per step** (the most-overstretched breaking edge),
and at most `max_scissions_per_step` composites break per step (excess defers a step). A
broken **bridge** splits the composite into its two connected halves (`reachable_mask` over
the remaining edges → fragment labels → `_apply_binary_splits` with **zero kick** — the
snapped spring just stops pulling, pairwise forces take over). A broken **ring** edge only
removes the edge (everything stays reachable → product 1 is empty → product 0, the whole
composite minus the edge, is written back). Scission emits `KIND_FISSION` events, so the
analysis pipeline (`transitions.py`) treats them as fission transitions and skips the
empty size-0 product of a ring break.

`bond_energy_scale` is deliberately set ABOVE the natural equilibrium-bond strain band
(measured mean ~0.25, p99 ~2.3 at `k_bond=20`) so the kinetic channel snaps only genuinely
overstretched bonds, not normal equilibrium bonds — a scale near the strain band (the
initial 2.0) caps every composite at a dimer.

## Liquid-Drop Stability (live fissility half-life)

`config.stability_mode` selects how a composite's half-life is set:

- **`"liquid_drop"` (default)** — `compute_liquid_drop_half_life` recomputes every
  composite's half-life **every step** (step Phase 6d, just before the decay roll) from a
  nuclear liquid-drop competition:
  - **Cohesion** `E_coh = Σ bond E_b − surface_energy_coeff · n^(2/3)` (aggregate bond
    dissociation energy minus a surface-tension penalty).
  - **Disruption** `E_rep` = the composite's internal hard-core repulsion PE — the
    "Coulomb" analog — accumulated for free by the force pass (`compute_all_forces` now
    returns `(forces, rep_pe)`; `rep_pe[i]` is particle `i`'s same-composite hard-core PE,
    summed per composite and halved since each pair is counted from both endpoints).
  - **Fissility** `x = E_rep / (2·E_coh)`; the half-life is
    `hl_min + (hl_max − hl_min) · t_coh · clip(1 − x, 0, 1)^fissility_exponent`, with
    `t_coh = clip(E_coh / (cohesion_hl_scale · n), 0, 1)`. Big / crammed / weakly-bonded
    composites get short half-lives and fission fast; this **replaces** the ad-hoc
    `composite_size_decay_scale` size penalty as the principled "big/repulsive things
    fission" law. The hash-BE → half-life value written at fusion/fission time is now just
    an initial placeholder until the first step overwrites it.
- **`"legacy"`** — the original fixed hash-BE → half-life formula (with the
  `composite_size_decay_scale` size penalty), set once at creation and never rewritten.
  `composite_size_decay_scale` is now legacy-mode-only (plus the placeholder values).

The one-step lag (PE comes from pre-integration positions) is negligible at `dt=0.06`.
In legacy mode `rep_pe` is dead-code-eliminated from the compiled step.

## Valence & Free Bonds (edge-based saturation)

Optional gate (`config.use_valence`, default True). Each species `s` gets a fixed
valence `v_s ∈ [1, max_valence]`, hashed from the species index via `_hash_to_valence`
(Fibonacci remix, decorrelated from BE *and* from bond rest-length). Valence is the
number of "hands" a particle of that species can use to hold neighbors —
molecular-valence analog (H=1, O=2, C=4 …).

**Accounting is edge-based, not tree-based.** Composites carry an explicit edge list
(`edges`, `edge_count`) — a real graph that may contain cycles, not just a spanning
tree. Free bonds are derived from actual edge incidence (`degree`), at two levels:

```
degree[i]            = number of edges incident on particle i   (compute_degree)
per-particle  free_bond[i] = v_{species[i]} − degree[i]
per-composite free_bonds[c] = Σ (v_s[m] − degree[m]) = Σ v_s − 2 × edge_count[c]
```

A free particle has `degree = 0`, so its free bonds are just `v_s`. The closed-form
`Σ v_s − 2×(n−1)` only holds while a composite stays a tree (`edge_count = n−1`);
once ring closure adds cycles, `edge_count > n−1` and the edge-count formula is the
authoritative one. The `free_bonds: (C,)` field is a cache of this, refreshed via
`compute_composite_free_bonds` and used as a cheap skip mask.

Three gates:

- **Fusion** (`attempt_fusion`): proximity is the nearest member-member contact
  within `fusion_radius`, not rep-to-rep. The two *contacting particles* `i` and `j`
  (the nearest member-pair) must **each** have `free_bond ≥ 1` (`v_s − degree ≥ 1`) —
  stricter than the old composite-level check. The new edge consumes one bond on each
  endpoint, so a composite with spare bonds *elsewhere* still can't fuse through a
  saturated contact member. The lowest-index "representative" is no longer a fusion
  gate — it survives only as a stable per-entity key for candidate dedup and conflict
  resolution. Conflict resolution is governed by `config.fusion_mode` (2026-06-12):
  `"matching"` (default) accepts a pair iff the two entities are each other's best
  candidate (mutual handshake) and applies the whole node-disjoint batch in one
  vmapped pass; `"scan"` is the legacy sequential greedy scan with
  one-fusion-per-entity-per-step `claimed` bookkeeping. Ring closure follows the
  same mode (mutual-nearest pairs in matching mode). Fission runs its heavy
  per-composite work over a compacted batch of at most `max_fissions_per_step`
  fissioning slots; excess fissions defer one step.
- **Ring closure** (`attempt_ring_closure`, Phase 6b): two members *of the same
  composite* within `fusion_radius`, both with a spare per-particle bond, form one
  extra internal edge (closing a ring), consuming 2 free bonds. Touches only
  `edges`/`edge_count`/`degree`/`free_bonds` — never the member list or
  `composite_id`. Gated by `allow_ring_closure AND bond_mode == "edges" AND
  use_valence` (skipped in `star_spring`/off modes, where the edge array is
  physics-inert and firing it would silently leak free bonds; skipped with valence
  off because the mechanic is defined by free-bond accounting and running it anyway
  let `max_valence` leak into valence-off dynamics), rate-limited by
  `max_ring_closures_per_step`.
- **Fission** (`apply_composite_decay`): a product whose own member multiset gives
  `free_bonds < 0` is structurally unsound (more edges required than the members
  offer) and **shatters** into free particles rather than forming a sub-composite.
  The fission kick still fires; particle conservation holds.

Toggled via Python `if config.use_valence:` since config is `static_argnums`, so XLA
traces only the live branch — zero runtime cost when off. BE-threshold preference is
unaffected; per-multiset specificity rides on the BE check, valence layers physical
saturation on top.

This *replaces* the earlier capacity-cap mechanic (commit `7ccad71`, since reverted):
caps were a static per-multiset upper bound on each species count; valence is a
dynamic free-bond counter driven by graph degree. Caps had the unphysical property
that a 2-particle composite could randomly roll a ceiling of [32, 32, 32]; valence
makes small composites of low-v species *immediately* saturated, as in real molecules.

## Angle Locking (VSEPR / harmonic bond geometry)

Without an angular term the edge springs only set bond *lengths*, so composites are
floppy — bonds pivot freely and structures collapse into long chains. `config.angle_mode`
adds a force between a composite's bonds so geometry holds real molecular angles
(`compute_angle_forces`, step.py, applied just before integration, gated on
`angle_mode != "off" AND bond_mode == "edges"` — the off path is dead-code-eliminated, so
zero cost when off; the live app `main.py` defaults to `vsepr`, headless/tests stay `off`):

- **`"vsepr"`** — bond directions repel via a chord-Coulomb law `U = k_angle/|û_i − û_k|`
  (the Thomson problem on a circle), so the bonds **spread evenly** to an emergent
  `2π/degree` rest angle (degree-2 → 180°, degree-3 → 120° Y, degree-4 → 90° cross). The
  rest angle *emerges* from repulsion rather than being prescribed, so it is **not
  frustrated** at degree ≥ 3 (4 vectors can't be pairwise-90° in 2D, but they *can* settle
  to even spacing). This is the general-purpose "un-floppy the chains" mode.
- **`"harmonic"`** — pulls `cos θ` toward `cos θ0` where θ0 is a hash-derived per-**central-
  species** rest angle (`_hash_to_rest_angle`, Fibonacci stream `0x85EBCA77 >> 15`,
  decorrelated from BE / valence / rest-length / bond-energy; mapped into
  `[theta_min_deg, theta_max_deg]`). A smooth cosine form (no atan2, no cusp at 180°).
  Intended for **degree ≤ 2** (prescribed bent low-valence shapes, water-analog); it
  over-determines at degree ≥ 3 (more angle constraints than 2D DOFs).

Mechanics shared by both: a per-particle neighbor list (`build_neighbor_list`, argsort +
segmented-rank, no scan) and an angle-triple list `(N, C(max_valence,2), 3)`
(`build_angle_list`, center j with neighbor pair i,k) are recomputed from the edge graph
each step — **no changes to fusion / fission / ring-closure**. Each triple force is purely
**tangential** (rotates bonds, never stretches them) and conserves linear & angular
momentum (`F_j = −(F_i + F_k)`), with min-image displacement matching
`compute_edge_bond_forces`. Stiffness is the runtime-tunable `PhysicsParams.k_angle`
(seeded from `config.k_angle`, exposed as the "angle k" slider in edges mode). The angle
potential is **not** part of `energy.py` conservation tracking (v1 scope).

## Visualization

- **Particles**: point sprites colored by species (HSV), sized by log(mass), brightness by speed
- **Composite modes** (toggle with `B` key or button):
  - **Bonds mode**: GL_LINES between composite members with periodic-boundary wrapping
  - **Merged mode**: single large point at center of mass
- **HUD overlay**: pygame RGBA surface uploaded each frame as an OpenGL texture on a
  fullscreen quad. Buttons on the left edge; key hints at the bottom.
- **Event sprites** (rebuilt 2026-06-12): expanding ring point sprites sourced from the
  kernels' real `ReactionEvent` batches (the live app runs `emit_events=True`;
  `make_run_n_steps_with_events` returns the stacked per-step events alongside the
  state). Gold = fusion (min-image midpoint of the two contact particles), cyan =
  fission (member COM of the parent slot), red = full dissolution (both fission
  products size 1). Age tracked in sim-time, birth staggered by step index within
  the frame batch. Sprite pool is a 200-row ring buffer (oldest overwritten —
  admission never stalls). Replaced the old comp_alive frame-diff detector, which
  missed parent-slot-reusing fissions and burst-oscillated when the pool saturated.
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
    num_species=3,             # current default; more species → richer chemistry
    num_particles=5_000,       # total particle pool (fixed; all always alive)
    interaction_radius=8.0,    # force cutoff
    fusion_radius=1.5,         # nearest member-member contact gap; must be < interaction_radius
    fusion_threshold=0.6,      # min binding energy to fuse [0,1]
    half_life_min=1.0,
    half_life_max=100.0,       # current experiment runs long-lived composites
    hash_modulus=100_000_007,  # changes the "universe" / chemistry

    # Composite stability (liquid-drop fissility law; "legacy" = fixed hash-BE hl)
    stability_mode="liquid_drop",
    surface_energy_coeff=0.5,      # a_s — cohesion penalty × n^(2/3)
    cohesion_hl_scale=1.0,         # per-member cohesion for max stability
    fissility_exponent=1.0,        # sharpness of the collapse as x → 1
    composite_size_decay_scale=0.05,  # legacy-mode hl size penalty + placeholder values

    # Bond-cut fission (2026-06-12)
    fission_label_iters=64,            # BFS / subtree-sum sweep cap (>= graph diameter)
    forbid_endothermic_fission=True,   # suppress decay rolls whose best cut has Q < 0

    # Valence saturation (per-species free-bond gate, on by default)
    use_valence=True,
    max_valence=4,             # per-species valence drawn from [1, max_valence]

    # Covalent bonds (edges mode) — sparse hash-derived springs between members
    bond_mode="edges",         # "edges" | "star_spring" | "off"
    k_bond=20.0,               # harmonic edge-spring stiffness
    # bond rest length is hash-derived per species pair, spanning
    # [repulsion_radius, fusion_radius] — no longer absolute config fields
    allow_ring_closure=True,   # let same-composite members form extra (cyclic) edges
    max_ring_closures_per_step=50,

    # Chemical bond scission (per-bond kinetic + thermal breaking; edges mode)
    enable_bond_scission=True,
    bond_energy_scale=10.0,        # E_b ceiling; sits above the natural strain band
    bond_temperature=1.0,          # kT for the Arrhenius thermal channel (0 = off)
    bond_break_attempt_rate=0.1,   # ν0 attempt frequency
    max_scissions_per_step=32,

    # Angle-locking (bond geometry; edges mode). Default "off" headless/tests;
    # the live app (main.py) sets "vsepr". See "Angle Locking" below.
    angle_mode="off",          # "off" | "vsepr" | "harmonic"
    k_angle=10.0,              # angle stiffness; seeds runtime PhysicsParams.k_angle
    theta_min_deg=90.0,        # harmonic θ0 band floor
    theta_max_deg=180.0,       # harmonic θ0 band ceiling
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
| `baseline` | legacy baseline, pinned to `num_species=12, half_life_max=15.0` (not the current config.py default, which is `num_species=3`) |
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
| `--windows N` | `5` | Tier 5 time-window count (mutually exclusive with `--window-width`) |
| `--window-width W` | none | Tier 5 fixed window width in steps (mutually exclusive with `--windows`) |
| `--override "k1=v1,k2=v2"` | none | Per-run config overrides (e.g. `"num_species=5,fusion_radius=3.0"`) |
| `--out PATH` | auto | Output path (default: `tests/reports/diag_<scenario>_<ts>.html`) |
| `--platform cpu\|gpu` | auto | Force JAX platform. Use `cpu` for tiny test runs, `gpu` for real diagnosis |
| `--from-cache` | off | Skip simulation; load the cached `RunResult` matching the other args and re-render the HTML |
| `--cache-path PATH` | auto | Override cache path (default derived from scenario+steps+seed+sample-every+overrides) |
| `--no-cache` | off | Don't write to cache after running (default: write, overwriting prior cache) |

### Cache

Every non-`--from-cache` run writes its `RunResult` (gzipped pickle) to
`tests/reports/cache/<scenario>_n<steps>_seed<seed>_every<K>[_ovr<hash>].pkl.gz`.
Re-rendering the report from cache is essentially instant — useful when iterating
on presentation code (`plots.py`, `report.py`, CSS) without burning GPU time on
identical sim runs. Cache filename is derived from the same args, so a follow-up
invocation with `--from-cache` and the same flags finds the right slot.

```bash
# First run: simulate + cache + render
.venv/bin/python -m halflife.analysis --scenario X --steps 3000 --platform gpu

# Tweak plot styling, then re-render from cache (no GPU, no sim):
.venv/bin/python -m halflife.analysis --scenario X --steps 3000 --from-cache
```

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

The **Tier 5** open-endedness section quantifies novelty accumulation over the run on
two type axes — composition (`species_hash`) and structure (Weisfeiler-Lehman bond-graph
hash). It shows the cumulative type-discovery curve, per-window novelty rate, Hill-number
diversity, window-to-window turnover (Jaccard / Bray-Curtis), per-window size facets, and
— folded into the structure axis (2026-06-12) — a per-window **bonded-particle degree
distribution** (5f) and a **topology split** (5g: chain / tree-branch / cyclic, shown by
composite *count* vs particle *mass* — the count-vs-mass contrast surfaces "mostly dimers
by count but mostly cyclic mesh by mass"). All of it is host-side post-processing on the
cached `RunResult`, so
`--from-cache --windows N` re-renders a different windowing instantly. Resolved at
`sample_every` cadence; structure metrics are only meaningful in `bond_mode="edges"` runs.

### Cost

- ~30 steps/sec on GPU at default `num_particles=5000`
- Memory: ~5 MB compact metrics + ~100 MB events + ~65 MB snapshots for 10k steps (well within RAM)
- Sub-second post-processing (transition matrices + compatibility matrices)
- Output HTML is typically 500KB-2MB depending on `--top-k` and event count

### Live-app cost

The diagnostic kernel emission is gated by `SimConfig.emit_events` (default `False`,
`static_argnums`); when False the emit branch is dead-code-eliminated before JIT — the
compiled kernel is bit-for-bit unchanged from the pre-pipeline version. **Since
2026-06-12 the live app (main.py `build_config`) deliberately sets `emit_events=True`**
to drive the renderer's event sprites from real events — measured cost +0.056 ms/step
(+3.0%) at 5k particles including the host transfer. Headless/test configs keep the
zero-cost False default.

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
- **Fast test runs (2026-06-12)**: test wall-time is dominated by per-process XLA compiles and
  Python-dispatched step loops at tiny N — the GPU sits mostly idle either way. Two fixes, both
  measured on the chemistry suite:
  - `tests/conftest.py` enables JAX's persistent compilation cache (`~/.cache/halflife-jax`,
    WSL-native FS) so reruns skip recompiles. Also wired into `main.py` and the analysis CLI —
    app restarts and repeat diagnostics skip the 10-30s warm-up.
  - Parallelize with workers: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -n 4` on GPU →
    278s → 129s (2.15×). The prealloc env var is REQUIRED with `-n`: by default each JAX
    process grabs 75% of VRAM and the workers OOM each other. Do NOT use CPU + `-n 8`:
    each worker's XLA spawns threads on all 16 cores and oversubscription makes it
    SLOWER (349s) than 4 GPU workers.
- **Diagnostic scripts**: live in `/tmp/` as throwaways. They need explicit
  `sys.path.insert(0, "/mnt/.../halflife-particle")` (not `dirname(__file__)`) since they're
  outside the project root.
