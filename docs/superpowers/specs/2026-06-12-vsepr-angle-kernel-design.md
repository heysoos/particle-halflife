# VSEPR Angle-Locking Kernel — Design Spec

**Date:** 2026-06-12
**Status:** Design (approved direction; awaiting spec review before plan execution)
**Author:** brainstormed with Claude

---

## 1. Goal

Add an angular force between the bonds of a composite so bonded geometry adopts
real **molecular angles** instead of being floppy. Today bonds are pure radial
harmonic springs with no angular term, so composites collapse into long straight
chains (next-nearest-neighbour repulsion straightens them) and the rare branches
have no restoring force to hold their shape. This kernel makes a degree-2 atom
prefer a defined angle, a degree-3 atom a Y, a degree-4 atom a cross — and makes
branches and rings stable once formed.

This is **one of two** complementary fixes for the chains problem. The other is the
fission bond-breaking redesign (see `notes/2026-06-01-fission-bond-breaking-redesign.md`):
that fixes the *kinetic / topological* bias (tip-accretion + slot-order path rebuild),
this fixes the *geometric* bias (floppy, straightened shapes). They are independent
and can land in either order.

## 2. Background — why chains, and what an angle term changes

From the diagnostic discussion (2026-06-12): chain dominance has three causes —
(1) fusion accretes at sterically-exposed tips; (2) the current fission rebuilds
products' bonds as a slot-order *path*, re-linearizing structure; (3) **no angular
term**, so what does form is floppy and gets straightened by the force kernel's
next-nearest-neighbour repulsion. This spec addresses (3). It does **not** fix
(1) or (2) — interior members stay sterically shielded — but it makes branches and
rings *stable* once they occur, and gives the structures real geometry.

A secondary motivation: bent, heterogeneous-species geometry is the prerequisite
for any *emergent* orientation-dependent intermolecular interaction (a crude
"polarity" arising from the existing asymmetric attraction matrix). True
electrostatic polarity (partial charges / dipoles) is explicitly **out of scope** —
it is a separate subsystem and would change the character of the sim.

## 3. Two force laws, one kernel

The kernel skeleton — build an angle list, vmap a per-triple force law, scatter
into an `(N, 2)` force buffer with min-image displacement — is **shared**. The
only thing that varies is the per-triple force law, selected by a new static
config field `angle_mode`:

- **`"vsepr"`** (the principled default): bond directions repel each other and
  spread out. The rest angle is *emergent* — `2π / degree` even spacing — with no
  target angle to prescribe and **no frustration** at degree ≥ 3. This is real
  VSEPR ("electron domains repel and spread"). Fixes the floppy-chain /
  branch-stability problem outright.

- **`"harmonic"`**: each bond pair is pulled toward a hash-derived **target angle**
  `θ₀(s)` keyed on the central species. This is the robust 2-D route to *prescribed*
  bent low-valence shapes (water-analog): set oxygen's θ₀ ≈ 120° and its two bonds
  bend. Uses the smooth **cosine** form, not the harmonic-in-θ form (which has a
  non-differentiable cusp at 180°, exactly the most common configuration here).

- **`"off"`** (default in headless/test configs): kernel is not called; behaviour
  is bit-identical to today. Selected by Python `if`, so XLA dead-code-eliminates
  the kernel entirely — zero cost when off, same pattern as ring closure.

### 3.1 Why VSEPR is not frustrated and harmonic-θ₀ is

A target-seeking well (`½k(θ−θ₀)²`) can demand an angle that not all bond pairs can
satisfy simultaneously: a degree-4 atom has C(4,2)=6 pairs all commanded to 90°, but
2-D has only two orthogonal axes, so a cross necessarily leaves two opposite pairs at
180° — perpetually strained, prone to buckling. VSEPR prescribes *no* angle; "spread
as far apart as possible" always has a clean answer (the Thomson problem on a circle:
even spacing 2π/n, unique up to rotation for every n). So a degree-4 VSEPR atom
relaxes to a clean cross with zero residual strain. `harmonic` mode is retained
*only* for the degree-≤2 prescribed-bend case, where there is a single angle to set
and frustration cannot arise.

## 4. Data flow

```
edges (C, E_max, 2)  ──build_neighbor_list──►  nbrs (N, max_valence)   [-1 padded]
nbrs                 ──build_angle_list────►  angles (N, P_max, 3)     [-1 padded]
                                               P_max = C(max_valence, 2)
angles + positions   ──compute_angle_forces─►  forces (N, 2)
                                               (per-triple law switched on angle_mode)
simulation_step: total_force += compute_angle_forces(...)   [when gated on]
```

Gate (Python-level, on the static config):

```python
if config.angle_mode != "off" and config.bond_mode == "edges":
    total_force = total_force + compute_angle_forces(state, config, physics)
```

`bond_mode == "edges"` is required because the angle list is derived from the edge
graph; in `star_spring`/`off` bond modes the edge array is physics-inert and angle
forces would be meaningless. (Valence accounting is *not* required — angles work on
the edge graph regardless of whether `use_valence` is on.)

## 5. Components

### 5.1 `build_neighbor_list(composites, config) → (N, max_valence) int32`

Per-particle adjacency, padded with -1. Built from the edge array, fully
vectorized (argsort + segmented rank, no scan). Each undirected edge contributes
its partner to *both* endpoints' neighbor rows. Column width is `max_valence`
because degree ≤ valence ≤ `max_valence` always (ring closure consumes free bonds,
so it never pushes degree past valence); any overflow is silently dropped via
`mode='drop'`.

### 5.2 `build_angle_list(nbrs, config) → (N, P_max, 3) int32`

For each particle `j`, enumerate the `C(max_valence, 2)` fixed column-pairs `(p, q)`
of its neighbor row and emit the triple `(nbrs[j,p], j, nbrs[j,q])`. Triples where
either neighbor slot is -1 are padded with -1 (invalid). `P_max` and the column-pair
index list are generated in Python from the static `max_valence`, so the enumeration
is unrolled at trace time. A free particle (no edges → empty neighbor row) yields all
-1 triples → zero force.

### 5.3 `compute_angle_forces(state, config, physics) → (N, 2) float32`

vmap over particles (each is a candidate center `j`); for each of its ≤ P_max
triples apply the selected force law and scatter the three forces into the buffer.
Min-image displacement matches `compute_edge_bond_forces`. Both laws are **purely
tangential** (perpendicular to each bond), so they rotate bonds without stretching
them — clean separation from the bond-length spring. Both conserve linear and
angular momentum per triple (`F_j = −(F_i + F_k)`, the exact gradient of a
rotation-invariant potential).

For a triple `(i, j, k)` with center `j`, let `r_ji`, `r_jk` be min-image
displacements, `L` their norms, `û = r/L` the unit bond directions, and
`c = û_i · û_k = cos θ`:

**VSEPR (chord-Coulomb repulsion):**
```
U = k_angle / d,      d = |û_i − û_k| = 2·sin(θ/2)
w = û_i − û_k
F_i = (k_angle / L_ji) · (w − û_i(û_i·w)) / (d² + ε)^{3/2}      (tangential, opens θ)
F_k = (k_angle / L_jk) · (−w − û_k(û_k·−w)) / (d² + ε)^{3/2}
F_j = −(F_i + F_k)
```
Diverges as `d → 0` (bonds aligning) — that divergence is what forbids overlap and
forces a *unique* even spread; softened by ε for numerical safety. Zero force at the
even-spread equilibrium and at 180° (a degree-2 atom rests straight).

**Harmonic-cosine (target angle):**
```
U = ½ · k_angle · (c − cos θ₀(s_j))²
F_i = −k_angle·(c − cosθ₀) · (û_k − c·û_i) / L_ji                (tangential, drives c→cosθ₀)
F_k = −k_angle·(c − cosθ₀) · (û_i − c·û_k) / L_jk
F_j = −(F_i + F_k)
```
`cos θ₀(s_j)` precomputed per central species (§5.4). Smooth everywhere including
θ = 180° (c = −1); no `atan2`, no sign-picking.

Numerical guards: `L += 1e-10` (short-bond safety, matching the bond kernel); ε on
the VSEPR core; optional force-magnitude clamp noted as a stability follow-up.

### 5.4 `_hash_to_rest_angle(species, config)` + `_species_rest_angles(config)`

Hash-derived per-species target angle for `harmonic` mode, cloning the exact recipe
of `_hash_to_rest_length` / `_hash_to_valence`: `_entity_hash_val` → Fibonacci
re-mix with its **own** constant (`0x85EBCA77`, shift 15 — distinct from the BE,
valence, and rest-length streams so θ₀ is decorrelated) → low digits → fraction →
mapped into `[theta_min, theta_max]` radians. Per *central species* only (not per
pair), so geometry is a property of the central atom — the water-correct choice. The
`(S,)` vector is precomputed once like `_species_valences`.

## 6. New configuration

`SimConfig` (frozen dataclass — new fields, all with defaults, so existing
construction is unaffected):

| field | type | default | notes |
|---|---|---|---|
| `angle_mode` | str | `"off"` | `"off" \| "vsepr" \| "harmonic"`. Static (`static_argnums`). |
| `k_angle` | float | `10.0` | seeds the runtime PhysicsParams value (like `k_bond`). |
| `theta_min_deg` | float | `90.0` | `harmonic` θ₀ band floor. |
| `theta_max_deg` | float | `180.0` | `harmonic` θ₀ band ceiling. |

`PhysicsParams` gains a runtime-tunable scalar **`k_angle`** (seeded from
`config.k_angle` in `initialize_physics_params`), so stiffness is adjustable via a
slider with no recompile, exactly like `k_bond` / `r_rest_scale`.

**No new buffer-size field.** `max_valence` (already static) determines both the
neighbor-row width and `P_max = max_valence·(max_valence−1)//2`.

**Live app:** `main.py build_config` sets `angle_mode="vsepr"` (so the user sees it),
mirroring the `emit_events=True` live-default pattern; headless/test configs keep the
`"off"` default. A `k_angle` slider is added to the renderer next to `k_bond`.

## 7. Testing strategy

Pure-CPU pytest (`JAX_PLATFORMS=cpu`), new file `tests/test_angle_kernel.py`:

- **Hash θ₀:** deterministic per species; in `[theta_min, theta_max]`; stream
  decorrelated from valence (different constant → different ordering).
- **Neighbor list:** hand-built chain / branch / ring composites produce the
  correct neighbor sets; free particles empty; degree never exceeds `max_valence`.
- **Angle list:** degree-2 center → 1 valid triple; degree-3 → 3 triples; degree-4
  → 6 triples; free particle → all padding; correct `(i, j, k)` identities.
- **VSEPR force (analytic):** two bonds at 90° → force opens toward 180°, equal and
  opposite tangential, `ΣF ≈ 0` (momentum); two bonds at 180° → ≈ 0 force
  (equilibrium); never NaN at 180°.
- **VSEPR relaxation (integration):** 4 bonds from a random start relax to a cross
  (pairwise angles → {90°×4, 180°×2}); 3 bonds → Y (120°).
- **Harmonic force (analytic):** force sign drives `cos θ → cos θ₀`; zero at θ₀;
  smooth (no NaN) at θ = 180°.
- **Harmonic relaxation:** a degree-2 atom relaxes to its species' θ₀.
- **Gating / off:** `angle_mode="off"` → `simulation_step` forces bit-identical to a
  pre-feature run; `"vsepr"` changes them; `bond_mode != "edges"` skips the kernel.
- **Step smoke:** a few hundred steps in each mode run without NaN / shape error.

A pre-build **baseline measurement** (recommended, not blocking): dump the current
degree distribution (fraction of particles at degree 1 / 2 / 3+) from a cached run,
to quantify chain dominance before/after.

## 8. Out of scope (explicit)

- **True electrostatic polarity** (partial charges, dipoles, orientation-dependent
  Coulomb). Separate subsystem; changes the sim's character.
- **Runtime lone-pair domains** as real repelling phantom directions. The 2-D
  geometry fights them (the cross has collinear opposite slots → tuned lone-pair
  repulsion tends to give *linear*, not bent; robust bent needs odd domain counts)
  and they need new per-atom orientation state. The robust 2-D bent-geometry route
  is `harmonic` θ₀, which folds the "domain count" story into a prescribed angle with
  no extra state. If pursued later, it is its own spec.
- **Energy-conservation accounting** for the angle potential in `energy.py`. v1
  treats it as a geometric force outside the soft energy budget (the bond springs'
  treatment is the reference). Revisit if drift becomes material.
- **Dihedral / torsion terms**, 3-D geometry, fission/fusion changes.

## 9. Risks & open questions

- **VSEPR core divergence** when two bonds nearly align (`d → 0`): physically the
  point (push apart) but numerically large; mitigated by ε softening and an optional
  magnitude clamp. Watch for transient force spikes right after a fusion places two
  bonds close.
- **`harmonic` per-central-species θ₀ still over-determines degree-4** (six pairs all
  read the same θ₀). `harmonic` is intended for degree ≤ 2; for degree ≥ 3 use
  `vsepr`. Mixed-degree composites in `harmonic` mode are an accepted rough edge.
- **`angle_mode="vsepr"` in the live app perturbs current experiment dynamics.** This
  is intended (it's the feature) but means cached baselines won't match; the toggle
  makes A/B easy.
- **Stiffness coupling:** `k_angle` vs `k_bond` ratio sets how much angles fight
  positions; sensible defaults TBD by live tuning (hence the slider).

## 10. Build order

config/physics → hash θ₀ → neighbor list → angle list → VSEPR force → harmonic
force → wire into step (gated) → live app default + slider → docs. TDD, one
component per task; see the implementation plan.
