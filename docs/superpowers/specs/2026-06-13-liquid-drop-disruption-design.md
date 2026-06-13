# Liquid-Drop Disruption Term — Design Spec

**Date:** 2026-06-13
**Status:** Approved design (pending user review of this doc)
**Touches:** `halflife/chemistry.py` (`compute_liquid_drop_half_life`),
`halflife/state.py` (`PhysicsParams`), `halflife/config.py` (`SimConfig`),
`halflife/renderer.py` (sliders), `CLAUDE.md` / `README.md`.

---

## 1. Problem

Two observed symptoms, one root cause:

1. **Half-lives pin to exactly `half_life_max` (200).** A large majority of live
   composites read the maximum half-life regardless of size or composition.
2. **Composites grow to exactly `max_composite_size` (256) trivially.** Giant
   composites form with no apparent selection pressure and park at the buffer
   ceiling.

### Root cause

`compute_liquid_drop_half_life` (chemistry.py) sets the live half-life from a
nuclear liquid-drop competition:

```
e_coh = Σ E_b(edges) − surface_energy_coeff · n^(2/3)
x     = e_rep / (2·e_coh)                                   # fissility / "Coulomb"
t_coh = clip(e_coh / (cohesion_hl_scale · n), 0, 1)
stab  = t_coh · clip(1 − x, 0, 1)^fissility_exponent
hl    = half_life_min + (half_life_max − half_life_min)·stab
```

Both modulators are saturated:

- **`t_coh → 1` always.** `e_coh ≈ (#edges)·⟨E_b⟩ ≈ n·⟨E_b⟩` and `⟨E_b⟩ ≈ 5`
  (bond energies uniform in `[0, bond_energy_scale=10]`), so
  `e_coh/(cohesion_hl_scale·n) ≈ ⟨E_b⟩/1.0 ≈ 5`, clipped to 1. Even a dimer
  saturates. Cohesion grows *linearly* with `n`, so the `/n` normalization
  cancels all size dependence.
- **`x → 0` always.** `e_rep` is the **same-composite hard-core repulsion PE**,
  which is non-zero only when two members sit closer than `repulsion_radius`
  (`interactions.py:174`). But hash-derived bond rest lengths span
  `[repulsion_radius, fusion_radius]`, so a relaxed bond sits *at or beyond* the
  hard core → `e_rep ≈ 0` → `x ≈ 0`.

Result: `stab = 1·(1−0)^1 = 1`, so `hl = half_life_max` for essentially every
well-relaxed composite. And because **no term in the model grows super-linearly
with `n`**, there is no critical size — composites accrete at their (always
free-bonded) surface until they hit the `max_composite_size` buffer.

The liquid-drop *structure* is correct (cohesion vs disruption), but the
disruption term is mis-specified: real liquid-drop / nuclear fission is driven by
a **long-range** Coulomb energy that grows faster than cohesion
(`∝ Z²/A^(1/3) ~ n^(5/3)`, summed over *every* pair). Our `e_rep` is short-range
and zero at equilibrium, so fissility never climbs with size.

---

## 2. Goal

Add the missing long-range disruption so that:

- `x` grows with composite size and **spreads off zero**, un-pinning half-lives.
- A tunable **critical size `n_crit`** emerges, beyond which composites fission
  — pulling the size distribution away from the 256 buffer on its own.
- Bond **quality** also modulates half-life (de-saturated cohesion).
- `disruption_scale = 0` reproduces today's behavior exactly (escape hatch).

No new fission code: the existing bond-cut fission, Q-value energetics, and
`forbid_endothermic_fission` machinery fire automatically once half-lives drop
and large splits become exothermic.

---

## 3. Design

### 3.1 Disruption energy — R_g monopole

Model every member as a unit "charge" (`q = 1`). The Coulomb self-energy of a
blob of total charge `Q = n` and characteristic size `R_g` scales as `Q²/R_g`
(the leading monopole term of the pairwise `Σ q_iq_j/r_ij` sum). Define:

```
E_coulomb(c) = disruption_scale · n² / (R_g + ε)          # ε = 1e-6
```

where `R_g` is the composite's radius of gyration (RMS distance of members from
their centroid). This is **O(n) per composite** (one pass over members), unlike
the O(n²) full pairwise sum, and is **shape-aware**: a compact blob has small
`R_g` → high disruption; an extended chain has large `R_g` → low disruption (so
stringy structures resist fission, compact ones don't — physically desirable).

**Total disruption** keeps the existing hard-core term as a secondary
"over-crammed" contribution (it is already computed for free by the force pass):

```
E_dis(c) = E_coulomb(c) + e_rep(c)
x        = E_dis / (2·e_coh)
```

`e_rep` catches the genuinely-overlapping case (members forced inside
`repulsion_radius`); `E_coulomb` is the dominant, size-growing term. When
`disruption_scale = 0`, `E_dis` collapses to the old `e_rep` and behavior is
unchanged.

### 3.2 Radius of gyration with periodic boundaries

A composite straddling the world-wrap edge has a naïve centroid in the middle of
the box, producing a garbage `R_g`. Compute `R_g` from member displacements
**unwrapped relative to a reference member**:

```
ref      = position of members[c, 0]                       # first valid member
d_i      = min_image(pos_i − ref)        for each valid member i   # (≤ MAX_COMPOSITE_SIZE, 2)
offset   = mean(d_i)                      over valid members        # centroid offset from ref
R_g      = sqrt( mean( |d_i − offset|² ) over valid members )
```

`min_image` uses the same periodic wrap as `compute_edge_bond_forces` /
`compute_angle_forces`. This is exact provided a composite spans less than half
the world in each axis — always true here (composites are ≪ the 200×112.5
world). For `boundary_mode != "periodic"`, `min_image` is the identity and the
formula is the ordinary RMS radius.

Invalid/padded member slots (`members[c,k] = −1`) are masked out of both means
(weight 0), exactly as `member_count` dictates. Single-member composites
(`n = 1`, transient) get `R_g = 0`; guarded by `R_g + ε` and by the fact that
size-1 "composites" are not alive in steady state.

### 3.3 De-saturated cohesion

Raise `cohesion_hl_scale` from `1.0` toward `⟨E_b⟩ ≈ 5.0` so
`t_coh = clip(e_coh/(cohesion_hl_scale·n), 0, 1)` becomes a genuine gradient in
`[0,1]` driven by **mean bond energy** rather than clipping to 1. Weakly-bonded
composites then get shorter half-lives independent of size; the disruption term
(§3.1) supplies the size axis. Together they restore the full
cohesion-vs-disruption competition.

### 3.4 Critical-size scaling (for tuning intuition)

With `e_coh ≈ ⟨E_b⟩·n` and a compact 2-D blob packed at member spacing `s`
(`R_g ≈ s·√(n/2π)`):

```
x ≈ disruption_scale · n / (2·⟨E_b⟩·R_g) ∝ √n
n_crit (where x → 1) ≈ 2·(⟨E_b⟩·s)² / (π · disruption_scale²) ≈ 16 / disruption_scale²
```

(using `⟨E_b⟩≈5`, `s≈1`). So `disruption_scale ≈ 0.4–0.56` lands `n_crit ≈ 50–100`.
These are order-of-magnitude; the **final default is tuned empirically** with the
diagnostic (§6), and the live slider (§3.5) is the real tuner.

### 3.5 Runtime tunability

Both new scalars are promoted to `PhysicsParams` (runtime-tunable, no recompile)
and exposed as sliders, matching `k_bond` / `k_angle`:

- `disruption_scale` — Coulomb constant `k`; sets `n_crit` live.
- `cohesion_hl_scale` — moved from a `config`-read to a `physics`-read inside
  `compute_liquid_drop_half_life` so it can be dialed live.

`compute_liquid_drop_half_life` already receives `physics`; it will read
`physics.disruption_scale` and `physics.cohesion_hl_scale` instead of (or seeded
from) the `config` values.

---

## 4. Data flow & integration points

- **`config.py`** — add `disruption_scale: float = 0.5` (final default tuned in
  §6). `cohesion_hl_scale` default `1.0 → 5.0`.
- **`state.py`** — add `disruption_scale` and `cohesion_hl_scale` to
  `PhysicsParams`, seeded in `initialize_physics_params` from config.
- **`chemistry.py:compute_liquid_drop_half_life`** — compute `R_g` (min-image)
  and `E_coulomb`; read `disruption_scale` / `cohesion_hl_scale` from `physics`;
  `x = (E_coulomb + e_rep)/(2·e_coh)`. The function already gathers per-composite
  member data for `e_coh`; member positions for `R_g` are a parallel gather over
  `composites.members` / `member_count`.
- **`step.py`** — no signature change: `compute_liquid_drop_half_life` is already
  called with `(particles, composites, rep_pe, config, physics)` in Phase 6d.
- **`renderer.py`** — add `disruption_scale` ("disrupt k") and `cohesion_hl_scale`
  ("cohesion") sliders. They are meaningful only in `stability_mode="liquid_drop"`;
  add to the general physics-slider block (always shown), like the other
  liquid-drop knobs are configured at construction. The fit-to-window logic
  (2026-06-13) absorbs the two extra rows.
- **`legacy` mode** unaffected: `compute_liquid_drop_half_life` is only called
  when `stability_mode == "liquid_drop"`. `cohesion_hl_scale`'s new default does
  not touch `legacy` (which uses `composite_size_decay_scale`).

---

## 5. Testing strategy

New `tests/test_liquid_drop_disruption.py` (CPU, `JAX_PLATFORMS=cpu`), plus
regression on `test_chemistry.py`:

1. **R_g correctness (open boundary).** Hand-build composites with known
   member geometry (collinear chain vs compact cluster); assert `R_g` matches the
   analytic RMS radius. Chain `R_g` > cluster `R_g` for equal `n`.
2. **R_g periodic wrap.** Place a composite straddling the world edge; assert
   `R_g` equals the unwrapped value (≈ the same composite away from the edge),
   not the inflated naïve-centroid value.
3. **Disruption grows with size.** Compact blobs of increasing `n` (fixed
   density) → `x` increases monotonically; assert `x(n=large) > x(n=small)`.
4. **Shape sensitivity.** Equal-`n` chain vs blob → blob has higher `x` (lower
   half-life) than the chain.
5. **Half-life un-pins.** A size-spread population no longer returns a constant
   `half_life_max`; assert a spread of half-lives across sizes.
6. **Backward compat.** `disruption_scale = 0` → half-lives identical to the
   pre-feature formula (bit-for-bit on a fixed state).
7. **Cohesion gradient.** With `cohesion_hl_scale = 5`, two composites of equal
   size but different mean bond energy get different half-lives.
8. **Smoke / no-NaN.** `simulation_step` over a populated state in liquid-drop
   mode with non-zero `disruption_scale` produces finite half-lives, positions,
   velocities.

**Empirical validation** (not a unit test): a diagnostic run (§6) confirms the
size histogram pulls off 256 and the half-life distribution spreads.

---

## 6. Tuning & validation procedure

1. Implement with `disruption_scale = 0.5` (placeholder).
2. Run the diagnostic on `current_experiment` for ~10–20k steps:
   `.venv/bin/python -m halflife.analysis --scenario current_experiment --steps 15000 --sample-every 250 --platform gpu`
3. Inspect the size histogram (Tier 1/5) and a dumped half-life distribution.
   Target: max size sits **well below 256**, half-lives span a range.
4. Adjust `disruption_scale` (raise to shrink `n_crit`, lower to grow it) and the
   live slider until the size distribution looks right; set the final config
   default accordingly. Update the `current_experiment` preset note in
   `cli.py` / `CLAUDE.md` if the default changes.

---

## 7. Out of scope (v1)

- **Force-coupled disruption** — a real long-range repulsion *force* between
  members (blobs physically strain and break). Deferred follow-up; v1 is
  energy-only (half-life), per the chosen design.
- **Full O(n²) pairwise Coulomb** — the monopole `n²/R_g` is the chosen
  approximation; the exact pairwise sum is a possible later upgrade for unusual
  geometries.
- **Species-dependent charge** — v1 uses uniform `q = 1` per member. A
  hash-derived per-species charge (richer chemistry) is a later extension.
- **Energy-conservation accounting** — the disruption energy is a half-life
  modulator, not tracked by `energy.py` (consistent with the existing
  liquid-drop term).

---

## 8. Risks & mitigations

- **`R_g` mis-tuned default** → `n_crit` too small (everything fissions to
  dust) or too large (no effect). Mitigated by the live slider and the §6 tuning
  loop; `disruption_scale = 0` is always a safe no-op.
- **Periodic-COM edge case** for a composite spanning > half the world. Cannot
  occur given composite size ≪ world; documented as an explicit assumption.
- **Cost.** One extra O(n) member gather per composite per step. Same order as
  the existing `e_coh` edge gather; negligible vs the force pass. No O(n²) work.
- **Cohesion default change** alters the *absolute* half-life scale for existing
  liquid-drop runs. Intended; documented in CLAUDE.md. Legacy mode untouched.
```
