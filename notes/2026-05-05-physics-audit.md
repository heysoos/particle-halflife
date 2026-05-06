# Half-Life — physics audit

**Date:** 2026-05-05
**Topic:** Detailed read-through of every physics module to nail down what the simulator *actually* does, surface dead code, and call out subtle behaviours.

## Summary

The simulation step is a **9-phase JIT'd graph** ([halflife/step.py:151](../halflife/step.py#L151)). Particles never die (the alive-mask machinery was removed in `b0c049f`); only composites do. Several earlier-design functions are left as dead code: `hash_multiset`, `hash_scalar`, `_hash_to_half_life`, `_hash_to_decay_products`, plus `composite_half_life_scale` and `max_decay_products` config knobs that nothing reads. The actual half-life formula is **driven by binding energy**, not the hash directly. Fusion only operates on each entity's "representative" (its lowest-index member), and a quick experiment shows free-particle access to composites is **spatially limited**, not BE-limited.

> **Status update 2026-05-06:** most of the dead-code, `r_attract`, and `total_energy`-quirk findings below have since been resolved or partially addressed. See [Status update — 2026-05-06](#status-update--2026-05-06) for the per-finding mapping to commits, plus two new bugs that surfaced when the test suite was unblocked.

## Contents

- [Status update — 2026-05-06](#status-update--2026-05-06)
- [The simulation step (9 phases)](#the-simulation-step-9-phases)
- [State model](#state-model)
- [Force model — Particle Life kernel + polarity + bonds](#force-model--particle-life-kernel--polarity--bonds)
- [Hash chemistry as actually implemented](#hash-chemistry-as-actually-implemented)
- [Fusion mechanics](#fusion-mechanics)
- [Decay and fission](#decay-and-fission)
- [Energy bookkeeping (and a subtle ordering issue)](#energy-bookkeeping-and-a-subtle-ordering-issue)
- [Boundaries](#boundaries)
- [Dead code findings](#dead-code-findings)
- [Experimental verification](#experimental-verification)
- [What this means for the size-plateau question](#what-this-means-for-the-size-plateau-question)
- [Nubs](#nubs)

## Status update — 2026-05-06

Five commits over 2026-05-05 → 2026-05-06 closed most of this audit's open items. Net diff is summarized here; the original sections below are kept intact for historical context with inline ✓ markers pointing at this section.

### Resolved

| Finding (audit section) | Status | Commit |
|---|---|---|
| All six dead chemistry/util/config items ([Dead code findings](#dead-code-findings)) | ✓ Commented out with revival banners; tests for dead code commented alongside | [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) |
| `r_attract[si,sj]` initialized but unused by kernel (also flagged in [Force model](#force-model--particle-life-kernel--polarity--bonds)) | ✓ Replaced. Per-pair `r_attract` and uniform `r_cutoff` matrices removed in favor of fractional `peak_fraction[S,S]` and `cutoff_fraction[S,S]`. Kernel now uses an asymmetric two-segment triangle (`r_peak` is no longer constrained to the midpoint). Each species pair now has its own force-shape, not just amplitude. | [`333df85`](https://github.com/heysoos/particle-halflife/commit/333df85) |
| "Tests pass on dead code" false-signal ([Dead code findings](#dead-code-findings)) | ✓ `test_half_life_distribution` commented out; `composite_half_life_scale` references in `test_chemistry.py` adjusted | [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) |
| Test-rot meta-issue (suite couldn't even run because production signatures grew a `physics: PhysicsParams` arg without updating callers) | ✓ Threaded `physics` through every test setup; suite went from 13/28 → 28/28 passing | [`c6a917a`](https://github.com/heysoos/particle-halflife/commit/c6a917a) |
| Velocity clamp ordering quirk ([Energy bookkeeping](#energy-bookkeeping-and-a-subtle-ordering-issue) — flagged that soft conservation rescales after the clamp) | ✓ Added a final magnitude-based clamp at the end of `simulation_step`, after fission kicks and soft conservation | [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa) |

### New findings (surfaced once the test suite ran)

| Finding | Detail | Commit |
|---|---|---|
| **Velocity clamp was per-component, not magnitude** | `jnp.clip(v, -V, V)` allowed diagonal motion to reach `\|v\| = √2 · max_velocity` — observed up to 11.26 with `max_velocity=8.0`. Replaced with vector-rescale form `v · min(1, V / \|v\|)` at both phase 4 and the new end-of-step clamp. | [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa) |
| **JAX `at[safe_pids].set(...)` duplicate-index race on particle 0** | Three call sites in fusion (`assign_i_member`, `assign_j_member`) and decay (`apply_composite_decay`) used `safe_pids = jnp.where(valid, pids, 0)` as the fallback for invalid entries, then read back `composite_id[0]` as the value. JAX scatters with duplicate indices have indeterminate behavior, so M−1 invalid slots writing the read-back value to index 0 raced against any real write to particle 0 with a different value. **Particle 0 was uniquely vulnerable** because every invalid lookup defaulted to 0; matched the test failure pattern (4 alive composites all claiming `member[0]=0` with `composite_id[0]=−1`). Fix: route invalid entries to OOB index `N` and use `mode='drop'`. Working precedent for this pattern was already in [`chemistry.py:481-483`](../halflife/chemistry.py#L481-L483) (the merged-members compaction) — just hadn't been applied to the composite_id writes. | [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa) |

### Still open (deliberately deferred)

- **`total_energy` storage is the pre-correction snapshot** ([Energy bookkeeping](#energy-bookkeeping-and-a-subtle-ordering-issue)) — not fixed. The end-of-step velocity clamp added in [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa) partially mitigates the consequence (velocities don't drift unboundedly anymore) but the recorded `total_energy` is still pre-correction.
- **Min-image displacement duplicated in 4 places** ([nub](#nubs)) — `pairwise_displacement` was commented out in [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) rather than adopted; revival is the path to dedup.
- **`find_neighbors_for_particle.pack_slot` complexity** ([nub](#nubs)) — untouched.
- **Rep-only fusion rule at higher densities** ([nub](#nubs)) — untouched.

### Adjacent change (UI, not physics)

- Slider panel reorganization: `×` suffixes dropped, sliders grouped by relevance with visual gaps, full-track click target instead of pixel-precise on the knob. [`196ba55`](https://github.com/heysoos/particle-halflife/commit/196ba55).

## The simulation step (9 phases)

[halflife/step.py:151–226](../halflife/step.py#L151) — a single JIT'd function. `config` is `static_argnums=(2,)`, everything else is dynamic.

1. **Build cell list** — particles → grid, [halflife/spatial.py:51](../halflife/spatial.py#L51)
2. **Find neighbors** — for each particle, scan 3×3 cell window, [halflife/spatial.py:208](../halflife/spatial.py#L208)
3. **Compute forces** — Particle-Life kernel + polarity scaling, [halflife/interactions.py:167](../halflife/interactions.py#L167); optional bond forces from [halflife/step.py:48](../halflife/step.py#L48)
4. **Integration** — explicit Euler: `v += (F/m)·dt; v *= damping; v = clip(v, ±max_velocity); x += v·dt`. [halflife/step.py:192](../halflife/step.py#L192)
5. **Boundaries** — periodic wrap or reflective bounce, [halflife/utils.py:129](../halflife/utils.py#L129)
6. **Fusion** — unified entity-entity, [halflife/chemistry.py:233](../halflife/chemistry.py#L233)
7. **Composite decay (fission)** — stochastic per composite, [halflife/chemistry.py:125](../halflife/chemistry.py#L125)
8. **Energy accounting** — recompute totals + soft correction, [halflife/energy.py](../halflife/energy.py)
9. **Increment ages and counters**

Note on phase 8 ordering: `current_energy` is computed *before* the soft correction is applied, but it's the value stored as `state.total_energy`. The correction targets the *previous* step's total. See [Energy bookkeeping](#energy-bookkeeping-and-a-subtle-ordering-issue) below.

## State model

Three NamedTuples in [halflife/state.py](../halflife/state.py):

- **`ParticleState`** (N=2000): position, velocity, species, energy, mass, age, composite_id. **No `alive` mask** — every slot is always "alive" since `b0c049f`.
- **`CompositeState`** (C=5000): members `(C, M=64)`, member_count, alive, binding_energy, half_life, age, species_hash, net_polarity. M=64 is a JAX buffer constraint, not a physics cap (per the design philosophy memory).
- **`WorldState`**: above two + time + rng_key + total_energy + step_count.

Two parameter bundles passed separately (so they're not part of `static_argnums` — slider changes don't recompile):

- **`InteractionParams`**: per-species-pair `attraction[S,S]`, `r_attract[S,S]`, `r_cutoff[S,S]`, plus per-species `polarity[S]`. Random-initialized (seed 42). ✓ *Updated 2026-05-06 in [`333df85`](https://github.com/heysoos/particle-halflife/commit/333df85): `r_attract`/`r_cutoff` replaced with fractional `peak_fraction[S,S]` and `cutoff_fraction[S,S]` (kernel multiplies them by `interaction_radius`).*
- **`PhysicsParams`**: 10 runtime-tunable scalars (damping, repulsion_strength, fusion_threshold, polarity_*_scale, binding_energy_scale, repulsion_radius, r_cutoff_scale, spring_k, attraction_scale).

A particle's composite membership is *only* the `composite_id` field on `ParticleState` (-1 = free). The composite's `members` array is the redundant reverse mapping. **Both must stay in sync** during fusion and fission.

## Force model — Particle Life kernel + polarity + bonds

### The kernel ([halflife/interactions.py:31](../halflife/interactions.py#L31))

Three regions, function of distance `r` between species `(si, sj)`:

| Region | Force magnitude |
|---|---|
| `r < repulsion_radius` (=0.8) | `−repulsion_strength · (1 − r/r_repulse)` (always repulsive) |
| `repulsion_radius ≤ r < r_cutoff` | `attraction[si,sj] · (1 − \|r − peak\|/half_width)` triangle, peak at `r_repulse + (r_cutoff − r_repulse)/2` |
| `r ≥ r_cutoff` | 0 |

Sign convention: positive → attractive (toward `j`), negative → repulsive. Final force vector is `−f_mag · d_hat` where `d_hat = (pos_i − pos_j)/r`.

**Note:** the per-species `r_attract[si,sj]` is sampled at init *but is not used in the actual kernel* — `peak` is computed solely from `r_repulse` and `r_cutoff`. That's a config-vs-implementation gap worth flagging. Effectively the peak position is uniform across all species pairs; only the *amplitude* (`attraction[si,sj]`) varies. (See [Nubs](#nubs).)

> ✓ **Resolved 2026-05-06 in [`333df85`](https://github.com/heysoos/particle-halflife/commit/333df85).** Kernel now uses an asymmetric two-segment triangle with `r_peak = interaction_radius * peak_fraction[i,j]` and `r_cutoff = interaction_radius * cutoff_fraction[i,j]`. Both halves of the triangle can have different widths (no longer fixed midpoint). Per-pair force-shape is real now, not just amplitude.

### Polarity scaling ([halflife/step.py:175–184](../halflife/step.py#L175))

Each particle gets an `attr_mod`:
- Free particle: `attr_mod = 1.0`
- Composite member: `attr_mod = composite.net_polarity` (mean polarity of members at formation)

Effective attraction between i and j: `aij · attr_mod_i · attr_mod_j · attraction_scale` ([halflife/interactions.py:116](../halflife/interactions.py#L116)). Result: a balanced (net_polarity ≈ 0) composite is **inert** — its members exert near-zero attraction on anything else. Polarized composites stay reactive.

The repulsive core is **not** scaled by polarity — it's a global hard core. So even inert composites still repel each other at close range.

### Bond forces ([halflife/step.py:48](../halflife/step.py#L48))

Optional, on by default (`use_bond_forces=True`). Each composite member is pulled toward the composite's center of mass (computed using min-image displacement so periodic-boundary composites don't tear apart) with `F = spring_k · (com − pos)`. Cost is O(C·M), no all-pairs. `spring_k=50.0` (in `PhysicsParams`, runtime-tunable).

## Hash chemistry as actually implemented

The current implementation does **not** use the polynomial-rolling `hash_multiset` from [halflife/utils.py:18](../halflife/utils.py#L18). That function is dead code (see [Dead code findings](#dead-code-findings)). The live path is a **commutative additive hash**:

```
entity_hash_val(s) = ((s+1)² · prime_a + (s+1) · prime_b) % modulus
H(entity) = sum(entity_hash_val(s) for s in members) % modulus
H(i ∪ j) = (H(i) + H(j)) % modulus     ← single addition, no sort, no scan
```

Defined at [halflife/chemistry.py:41–74](../halflife/chemistry.py#L41). The commutative form is a perf win (commit `086e9e1`): merging two entities is O(1) instead of O(M log M).

### From hash to binding energy ([halflife/chemistry.py:86](../halflife/chemistry.py#L86))

```python
h2 = (h * 2_654_435_761) ^ (h >> 13)        # Fibonacci hash mix
BE = ((h2 % 1000) / 999) · binding_energy_scale     # ∈ [0, scale]
```

The Fibonacci mix is a bug fix (in-line comment): the additive `entity_hash_val` produces values that are large multiples of `prime_a ≈ 10⁶`, so naive `% 1000` returns 0 for most inputs. Mix first, then mod.

### From hash to half-life

**Not used in production.** [halflife/chemistry.py:77](../halflife/chemistry.py#L77) `_hash_to_half_life` exists, but the actual fusion path computes half-life from binding energy directly:

```python
t = clip((BE − fusion_threshold) / (1 − fusion_threshold), 0, 1)
hl_base = lerp(half_life_min, half_life_max, t)              # high BE → long life
size_penalty = 1 + composite_size_decay_scale · max(0, M−2)  # bigger → shorter life
hl = hl_base / size_penalty
hl_eff = hl · (1 + polarity_stability_scale · (1 − |net_polarity|))   # neutral → boost
```

[halflife/chemistry.py:436–458](../halflife/chemistry.py#L436). So the design got reshaped: BE drives stability, not the species multiset directly. The hash still influences BE, so via that chain the species set still matters — but the mapping is now `species → BE → half-life` rather than `species → half-life` independently.

## Fusion mechanics

[halflife/chemistry.py:233](../halflife/chemistry.py#L233). The **single most subtle module** in the codebase.

### The "representative" trick

Each entity (free particle or composite) is identified by a single particle, its **representative**:
- Free particle: itself
- Composite: `members[c, 0]` — the first member, fixed at composition time

Only representatives participate in the fusion scan ([halflife/chemistry.py:286](../halflife/chemistry.py#L286), `i_is_rep` gate). This avoids double-counting (an N-member composite would otherwise contribute N candidates).

Implication worth flagging: **only the representative's local neighborhood drives fusion**. If the rep ends up surrounded by other composite members, free particles approaching from outside have to reach the rep specifically.

### Per-step fusion loop

1. **Compute candidates in parallel** (`find_entity_partner`, vmapped over all reps): for each rep, scan its neighbor list, compute `merged_h`, `BE_eff = BE + polarity_fusion_scale·(−p_i·p_j)`, and pick the neighbor with highest BE that exceeds `fusion_threshold` and is within `fusion_radius`. Returns one candidate per rep.
2. **Sample at most `max_fusions_per_step` candidates** ([halflife/chemistry.py:369–389](../halflife/chemistry.py#L369)) using a *random* shuffle (not lowest-index priority — explicitly noted in code; the alternative biased version is left in a comment).
3. **Conflict resolution** via `jax.lax.scan` ([halflife/chemistry.py:394](../halflife/chemistry.py#L394), `fusion_scan_body`). Each candidate is processed sequentially, with a `claimed` mask preventing double-fusion in one step. Four cases handled in one branch via nested `jnp.where`:
    - free + free → new composite slot (consumes from `free_comp_slots` pre-computed pool)
    - free + composite → grow `cj`, target = `cj`
    - composite + free → grow `ci`, target = `ci`
    - composite + composite → merge into `min(ci, cj)`, mark `max(ci, cj)` dead
4. **Member compaction**: concat `i_members ∥ j_members` (size 2M=128), cumsum to compact valid IDs to the front, drop overflow. [halflife/chemistry.py:473–483](../halflife/chemistry.py#L473).
5. **Sync** `composite_id` for every member of both sides to point at the new target.

The `would_overflow = (cnt_i + cnt_j) > M` check ([halflife/chemistry.py:333](../halflife/chemistry.py#L333)) prevents fusions that would exceed buffer size M=64. With current dynamics composites max around 8, so M is not the cap.

### Performance note

Fusion is **55% of step time** per [tests/README_PERFORMANCE.md](../tests/README_PERFORMANCE.md). The scan over `max_fusions_per_step=200` candidates is the dominant cost.

## Decay and fission

[halflife/chemistry.py:125](../halflife/chemistry.py#L125). Composites only — **free particles never decay**.

Per step, each alive composite `c`:
- `P(decay) = 1 − exp(−dt · ln 2 / half_life)`
- If decays:
  - `composites.alive[c] = False`
  - All members get `composite_id = −1`
  - Each member receives a radial velocity kick away from the composite's CoM
  - Per-member kinetic injection: `KE = binding_energy · (1 − fission_cost) / n_members`, speed `= √(2·KE)`

So `fission_cost` (=0.5) means **half the binding energy is dissipated** as the composite breaks up; the other half goes back into kinetic energy. (No transmutation — the released particles keep their original species.)

## Energy bookkeeping (and a subtle ordering issue)

[halflife/energy.py](../halflife/energy.py) defines:
- `KE = Σ 0.5 · m · |v|²`
- `BE = Σ binding_energy · alive` (treated as a stored potential)
- `total = KE + BE`

The soft correction scales velocities by `√(target_KE / current_KE)`, clamped to ±1% per step.

### The ordering quirk

In [halflife/step.py:210–223](../halflife/step.py#L210):

```python
current_energy = compute_total_energy(state)              # snapshot
state = apply_soft_energy_conservation(state, state.total_energy)   # uses PREVIOUS step's total
...
final_state = state._replace(..., total_energy=current_energy, ...)  # stores pre-correction snapshot
```

So:
- The correction targets *last* step's total — fine, that's the conservation law's reference.
- But the stored `total_energy` is the snapshot taken **before** the correction is applied, not **after**. So the next step's reference is the pre-correction value. The ±1% velocity scaling does change KE, but that change is never reflected in `total_energy`.

Effect: total energy walks slowly because the recorded "target" is always the pre-correction state, but the velocities have already been nudged. Over many steps the system tracks toward whatever balance the chemistry phases produce. Not necessarily wrong (the design is "soft" conservation, not strict), but worth being aware that **the recorded `total_energy` is not literally the energy of the state being returned**. If anything in the future tries to use `total_energy` for a precise energy-balance calculation, it'll be off by the correction amount.

> ⚠ **Partially resolved 2026-05-06 in [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa).** A separate consequence of this ordering — that the soft conservation could compound velocities back over `max_velocity` (~1% per step → ~21× over 100 steps) — was fixed by adding a final magnitude-based velocity clamp at end-of-step. The recorded `total_energy` is still pre-correction, so any precise energy-balance calculation downstream is still affected. Also note: the phase-4 velocity clamp was changed from per-component (`jnp.clip(v, -V, V)`) to magnitude-based (`v · min(1, V/|v|)`) since the per-component form let diagonal motion reach `|v| = √2 · V`.

## Boundaries

[halflife/utils.py:84–141](../halflife/utils.py#L84). Two modes, set at config:
- **Periodic** (default) — `pos % world_size`, plus minimum-image displacement in force/fusion/bond computations.
- **Reflective** — flip position and velocity sign at walls.

The min-image convention is duplicated **four times** across the code: `pairwise_displacement` in utils, the inline form in `pairwise_force`, in `find_neighbors_for_particle`, and in `compute_bond_forces`. They look correct but it's mild redundancy.

## Dead code findings

The chemistry module shows clear evolutionary scarring — earlier designs that got replaced but the old code is still around. **All six items below were commented out (with revival banners) in [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) on 2026-05-06 — line numbers preserved for historical reference.**

| Dead symbol | Location | Replaced by |
|---|---|---|
| ✓ `hash_multiset` (polynomial rolling hash) | [halflife/utils.py:18](../halflife/utils.py#L18) | `_entity_hash_val` + commutative sum at [halflife/chemistry.py:41](../halflife/chemistry.py#L41) |
| ✓ `hash_scalar` | [halflife/utils.py:49](../halflife/utils.py#L49) | `_entity_hash_val` |
| ✓ `_hash_to_half_life` | [halflife/chemistry.py:77](../halflife/chemistry.py#L77) | BE-based formula in `fusion_scan_body` |
| ✓ `_hash_to_decay_products` | [halflife/chemistry.py:97](../halflife/chemistry.py#L97) | Nothing — current decay just releases existing members, no transmutation |
| ✓ `composite_half_life_scale` (config knob) | [halflife/config.py:64](../halflife/config.py#L64) | Only used by dead `_hash_to_half_life` and tests |
| ✓ `max_decay_products` (config knob) | [halflife/config.py:84](../halflife/config.py#L84) | Only used by dead `_hash_to_decay_products` |

The dead `_hash_to_*` functions are still **unit-tested** in `tests/test_hash.py` and `tests/test_chemistry.py`. Tests pass on dead code; this is a **false signal** that the chemistry is "covered." ✓ *Resolved 2026-05-06 in [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) — `test_half_life_distribution` commented out, `composite_half_life_scale` references in `test_chemistry.py` adjusted.*

Earlier-design implication: the system was originally going to derive *both* half-life and decay product species from the hash (true Sayama-style hash chemistry). It pivoted to BE-driven half-life and no transmutation. That pivot has design consequences worth noting:

- "Different hash_modulus = different universe" is **less true now**: the hash only determines BE; everything else (half-life, fission products) is downstream of BE or just "release members unchanged."
- There's no transmutation. A composite of species `{A, B, C}` decays into a free A, a free B, and a free C — never into, say, a D and an E. The species pool is conserved per particle for life. (Particles are conserved entities — only their bonded/free state changes.)

## Experimental verification

Two micro-experiments run with the default config to test understanding (full code in shell history; results below).

### 1. Pair binding-energy distribution

Computed `BE` for all S² = 4096 free+free pairs (S=64 species):

```
min=0.000  max=1.000  mean=0.503  std=0.287
> fusion_threshold (0.2): 81.1%
> 0.5: 49.8%
Symmetric? True (sum-based hash is commutative)
Histogram: roughly uniform across [0,1] in 10 bins
```

**Verdict:** the hash chemistry is *not* biased against fusion at the pair level. ~81% of pairs satisfy the BE threshold. So the size plateau is **not explained by hypothesis #1** (BE filtering out fusions), at least at the free+free stage. Worth re-running for free+composite and composite+composite pairs, but the math is the same — additive hash means the BE distribution should remain roughly uniform regardless of size.

### 2. Spatial access at saturation

After 500 steps from the default config:
- 504 alive composites, 737 free particles
- Size hist: `[0,0,324,121,46,10,2,1,0,0]` — overwhelmingly size-2 (mean 2.47)
- Free particles within `fusion_radius=1.0` of any composite member: **0**
- At `fusion_radius=2.0` (just for visibility): 0.16 free contacts per composite
- Rep-only access vs any-member access: 1.14× advantage for "any" — i.e., only ~14% more contacts when you allow all members to receive

**Verdict:** the dominant constraint is **spatial isolation** (hypothesis #3 from the prior PHASE1 analysis). At saturation, free particles aren't reaching composites at all; whether the rep-only vs any-member rule applies barely matters because the contact rate is near zero anyway. This is a stronger result than I expected — the rep-only constraint is an interesting design choice but not the limiting factor.

This shifts the priority for "break the plateau" work toward whatever increases free→composite spatial contact: lowering `damping`, raising `init_speed`, increasing `fusion_radius`, or boosting per-composite reactivity (e.g., make polarized composites' attraction larger so they actively pull in free particles).

## What this means for the size-plateau question

Combining the audit with the experiments:

- **BE threshold** is *not* the dominant cause of the plateau (81% pass at pair level).
- **Rep-only fusion** is a real architectural constraint but only buys ~14% access difference vs. an "any-member" rule under current dynamics.
- **Spatial isolation at saturation** is the actual choke point. Once particles bond, they don't move toward more bonding partners — net_polarity-scaled attraction in particular *kills* their reactivity if they happen to form a balanced composite (`net_polarity ≈ 0` → `attr_mod ≈ 0` → no force on anyone).
- **Size-decay penalty** (`composite_size_decay_scale=0.05`) means anything bigger than 2 is half-life-penalized. With BE-driven `hl_base` already capped at `half_life_max=500` and a size penalty, larger composites are increasingly unlikely to survive long enough to absorb more.

The interesting next moves:
1. Run with `composite_size_decay_scale=0.0` and see if the plateau lifts. The existing sweep covers this — read [tests/reports/composite_statistics_20260327_211754.html](../tests/reports/composite_statistics_20260327_211754.html) before re-running.
2. Test whether *polarized* composites (large `|net_polarity|`) preferentially grow vs balanced ones — confirms the "neutralization-as-inertness" mechanism is the real sink.
3. Consider whether the design intent was for inert composites to be the goal (autopoiesis-like steady states) or if growth is the goal (autocatalysis). These pull in opposite directions.

## Nubs

- ✓ ~~`r_attract[si, sj]` is initialized but **not used by the kernel** — the peak position is computed from `r_repulse` and `r_cutoff` only ([halflife/interactions.py:66–67](../halflife/interactions.py#L66-L67)). Either the kernel intent is wrong (should use `r_attract` for peak) or `r_attract` should be deleted. Worth deciding.~~ **Resolved 2026-05-06 in [`333df85`](https://github.com/heysoos/particle-halflife/commit/333df85): replaced with fractional `peak_fraction[S,S]` and `cutoff_fraction[S,S]`; kernel now actually uses them.**
- ✓ ~~Dead chemistry functions (`hash_multiset`, `_hash_to_half_life`, `_hash_to_decay_products`) still have passing unit tests. Decide: delete dead code + tests, or revive the older design (transmutation on decay would be a significant qualitative addition).~~ **Resolved 2026-05-06 in [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f): commented out with revival banners (chose "preserve for revival" over delete since transmutation may be revived to break the size plateau). Tests for dead code commented alongside.**
- ⚠ `total_energy` storage is the *pre-correction* snapshot; the +/-1% velocity rescale isn't reflected. Document or fix if anything uses `total_energy` for a precise balance. **Partially mitigated 2026-05-06 in [`ddce9fa`](https://github.com/heysoos/particle-halflife/commit/ddce9fa): end-of-step magnitude clamp prevents the velocity-compounding consequence. The bookkeeping itself is unchanged.**
- ?? Min-image displacement is duplicated in 4 places. Consider extracting `pairwise_displacement` into the hot paths instead of inlining. **Note 2026-05-06: `pairwise_displacement` was commented out in [`98abb0f`](https://github.com/heysoos/particle-halflife/commit/98abb0f) since it had zero callers; revival is the path to dedup.**
- ?? `find_neighbors_for_particle.pack_slot` is O(max_neighbors × max_candidates) per particle — flagged in [PLAN.md](../PLAN.md) but not yet addressed. Could simplify to O(N · K) with a cumsum compaction (same trick used for member merging in fusion).
- ?? Experiment idea — does the rep-only fusion rule become important at higher densities? Re-run the spatial access experiment with `num_particles` doubled.

### Nubs added 2026-05-06

- ?? **JAX `at[].set()` duplicate-index pattern** is now a known footgun. Three sites used `safe_pids = where(valid, pids, 0)` with `mode='drop'`-less scatter, allowing M−1 invalid writes to race against any real write to particle 0. Worth a one-time grep across the codebase to make sure no other call site has the same shape (look for `jnp.where(..., ..., 0)` immediately followed by `.at[...].set(...)`).
- ?? **Velocity clamp semantics** are now magnitude-based, not per-component. If anywhere else uses `jnp.clip(velocity, -V, V)` directly (e.g., a future debugging probe), it'll silently re-introduce the diagonal-motion sqrt-2 issue.
