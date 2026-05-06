# Composite dynamics — what's a composite, and why are most fixed?

**Date:** 2026-05-06
**Topic:** Two questions: (1) why do composites seem fixed in place after the kernel-fix? (2) Are composites behaving as their constituent particles or as larger entities?

## Summary

**Composites are not first-class objects in the force code.** Every member is a particle that independently computes pairwise forces from its neighbors; the composite-ness only enters through (a) `attr_mod = net_polarity` scaling each member's attractions, and (b) bond springs pulling each member toward the COM. Surprisingly, **internal member-member forces do not cancel** — the attraction matrix is asymmetric (Newton's 3rd law is violated *on purpose* in Particle Life), so a composite gets a residual "self-propulsion" force from its own members. This dominates the force budget; **composite-composite forces are essentially zero** because composite COM-COM distances exceed `interaction_radius=4` for almost all pairs. Result: **76% of composites are completely stationary, 3% move long-range** (max 69 units in 300 steps). Visually you mostly see the static ones, with their members wobbling hard around a fixed COM.

## Contents

- [Architecture answer: composites are member-collections, not particles](#architecture-answer-composites-are-member-collections-not-particles)
- [Internal forces DO cancel — earlier "self-propulsion" claim was a bug](#internal-forces-do-cancel--earlier-self-propulsion-claim-was-a-bug)
- [Cell-list overflow bug (the actual cause)](#cell-list-overflow-bug-the-actual-cause)
- [Composite-composite forces are nearly zero](#composite-composite-forces-are-nearly-zero)
- [The bimodal mobility distribution](#the-bimodal-mobility-distribution)
- [Why visually it looks like jiggling-in-place](#why-visually-it-looks-like-jiggling-in-place)
- [What would make composites actually interact](#what-would-make-composites-actually-interact)
- [Nubs](#nubs)

## Architecture answer: composites are member-collections, not particles

The user's question — "are they behaving as constituent particles or as a whole?" — has a precise answer in the code. **Force calculation is purely per-particle.** [halflife/interactions.py:127–164](../halflife/interactions.py#L127-L164) `compute_forces_for_particle` is vmapped over every particle index, and each particle scans *its own* neighbors and sums pairwise forces. No code path treats a composite as a single entity for force purposes.

The composite-ness only enters at three points:

1. **`attr_mod` scaling.** [halflife/step.py:175–184](../halflife/step.py#L175-L184) sets `attr_mod[i] = composite.net_polarity[c]` for composite members and `1.0` for free particles. Then in [halflife/interactions.py:116](../halflife/interactions.py#L116), the *attractive* component of each pairwise force is scaled by `attr_mod_i · attr_mod_j`. This is the only place a member's force calc "knows" it's in a composite.
2. **Bond springs.** [halflife/step.py:48–114](../halflife/step.py#L48-L114) adds `F = spring_k · (com − pos)` per member. This is composite-aware (uses the composite's COM) but it acts on individual members.
3. **COM aggregation through summing.** Once each member has its force, integration happens per-particle. Bonds keep the cluster moving roughly together; the "composite COM motion" emerges from `mean(F_member) / mean(mass)` rather than being explicitly computed.

**So: the answer is hybrid.** Force calc is per-particle (composites *are* their constituent particles). COM motion is emergent (composites *appear* to move as a unit because bonds rigidify them). There's no third-mode where a composite acts as a single particle of species "S_composite" and polarity `net_polarity` — even though the polarity scaling tries to gesture at that.

A consequence: **the species-pair attraction matrix is sampled at the member level, not the composite level.** Two composites both with `net_polarity = 0.3` and the same member species set will interact differently than two free particles with charges `+0.3, +0.3` would, because the actual force comes from member-by-member pairings using `attraction[species_a, species_b]` for specific member species.

## Internal forces DO cancel — earlier "self-propulsion" claim was a bug

**RETRACTED 2026-05-06 (later same day):** I originally claimed here that "Newton's 3rd law is deliberately violated at the pairwise level" and that internal asymmetric forces produce a self-propulsion residual that drives composite COM motion. That claim was wrong, and the experimental "evidence" was a simulation artifact, not real physics. See the [Cell-list overflow bug](#cell-list-overflow-bug-the-actual-cause) section below for what's really going on. Leaving the wrong claim in place (struck through below) so the trail of reasoning isn't lost.

~~Look at the kernel: force on `i` from `j` is `-f_mag(species_i, species_j) · d_hat`. Force on `j` from `i` is `-f_mag(species_j, species_i) · (-d_hat) = f_mag(species_j, species_i) · d_hat`. For these to cancel, `f_mag(s_i, s_j)` would need to equal `f_mag(s_j, s_i)`. The matrix is asymmetric, so the residual gives a self-propulsion force.~~

The math *for the attractive component* is right — asymmetric `A[i,j]` does mean `F(a→b) + F(b→a)` for the attractive part is `(A[s_b,s_a] − A[s_a,s_b]) · attr_mod_i · attr_mod_j · f_triangle(r) · d_hat_ab`. **But** that's scaled by `attr_mod_i · attr_mod_j = net_polarity²`, which is ~10⁻⁶ for a near-neutral composite. The repulsive part is symmetric and cancels exactly. So the per-pair residual is *tiny* (~10⁻⁶), and across ~76 pairs randomly summed it's still ~10⁻⁵ — nowhere near the 38 N I attributed to it.

The 38 N came from somewhere else.

## Cell-list overflow bug (the actual cause)

[halflife/spatial.py:51](../halflife/spatial.py#L51) — the cell list has a fixed `cell_capacity = 8` particles per cell. When more than 8 particles cluster into one cell, **excess particles are silently dropped from the cell list** ([halflife/spatial.py:103–119](../halflife/spatial.py#L103-L119)). The `did_overflow` flag is set on the returned `CellList` but **never checked anywhere downstream**.

Composites tend to be very tight (members < 1 unit apart due to fusion + bond springs balancing repulsion). Composite 6 has 20 members crammed into a 0.7-unit-wide ball — they're all in one cell. The cell records only 8 of them; the other 12 are invisible to the spatial index.

Consequence for force computation:

- Member 1225's neighbor list contains 8 of the 19 other members.
- Member 556's neighbor list contains a different (overlapping but not identical) 8.
- Pair (1225, 556) might be in 1225's list but not 556's, or vice versa.
- For one-sided pair-views, the `F(a→b)` contribution exists but the `F(b→a)` cancellation doesn't.
- These uncanceled views accumulate into a fictitious net force on the composite COM.

Verified empirically:

| group | n | mean &#124;F_COM&#124; | median | max | fraction &#124;F&#124; > 1 |
|---|---|---|---|---|---|
| composites in overcrowded cells | 42 | **15.5** | 14.7 | 38.3 | **98%** |
| composites in clean cells | 220 | **0.016** | 0.000 | 2.97 | **0%** |

**Overcrowded composites have 956× the COM force of clean ones.** The "swimmers" I identified yesterday as "active 3% with non-balancing compositions" were actually the 16% of composites whose members happen to all sit in one cell. The "static blobs" were the 84% whose members are spread across enough cells that the cell list captures them all.

In other words: composites with full spatial visibility correctly have ~0 internal net force (Newton's 3rd does its job, and polarity scaling kills the asymmetric attractive residual). Composites that overflow the cell get spurious forces from the missing pair-views.

### Why this matters for the user's question

Going back to the original prompt — *"why are composites fixed in place?"*:

- The **84% that are clean** ARE genuinely fixed because internal forces correctly cancel and composite-composite forces are ~0 due to spatial isolation. This is real physics and matches the visual.
- The **16% that overflow** are moving due to a bug, not real dynamics. If you saw any composites swimming in your viz, those were the buggy ones.

So fixing the cell-overflow bug *will not* unstick composites — it'll likely make them stick *more*, by removing the only source of motion they currently have. The path to interesting composite-composite dynamics still goes through the items in [What would make composites actually interact](#what-would-make-composites-actually-interact), but the diagnosis is sharper now: composites are correctly inert in this configuration; we need to *add* genuine inter-composite forces, not just remove a bug.

### Suggested fixes for the cell overflow

1. **Increase `cell_capacity` aggressively.** Default 8 is way too low. Composites of 20+ members are common; capacity 32 or 64 would absorb that. Cost: linear in the cell-list memory size (`num_cells × cell_capacity` ints) — for 50×50 cells × 64 = 160k ints, peanuts.
2. **Assert on `did_overflow`.** The flag exists; just check it after `build_cell_list` and crash with a useful message if set. Currently it's silently ignored.
3. **Adaptive cell sizing.** Detect overflow and grow the buffer dynamically. More invasive; probably overkill.
4. **Use a sort-based neighbor approach.** Particles sorted by cell, then linear scan within neighborhood. No fixed cap. More work; the current cell list is already fast.

(1) + (2) is the cheapest fix and probably sufficient. Worth doing soon — the wrong forces from this bug are coloring every analysis we run on settled states.

## Composite-composite forces are nearly zero

Why? Spatial geometry.

```
interaction_radius (force cutoff): 4.0
typical composite radius (max member-to-COM): ~3.0
typical inter-composite COM-COM gap: 5–7 units
```

So two composites typically have COM-COM distance > `r_cutoff`. Their members' pairwise distances *might* still be inside `r_cutoff` if the closest faces are within ~1–2 units, but in the settled state most composites have at least one full radius of empty space between them.

Quantified: across 11 composite pairs whose COMs were within `1.5 × r_cutoff` (= 6 units), the actual inter-composite force was **0.0000** in every case. Even when geometrically possible, the members don't end up close enough to interact across composite boundaries.

This is also why my prior "polarity-induced inertness" finding (hypothesis c) doesn't really matter for current dynamics: the attraction it would kill isn't being computed in the first place. Composite-composite attraction would only matter if composites got within interaction range, which they currently don't.

## The bimodal mobility distribution

Tracked all 245 alive composites' COM drift over 300 steps in the settled state:

| drift | composites | description |
|---|---|---|
| < 0.1 units | **76%** | completely stationary |
| < 1.0 unit | 81% | barely moving |
| ≥ 3.0 (one radius) | 12% | actually mobile |
| > 10 | **3%** | long-range swimmers |
| max | **69.4** | one composite traversed a third of the world |

The distribution is **bimodal**: a fat majority of "static blobs" and a thin tail of "active swimmers." The visualization is dominated by the static ones (they don't move offscreen, they just sit there) but a few are doing real motion.

What separates them? **The species composition's contribution to internal asymmetry.** A composite whose member-species combination produces near-zero `F_self` ends up frozen at force equilibrium (springs balance internal asymmetry → zero net force → no acceleration → damping kills any residual). A composite whose composition produces large `F_self` keeps accelerating, hits `max_velocity=8`, and swims.

Sample from the same run (size 3+ composites, sorted by motion):

```
cid  size |np|   |F_internal|  |v_COM|
  3   12  0.278    0.0000      2.250    ← active swimmer
 12   10  0.069   12.6598      2.488    ← active swimmer
 17   16  0.057   20.7375      2.014    ← active swimmer
 14   16  0.037   17.7020      1.836    ← active swimmer
  6   20  0.001   38.3331      0.284    ← still trying to swim, but heavy
  1    8  0.346    0.0000      0.007    ← static
  2    8  0.013    0.0000      0.000    ← static
  7    5  0.464    0.0000      0.000    ← static
```

Note: polarity (`|np|`) is **not** the deciding factor. Composite 7 has `|np|=0.464` (very polar) but is static; composite 6 has `|np|=0.001` (totally neutral) yet has the highest internal force. The asymmetric-matrix self-force is composition-dependent in a way that doesn't track polarity.

## Why visually it looks like jiggling-in-place

Even for the active swimmers, the per-member force magnitude is much larger than the per-COM aggregate. Measured at settled state:

```
member speed RELATIVE to COM (= wobble):  mean=0.674, p90=2.494
COM speed (translation):                   mean=0.183, p90=0.690
                                           wobble dominates COM motion ~3.7×
```

So even the moving composites look mostly like jiggling clusters. Each member oscillates hard around the local COM (driven by intra-composite repulsion balanced by bond springs, with damping bleeding energy slowly), while the COM creeps along.

User's observation reconciled: **"composites are fixed in place"** is true for ~76% of them — and the rest are moving slowly enough that wobble drowns the translation visually.

## What would make composites actually interact

In rough order of "smallest change → biggest change":

1. **Bump `interaction_radius` to 6–8** (with matching `cell_size`). Lets the closer faces of nearby composites overlap in interaction range. This is the cheapest fix and would directly address composite-composite force = 0.

2. **Compute force on composite COM as if it were a single particle** at the COM, with effective species and polarity derived from members. Then *also* compute member forces for internal dynamics. This makes composite-composite interactions explicit and decouples them from member-level forces. Big architecture change but matches the user's intuitive model.

3. **Distinguish self-force from external-force.** Separate the contributions: internal forces (cancel out for COM) → just contribute to wobble; external forces (other composites, free particles) → drive COM motion. This requires per-pair bookkeeping but cleanly decouples wobble from translation.

4. **Add a long-range potential.** Currently the kernel is a triangle that goes to 0 at `r_cutoff`. A 1/r-style tail (e.g., gravity or Coulomb) would give composites a long-range "presence" they currently lack. Particle Lenia uses smooth long-range kernels for this reason.

5. **Symmetrize the attraction matrix.** Would eliminate the self-propulsion (internal forces would cancel) and make composites genuinely depend on external forces for motion. Loses the chasing dynamics that make Particle Life interesting at the free-particle level. Probably not worth it.

The user's instinct toward "make composite-composite physics richer" probably maps to (1) or (2). (1) is a one-line config change worth trying first.

## Nubs

- ?? Plot the actual `F_internal` distribution by composite — is it bimodal (matching the static/swimmer split) or continuous? If continuous, there might be a sharp threshold where damping wins vs internal force wins.
- ?? Self-propulsion direction relative to species composition: is there a deterministic mapping from member-species multiset to swim direction? Could you "evolve" composites that swim in particular ways?
- ?? Test whether changing `cell_size` independently of `interaction_radius` helps (currently they're tied at 4.0 by config convention; the spatial code assumes `cell_size ≥ interaction_radius` for correctness).
- ?? What does the "as-if-single-particle" force on the static composites look like? My experiment showed `|F_asif| = 0.0000` for all 11 nearby pairs — so the as-if-particle approximation also predicts they won't interact at these distances. Confirms that geometry, not polarity, is the limiting factor in *this* config.
- ?? Cross-reference with [notes/2026-05-05-boring-dynamics-investigation.md](2026-05-05-boring-dynamics-investigation.md) — yesterday's finding (polarity-induced inertness) is real but masked here by the more dominant "composites just don't reach each other" effect.
