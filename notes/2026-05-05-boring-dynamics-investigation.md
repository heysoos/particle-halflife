# Why are long-term dynamics boring? — investigation

**Date:** 2026-05-05
**Topic:** After bumping fusion radius and other UI sliders, compounds form readily but the system equilibrates into a static "lattice of repelling jelly-blobs." Why?

## Summary

The screenshot config has **`repulsion_radius=8.0` cranked above `interaction_radius=4.0`**, which makes the Particle-Life kernel's attraction zone `[r_repulse, r_cutoff)` **empty**. The kernel's `(r >= r_repulse) & (r < r_cutoff)` gate is never true, so `f_attract = 0` for every pair. The only forces left are hard-core repulsion + fusion + bond springs. Composites form (via fusion proximity) but can never *attract* each other — explaining the "they avoid each other" observation. Hypothesis (a) about the rep-particle is largely wrong for *force dynamics* (forces use all members). Hypothesis (b) about a boring attraction matrix is wrong — the matrix is fine; it's just unused. Polarity-induced inertness (hypothesis c, my own addition from the prior audit) is a real but secondary effect.

## Contents

- [Headline: kernel is degenerate at user's settings](#headline-kernel-is-degenerate-at-users-settings)
- [Hypothesis (a) — does the representative limit interaction?](#hypothesis-a--does-the-representative-limit-interaction)
- [Hypothesis (b) — is the attraction matrix boring?](#hypothesis-b--is-the-attraction-matrix-boring)
- [Hypothesis (c) — polarity-induced inertness](#hypothesis-c--polarity-induced-inertness)
- [Direct test: corrected kernel produces livelier dynamics](#direct-test-corrected-kernel-produces-livelier-dynamics)
- [What I would change](#what-i-would-change)
- [Nubs](#nubs)

## Headline: kernel is degenerate at user's settings

The Particle Life kernel in [halflife/interactions.py:31](halflife/interactions.py#L31) has three regions:

```
r < r_repulse           →  hard-core repulsion (always negative)
r_repulse ≤ r < r_cutoff →  triangle attraction (sign = species attraction[si,sj])
r ≥ r_cutoff             →  zero
```

**Required invariant:** `r_repulse < r_cutoff`.

Screenshot values: `r_repulse = 8.0` (10× default), `r_cutoff = 4.0` (default `interaction_radius`). The attraction zone is the interval `[8.0, 4.0)` which is **empty**. `f_attract` is gated by `(r >= r_repulse) & (r < r_cutoff)` ([halflife/interactions.py:68–72](halflife/interactions.py#L68-L72)) — that condition is *never* true.

Even worse, the `half_width = (r_cutoff - r_repulse) * 0.5` becomes **negative** ([halflife/interactions.py:66](halflife/interactions.py#L66)). That doesn't crash because the gate masks the result, but it's a sign that the kernel doesn't validate its preconditions.

### What's left in the force budget

For any pair within neighbor range (distance < `interaction_radius` = 4.0):

- `r < r_repulse=8.0` is always true → `f_repulse = -repulsion_strength · (1 - r/r_repulse)` is always nonzero
- `f_attract = 0` always

So the kernel produces **only repulsion** — a hard-core blob that pushes everything apart at any distance < 4.0. Composites still form because fusion has its own radius gate (`fusion_radius=2.5`) and once two particles fuse, bond springs hold them together. But there is no medium-range attractive force pulling things together.

That's the smoking gun for "compounds bounce around with spring physics around themselves and seem to avoid other composites."

### Verification — geometry matches the prediction

Inter-member distances inside size-4+ composites at this config:

```
mean min member-member distance: 3.41   (just outside the bond springs' equilibrium)
mean max member-member distance: 6.30   (still inside r_repulse=8, so strong repulsion across)
99.6% of intra-composite pairs are within r_repulse=8 → constant strong repulsion
54.9% are within r_cutoff=4 → spatially neighbors but no attraction available
```

So inside a composite, members feel huge repulsion (~12 force units each) balanced by `spring_k=5` bond pulls — they oscillate around the fixed-point but don't go anywhere. **Net force on the composite COM ≈ 0** because internal forces cancel pairwise (Newton's third law) and external attraction is zero.

## Hypothesis (a) — does the representative limit interaction?

**Short answer: only for fusion, not for forces.** I grepped: `is_rep` / `all_reps` appear *only* in [halflife/chemistry.py](halflife/chemistry.py) (the fusion module). Force computation in [halflife/interactions.py](halflife/interactions.py) and bond springs in [halflife/step.py](halflife/step.py#L48) operate on every particle individually.

So a composite *member* — any of them, not just the rep — feels and exerts force from/on:
- Other members of its own composite (cancel pairwise)
- Members of other composites (these would drive COM-COM motion)
- Free particles (these would drive composite mobility)

Where the rep does limit things: **fusion access**. Only the rep particle's neighborhood is scanned for partners. So a composite can only grow when a free particle wanders close to its lowest-index member specifically. Other members standing closer to a free particle don't generate fusion candidates. That matters for *growth* but not for *motion*.

So: hypothesis (a) is the wrong target for the boring-dynamics complaint, but it's a real architectural bias for fusion. (The prior audit at [notes/2026-05-05-physics-audit.md](notes/2026-05-05-physics-audit.md#fusion-mechanics) covers this in more detail.)

## Hypothesis (b) — is the attraction matrix boring?

**Not boring.** The 64×64 random matrix from [halflife/state.py:188](halflife/state.py#L188) at the default seed:

```
mean:    -0.0025         (balanced — no global bias)
std:      0.5802         (full dynamic range)
fraction strongly attractive (>0.5):  25.0%
fraction strongly repulsive (<-0.5):  25.2%
asymmetry: mean |A[i,j] - A[j,i]| = 0.6722   ← good!
```

That asymmetry (0.67) is the key indicator: lots of pairs where A attracts B but B repels A — the "chasing" dynamic that makes Particle Life interesting. It's a perfectly fine substrate for rich behaviour.

**The matrix is unused at the user's settings** because the attraction zone is empty. Every entry of the matrix multiplies a 0. So the matrix's quality doesn't matter — the kernel never reaches `f_attract`.

If/when the kernel is fixed, the matrix is *fine* as-is. Structured matrices (Perlin, sparse, block-diagonal) would change the *flavor* of behaviour but aren't necessary to escape the current "boring" regime.

## Hypothesis (c) — polarity-induced inertness

This is the one I added in the prior audit. Each composite member's force on others gets scaled by `attr_mod_i · attr_mod_j` where `attr_mod = composite.net_polarity` ([halflife/step.py:178–184](halflife/step.py#L178-L184)). As composites grow, member polarities average toward 0 by central-limit dynamics, so `attr_mod² → 0` and the composite becomes inert *with respect to attraction*.

Measured at the user's settled state:

| size | n | mean \|net_polarity\| | mean attr_mod² |
|---|---|---|---|
| 2 | 341 | 0.368 | 0.206 |
| 3 | 123 | 0.333 | 0.164 |
| 4 | 55 | 0.273 | 0.106 |
| 5 | 19 | 0.250 | 0.095 |
| 6 | 6 | 0.291 | 0.129 |
| 7 | 5 | 0.178 | 0.035 |
| 8 | 1 | 0.012 | 0.0001 |
| 9 | 1 | 0.072 | 0.005 |

A size-7 composite has `attr_mod² ≈ 0.035` — it generates **3.5% of the attraction it would as a free particle**. Composite-composite pairs are doubly affected:

```
pair_product = attr_mod_i · attr_mod_j
  mean = 0.006   (basically zero)
  median |product| = 0.067
  fraction with |product| < 0.10:  60.4%
  fraction with |product| > 0.50:   2.5%
```

So even **with a working kernel**, attraction between most composites is reduced to ~6% of what it would be between free particles. This is a genuine architectural cause of boring composite-composite physics — it just isn't *the* cause at the user's settings, where attraction is already zero.

Worse, there's a **positive feedback loop**: the polarity_stability_scale boosts half-life for neutral composites (`hl_eff = hl · (1 + scale · (1 − |net_pol|))`). So balanced composites (which are inert) live *longer* than polarized composites (which are active). The system drifts toward a population dominated by long-lived, inert blobs — exactly what the screenshot looks like.

## Direct test: corrected kernel produces livelier dynamics

Compared three configs, all run for 1500 steps then tracked COM drift over the next 200 steps:

| config | composites | size hist (2..11) | drift median | drift max |
|---|---|---|---|---|
| **User's** (`r_repulse=8 > r_cutoff=4`) | 547 | 338,122,55,19,6,5,1,1,0,0 | 0.31 | **1.1** |
| **Corrected** (`r_repulse=1, r_cutoff=6`) | 431 | 118,80,67,57,44,23,13,9,3,4 | 0.08 | **7.9** |
| **Default** (everything stock) | 356 | 84,66,37,47,24,19,24,10,9,9 | 0.004 | **5.9** |

Two things stand out:

1. The user's config gives a **homogeneous "everyone stuck"** signature (mean ≈ median ≈ 0.31, max only 1.1 — about a third of a composite radius). All composites move similar tiny amounts.
2. The corrected and default configs both give a **heavy-tailed** distribution (median tiny, max several radii). Most composites are stuck *but some are not* — that's the signature of interesting heterogeneous dynamics where polarized composites actively move and explore while neutral ones sit still.

Also notable: the size distribution in the corrected/default configs is **much more diverse** — sizes 2 through 11+ all populated, vs. the user's config dominated by size 2.

So fixing the kernel doesn't just unstick the dynamics — it produces qualitatively different (richer) emergent structure.

## What I would change

In rough priority order:

1. **Validate the kernel preconditions.** Add an assertion or warning when `r_repulse >= r_cutoff` somewhere visible (e.g., at the start of `simulation_step` or as a `__post_init__` on `SimConfig` and on `PhysicsParams` updates). Right now the user can silently destroy attraction with one slider and never know. A check that prints "WARNING: r_repulse=8 >= r_cutoff=4 — attraction zone empty" would be enough.

2. **Make `r_repulse` a fraction of `r_cutoff`, not absolute.** E.g., `repulsion_radius = repulsion_fraction * interaction_radius` with `repulsion_fraction ∈ [0, 0.5]`. This makes the invariant impossible to violate via UI.

3. **Reconsider the polarity scaling.** Three options worth considering:
    - **Remove it entirely.** Lets large composites stay reactive; shifts the inertness/stability tradeoff to half-life only.
    - **Use `(1 + |net_polarity|) / 2`** instead of `net_polarity`. Range [0.5, 1.0] — composites are slightly less reactive than free particles but never fully inert. Loses the "ionic-bonding" flavor, gains stable mobility.
    - **Use `max(|p_i| for i in members)`** instead of mean. The most polarized member's charge dominates — composites stay active as long as they include at least one strong species.
   The current formulation maximizes the inertness/stability feedback loop, which makes neutral blobs dominate over time.

4. **Increase `interaction_radius` relative to composite size.** Composites at this config have radius ~3–4. With `interaction_radius=4`, two adjacent composites can barely "see" each other (their members must be within 4 units of each other). Bumping to 6–8 would let composite-composite interactions actually happen.

5. **Optional — sparsify the attraction matrix.** Once (1)–(3) are fixed, try setting ~50% of off-diagonal entries to 0 (or use a sparse mask + Perlin noise). This is well-trodden Particle Life territory and tends to produce more visually distinct cluster types. Not necessary, but a follow-up if the basic dynamics are still flat after the kernel fix.

The fastest one-line fix to test in the live UI: drag `repulse r` back toward 1.0–2.0 (its current 8.0 is what's killing attraction). Everything else can stay cranked.

## Nubs

- ?? Test whether removing polarity scaling entirely produces interesting dynamics or causes runaway mega-composites. The screenshot already shows sizes up to 40 with polarity scaling — without it, would composites grow even larger and interact more, or would they collapse into one giant blob?
- ?? With kernel fixed, run a parameter sweep on `interaction_radius / fusion_radius / repulsion_radius` ratios. There's likely a sweet spot for "rich emergent dynamics" that the current default config doesn't hit.
- ?? Stability feedback loop (neutral composites live longer than polarized) probably drives the population toward inert blobs over time even with valid kernel. Worth testing with `polarity_stability_scale=0` to isolate.
- ?? `find_neighbors_for_particle` uses `r2 = config.interaction_radius**2` to filter — so the cell-list already throws away pairs beyond `r_cutoff`. That's correct, but it means `r_repulse > r_cutoff` doesn't even trigger longer-range repulsion (any pair with `r_cutoff ≤ r < r_repulse` is dropped at the neighbor-finding stage, before the kernel sees it). So the user's config doesn't even produce *more* repulsion — it produces *only* repulsion at short range, exactly the same as the default would, just with a stronger gradient.
