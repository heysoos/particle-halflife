# Ideas for richer physics — TODO walkthrough

**Date:** 2026-05-06
**Topic:** For each item in [PLAN.md](../PLAN.md) "Next Session" TODO, think through *why* the current setup limits emergent richness, sketch 2–4 alternatives, and weigh perf cost.

---

## Suggested attack order

The TODO list isn't ordered by priority. I'd tackle them in this order:

1. **Cell-capacity overflow (#3)** — *first*, because it's a 1-line fix and 16% of composites are silently corrupted by it. Without this, every dynamics measurement has a 956× force outlier mixed in. Cheapest, biggest signal cleanup.
2. **Polarity scaling (#1)** — second. Most directly tied to the boring-dynamics issue; trivial code change; fast feedback loop.
3. **Force kernels (#4)** — third. Bigger perturbation but the most likely source of qualitatively new behavior. Worth the time.
4. **Composite interaction range + rep (#2)** — last. Most invasive; partially overlaps with #4 (long-range kernels would also let composites "see" each other).

---

## 1. Polarity scaling — break the inertness/stability feedback loop

### Current behavior

[step.py:178-179](../halflife/step.py#L178-L179) sets each member's `attr_mod = composite.net_polarity` (mean of member polarities). [interactions.py:122](../halflife/interactions.py#L122) then computes `eff_attraction = a_ij * attr_mod_i * attr_mod_j`. Neutral composites (`net_polarity ≈ 0`) lose **all attraction** to anything except their own bonds + universal repulsion. They also get a half-life *bonus* via `polarity_stability_scale` ([config.py:94](../halflife/config.py#L94)).

### Why this kills richness

It's a positive feedback into boring outcomes:

- Long-lived neutral composite forms.
- Neutral composite → no attraction to neighbors → cannot fuse, cannot break, cannot rearrange.
- Population is dominated by long-lived inert blobs that just sit there.

The intended physics — "neutral things are chemically passive" — is correct in spirit but the multiplicative form `mod_i * mod_j` collapses to zero too aggressively, and the stability bonus rewards exactly the configurations that are inert.

### Alternatives

**A. Remove polarity scaling entirely (`attr_mod = 1`).** All composites obey species-matrix forces, polarity only affects fusion preference + half-life. Clean baseline; lets us see what the substrate actually does without polarity confounds. Likely *more* dynamics, possibly less specialization.
*Perf:* zero. *Risk:* loses a chemistry knob; all composites become reactive.

**B. `attr_mod = (1 + |net_polarity|) / 2`.** Neutral composites attract at half strength; polarized ones at full. Symmetrizes the magnitude — neutrals stay reactive but less than polarized.
*Perf:* zero. *Risk:* loses the sign-flip behavior of polar-polar pairs.

**C. `attr_mod = max(|p_i|)` over members.** Composite is "as polar as its most polar member" — much harder to neuter by averaging out. A 50/50 mix of +1 and −1 species has `net_polarity=0` but `max|p|=1`.
*Perf:* one extra reduction per composite per step (negligible). *Risk:* might over-activate everything.

**D. Decouple sign from magnitude: `attr_mod = sign(net_polarity)` (or `tanh`)`.** Sign carries the polar/nonpolar interaction asymmetry; magnitude stays bounded. Neutral composites have `attr_mod = 0` (still inert), but you can pair this with a nonzero floor.
*Perf:* zero.

**E. Stop coupling stability to neutrality.** Drop `polarity_stability_scale`, or invert it (neutral composites are *short-lived*, polar ones live longer because they have an interaction story). Removes the feedback loop at its source rather than fighting it via attr_mod.
*Perf:* zero. *Risk:* changes composite lifetime distribution dramatically.

### Recommendation

Try **B** first (one-line change, preserves the spirit, guarantees nonzero attraction). If composites are still boring, try **A** (no polarity scaling at all) as a clean baseline. Independently consider **E** — the stability bonus is doing more harm than the attr_mod problem in my read.

---

## 2. Composite interaction range + the "representative" trick

### Current behavior

Two related shortcuts:

- **Force range:** `interaction_radius = 4.0` ([config.py:42](../halflife/config.py#L42)) is barely larger than a packed 4–6-member composite's diameter. Composites only "see" neighbors via their outermost members. Two composites passing within 4 units of edge-to-edge can interact; beyond that, nothing.
- **Rep trick:** [chemistry.py:267-268](../halflife/chemistry.py#L267-L268) — only the lowest-index member of each composite is checked as a fusion candidate. The other N-1 members never trigger fusion themselves, so a 20-particle composite has the same fusion "reach" as a single particle.

### Why this kills richness

- Composites cannot form long-range structure (chains, networks, halos). The world looks like ≤20-particle clusters separated by emptiness.
- The rep is an arbitrary architectural choice (lowest index) that biases growth direction. A composite is unlikely to grow on the side opposite its rep.
- Gives a false impression that composites are "stable" when really they're just isolated by a too-short force horizon.

### Alternatives

#### Range — three options

**A. Bump `interaction_radius` to 8 or 12.** Simplest. But cell_size needs to track interaction_radius for the 3×3 scan to be valid, so cells get bigger → fewer cells → more particles per cell → makes #3 (cell_capacity) worse. Also: scan radius is fixed 3×3 cells; if you decouple `cell_size < interaction_radius` you need a 5×5 or 7×7 scan in [spatial.py:169-173](../halflife/spatial.py#L169-L173).
*Perf:* with 3×3 scan and `cell_size = interaction_radius = 12`, neighbor count grows ~9× (radius² scaling). compute_forces is O(N · neighbors) so step time roughly 9× in the force phase. Currently force phase is ~9% of step (~1ms), so step goes 11.6→~9ms+8ms = 17.6ms. Painful but survivable.

**B. Two-tier forces: short-range (current) + long-range composite-only.** Keep the per-particle short-range kernel as is. Add a second pass: for each composite C, attract toward neighboring composite COMs at long range (e.g., via a separate cell list at composite scale, cell_size=20). Roughly:
```
F_long(C, C') = G * net_pol(C) * net_pol(C') / r²
```
Like gravity, but only between composites.
*Perf:* O(C · neighbors_C). With ~200 composites and modest density it's small (<<1ms). Adds an extra cell-list build at composite scale.
*Risk:* introduces a new physical layer to tune; could destabilize.

**C. Per-species-pair `cutoff_fraction`** ([state.py:170](../halflife/state.py#L170)) **already exists** — current cutoff is `interaction_radius * cutoff_fraction[i,j]` with fractions in [0.4, 1.0]. So pairs already vary in reach. The lever is the *unit* (`interaction_radius`) being too small. So really this is just A in disguise unless we widen the fraction range to e.g. [0.4, 2.0] (allowing some pairs to reach beyond cell-size, which breaks cell-list correctness).

#### Rep trick — three options

**D. Drop is_rep filter, every member fuses.** [chemistry.py:298](../halflife/chemistry.py#L298) currently constrains the scan to representatives. Remove that filter and every particle scans for partners. Will produce duplicate (composite-A, composite-B) candidate pairs (one per A-member that sees a B-member); dedupe at composite-id level by reducing to e.g. min-priority-per-composite-pair before applying.
*Perf:* a 20-particle composite goes from 1 fusion-scan to 20. But each scan was already vmapped over neighbors; we just stop short-circuiting. Step-time cost = roughly the fraction of particles that are non-rep composite members (~50% in current sims). So fusion phase ~1.5× longer, total step ~+5%.
*Risk:* dedup logic must be careful not to apply two merges to the same composite in one step.

**E. Geometric rep (closest-to-COM).** Instead of `mids[0]`, pick the member nearest the COM as the representative. Less arbitrary, removes the directional bias.
*Perf:* tiny extra cost to compute COM-distance per member.
*Risk:* still has the "only one member fuses" problem; just shifts where the bias points.

**F. Rotating rep.** Each step, the rep is `mids[step_count % member_count]`. Cheap, breaks the directional bias over time without changing scan structure.
*Perf:* zero. *Risk:* introduces step-count-dependent stochasticity which complicates reproducibility/debug.

### Recommendation

Pair **A (bump interaction_radius)** with **D (drop rep)**. Both move toward the same goal — composites genuinely interact at composite scale. After #3 is fixed (cell_capacity), the perf hit from A is the only real cost. **B (two-tier)** is the more elegant solution but requires real implementation work; revisit if A+D doesn't produce richer dynamics.

---

## 3. cell_capacity overflow — the cheap critical fix

### Current behavior

[config.py:39](../halflife/config.py#L39) — `cell_capacity = 8`. [spatial.py:87](../halflife/spatial.py#L87) computes `did_overflow` but nothing checks it. [spatial.py:103-119](../halflife/spatial.py#L103-L119) silently truncates particles past slot 7 in any over-full cell. Audit data: ~16% of composites have members crammed into a single cell exceeding 8, producing fictitious 956× forces from asymmetrically truncated neighbor lists.

### Why this kills richness

It's not a richness problem per se — it's a **measurement** problem. Until this is fixed, every claim about "composite dynamics" is contaminated by an artifact that mostly looks like spurious self-propulsion. We can't honestly evaluate #1, #2, or #4 with the bug present.

### Alternatives

**A. Bump `cell_capacity` to 32 (or 64).**
*Perf:* memory is `num_cells × cap × 4 bytes`. With 50×50 cells, cap=32 → 320KB; cap=64 → 640KB. Negligible on GPU. Compute cost: each [find_neighbors_for_particle](../halflife/spatial.py#L129) scans 9×cap candidates and vmaps `check_candidate` over them; cap=32 → 288 candidates (4× current 72), cap=64 → 576 (8×). The check is cheap (one min-image distance) but the vmap output is `(max_candidates, 2)`. Forces phase is currently ~9% of step time. Estimate +20–40% on neighbor-find phase, total step +2–4%. **Worth it.**
*Bonus:* `max_neighbors = 256` ([config.py:43](../halflife/config.py#L43)) is the buffer compute_forces vmaps over. With cap=32 the candidate buffer is 288, > max_neighbors, so we'd lose them again at the pack stage [spatial.py:194-205](../halflife/spatial.py#L194-L205). Bump `max_neighbors` to 288 or 320.
*Required:* convert `did_overflow` to a runtime check — `jax.debug.check` or print on first occurrence. Silent truncation should never be possible going forward.

**B. Shrink `cell_size` so each cell holds fewer particles.** With `cell_size = 2` (and `interaction_radius = 4`), each cell halves in area → ~¼ the particles. But the 3×3 scan no longer covers `interaction_radius`; you'd need 5×5. Cost is similar to A (more cells × less per cell) but more complex. Not recommended unless we also do #2A (bigger interaction_radius), in which case decoupling cell_size from interaction_radius is natural.

**C. Sparse cell list (variable capacity).** Out of scope for now; would need a different data structure (segment-pointer list rather than dense `(num_cells, cap)`).

### Recommendation

**A**, with both `cell_capacity = 32` and `max_neighbors = 320`, plus an assert (or `jax.debug.print`) when `did_overflow` is True. 1–2 lines of config change + one assertion. Do this first.

---

## 4. Force kernel audit + alternatives — the biggest lever

### Current behavior

[interactions.py:34-79](../halflife/interactions.py#L34-L79) — `particle_life_force` is a piecewise-linear triangle:
```
r < r_repulse:           strong repulsion (linear ramp from -strength at r=0)
r in [r_repulse, r_peak]: linear ramp from 0 to attraction[i,j]
r in [r_peak, r_cutoff]:  linear ramp from attraction[i,j] back to 0
r >= r_cutoff:            zero
```

`r_peak[i,j]` and `r_cutoff[i,j]` are **per-species-pair fractions** of `interaction_radius`, sampled in [0.3, 0.95] then sorted ([state.py:207-211](../halflife/state.py#L207-L211)). `attraction[i,j]` is a uniform `[-1, 1]` matrix, **asymmetric** (A→B can attract while B→A repels). Verified used in [interactions.py:117](../halflife/interactions.py#L117). Asymmetry is preserved end-to-end because force is computed per-particle vmap-over-neighbors, so A computes its own force from B and vice versa.

### What I want to verify before changing anything

- [ ] `attraction[i,j]` is signed and asymmetric → ✓ ([state.py:200](../halflife/state.py#L200))
- [ ] `attraction` is consulted for **every** particle pair (not e.g. only same-composite or only free) → ✓ ([interactions.py:117](../halflife/interactions.py#L117))
- [ ] `peak_fraction[i,j]` is consulted per-pair → ✓
- [ ] `r_attract[i,j]` mentioned in PLAN.md as "dead arg" — check if it's actually still in `InteractionParams`. → already removed; current names are `peak_fraction` and `cutoff_fraction`. The PLAN.md note is stale; confirm and update.

### Why the current kernel limits richness

- Linear triangle has discontinuous derivative at `r_repulse`, `r_peak`, `r_cutoff`. Particles "snap" through transitions; doesn't produce smooth orbits or rings.
- All species pairs share the same *shape* (triangle), only width and amplitude differ. Specialization is amplitude-coded.
- No long-tail behavior — force is exactly zero outside cutoff, so composites can't have a "halo" of weak attraction.

### Alternative kernels

**A. Lennard-Jones (6-12).**
```
F(r) = 24 * eps[i,j] * ((sigma[i,j] / r)^7 - 2 * (sigma[i,j] / r)^13)
```
Smooth, has a natural well at `r = 2^(1/6) * sigma`, repulsive core is automatic, attractive tail decays as `1/r^7` (effectively local). Per-pair: `eps[i,j]` (well depth, signed) and `sigma[i,j]` (zero-crossing).
*Pro:* physical, smooth, classic.
*Con:* `r^13` is numerically nasty for small `r`; needs floor. Tail isn't strictly zero; needs a soft cutoff at `interaction_radius`. Two parameters per pair.
*Perf:* ~same as current (still O(1) per pair).

**B. Gaussian-ring (Lenia-style).**
```
F(r) = -A[i,j] * exp(-(r - mu[i,j])^2 / (2 * sigma[i,j]^2))
```
A bump centered at `mu[i,j]` with width `sigma[i,j]`. Combine with current hard-core repulsion. Each pair gets its own ring.
*Pro:* smooth, parametric well, rings are conducive to ring-shaped composites and rotating structures (Lenia's signature).
*Con:* doesn't have a self-repulsive core; needs separate repulsion. Three params per pair.
*Perf:* one `exp` per pair; still O(1) but exp is ~5× a multiply on GPU.

**C. Step kernel (block-constant).**
```
F(r) = -A[i,j]    if r in [r_in[i,j], r_out[i,j]] else 0
```
Hard-edged annulus of constant force. Combined with current repulsion.
*Pro:* very crisp; same param count as current.
*Con:* discontinuous → integrator stiffness, particles "kick" through edges.
*Perf:* zero cost.

**D. Structured/sparse `attraction` matrix.** Don't change the kernel shape; change *how* `attraction[i,j]` is sampled. Options:
- **Block-diagonal:** group species into k blocks, attraction strong within block, weak across. Forces compartmentalization.
- **Sparse:** only `~10%` of pairs have non-zero attraction; rest are repulsive. Should produce specific reaction networks.
- **Cycle structure:** `attraction[i, (i+1) % S]` is large positive (chase patterns).
- **Dipole-like:** based on polarity, `attraction[i,j] = -p[i] * p[j]` (opposites attract).
*Pro:* zero kernel change, just init code change. Direct route to "this universe has actual chemistry rather than uniform noise."
*Con:* more design surface; need to think about what structure to impose.
*Perf:* zero.

**E. Perlin-driven spatial heterogeneity.** `attraction[i,j]` modulated by the Perlin noise value at the particle position. Different regions of the world have different chemistry → ecological niches.
*Pro:* spatial differentiation, niches.
*Con:* need to evaluate Perlin per particle per step (added cost in inner loop). Or precompute on a coarse grid once per N steps.
*Perf:* nontrivial. Defer.

### Recommendation

Do this in two passes:

**Pass 1 — kernel registry** ([config.py](../config.py) + [interactions.py](../halflife/interactions.py)):
Add `force_kernel: str = "triangle"` to SimConfig. Implement `triangle`, `lj`, `gaussian` as separate functions selected via static-arg dispatch. Toggle from sliders/keys for live A/B comparison. This is a 50-line refactor.

**Pass 2 — structured matrix** ([state.py:initialize_interaction_params](../halflife/state.py#L174)):
Add an `attraction_init: str` flag — `"uniform"` (current), `"block"`, `"sparse"`, `"cycle"`, `"dipole"`. Each is just a different sampling strategy for the (S, S) matrix.

The two passes are independent — kernel **shape** vs. matrix **structure** — and both contribute to richness independently.

---

## Cross-cutting perf budget

If I do all four:

| Change | Step-time delta |
|---|---|
| #3 cell_capacity 8→32 | +2–4% |
| #1 polarity rescale | 0 |
| #4 kernel switch (LJ or gaussian) | +0–5% |
| #2A interaction_radius 4→8 | +50% (force phase 9× larger) |
| #2D drop rep, every member fuses | +5% (fusion phase ~1.5×) |

Conservatively: ~+60% step-time worst case (11.6 → 19 ms/step at 2k). Manageable. If we want to keep 10k-particle headroom, #2A is the one to gate behind a flag.

**Most bang for buck:** #3 (free) + #1 (free) + #4 sparse-matrix init (free). All zero perf cost. Try those first; if dynamics are still boring, escalate to #2A.
