# Composite Reaction Network Evolution — Design Specification

**Date:** 2026-03-26
**Status:** Design Review
**Goal:** Enable open-ended composite evolution through hierarchical growth, multi-product reactions, and heredity

---

## Executive Summary

This design enables the Half-Life particle simulator to evolve composites over time through:
1. **Composite growth** — composites absorb free particles and other composites
2. **Multi-product reactions** — C+C interactions produce 1-5 offspring (fused, fissioned, or hybrid)
3. **Hereditary properties** — offspring inherit from parents through hash-based determinism
4. **Evolutionary dynamics** — selective pressure emerges from energy, stability, and polarity landscapes

The design is incremental: Phase 1 verifies current mechanics, Phase 2 adds growth, Phase 3 adds reactions, Phase 4 adds heredity tracking. Each phase is independently testable and does not require recompilation (except Phase 1-3, which modify simulation logic).

---

## Current State: Binding Energy System

Before describing changes, we clarify the existing binding energy (BE) model:

**Definition:**
- Derived deterministically from hash of a composite's member species: `H = hash(species_0, species_1, ..., species_k)`
- Normalized to [0, 1] then scaled by `physics.binding_energy_scale` → always non-negative

**Role in fusion:**
- Fusion occurs if `BE_merged > fusion_threshold`
- Adjusted by polarity bonus: `BE_eff = BE + polarity_fusion_scale * (-pi * pj)`

**Role in fission:**
- Released as kinetic energy: `kick = sqrt(2 * BE * (1 - fission_cost) / n_members)`
- Members scattered radially outward with speed proportional to BE

**Key property:** Same member set always yields same BE (deterministic, reproducible)

---

## Phase 1: Verification & Debugging

**Objective:** Understand why current composites don't grow larger and whether composite-composite fusion works correctly.

**Current behavior (observed):**
- Composites form but don't exceed small sizes (max ~5-10 members)
- Composite-composite fusion is implemented but effectiveness unclear
- Decay events are silent (not visually observable)

**What to instrument:**

1. **C+C fusion event tracking**
   - Count successful C+C fusions per step
   - Log binding energies of fusing pairs (to understand threshold behavior)
   - Track composite size distribution before and after fusion

2. **Visual feedback for C+C fusion**
   - Event sprite (existing system): render gold ring at fusion point when two composites merge
   - Console log: print "C+C fusion: composite A (5 members) + composite B (3 members) → C_new (8 members)"
   - Stats panel: add counter for "C+C fusions this session"

3. **Composite growth profiling**
   - Track over time: max composite size, mean composite size, distribution of sizes
   - Measure: how often do composites exceed size 5? Size 10? Size 20?
   - Hypothesis: if growth is rare, either BE is too low (energetically unfavorable) or composites don't encounter each other spatially

4. **Debug visualization (optional)**
   - Highlight composites that are close to fusion threshold (render with different color/brightness)
   - Show BE values above composites during simulation (overlay text on HUD)

**Deliverable:**
- A clear understanding of: Does C+C fusion happen? At what rate? Why does composite size plateau?
- Data to inform Phase 2 tuning (should we adjust fusion threshold, increase particle count, change initial layout?)

**Implementation notes:**
- No changes to core simulation logic, only observation and rendering
- Use existing event sprite system (gold ring for fusion)
- Add temporary logging/stats output

**Performance impact:** None (measurement only)

---

## Phase 2: Composite Growth (Absorption of Free Particles)

**Objective:** Enable composites to absorb free particles and grow larger.

**Current limitation:**
- `attempt_fusion()` fuses two free particles and fuses free particle with composite member
- But a free particle doesn't join an existing composite unless within `fusion_radius` of a member

**Design change:**

When a free particle p is within `fusion_radius` of ANY member m of composite C:

1. **Compute binding energy of augmented composite:**
   - `H_new = H_composite + hash(species_p)` (commutative sum)
   - `BE_new = _hash_to_binding_energy(H_new, config, physics)`
   - Check: `BE_new > fusion_threshold`

2. **If threshold passed, add particle to composite:**
   - Add p to C's `members` array
   - Increment C's `member_count`
   - Update C's `species_hash = H_new`
   - Recompute C's half-life from new hash
   - Update C's `net_polarity = (old_polarity * old_count + p_polarity) / (old_count + 1)`
   - Position p near the interaction site (no change to position, or small random jitter)
   - Set p's `composite_id = C`

3. **Size cap:**
   - If C is already at `max_composite_size`, the particle bounces off (fusion doesn't happen)
   - This prevents unbounded growth and computational overflow

4. **Conflict resolution:**
   - Use sequential scan (same as C+F fusion): first match wins
   - A free particle can only join one composite per step
   - A composite can absorb multiple free particles if they're in separate scan iterations

**Hash update on absorption:**
- Composite's hash changes as it grows: `H_new = H_old + hash(p_species)`
- This means composite's BE, half-life, and properties change dynamically
- Biological realism: a composite's stability depends on what it absorbs

**Polarity inheritance on absorption:**
- `new_net_polarity = (old_polarity * member_count + p_polarity) / (member_count + 1)`
- Balanced composite that absorbs polarized particle becomes slightly polarized
- Polarized composite that absorbs opposite particle becomes more balanced

**Implementation:**
- Modify `attempt_fusion()` to check for free particle + composite interactions
- Reuse existing neighbor-finding (already done per particle)
- Add size-cap check before insertion
- Update hash/polarity/half-life on successful absorption

**Testing:**
- Measure composite size growth over time (should exceed Phase 1 values)
- Verify no composites exceed `max_composite_size`
- Confirm energy conservation (soft correction should handle dynamics)

**Performance impact:** Minimal
- One additional hash computation per successful absorption
- No new O(N²) operations
- Already checking neighbors in fusion scan

**Risks & mitigations:**
- **Unbounded growth:** Mitigated by `max_composite_size` cap
- **Energetic instability:** Composites that absorb incompatible particles may destabilize; natural selection will filter them out

---

## Phase 3: Multi-product Reactions (Composite + Composite)

**Objective:** Enable composite-composite interactions to produce multiple offspring (1-5 products), creating complex evolutionary dynamics and composite lineages.

**Current behavior:**
- Two composites within fusion radius fuse into one larger composite
- When that composite decays, members are released as free particles (structure lost)

**Design change:**

When two composites C_a and C_b come within `fusion_radius` and both have `BE > fusion_threshold`:

1. **Compute reaction hash:**
   ```
   H_reaction = hash(C_a.species_hash, C_b.species_hash, REACTION_TYPE, config.hash_modulus)
   ```
   - Use a distinct hash function or REACTION_TYPE constant to distinguish from regular fusion
   - Commutative: ensures same pair always produces same products

2. **Determine number of products:**
   ```
   num_products = 1 + (H_reaction % 4)  # range [1, 5]
   ```
   - Can adjust modulus (4) if different range is preferred

3. **Assign members to each product:**
   - For product k (0 to num_products-1):
     ```
     seed_k = hash(H_reaction, k, config.hash_modulus)  # unique seed per product
     # Assign approx 50% of A's members, 50% of B's members to product k
     bit_mask_a = (seed_k >> (k * 4)) % (2^member_count_a)  # bits unique to k
     selected_a = select by mask: [member_i from A where (i & bit_mask_a) < count/2]

     seed_k' = (seed_k * PRIME) % hash_modulus  # new seed for B
     bit_mask_b = (seed_k' >> (k * 4)) % (2^member_count_b)
     selected_b = select by mask: [member_j from B where (j & bit_mask_b) < count/2]

     product_members[k] = selected_a ∪ selected_b
     ```
   - Use hash bits to deterministically select which members from each parent
   - Split roughly evenly between parents (50/50), but can skew if one parent is much larger
   - Ensure no member is used twice (global tracking across all products)
   - If members run out, remaining products can be free particles or smaller composites

4. **Compute product properties:**
   - For each product k with members `members_k`:
     ```
     H_k = hash(members_k)
     BE_k = _hash_to_binding_energy(H_k, config, physics)
     half_life_k = _hash_to_half_life(H_k, config)
     polarity_k = blend_polarity(C_a.net_polarity, C_b.net_polarity, H_k, blend_mode)
     ```

5. **Product positioning and velocity:**
   - Position: start at center of mass between C_a and C_b's representative particles
   - Jitter: add random displacement ±0.5 units per product (prevents clustering); wrap through periodic boundaries
   - Velocity: each product inherits `(m_a * v_a + m_b * v_b) / (m_a + m_b)` (momentum conservation), then add energy kick
   - Energy kick: if `(BE_a + BE_b) > (sum of BE_k)`, release difference as kinetic energy
     - Direction: radial from interaction center, or along A→B axis (configurable)
     - Magnitude: `kick_mag = sqrt(2 * energy_released / product_mass)` per product
     - If reaction is endergonic (sum BE_k > BE_a + BE_b), products are slower (no kick)

6. **Create products in free slots:**
   - Use `find_free_slots()` to allocate `num_products` new composite slots
   - If insufficient free slots, reaction is skipped
   - Insert each product into CompositeState

7. **Conflict resolution:**
   - Use `lax.scan` with max `max_reactions_per_step` (configurable in SimConfig, default 50)
   - Sequential scan ensures each composite participates in at most one reaction per step
   - Biased: low-index composites have priority (acceptable for determinism; can randomize if needed)
   - If no free composite slots remain, reaction is skipped (checked before creating products)

**Polarity blending:**
New function `blend_polarity(pol_a, pol_b, product_hash, mode)`:
- **mode="mean":** `(pol_a + pol_b) / 2` — offspring are intermediate
- **mode="xor":** `pol_a * pol_b` — opposite parents → neutral offspring
- **mode="hash_weighted":** hash bits determine interpolation weight
- **mode="nonlinear":** `sign(pol_a + pol_b) * sqrt(pol_a² + pol_b²)` — emergent states

Recommendation: use **"xor"** mode for interesting dynamics (opposite composites → neutral, stable offspring).

**Example reaction:**
```
Composite A: [sp1, sp3, sp5]  (3 members, BE=0.6, polarity=+0.8)
Composite B: [sp2, sp4]       (2 members, BE=0.4, polarity=-0.7)
Hash → num_products = 3

Product 0: [sp1, sp2]         (2 members, BE=0.55, polarity≈0.0 via xor)
Product 1: [sp3, sp4]         (2 members, BE=0.50, polarity≈0.0)
Product 2: [sp5]              (1 member, BE depends on hash, maybe becomes free particle)

Energy released: (0.6 + 0.4) - (0.55 + 0.50 + BE_prod2) = 0.05 units
→ Products kicked outward
```

**Energy conservation:**
- Total kinetic + binding energy before = after (within soft correction tolerance)
- Exergonic reactions (delta_BE > 0) release energy → faster products
- Endergonic reactions (delta_BE < 0) absorb energy → slower products
- Soft energy conservation rescales as needed to prevent drift

**Implementation complexity:**
- New function: `compute_reaction_products(c_a, c_b, config, physics)` → list of new composites
- New function: `blend_polarity(pol_a, pol_b, hash, mode)` → scalar polarity
- Modify `step.py`: add reaction scan before current fusion scan
- Reuse existing event sprite system (render different color for product spawns, e.g., magenta)

**Testing:**
- Composite diversity: count number of distinct composite types over time (should increase)
- Composite lineage: track parent-child relationships (implemented in Phase 4)
- Energy conservation: verify total energy stays bounded over long runs
- Product sizes: histogram of composite member counts (should see peak shift upward)

**Performance impact:** Moderate
- One reaction scan per step (max 50 reactions): O(50 * member_assignment)
- Member assignment is O(max_composite_size²) worst-case, but sparse in practice
- Total: ~0.5-1ms per step (acceptable, to be verified via profiling)

**Risks & mitigations:**
- **Combinatorial explosion:** If products proliferate too fast, composites list fills up
  - Mitigation: cap alive composites at `max_composites`; if full, reactions skipped
- **Stability collapse:** If reaction products are too fragile, composites die faster than forming
  - Mitigation: tune blend modes and hash functions; natural selection will stabilize
- **Energy drift:** If products' total BE ≠ parents' BE, energy can accumulate
  - Mitigation: soft energy conservation bounds drift; acceptable within 1-2% per 1000 steps

---

## Phase 4: Composite Heredity & Evolutionary Dynamics

**Objective:** Track composite lineages and enable inheritance of properties from parents, creating recognizable evolutionary patterns.

**Current state (after Phase 3):**
- Composites fuse/fission to produce multiple offspring
- Offspring properties are deterministic (from hash)
- But "inheritance" is implicit; hard to see parent-child relationships

**Design additions:**

**1. Parent tracking (optional metadata):**
- Add to `CompositeState`: `parent_ids: (max_composites, 2) int32`
  - `parent_ids[c, 0]` = first parent composite index
  - `parent_ids[c, 1]` = second parent composite index
  - `-1` if no parent (initial composites)
- Set on product creation in Phase 3 reaction scan
- Used for post-run lineage analysis, not sim logic

**2. Polarity inheritance modes:**
Implemented in `blend_polarity()`:
- **"xor" mode (recommended):**
  - `polarity_product = pol_a * pol_b`
  - Opposite parents (pol_a ≈ +1, pol_b ≈ -1) → neutral offspring (≈ -1, but clamped to [-1, 1])
  - Same polarity parents → polarized offspring
  - **Benefit:** Creates ecological niche: neutral composites are stable and inert; polarized ones are reactive and fragile

- **"mean" mode:**
  - `polarity_product = (pol_a + pol_b) / 2`
  - Offspring are intermediate between parents
  - Less extremal but more predictable

- **"hash_weighted" mode:**
  - Bits of product hash determine weight: `w = (H_product % 1000) / 999.0`
  - `polarity_product = (1-w) * pol_a + w * pol_b`
  - Adds variation even from same parent pair

**3. Stability through composition:**
- Certain member combinations have high BE → easy to form, unstable (fission easily)
- Others have low BE → hard to form, stable (persist long)
- Over generations, composites "discover" stable member sets
- Example: if [sp1, sp2, sp3] yields high half-life and neutral polarity, composites with these members persist and diversify

**4. Evolutionary feedback loop:**
```
composite properties (polarity, BE, half-life)
    ↓
interaction likelihood and fission energy
    ↓
which partners they meet and how energetically
    ↓
offspring properties (inherited via hash)
    ↓
selective pressure → stable lineages emerge
```

**5. Post-run analysis (not real-time):**
- Script: `scripts/analyze_lineages.py` (to be written later)
- Reads parent_ids from logged composites
- Reconstructs lineage trees: ancestor → descendants
- Computes:
  - Number of distinct lineages
  - Lineage depth (generations)
  - Phenotypic diversity (how different are composites from their lineage start?)
  - Fitness proxies (which lineages had most descendants?)

**Implementation:**
- Minimal sim changes: just set `parent_ids` on product creation
- No performance impact (metadata, not computed during sim)
- Logging: option to save CompositeState with parent_ids to HDF5 or pickle every N steps
- Analysis: Python post-processing script (not in JAX)

**Testing:**
- Lineage diversity: compute number of distinct parent pairs over time (should increase)
- Survival rates: some lineages should outcompete others
- Property correlation: do neutral composites live longer? (hypothesis from Phase 4 design)

**Risks & mitigations:**
- **Memory overhead:** parent_ids array is small (2 int32 per composite = negligible)
- **Analysis complexity:** requires post-run scripts; not real-time feedback
  - Mitigation: add live lineage display to HUD (e.g., "5 lineages, max depth 12")

---

## Testing & Validation Strategy

**Phase 1: Verification (no code changes)**
- Run existing simulator with instrumentation
- Measure: C+C fusion rate, BE distribution, composite size growth
- Success criteria:
  - C+C fusions occur (> 0 per step)
  - Max composite size > 10 (or understand why not)

**Phase 2: Composite Growth (minimal code changes)**
- Enable free particle absorption in fusion logic
- Run with Phase 1 instrumentation
- Measure: composite size growth, absorption rate, polarity changes
- Success criteria:
  - Max composite size increases significantly vs. Phase 1
  - No composites exceed `max_composite_size`
  - Energy remains bounded (< 10% drift)

**Phase 3: Multi-product Reactions (significant logic changes)**
- Implement reaction scan in `step.py`
- Run with Phase 1+2 instrumentation plus reaction tracking
- Measure: number of products per reaction, product sizes, energy conservation, reaction rate
- Success criteria:
  - Reactions produce 1-5 products (as designed)
  - Energy delta is < 5% per reaction (soft conservation corrects it)
  - Composite diversity increases over time
  - Frame time remains < 5ms (measure before/after)

**Phase 4: Heredity & Lineage Tracking (metadata only)**
- Add parent_ids logging
- Implement post-run lineage analysis script
- Run simulator, then analyze lineage trees offline
- Measure: number of lineages, depth, phenotypic diversity
- Success criteria:
  - Multiple lineages emerge
  - Lineage depth > 5 (non-trivial evolution)
  - Lineages have recognizable "style" (family resemblance)

**Long-run emergent behavior test (all phases):**
- Run simulator for 10,000+ steps with all phases enabled
- Observe for:
  - Self-organizing structures (clusters of compatible composites)
  - Autocatalytic loops (composites that enable each other's formation)
  - Evolutionary arms races (polarity dynamics driving competition)
  - Niche specialization (different composite types in different regions)

---

## Performance Considerations

**Current baseline:** ~3.9ms per step (2K particles, Phase 0 code)

**Expected impact by phase:**

| Phase | Operation | Complexity | Est. cost | Cumulative |
|-------|-----------|-----------|-----------|-----------|
| 1 | Instrumentation only | O(1) | < 0.1ms | 3.9ms |
| 2 | Free particle absorption | O(N * K) | ~0.2ms | 4.1ms |
| 3 | Reaction scan + member assignment | O(R * M²) | ~0.8ms | 4.9ms |
| 4 | Parent tracking (metadata) | O(1) | < 0.1ms | 5.0ms |

**Assumptions:** R = 50 reactions/step, M = 8 avg composite size, K = 16 neighbors

**If target framerate is 60 FPS:** 16.7ms per frame, allowing ~10-12 simulation steps before rendering.
At 5ms/step → 3 steps/frame, stable 20 FPS (acceptable for long-run observation).

**Profiling strategy:**
- Phase 1: use existing profiler harness (PLAN.md Phase 6)
- Phase 2: re-run profiler, measure free particle absorption cost
- Phase 3: add reaction scan to profiler, measure worst-case (max reactions)
- Phase 4: no profiling needed (metadata only)

**Optimizations if needed:**
- Compact alive composites before reaction scan (same as Phase 6 optimization)
- Vectorize member assignment (currently scalar per product)
- Cache hash values (reuse if same hash appears multiple times)

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Unbounded composite growth** | Composites fill max_composite_size, new growth blocked | Size cap enforced; natural selection will form small stable composites |
| **Composite pool fills** | Reactions can't spawn products; evolutionary stall | Cap alive composites at max_composites; use soft limit with death pressure |
| **Energy drift** | Total energy accumulates or disappears | Soft energy conservation already present; Phase 3 reactions tested for <5% delta |
| **Polarity blend instability** | Some blend modes create fragile composites | Test multiple modes (xor, mean, etc.); natural selection filters instability |
| **Reaction combinatorial explosion** | Too many products, performance collapse | Cap num_products at 5; cap reaction rate at max_reactions_per_step |
| **Lineage metadata bloat** | parent_ids array grows memory footprint | Small cost (2 int32 per composite); acceptable |

---

## Scope & Out of Scope

**In scope:**
- Composite growth through free particle absorption
- Multi-product composite-composite reactions
- Deterministic product generation via hash functions
- Polarity inheritance and blending
- Lineage tracking and post-run analysis
- Energy conservation with soft bounds

**Out of scope (Phase 5+):**
- Negative binding energy (endergonic reactions with activation barriers)
- NCA-style learned internal dynamics on composites
- Mass-conserving advection (FlowLenia inspiration)
- Interactive parameter tuning (sliders for reaction rates, blend modes, etc.)
- Real-time lineage visualization during simulation

---

## Implementation Order

1. Phase 1: Add instrumentation and profiling (days 1-2)
2. Phase 2: Implement free particle absorption (days 2-3)
3. Phase 3: Implement multi-product reactions (days 3-5)
4. Phase 4: Add parent tracking and lineage analysis (days 5-6)
5. Long-run testing and emergence observation (days 6+)

Each phase is independently testable and does not block the next (except Phase 1 results informing Phase 2 tuning).

---

## Conclusion

This design provides a clear, incremental path to composite-level evolution while maintaining performance and reproducibility. The use of hash functions ensures determinism (same compositions always produce same properties), while allowing rich variation through heredity and selective pressure. The four phases build toward open-ended emergent behavior: from verification, to growth, to reactions, to heredity.

Success looks like: composites forming diverse lineages, exploiting energetic/spatial niches, adapting to their environment through survival and fission of offspring with inherited properties, and ultimately displaying recognizable evolutionary dynamics at the composite level.
