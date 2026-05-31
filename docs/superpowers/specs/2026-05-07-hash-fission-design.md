# Hash-determined fission — design specification

**Date:** 2026-05-07
**Status:** Design review
**Supersedes (in part):** [2026-03-26-composite-reaction-network-design.md](2026-03-26-composite-reaction-network-design.md) — Phases 3 (multi-product reactions) and 4 (heredity/polarity blending) are replaced by this simpler design. Phase 2 (composite growth via free-particle absorption) is already implemented in the unified fusion path.

## Goal

Make composite **fission** (decay) the primary engine of reaction-network richness. The hash already determines composite identity; extend it to determine *how* a composite breaks apart. Combined with the existing fusion mechanic, this gives a complete hash-defined reaction network — no hand-tuned rules, no transmutation, particles fully conserved.

This commits the project to **path 1** (hash-chemistry maximalism, see [notes/2026-05-05-boring-dynamics-investigation.md](../../../notes/2026-05-05-boring-dynamics-investigation.md)). Polarity machinery is removed in the same change.

## Core design

When a composite with `n` members and species hash `h` decays, the hash determines:

1. **K** — the number of products, an integer in `[2, n]`.
2. **assignment[i]** — for each member slot `i ∈ [0, n)`, which product `k ∈ [0, K)` it goes to.

After fission:
- Each product collects its assigned members.
- Products of size 1 → free particles (`composite_id = -1`).
- Products of size ≥ 2 → new composites occupying free composite slots, with hash recomputed via the existing commutative sum.
- Particle species are **never** modified. Only their bonding state and which composite they belong to.

Same composite hash → same K and same assignment → same fission outcome every time. This is what makes it "chemistry."

### Worked example

Composite of 5 members with species `[A, A, B, C, D]` and hash `h_AABCD`:

```
hash → K = 2
hash → assignment = [0, 1, 0, 0, 1]   (members go to product 0 or 1)

Product 0: members [A, A, C]   → new composite (hash = sum of A+A+C entity hashes)
Product 1: members [B, D]      → new composite (hash = sum of B+D entity hashes)
```

Two new composites. Particle count and species composition are exactly preserved.

A different hash for the same composite shape might give:

```
hash' → K = 3
hash' → assignment = [0, 1, 2, 0, 1]

Product 0: [A, C]   → composite
Product 1: [A, D]   → composite
Product 2: [B]      → free particle
```

Different "universe" (different `hash_modulus`) → different fission rules entirely.

## Design decisions

### (a) Distribution of K — start with **binary fission**

`K = 2` always, for the first iteration. Justifications:
- Real-world precedent: nuclear fission, cell division, radioactive decay are all overwhelmingly binary.
- Easier to reason about, debug, and analyze. Heavy-tailed multi-product splits can be added later without breaking the framework.
- Reduces the design surface — no need to decide "how is K distributed" yet.

If binary feels too restrictive after testing, generalize to `K ∈ [2, min(n, max_decay_products)]` with `K = 2 + ((h >> 16) % min(n-1, max_decay_products-1))`.

### (b) Both products must be non-empty

A binary split with `assignment` derived from hash bits could produce `[0, 0, 0, ..., 0]` — all members in product 0, product 1 empty. That degenerates to "didn't decay." Enforce non-empty splits:

1. Sort members by `hash_mix(h, member_slot_index) % LARGE_PRIME`.
2. Assign the first `pivot` members to product 0, the rest to product 1, where `pivot ∈ [1, n-1]` is hash-derived.
3. The pivot itself can be: `pivot = 1 + ((h >> 12) % (n - 1))` — uniformly anywhere in `[1, n-1]`.

This guarantees both products have ≥1 member.

### (c) K = 1 is reserved — means "didn't decay this step"

The half-life roll already decides whether a composite decays this step. If a roll passes, fission must produce ≥ 2 products. K = 1 doesn't appear in this design.

### (d) Geometry — products fly apart as cohesive groups

Currently fission gives every member a radial kick from the parent's COM. With multi-product fission:
- Compute each product's center of mass (using min-image displacement, as the bond-force code already does).
- Compute the **vector between product COMs**.
- All members of product 0 get a velocity kick along (COM_0 − COM_1); all members of product 1 get the opposite.
- Magnitude per product: `v = sqrt(2 · E_kick / M_product)` where `E_kick = parent.binding_energy · (1 − fission_cost) / K` and `M_product = sum of member masses`.

This preserves momentum (equal-and-opposite kicks weighted by mass) and gives each product a coherent COM velocity — they actually fly apart as units, rather than each member scattering independently.

The springs hold each product together internally (existing bond-force machinery, no change needed).

### (e) Drop polarity entirely in the same change

Per path-1 commitment: polarity is hand-tuning that fights the hash-chemistry direction. Remove:

- `InteractionParams.polarity` field
- `CompositeState.net_polarity` field
- `PhysicsParams.polarity_fusion_scale`, `polarity_stability_scale`
- `SimConfig.polarity_fusion_scale`, `polarity_stability_scale`
- The `attr_mod` machinery in [step.py:175-184](../../../halflife/step.py#L175-L184)
- The polarity bonus on binding energy in `find_entity_partner` ([chemistry.py:319-330](../../../halflife/chemistry.py#L319-L330))
- The neutrality boost on half-life in `fusion_scan_body` ([chemistry.py:446-458](../../../halflife/chemistry.py#L446-L458))
- The two slider rows for polarity in the UI

Pairwise force becomes pure Particle-Life: `f = particle_life_force(r, attraction[si, sj], ...)`. No multiplicative scaling.

### (f) Drop species count to 12 for the rebuild

Default `num_species` from 64 → 12. Rationale (from prior session):
- 64 species washes out PL force kernel signal — every neighborhood contains all species in random proportion.
- 12 species gives ~150 possible reactions (counting fusion + fission outcomes) — large enough for emergence, small enough to actually study and visualize.
- Easier to detect autocatalytic loops in data.
- Trivially scalable later if dynamics feel sparse.

## State changes

### Removed

- `CompositeState.net_polarity` (see (e))
- `InteractionParams.polarity`
- `PhysicsParams.polarity_fusion_scale`, `polarity_stability_scale`
- `SimConfig.polarity_fusion_scale`, `polarity_stability_scale`
- (Already commented out in 98abb0f, can now be deleted) `_hash_to_half_life`, `hash_multiset`, `hash_scalar`

### Repurposed

- `_hash_to_decay_products` → renamed `_hash_to_partition`. New signature:
  ```python
  def _hash_to_partition(h: jnp.ndarray, n_members: jnp.ndarray, config) -> jnp.ndarray:
      """
      Returns assignment: (max_composite_size,) int32, with values in [0, K).
      For binary fission, K = 2. Slot i's value is which product member i belongs to.
      Slots beyond n_members are -1 (unused).
      """
  ```
  - Computes pivot from hash bits.
  - Sorts member slot indices by `hash_mix(h, slot_index)`.
  - Assigns first `pivot` slots to product 0, rest to product 1.

### Modified

- `apply_composite_decay` in [chemistry.py:125](../../../halflife/chemistry.py#L125):
  - For each fissioning composite, compute partition assignment.
  - Allocate `K - 1` new composite slots (one product reuses the parent's slot, the rest need fresh slots) via `find_free_slots`.
  - Group members by assignment.
  - For each product:
    - If size = 1: set member's `composite_id = -1` (free particle).
    - If size ≥ 2: write members to a composite slot, set `member_count`, recompute `species_hash` via commutative sum, derive `binding_energy` via `_hash_to_binding_energy`, derive `half_life` from BE (existing formula minus the polarity-stability term).
  - Apply velocity kicks to all members based on their product's COM-vs-other-COM direction (per (d)).
  - Mark the parent composite as dead.

### `SimConfig` — new

- Keep `max_decay_products` as the upper bound (currently dead, becomes live again). Default 2 to enforce binary fission. Bump to 3+ to allow higher-K fission later.

## Algorithm sketch

```python
def apply_composite_decay(state, config):
    # 1. Roll for decay (existing) — produces 'fissions' boolean mask over composites
    fissions = composites.alive & (rng < decay_prob)

    # 2. For each fissioning composite, compute partition
    def compute_split(c):
        h = composites.species_hash[c]
        n = composites.member_count[c]
        assignment = _hash_to_partition(h, n, config)  # (M,) int32, values 0..K-1 or -1
        return assignment

    all_assignments = vmap(compute_split)(arange(max_composites))  # (C, M)

    # 3. Allocate fresh composite slots for product k=1 of each fissioning composite
    # (k=0 reuses the parent slot)
    n_new_slots_needed = sum(fissions)
    new_slots = find_free_slots(composites.alive, max_composites)  # already-allocated machinery

    # 4. For each fissioning composite, populate two products
    def write_products(c, slot_for_k1):
        if not fissions[c]: return no-op
        assignment = all_assignments[c]
        members = composites.members[c]
        n = composites.member_count[c]

        # Product 0 → reuse slot c
        product_0_members = members where assignment == 0
        product_0_size = count
        write to composites at slot c: members, member_count, hash, BE, half_life

        # Product 1 → use slot_for_k1
        product_1_members = members where assignment == 1
        ... same write ...

        # Compute COMs of each product (min-image), apply velocity kicks
        ...

        # Update composite_id of all member particles
        ...

    # 5. Mark parent composites as dead — actually for K=2 the parent's slot becomes product 0,
    #    so don't kill it; only kill it if a product is empty (which can't happen by (b))
```

The `K-1` extra slots needed per fission is exactly the same allocation pattern as fusion (which allocates 1 slot per free+free pair). The `find_free_slots` machinery already supports it.

## What this gives us — and what it doesn't

**What it gives:**
- A real reaction network. Composites have deterministic decay paths. Selection of stable composite types over time. Autocatalytic cycles possible (composite types that fission into pieces that re-form themselves).
- Particle conservation. No species can dominate the pool because particle species are immutable.
- A clean, hash-defined "universe": changing `hash_modulus` changes every reaction in the system at once.
- One coherent system instead of two half-committed ones (PL + polarity).

**What it does not give:**
- Heredity in the traditional sense. There's no parent metadata, no lineage tracking, no "DNA." If you want that later, the old design doc's Phase 4 (parent_ids on CompositeState) is a clean extension.
- Cross-composite reactions like `Ca + Cb → Cx + Cy + Cz`. Currently only fission-time multi-product is in scope; if two composites collide and fuse, they make ONE bigger composite (existing fusion code), which can later fission into pieces. Direct C+C → multiple products is in the old design but is an additional layer of complexity for marginal gain — skip for now.

## Open questions

1. **Pivot distribution.** A uniform `[1, n-1]` pivot on a 5-member composite gives equal probability to splits {1+4, 2+3, 3+2, 4+1}. That seems fine. But for very large composites (n > 10), uniform pivot gives lots of asymmetric splits — is that desirable? Maybe bias toward symmetric (`pivot ≈ n/2`) for nuclear-fission-like behavior. *Default: uniform. Revisit if the size distribution looks wrong.*

2. **Recursive fission.** A 4-member composite fissions into a 3-member + 1 free. The 3-member then has its own half-life; eventually it fissions too. Is this what we want, or should fission cascade in one step? *Default: cascade across steps. The 3-member behaves like any other composite — it'll fission when its own dice come up.*

3. **Energy budget.** Currently `binding_energy * (1 - fission_cost)` becomes kinetic energy on fission. With multi-product fission, do all products share that pool, or does each product get its own pool from "its share" of the parent's BE? *Default: total parent BE × (1 − fission_cost) is split equally among the K products as kinetic kick. Simple, conservation-friendly.*

4. **Should `find_entity_partner` still allow C+C fusion?** Yes — necessary for composites to grow beyond size 2. C+C fusion is unchanged by this design; only decay changes. C+C fusion produces ONE product (the merged composite); fission later breaks it back into pieces.

5. **What replaces the polarity-stability bonus on half-life?** It currently lengthens the half-life of neutral composites. Without it, half-life is purely BE-driven. That's fine — the prior audit noted that the polarity bonus *compounded* the inertness problem anyway. *Default: drop it, no replacement.*

## Testing & validation

After implementation, before declaring success:

1. **Conservation check.** Total count of particles per species must be exactly constant across thousands of steps. (Currently it is, trivially. After this change it should still be — make sure fission doesn't lose particles to padding/overflow.)

2. **Determinism check.** Same composite (same member set) decaying twice in different runs (with the same RNG-seeded "did it decay" roll) should produce identical products.

3. **Reaction-network observability.** Add a counter for "distinct composite types observed" (where "type" = sorted member-species tuple). Should grow over time and approach some saturation governed by hash and species count.

4. **Autocatalytic loop detection.** Track composite types that appear, disappear, and reappear. Stable autocatalytic loops show as composite types whose population persists despite individual composites dying — i.e., the type's birth rate ≈ its death rate.

5. **Visual sanity.** With 12 species and binary fission, you should see a population of small-to-medium composites (sizes 2–8) with constant flux: composites forming via fusion, breaking via fission, sometimes reforming into similar shapes. The population of distinct composite types should be much smaller than the theoretical max — selection in action.

6. **No runaway behavior.** Populations of any single composite type shouldn't asymptote to 100%. If they do, the hash is producing self-replicating fixed points that don't generate diversity — interesting but a sign that the hash needs different constants.

## Implementation order

Suggested sequence (each step is independently testable):

1. **Cleanup pass.** Delete commented-out dead code (`hash_multiset`, `hash_scalar`, `_hash_to_half_life`). Delete polarity from `InteractionParams`, `CompositeState`, `PhysicsParams`, `SimConfig`. Delete `attr_mod` machinery in `step.py`. Delete polarity bonuses in `chemistry.py`. Delete polarity sliders in UI. Run tests.
2. **Reduce species.** Set `num_species = 12`. Run, sanity-check.
3. **Implement `_hash_to_partition`.** Pure function, unit-testable on its own. Verify it produces non-empty splits, deterministic for same hash.
4. **Modify `apply_composite_decay`.** Wire `_hash_to_partition` into the fission pipeline. Allocate slots, populate two products, kick each product's COM. Run, watch for: particle count conservation, reasonable composite sizes, no NaN velocities.
5. **Visual verification.** Watch the live sim. Composites should be forming, fissioning into pieces of various sizes, and refusing/reforming over time. The composite-size histogram should be much more dynamic than the static lattice from before.
6. **Quantitative tests.** Run the conservation, determinism, and observability checks above.

After all six steps pass, the rebuild is complete and you can decide whether to layer further features (heredity tracking, multi-product fission, sparse interaction matrix) on top.

## Estimated scope

About a 200-line diff. Most of it is *deletions* (polarity machinery). The new logic in `apply_composite_decay` is ~50 lines. `_hash_to_partition` is ~20 lines. Plus the cleanup of `InteractionParams`, `CompositeState`, `PhysicsParams`, and the UI panel.

No new tests would be required for the cleanups themselves — the existing test suite (now passing per `c6a917a`) should catch regressions. New tests would only be needed for `_hash_to_partition` and the new fission behavior.
