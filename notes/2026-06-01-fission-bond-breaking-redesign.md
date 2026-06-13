# Fission & bond-breaking redesign (chemical + nuclear channels)

**Date:** 2026-06-01
**Topic:** Root-caused the "long-distance bonds" bug, then designed a principled bond-breaking / fission overhaul. Mid-brainstorm — design not yet written up as a spec. This note lets a cold session resume.

**Status (2026-06-12):** Implementation underway — see
`docs/superpowers/plans/2026-06-12-fission-bond-breaking-redesign.md`.
Commit 1 (bond-cut fission) landed; benchmark: 29.1 → 23.2 steps/s (1.25×),
max bond length 4.1 (single-digit, bug gone).

## Summary

User reported routinely seeing very long bonds (spanning much of the screen) despite `fusion_radius=1.5`. Investigation **confirmed the root cause**: fission rebuilds each product's bond graph as a path through members *in slot order, with no spatial constraint*, minting edges up to ~34 units long. We then brainstormed a full redesign where bond breaking becomes physical, splitting into two orthogonal channels — **chemical** (per-bond scission) and **nuclear** (whole-body fission via a liquid-drop model). A series of physics-design decisions are locked in (below). We were about to present the full 3-commit design when the user paused to take notes. **Nothing has been implemented yet.**

## The bug (root cause — confirmed with evidence)

[`_path_edges_from_members`](halflife/chemistry.py#L507), called by [`apply_composite_decay`](halflife/chemistry.py#L371), rebuilds a fission product's edges as `(members[k], members[k+1])` in **compacted slot order**. Slot order is *not* spatially coherent (composites grown by comp+comp merges concatenate distant regions; repeated fission scrambles it further), so consecutive-in-slot members can be far apart → long bonds. [`_hash_to_partition`](halflife/chemistry.py#L310) also assigns members to products by hashing slot indices, ignoring topology.

**Why it's visible / persistent:** the harmonic edge spring ([step.py:117](halflife/step.py#L117) `compute_edge_bond_forces`) can't break — it just keeps pulling — and `max_velocity=8`, `dt=0.06` cap closing speed at ~0.48/step, so a freshly-minted long bond takes ~30+ steps to relax. Since *everything decays*, fission continuously replenishes them → steady-state population of long lines.

**Ruled out (all verified sound):** fusion edge creation (gated by `fusion_radius`, connects contacting pair), ring closure (distance-gated), the edge-spring force (masks by `alive & e_idx<count`, min-image), the renderer (min-images bond lines — long lines are *real*, not wrap artifacts), edge integrity (0 edges reference non-member particles).

### Evidence runs (throwaways in /tmp, CPU, seed 0, default config)
- Edge-length distribution at 800 steps: mean ~1.5, but a tail — max grew 10.8 → 27.6 over 800 steps; long edges (>8) live **only** in large composites (size 48–127), never small ones.
- Decisive test: ran the *actual* fission code path over all alive composites → longest edge fission *would* mint = **34.1 units** (composite 61, size 78). Confirmed fission is the creator. (A single-composite test initially mis-refuted this because the largest composite happened to have spatially-coherent slot order — had to test all composites.)

## Design decisions locked in (the physics)

The unifying insight: **harmonic springs are bottomless wells** — infinite depth, so bonds can *never* break by being pulled. That single fact causes both the long-bond bug *and* the user's larger gripe that "bonds only break via half-life, never kinetically." Real bonds are **Morse-like** (finite dissociation energy). Fixing the well depth fixes both.

Two **orthogonal** breaking channels (user explicitly wants to keep both):

| | **Chemical (per-bond)** | **Nuclear (whole-body)** |
|---|---|---|
| Trigger | one bond's strain/energy crosses its dissociation threshold | the whole composite is collectively unstable |
| Driven by | local bond energy `E_b` + local strain | global balance: cohesion vs internal repulsion + size |
| Event | sever that one bond | drop necks & splits into two large fragments |
| Analog | molecule loses an H | U-235 spontaneously fissions |

Decisions, in the order settled:

1. **Keep both channels** (not replace half-life). They're distinct failure modes, not redundant.
2. **Half-life derived from aggregate of the composite's bond energies** (cohesion) — same hash chemistry, routed through bonds. One energy scale across both channels.
3. **Whole-body fission = nuclear liquid-drop model.** Stability = competition of cohesion (bond aggregate + `A^2/3` surface term) vs **disruption** (internal member–member repulsion, the Coulomb analog — *already computed in the force kernel*) + size. Fission **rate** ← fissility `x = E_repulsion / (2·E_cohesion)`; half-life collapses as `x→1`. This replaces the ad-hoc `composite_size_decay_scale` with a principled "big/repulsive things fission" law.
4. **Fracture rule = max fragment stability (shell-effect analog).** *Key unification:* the hash-BE landscape **is** the sim's "shell structure / magic numbers" — some multisets are hash-favored (high BE). Fission fractures along the bond-cut that **maximizes total product binding energy**. Gives emergent **asymmetric** fission (favoring a "magic" fragment), makes fission/fusion inverses on one energy landscape, and — because the cut is *along bonds* — keeps fragments connected with short bonds (the bug fix). For a tree, candidate cuts = the `n−1` edges; additive commutative hash lets all cuts be scored in one `O(M)` subtree-sum pass.
5. **Fission energy = Q-value.** Kick energy = `[BE(frag0)+BE(frag1)] − BE(parent)` — the binding *gained*. Self-consistent with the max-stability fracture (best cut also releases most energy). Replaces `fission_cost`. Endothermic splits (Q<0) release no kick and arguably shouldn't fire spontaneously (barrier). Soft-conservation corrector ([energy.py](halflife/energy.py)) absorbs drift.

## Proposed 3-commit decomposition (NOT yet confirmed with user)

- **Commit 1 — fission fracture fix (the bug).** Replace `_path_edges_from_members` arbitrary path with **bond-cut fission**: max-fragment-stability cut + connected-components split, keeping surviving intra-fragment bonds. Triggered by the *existing* half-life mechanism (unchanged formula for now). Q-value kick. Kills long bonds. **Includes a micro-benchmark of the connected-components labeling pass** — it's the load-bearing perf assumption for steps 2–3.
- **Commit 2 — per-bond chemical scission.** Hash-derived per-bond dissociation energy `E_b` (decorrelated, like [`_hash_to_rest_length`](halflife/chemistry.py#L115)). Kinetic break (strain > `E_b`) + thermal break (prob ∝ `exp(−(E_b−strain)/kT)`). Broken bond → connected components → spin off fragments. Reuses commit 1 machinery.
- **Commit 3 — liquid-drop nuclear stability.** Replace half-life formula with the fissility rate law (cohesion vs internal repulsion + size); fold `composite_size_decay_scale` in. Optionally swap harmonic → Morse force so kinetic breaking is intrinsic to the potential.

## Compute analysis

Array sizes: `N=5000`, `max_neighbors=256`, `max_composites=3000`, `max_composite_size=128`, `e_max=256`. Baseline hot path ~1.3M pairwise/step; **sim is GPU latency/memory-bound (~30 steps/s), not flop-bound** — so op-counts mislead on wall-clock.

- **Cheap:** chemical kinetic check piggybacks on `compute_edge_bond_forces` (already has bond length `r`) → ~free. Thermal `exp` over `C·e_max≈768K` → ~0.5× a force pass. Fragment scoring `O(M)` via additive hash → cheap. **Coulomb term: reuse the force kernel's member–member pairwise work** (members are neighbors) → ~free (standalone it'd be `O(M²)·C ≈ 49M`/step).
- **The one real cost: connected-components labeling.** Static-shape label propagation = `M × e_max × C ≈ 98M` ops/step naive (~75× the force kernel in op-count). Mitigations: (a) parallel vmap + latency-bound GPU → realistic wall-clock hit likely **1.5–3×**, not 75×; (b) **cap propagation iterations at K≈16–32** — safe *because* the bug fix makes composites compact (small diameter). Net realistic target: **<2× slower once tuned.** Needs measurement, hence the commit-1 benchmark.

## Open questions / where we left off

- **Next action:** present the full 3-commit design for approval (was about to when paused). Commit decomposition above is proposed, not confirmed.
- Morse force vs. harmonic+threshold for kinetic breaking — flagged as "eventual" (commit 3 optional), not decided.
- Thermal rate law exact form and the global temperature `T` knob — not finalized.
- Connected-components iteration cap `K` — concrete value TBD (depends on post-fix composite diameter; measure).
- Chemical channel breaking *multiple* bonds across composites in one step needs general (multi-fragment) component labeling, not just binary — confirm scope.
- Should endothermic whole-body fission be fully forbidden (barrier) or just kick-less? Leaning forbidden.

## Nubs
- ?? `r_rest` band changed mid-session to `[repulsion_radius, fusion_radius]`; `r_rest_min/max` removed from config — user was actively editing `chemistry.py`/`config.py`. Re-confirm current bond rest-length code before implementing.
- benchmark the component-labeling pass in isolation before building commits 2–3 on it
- check `max_composites=3000` headroom: emergent multi-fragment fission could spawn more composites/step than today's binary split — watch free-slot exhaustion
- "magic number" tuning: which `hash_modulus` / `num_species` actually produce a rugged-enough BE landscape for interesting asymmetric fission?
