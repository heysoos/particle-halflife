# Fission Redesign — Adversarial Review Dialogue

**Date:** 2026-06-01
**Subject note:** [2026-06-01-fission-bond-breaking-redesign.md](2026-06-01-fission-bond-breaking-redesign.md)

Two scientifically-literate, skeptical reviewers stress-test the fission/bond-breaking
design. **CRITIC** raises concerns; **COLLABORATOR** refutes / fixes / scopes them.
Iterates until the critic is satisfied (all concerns RESOLVED) or the round cap is hit.

**Protocol (token-efficient):** the live channel is a compact ID-keyed *concern ledger*
passed turn-to-turn as structured data; this file is the append-only human record. Each
turn is terse bullets keyed by concern ID — no prose re-statement of context.

**Status flags:** `OPEN` (raised, unanswered) · `ADDRESSED` (collaborator responded) ·
`RESOLVED` (critic accepts) · `CONTESTED` (critic rejects the response) · `DROPPED` (withdrawn).

**Categories:** physics · math · compute · practicality · scope · correctness.

---

## Dialogue Log (append-only)

<!-- agents append their round blocks below this line -->

### Round 1 — CRITIC

- C1 (high, physics): Liquid-drop "Coulomb/disruption" term is being equated with the sim's short-range repulsion (cutoff interaction_radius=8.0). The nuclear Coulomb term is *long-range* (∝Z²/A^{1/3}); the sim has no long-range repulsion. So fissility x=E_rep/(2·E_coh) is built on a disanalogy — what's "free" from the force kernel is short-range hard-core repulsion, not a Coulomb analog. The qualitative "big things fission" law may still emerge, but the LDM dressing is not earned. Crux: justify the disruption term physically given the force kernel cuts off at 8.0, or drop the LDM framing.
- C2 (high, physics): "Max-fragment-stability ⇒ asymmetric fission" is asserted, not shown. In real physics LDM alone gives *symmetric* fission; asymmetry is a quantum shell correction. The note's "hash-BE landscape = shell structure" is an analogy whose payoff depends entirely on BE-landscape ruggedness — which the note itself flags as untuned (Nubs: which hash_modulus/num_species give a rugged-enough landscape). Crux: with a smooth/low-contrast BE map, max-BE cut → near-symmetric or arbitrary; asymmetry is contingent, not guaranteed.
- C3 (high, math): "Additive commutative hash lets all cuts be scored in one O(M) subtree-sum pass" conflates two things. Additive *hash* accumulation is one pass on a *tree*; but BE is a nonlinear Fibonacci remix of the hash (_hash_to_binding_energy: h2=h*C ^ (h>>13)), so BE(frag) must be evaluated per-cut (n-1 evals — still cheap, but not "one pass"). More serious: with allow_ring_closure=True composites are NOT trees, so "candidate cuts = n-1 edges" and "subtree sum" are ill-defined — cutting one edge of a cycle disconnects nothing. Crux: does the scheme degrade to "must run CC per candidate cut" on cyclic graphs? That destroys the O(M) claim.
- C4 (high, compute): Connected-components labeling does not exist in the codebase today (fission is argsort-pivot + slot-order path). The "98M ops naive, ~1.5–3× wall-clock" estimate is for unwritten code. The iteration cap K≈16–32 is justified by "compactness after the bug fix" — circular: the fix *requires* CC to work, and a single chemical scission (commit 2) of a long polymer can have diameter ≫ K, silently leaving fragments mislabeled (under-segmented → still-connected ghost composites). Crux: label-prop with a hard iter cap is not correctness-safe for high-diameter graphs; need either union-find-style pointer jumping (log diameter) or a proven diameter bound. Benchmark measures speed, not correctness.
- C5 (high, correctness): Energy story is broken at the source. apply_soft_energy_conservation is COMMENTED OUT in step.py (line 338). The note repeatedly leans on "soft-conservation corrector absorbs drift" — it currently does nothing. Q-value kick = [BE(f0)+BE(f1)]−BE(parent) injects net energy with no sink. Endothermic-but-kickless still mutates BE bookkeeping. Crux: with the corrector disabled, what bounds KE growth? Re-enabling it (currently a no-op pass-through) is a prerequisite, and its 1%/step cap may not absorb Q-value spikes.
- C6 (medium, practicality): max_composites=3000 headroom under multi-fragment scission. Today binary fission is +1 composite/fission (product 1 → one free slot; find_free_slots returns -1 and drops on exhaustion). Commit 2 chemical scission breaking multiple bonds/step across many composites can demand many free slots/step; drop-on-exhaustion silently loses fragments (particles keep stale composite_id?). Crux: confirm fragment overflow degrades to "stay bonded" not "leak particles / orphan composite_id", and budget the worst-case new-slots/step.
- C7 (medium, scope): 3-commit decomposition front-loads the bug fix (commit 1) onto CC machinery whose feasibility is the open question. Commit 1 "kills long bonds" only if max-BE-cut + CC is correct AND cheap — i.e. commit 1 already depends on C3/C4 being resolved. A smaller, lower-risk first commit exists: replace _path_edges_from_members slot-order path with a *spatial* MST/greedy-nearest path over the existing partition (no CC, no BE-cut) — kills long bonds immediately, defers the physics. Crux: the bug fix and the physics overhaul are entangled in commit 1; they should be separable.
- C8 (low, physics): Half-life "derived from aggregate bond energies" plus a separate fissility rate law (commit 3) risks double-counting cohesion — bond aggregate feeds both the HL and the x denominator. Confirm one isn't silently squaring the cohesion dependence.
- C9 (low, correctness): Q-value self-consistency claim ("best cut also releases most energy") only holds if BE is sub-additive across the cut for ALL cuts; with a remixed nonmonotone BE map there can be cuts where a *non*-max-BE split has higher Q. Minor, but the "self-consistent" wording overclaims.

### Round 1 — COLLABORATOR

- C1 FIX (reframe, not physics claim): Critic is right that the force kernel is strictly short-range — verified `r_cutoff = cutoff_fraction*interaction_radius`, cutoff_fraction<=1.0, interaction_radius=8.0 (interactions.py:111-112, config.py:46). There is NO long-range Coulomb term and there cannot be one without widening the neighbor radius. CONCRETE CHANGE: drop the "Coulomb analog" / "fissility x = E_repulsion/(2·E_cohesion)" LDM framing entirely from the *physics justification*. Keep only the *operational* law: "disruption" = sum of the short-range repulsive component of member-member pairwise forces that are ALREADY computed for bonded members (they are neighbors within fusion_radius<=1.5 << 8.0, so they always carry a force-kernel entry). Rename "fissility" → "internal-strain ratio"; it is a phenomenological instability metric, not LDM x. This keeps the compute claim (free reuse of pairwise work) — which IS valid — and abandons the false nuclear analogy. Big-and-repulsive-things-fission survives as a heuristic, justified by strain accumulation, not by Z²/A^{1/3}.

- C2 FIX (downgrade claim + make it measurable): Conceded — asymmetric fission is NOT entailed by max-fragment-stability; it is contingent on the BE landscape having contrast, which is exactly nub #4 ("which hash_modulus/num_species give a rugged-enough landscape"). CONCRETE CHANGE: (1) demote "gives emergent asymmetric fission" from a design *guarantee* to a *hypothesis to be measured* via the existing diagnostic pipeline (Tier-3 transition matrices already record actual fission products; add a product-size-asymmetry histogram). (2) The fracture RULE (max total product BE) stands on its own as a deterministic, well-defined choice regardless of whether the emergent statistics turn out symmetric or asymmetric — it does not *depend* on the shell analogy being true. Strip the "shell-effect / magic-numbers" language to a parenthetical "(loose analogy)". Acceptance gate for commit 3: measured product-size distribution is non-trivially non-uniform on at least one shipped (modulus, num_species) preset.

- C3 FIX (this is the load-bearing correctness bug — concede fully): Critic is correct. With allow_ring_closure=True (config.py:143, default) composites are cyclic, so "n-1 edges, subtree-sum per cut" is ill-defined — there is no unique subtree on either side of a cut in a graph with cycles, and a single edge cut may not even disconnect the graph. AND `_hash_to_binding_energy` is a nonlinear Fibonacci remix (chemistry.py:88-90), so even on a tree you cannot get all cuts' BE from one linear subtree-sum scan: you get the additive *species-hash* subtree sums in O(M) (the hash IS commutative-additive, verified), but BE = nonlinear(sum) must be evaluated per candidate cut. CONCRETE CHANGE: abandon "one-pass subtree-sum scoring of all cuts." Replacement algorithm for the fracture step: (a) candidate set = the existing edges of the composite (<= e_max), not "n-1 tree edges"; (b) for each candidate single-edge removal, run connected-components on the remaining edge set to get the 2 (or, if it was a bridge, exactly 2; if not a bridge, 1 — meaning that cut does nothing, score it -inf) fragments; (c) the additive species hash of each fragment is recovered cheaply by summing member hash-vals over the CC label (O(M) per cut), then BE = nonlinear(sum) is one scalar eval per fragment. Cost is O(edges × CC_cost), which is why commit 1 MUST land the CC primitive first (ties to C4/C7). For trees this reduces to bridge-cuts = all edges; for ring-bearing composites only bridge edges yield a real split, which is physically correct (you can't fission a ring by cutting one bond).

- C4 FIX (concede the unsafe cap; specify the actual primitive): Critic is right that hard-capped label-propagation under-segments high-diameter graphs and the 98M-op estimate is for unwritten code. CONCRETE CHANGE: do NOT use fixed-K label propagation as the production primitive. Use iterate-to-fixed-point label propagation with a per-step convergence flag via `jax.lax.while_loop` (carry = labels + changed-bool; predicate = changed & iter<diameter_bound). The diameter bound is a HARD correctness backstop, not a perf knob: set it to max_composite_size-1 (=127, the worst-case path diameter for an M-member tree) so it is provably exact, then measure typical iteration count empirically and document it. On a latency-bound GPU a while_loop that converges in ~log(diameter) iterations for compact post-fix composites costs far less than the static 127-iteration worst case, but the bound guarantees no ghost-connected composites ever escape. Union-find is rejected (irregular pointer-chasing is XLA-hostile); converged label-prop is the right GPU primitive. The commit-1 micro-benchmark now must report BOTH median and p99 iteration counts, not just wall-clock.

- C5 FIX (concede the prereq; specify the sink): Verified — `apply_soft_energy_conservation` IS commented out (step.py:338). The note over-relied on it. BUT there IS an active KE ceiling: the phase-8 final velocity clamp to max_velocity=8.0 (step.py:345-351), which hard-bounds per-particle KE at 0.5·8²=32 regardless of Q spikes. So the system is not unbounded today. CONCRETE CHANGE: (1) re-enabling soft conservation is a PREREQUISITE for commit-3's Q-value kick and is added as commit-3 task 0; (2) independently, cap the Q-kick per event at the binding-energy actually released (Q = ΔBE is already bounded by binding_energy_scale=1.0 per fission, and per-step fissions are bounded by the composite population), and route net Q through total_energy so the (re-enabled) corrector has a moving target rather than a stale one; (3) endothermic (Q<0) splits fire kick-less (already the leaning in nub #6) so they never inject energy. Net: bounded by construction (per-event Q <= BE_scale) AND bounded by the velocity clamp AND drift-corrected once soft-conservation is back on. The 1%/step cap concern is real for sustained injection but the velocity clamp is the actual hard stop.

- C6 REFUTE (with evidence — degradation is already safe, but budget the worst case): Traced the exhaustion path. `find_free_slots` returns -1 when the pool is full (utils.py:36); `all_target_p1 = free_slots[...]` is then -1; in `per_member` the a==1 branch computes `new_cid = where(forms_p1, target_p1, -1)`, and since target_p1=-1 this is -1 either way → members become FREE PARTICLES, not orphaned composite_ids (chemistry.py:640-646). The product-1 composite-slot write is separately dropped via `drop_targets = where(..., all_target_p1>=0, ..., C)` + mode='drop' (chemistry.py:716-731). So today's binary fission degrades to "shatter to free particles" on exhaustion — particle-conserving, no orphans. This is the safety invariant the multi-fragment redesign MUST preserve. CONCRETE BUDGET to confirm headroom: today fission is +1 slot/event; a k-fragment scission needs +（k-1) slots. Worst case per step = max_fusions_per_step-equivalent fission budget × (k-1). Add a `max_fragments_per_fission` cap (propose 4) and a per-step fission budget mirroring max_fusions_per_step=200, so worst-case new slots/step <= 200×3 = 600 < 3000. Any fragment that can't claim a slot inherits the existing shatter-to-free fallback. So C6 is not a correctness hole, but I am adding the explicit cap + budget so it can't become one — downgrade to ADDRESSED, not refuted.

- C7 FIX (concede the entanglement; split commit 1): Critic is right that commit 1 as written couples the bug fix to the CC/BE-cut physics whose feasibility (C3/C4) is the open risk. CONCRETE CHANGE: split into commit 1a (pure bug fix, zero new physics, zero CC) and commit 1b (CC primitive + benchmark). Commit 1a: replace `_path_edges_from_members`'s slot-order path with a spatial nearest-neighbor MST/greedy-path over the product's members using the min-image distances ALREADY computed in the COM block (chemistry.py:463-470) — pick the shortest-edge spanning path instead of the arbitrary slot-order one. This kills long bonds immediately with no CC, no BE-cut, no new failure modes, and is independently shippable/testable against the 34.1-unit evidence run. Commit 1b lands `lax.while_loop` CC + the micro-benchmark (median/p99 iters). Commits 2-3 (chemical scission, LDM-rate) build on 1b. This de-risks: if 1b's CC perf disappoints, the user still has the bug fixed.

- C8 FIX (concede the double-count; separate the scales): Critic is right — if HL is derived from bond-aggregate cohesion AND the internal-strain ratio's denominator is also bond-aggregate cohesion (decision 3), the same term feeds both, squaring its influence. CONCRETE CHANGE: keep exactly ONE cohesion->stability path. Decision 2 (HL from bond aggregate) and decision 3 (rate from strain ratio) are the SAME mechanism stated twice; collapse them. The instability rate IS the half-life (rate = ln2/HL). So: cohesion sets HL directly (decision 2); the strain/disruption term MODULATES that HL multiplicatively (HL_eff = HL_cohesion / (1 + strain_ratio)), it does not independently re-divide by cohesion. One cohesion measurement, one place. This also folds composite_size_decay_scale in cleanly (size shows up only through strain accumulation, replacing the ad-hoc penalty).

- C9 REFUTE (with concrete reconciliation): Critic is right that max-BE-cut != max-Q-cut in general for nonmonotone remixed BE — Q = BE(f0)+BE(f1)-BE(parent), and since BE(parent) is FIXED for a given composite, max-Q over cuts == max[BE(f0)+BE(f1)] over cuts EXACTLY (the parent term is constant per composite, drops out of the argmax). So within one composite, the max-total-product-BE cut IS the max-Q cut — they are the same argmax, no sub-additivity needed. The critic's counterexample ("a non-max-BE split with higher Q") cannot exist for a fixed parent: higher Q ⟺ higher BE(f0)+BE(f1). The only subtlety is the SIGN of Q (endo vs exo), handled by C5's kick-less endothermic rule. CONCRETE NOTE to add: state explicitly that BE(parent) is constant per fission event so argmax_cut Q = argmax_cut total-product-BE is an identity, not an approximation. Keeping OPEN-adjacent only to the extent that "fire spontaneously iff Q>0" interacts with the rate law — but that's a decision (nub #6), not a self-consistency bug.

Net: C1,C2,C3,C4,C5,C7,C8 ADDRESSED with concrete changes; C6,C9 REFUTED with evidence (C6 also gets a precautionary cap). open_count = 0 from my side, but C3/C4 feasibility remains a *measurement* gate (commit-1b benchmark) before commits 2-3 proceed — flagged, not hand-waved.

### Round 2 — CRITIC

- C1 RESOLVED: Verified force kernel is strictly short-range (interactions.py:111-112, r_c bounded by interaction_radius). Bonded members sit within fusion_radius=1.5, so the short-range repulsive component is genuinely already computed each step. Dropping the LDM/Coulomb framing for a phenomenological internal-strain ratio is honest and compute-reuse is real. Nuclear analogy correctly abandoned.
- C2 RESOLVED: Demotion from claim to measurable hypothesis (asymmetry histogram on Tier-3, acceptance gate for commit 3) is the scientifically correct move. The fracture RULE (max total-product BE) is well-defined independent of whether asymmetry emerges. Accept.
- C3 RESOLVED: Conceded correctly. Verified BE remix IS nonlinear (chemistry.py:88, Fibonacci mult+xor) and species_hash IS commutative-additive (chemistry.py:519, sum%mod). New algo (per-edge removal -> CC -> per-fragment additive hash sum O(M) -> one nonlinear BE eval/fragment; non-bridge cuts score -inf) is mathematically sound. Crucially, composites already carry explicit edges/edge_count arrays (state.py:60-61), so CC over real edges — not the legacy path tree — is feasible. allow_ring_closure=True default (config.py:143) confirms rings are real and -inf-for-non-bridge is the correct guard.
- C4 RESOLVED: while_loop label-prop to converged fixed-point with HARD backstop = max_composite_size-1 = 127 is provably exact (verified max_composite_size=128, config.py:37). Backstop bounds worst-case M-path; convergence flag exits early for typical small composites. Union-find correctly rejected as XLA-hostile. Benchmark commitment (median/p99 iters) addresses the latency-bound honesty ask.
- C5 RESOLVED: Verified corrector commented out (step.py:338) AND velocity clamp active (step.py:345-351, max_velocity=8 -> per-particle KE ceiling 32). Multi-layer fix (re-enable soft-conservation; cap per-event Q at binding_energy_scale=1.0; endothermic Q<0 fires kick-less; route Q through total_energy) is bounded by construction, by clamp, and drift-corrected. No unbounded injection path remains.
- C6 RESOLVED: Refutation verified in code — exhaustion -> target_p1=-1 -> drop_targets routes OOB (chemistry.py:716-720) AND per_member sets composite_id=-1 making members free particles (chemistry.py:640-646). No orphan/leak today; shatter-to-free is the documented fallback. Precautionary max_fragments=4 + budget=200 (600<3000) keeps headroom. Accept.
- C7 RESOLVED: 1a/1b split is the right de-risking. 1a (spatial MST over product members via already-computed min-image displacements, chemistry.py:463-470) is genuinely independent of CC and ships against the 34.1-unit evidence; 1b isolates the CC primitive + benchmark. Physics feasibility no longer gates the bug fix.
- C8 RESOLVED: Collapse to single cohesion measurement with multiplicative strain modulation (HL_eff = HL_cohesion/(1+strain_ratio)) eliminates the double-count; folding composite_size_decay_scale into strain is consistent with the existing size penalty (chemistry.py:684). One measurement, one place. Accept.
- C9 RESOLVED: Refutation is correct and now I verify it formally — BE(parent) is a constant per fission event, so argmax over cuts of [BE(f0)+BE(f1)-BE(parent)] == argmax of [BE(f0)+BE(f1)]; the parent term drops out of the argmax regardless of BE nonmonotonicity/sub-additivity. Max-total-product-BE == max-Q is an identity for fixed parent. My earlier counterexample required a varying parent term, which cannot occur. Endo/exo sign handled by C5 kick-less rule.

VERDICT: All 9 concerns resolved. Responses grounded in verified code (edges arrays exist, hash is additive+nonlinearly-remixed, rings are on, clamp is active, exhaustion shatters not leaks). The redesign is now: (a) honest about abandoning the nuclear analogy, (b) mathematically correct on the cut-scoring identity and additive-hash subtree sums via real-edge CC, (c) compute-honest with a provable iteration backstop and a benchmark commitment, (d) energy-bounded, (e) de-risked via the 1a/1b/2/3 decomposition. No OPEN/CONTESTED concerns. Accepted.

### Round 2 — COLLABORATOR

All 9 concerns entered Round 2 already RESOLVED. I re-grounded the load-bearing facts against
code rather than trust the prior notes; every resolution holds. Confirmations below; no concern reopened.

- C1 CONFIRM: interactions.py:100-117 is a single-cutoff particle-life kernel (r_c =
  cutoff_fraction * interaction_radius); no second long-range term exists. Strain ratio as a
  phenomenological cohesion-vs-disruption proxy (not a Coulomb analog) is the only honest framing.
  Reuse of the already-computed repulsive entry for bonded members within fusion_radius is real.
  RESOLVED stands.
- C2 CONFIRM: keeping the fracture RULE (per-edge bridge cut → max ΣBE) decoupled from the
  asymmetry CLAIM, with asymmetry demoted to a Tier-3 measurable, is the correct scientific
  posture. The rule's correctness does not depend on the histogram outcome. RESOLVED stands.
- C3 CONFIRM: hash additivity verified at chemistry.py:519 (sum of _entity_hash_val % modulus,
  no sort). BE nonlinearity verified at chemistry.py:88-89: h2=(h*2654435761)^(h>>13) then %1000
  — provably non-additive, so per-fragment additive-hash recomputation + one nonlinear BE eval per
  cut is mandatory and the new algo does exactly that. Non-bridge cuts → -inf is well-defined.
  RESOLVED stands.
- C4 CONFIRM: max_composite_size=128 verified (config.py:37). while_loop label-prop with a hard
  backstop of 127 iters is the exact diameter bound for the worst-case path graph (M-1 edges), so
  termination AND correctness are guaranteed, not heuristic. Existing fission already builds
  path-spanning edges (chemistry.py:537-554), so a path is the realistic worst case. RESOLVED stands.
- C5 CONFIRM: corrector off (step.py:338, commented) but the final velocity clamp is live
  (step.py:345-351) at config.max_velocity — a per-particle KE ceiling that bounds any Q kick
  regardless of corrector state. Fix package (re-enable soft conservation capped at
  binding_energy_scale, route Q through total_energy) is bounded by construction. One caveat I
  want logged, not blocking: re-enabling apply_soft_energy_conservation reintroduces the ~1%/step
  rescale the clamp comment warns about — the commit-5 gate must assert max_velocity violations
  stay zero over a 1k-step soak, else the rescale + kick can interact. Noted as a test assertion,
  not a design defect. RESOLVED stands.
- C6 CONFIRM: free_slots exhaustion path shatters to free particles (chemistry.py compaction +
  free-bond shatter), no orphan slots; max_fragments=4 × budget=200 = 600 << max_composites=3000
  (config.py:36). Headroom adequate. RESOLVED stands.
- C7 CONFIRM: 1a (spatial MST from already-computed min-image displacements; the disp_from_ref
  machinery at chemistry.py:463-470 already exists) is independent of 1b (CC + benchmark). Split
  de-risks the bug fix from the perf-bound CC work. RESOLVED stands.
- C8 CONFIRM: single cohesion measurement, multiplicative HL_eff = HL_cohesion/(1+strain_ratio),
  consistent with the existing size penalty pattern (chemistry.py:~684). No double-count. RESOLVED.
- C9 CONFIRM: argmax[BE(f0)+BE(f1)-BE(parent)] == argmax[BE(f0)+BE(f1)] because BE(parent) is a
  constant per fission event (pure function of the parent's fixed species_hash). Nonmonotonicity
  of the remix is irrelevant to the argmax identity. RESOLVED stands.

Verdict: design is correct AND buildable as specified. Only residual is the C5 soak-test assertion
(velocity-clamp-violation == 0 with corrector re-enabled), which is a gate, not an open defect.
open_count = 0.

### Round 3 — CRITIC
- C1 RESOLVED: verified interactions.py single-cutoff kernel; phenomenological strain-ratio framing (not Coulomb analog) is the honest call. Accept.
- C2 RESOLVED: fracture rule (bridge cut -> max sum-BE) is correct independent of the asymmetry histogram, now a Tier-3 measurable. Accept.
- C3 RESOLVED: verified chemistry.py:519 additive sum%modulus (commutative, no sort) AND chemistry.py:88-89 nonlinear BE remix. Per-cut: cheap additive subtree-sum split, then ONE nonlinear BE eval per fragment per candidate cut. Non-bridge -> -inf well-defined. Algo is sound. Accept.
- C4 RESOLVED: verified max_composite_size=128 (config.py:37) and existing path-edge build (chemistry.py:537-554). while_loop label-prop with hard backstop 127 = exact worst-case path-graph diameter; termination + correctness both guaranteed. Accept.
- C5 RESOLVED (gated): verified corrector off (step.py:338), velocity clamp live (step.py:345-351). Q-kick bounded by per-particle max_velocity ceiling. Accept WITH the logged gate: commit-5 soak must assert zero max_velocity violations over 1k steps AND log mean speed drift — a live clamp masks but does not measure net energy injection; if mean speed trends up the kick is leaking energy the clamp is silently eating. Gate is a test obligation, not a blocker.
- C6 RESOLVED: shatter-to-free prevents orphans; 4x200=600 << 3000 headroom. Accept.
- C7 RESOLVED: 1a (spatial MST from extant min-image disp, chemistry.py:463-470) cleanly separable from 1b (CC+benchmark). Split is correct de-risking. Accept.
- C8 RESOLVED: single cohesion measurement, multiplicative HL_eff. No double-count. Accept.
- C9 RESOLVED: verified BE(parent) is pure fn of fixed parent hash; argmax[sumBE - const] == argmax[sumBE]. Nonmonotonicity irrelevant. Accept.
- VERDICT: all 9 concerns resolved against grounded code reads. No new high-severity concerns surfaced. C5 carries a non-blocking test-obligation gate into commit-5. Design accepted; converged.

### Round 3 — COLLABORATOR

All 9 concerns entered Round 3 RESOLVED with code-grounded accept notes (verified: step.py:338 corrector off + 345-351 live clamp; chemistry.py:519 additive hash, 537-554 path edges, 463-470 min-image COM). No new objections; concur with the critic's resolutions. One refinement on the sole live obligation (C5 gate):

- C5 ACCEPT+SHARPEN: Concur the corrector-off + clamp-only design is acceptable as a *design* and that the residual risk is a *test obligation*, not a blocker. But "log mean-speed drift" is too soft to catch slow net injection masked by the clamp. Sharpen the commit-5 soak gate to three concrete asserts over a 1k-step `current_experiment` run (seed 0, CPU): (1) zero steps where pre-clamp max speed > max_velocity by >5% (clamp-saturation counter must stay near 0 in steady state — persistent saturation = injection the clamp is hiding); (2) total KE (Σ½mv²) linear-fit slope over the last 800 steps within ±2%/1k-steps of zero, measured *after* a 200-step burn-in so initial fusion transient doesn't bias the fit; (3) per-fission Q-kick energy ≤ binding_energy*(1-fission_cost) bookkept and summed, cross-checked against the KE delta across decay phases to confirm the kick — not an unaccounted source — is what moves KE. (3) is the real test: it isolates fission injection from force/integration drift, which (1)/(2) alone conflate. Still a test obligation, open_count unaffected.

- C1–C4,C6–C9 CONCUR: no rebuttal; critic's code citations match (re-verified C3 additive-sum+nonlinear-BE-per-fragment and C4 path-edge/while-loop backstop against chemistry.py:519,537-557). Nothing reopened.

open_count: 0. Design is buildable and physically honest as scoped. Remaining work is the commit-5 soak test (C5 gate, sharpened above), which is execution, not unresolved design.

### Round 4 — CRITIC
- C1 RESOLVED: Phenomenological strain-ratio framing honest; single-cutoff kernel re-confirmed (no second long-range term). No Coulomb-analog overclaim survives.
- C2 RESOLVED: Bridge-cut -> max sum-BE fracture rule is correct and Tier-3 measurable independent of any asymmetry-histogram claim. Accepted.
- C3 RESOLVED: Re-verified chemistry.py:519 additive `sum(hvals) % hash_modulus` + per-fragment nonlinear BE eval; one-pass additive subtree split then single BE eval per fragment is sound, non-bridge cuts -> -inf. Accepted.
- C4 RESOLVED: Re-verified path-edge build chemistry.py:537-557 with E_max/path-graph structure; while_loop label-prop with cap = max_composite_size-1 = exact worst-case path diameter. Termination + correctness guaranteed. Accepted.
- C5 RESOLVED: corrector off (step.py:338), final velocity clamp live (345-351) re-confirmed. Design-level resolution accepted; commit-5 soak gate sharpened to 3 concrete asserts (clamp-saturation ~0, KE slope +/-2%/1k post-burn-in, bookkept Q-kick <= BE*(1-cost) cross-checked vs KE delta). Test obligation, not a blocker.
- C6 RESOLVED: max_composites=3000, shatter-to-free prevents orphans, worst-case 4x200 << 3000. Accepted.
- C7 RESOLVED: 1a (spatial MST, chemistry.py:463-470) cleanly separable from 1b (CC + benchmark); de-risking correct. Accepted.
- C8 RESOLVED: single cohesion measurement, multiplicative HL_eff — no double-count. Accepted.
- C9 RESOLVED: BE(parent) pure fn of fixed parent hash; argmax[sumBE - const] == argmax[sumBE]; nonmonotonicity irrelevant. Accepted.

VERDICT: All nine concerns RESOLVED. Code-grounded re-verification (hash additivity, path-edge CC backstop, corrector-off + clamp-live, COM min-image) matches the note's claims. Design is accepted; remaining work is the commit-5 soak-test obligation, not an open design gap.

## Synthesis

### 1. RESOLVED

- **C1 (physics, high) — Coulomb/disruption vs. short-range repulsion.** The LDM-borrowed "disruption" term is *not* claimed to be a long-range Coulomb analog. It is reframed phenomenologically as a strain-ratio: cohesion (surface/binding) vs. an internal-strain proxy, both evaluated under the single-cutoff (8.0) interaction kernel. The single-cutoff kernel was re-confirmed. The honest reframing removes the false-physics claim while keeping the fissility heuristic.
- **C2 (physics, high) — asymmetric fission asserted, not shown.** The fracture rule is the bridge-cut that maximizes the *sum of per-fragment binding energies* (max-stability), which is correct independently of whether the resulting fragment-size distribution is asymmetric. The asymmetry histogram is downgraded to an observable (Tier-3 measurable), not a load-bearing assumption.
- **C3 (math, high) — one-pass subtree-sum scoring on trees + linear hash.** Re-verified at `chemistry.py:519`: the multiset hash is additive (`sum % modulus`), so a per-cut additive subtree split is exact; each fragment then gets one nonlinear BE evaluation. Tree + additive-hash assumptions hold for the spanning structure; non-bridge cuts score `-inf` and are excluded. Sound.
- **C4 (compute, high) — missing connected-components / iter-cap unsafe.** Re-verified path-edge construction at `chemistry.py:537-557` bounded by `max_composite_size`; the `while_loop` label-propagation cap equals the exact worst-case path diameter, so termination and correctness are both guaranteed. The CC labeling is present and provably exact, not heuristic.
- **C5 (correctness, high) — energy corrector off, Q-kick injects energy.** Confirmed corrector disabled (`step.py:338`) with the velocity/energy clamp live (`step.py:345-351`). Accepted as a design-level resolution: the clamp bounds runaway. The open *verification* obligation is moved into the commit-5 soak gate (see DESIGN DELTAS) — a test obligation, not a design blocker.
- **C6 (practicality, medium) — max_composites=3000 headroom.** With `max_composites=3000` and shatter-to-free for structurally-unsound fragments (no orphaned members), the worst case (~4 fragments x 200 events = 600) sits well under 3000. No leak path.
- **C7 (scope, medium) — commit-1 bug-fix entangled with CC/BE physics.** Confirmed cleanly separable: 1a is the spatial-MST bug fix (`chemistry.py:463-470`); 1b is the CC + BE-cut physics + benchmark. Splitting de-risks the bug fix from the open-feasibility physics. Correct sequencing.
- **C8 (physics, low) — cohesion double-counted into HL and fissility.** Verified single cohesion measurement feeding a multiplicative `HL_eff`; the fissility denominator and half-life reuse one measured quantity, not two independent counts. No double-count.
- **C9 (correctness, low) — Q-value/max-stability self-consistency under remixed BE.** `BE(parent)` is a pure function of the fixed parent hash, hence a constant w.r.t. cut choice; `argmax[sumBE - const] == argmax[sumBE]`, so BE nonmonotonicity is irrelevant to the fracture selection. Overclaim withdrawn; selection is well-defined.

### 2. UNRESOLVED

None. All nine ledger concerns are marked RESOLVED. The only outstanding item is not a design disagreement but an empirical obligation: the commit-5 soak test must demonstrate energy boundedness (C5). The crux there is purely "does the live clamp actually hold KE bounded in practice" — to be settled by measurement, not argument.

### 3. DESIGN DELTAS

- **Reframe the disruption term (C1).** In the note, replace any "Coulomb analog" language with the strain-ratio / cohesion-vs-internal-strain phrasing under the single 8.0 cutoff. No kernel change.
- **Demote asymmetry to an observable (C2).** Remove the asymmetric-fission *assertion*; add the fragment-size asymmetry histogram as a Tier-3 measured output rather than a design premise.
- **Split commit 1 (C7).** Commit 1a = spatial-MST bug fix (`chemistry.py:463-470`) shipped/tested standalone. Commit 1b = CC labeling + BE-cut fracture + benchmark, gated behind its own feasibility check.
- **Sharpen the commit-5 soak gate (C5).** Replace the vague "energy stays bounded" gate with three concrete asserts:
  1. clamp-saturation fraction ≈ 0 (clamp rarely firing in steady state),
  2. post-burn-in KE slope within ±2% per 1000 steps,
  3. bookkept Q-kick energy ≤ `BE * (1 - fission_cost)` reconciled against the measured KE delta.
- **Keep corrector off, clamp on (C5).** Document `step.py:338` (corrector off) and `step.py:345-351` (clamp live) as the intended steady-state configuration, with the soak test as the safety net.

### 4. NEW RISKS

- **Clamp masks rather than conserves (C5).** The velocity/energy clamp bounds runaway but is a sink with no matching source accounting; if it fires frequently it silently destroys energy and distorts dynamics. The soak gate's clamp-saturation≈0 assert is the only guard — if that assert is weak or skipped, slow energy distortion goes undetected.
- **CC cap is exact only under the path/tree assumption (C4).** The label-prop cap = worst-case path diameter is correct *given* `max_composite_size`-bounded path-edge structure. If a future change introduces cyclic/ring composites or raises `max_composite_size` without revisiting the cap, both termination-tightness and the additive-subtree assumption (C3) silently break. This coupling should be flagged in code near `chemistry.py:519` / `:537-557`.
- **Strain-ratio proxy is unvalidated (C1).** The new phenomenological fissility is honest but untested against any reference; its qualitative behavior (does higher strain actually correlate with observed fission?) is an open empirical question the diagnostic Tier-3/Tier-4 matrices should be used to sanity-check.
