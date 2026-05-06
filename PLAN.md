# Half-Life Particle Simulator — Project Roadmap

## Status (2026-05-06)

Simulator is running interactively at ~40 FPS with 2k particles, 200+ composites, full live-tunable physics via UI sliders. Core build phases (0–4) are complete; current work is on physics tuning and emergent-dynamics investigation.

---

## Next Session — TODO (queued 2026-05-06)

After today's audit ([notes/2026-05-06-composite-dynamics-architecture.md](notes/2026-05-06-composite-dynamics-architecture.md)), four items to attack tomorrow:

- [ ] **Rethink polarity scaling.** Current `attr_mod = net_polarity` makes neutral composites attractively inert in every direction, which biases the population toward boring blobs. Try alternatives: remove polarity scaling entirely, use `(1 + |net_pol|)/2` so composites stay reactive, or use `max(|p_i|)` instead of mean. Goal: stop the inertness/stability feedback loop. Context: [boring-dynamics note → polarity hypothesis](notes/2026-05-05-boring-dynamics-investigation.md#hypothesis-c--polarity-induced-inertness).

- [ ] **Composite interaction range + drop the "representative" trick.** Two coupled changes: (a) bump `interaction_radius` so composites can actually see each other (currently 4.0 ≈ composite diameter); (b) replace the rep-only fusion access ([halflife/chemistry.py:265–272](halflife/chemistry.py#L265)) — every member should be able to participate in fusion, not just the lowest-index one. The rep is an arbitrary architectural choice that limits growth.

- [ ] **Fix `cell_capacity` overflow.** Currently 8 — way too low for our typical compact composites (20+ members in one cell is common). 16% of composites are silently corrupted by truncated neighbor lists, producing fictitious "self-propulsion" forces (956× clean composites). Cheapest fix: bump capacity to 32 or 64, assert on `did_overflow`. See [composite-dynamics note → cell-list overflow bug](notes/2026-05-06-composite-dynamics-architecture.md#cell-list-overflow-bug-the-actual-cause).

- [ ] **Audit + experiment with force kernels.** Verify the attraction matrix is being used as intended (signed values? actually consulted? `r_attract[i,j]` per-pair is sampled but not used by the kernel — known dead arg). Then sketch 2–3 alternative kernels (e.g. Lennard-Jones, smooth Lenia-style, structured/sparse attraction matrix, Perlin-driven) and toggle between them via config to compare emergent richness. Goal: figure out which kernel substrate actually produces interesting dynamics.

---

## Phase 5: Polish and Optimization

- [x] Async rendering overlap (commit `086e9e1`)
- [x] Interactive parameter controls / live sliders (`bb7d951`, `2b44ab9`, `196ba55`)
- [x] Statistics overlay — FPS, composites, energy, histogram, sparklines (`c7142af`, `bb7d951`)
- [x] Profiler infrastructure with phase-level timing (`edd8465`–`746b4ac`)
- [ ] Frame recording (save to disk for videos)
- [ ] Scale up to 10k+ particles (gated on cell_capacity fix in Next Session TODO)

---

## Phase 6: Evolution and Emergence

- [x] Parameter sweep infrastructure ([tests/test_composite_statistics.py](tests/test_composite_statistics.py) generates HTML reports in [tests/reports/](tests/reports/))
- [ ] Group fitness metrics
- [ ] Spatial compartmentalization detection
- [ ] NCA-style learned update rules — would need to restore the `internal` field removed in `b0c049f`
- [ ] Mass conservation mode (FlowLenia / reintegration-tracking inspired)
- [ ] Optimization loops (evolve interaction matrices for specific goals)

---

## Implementation Log

Detailed history is in `git log`; thematic recap in [notes/2026-05-05-project-status-recap.md](notes/2026-05-05-project-status-recap.md). High-level arc:

- **Phase A — Initial build & first-run debugging.** Modules written; first run hit 4 perf bottlenecks, fixed → 30ms → 3.9ms/step.
- **Phase B — Polarity chemistry, UI, events, stats.** Live HUD, event sprites, sparklines, polarity-based fusion preference and stability. Closed with a 10–100× perf jump (`086e9e1`: commutative hash, COM-spring bonds, async pipeline).
- **Phase C — Live tuning UX + fusion-scan optimization.** Log-scale sliders, fusion scan rewrites, dead-particle machinery removed (`b0c049f`), `lax.switch` → `jnp.where` (4× speedup).
- **Phase D — Profiling instrumentation (2026-03).** Profiler module, C+C fusion detection, baseline performance docs.
- **Phase E — Composite statistics analysis (2026-03-27).** HTML sweep reports over `(fusion_threshold, interaction_radius, composite_size_decay_scale)` (six runs in `tests/reports/`).
- **Phase F — Physics audit (2026-05-05/06).** Diagnosed degenerate-kernel issue at user's UI settings, dead-code cleanup (`98abb0f`), fusion_radius bump 1.0→4.0 (`86eb78f`), discovered cell_capacity overflow bug. Notes: `notes/2026-05-05-*`, `notes/2026-05-06-*`.

Performance baseline (2026-03-27): 11.6 ms/step fused, 85 steps/sec at 2k particles. Bottleneck is `attempt_fusion` at 55% of step time. Full breakdown in [tests/README_PERFORMANCE.md](tests/README_PERFORMANCE.md).
