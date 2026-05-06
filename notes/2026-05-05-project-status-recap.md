# Half-Life simulator — return-to-project status recap

**Date:** 2026-05-05
**Topic:** Where past sessions left off, what's open, and what to pick up next.

## Summary

Project has matured from "all modules written, untested" (2026-03-14) through several optimization passes and into a Phase 1 *instrumentation/profiling* effort that wrapped on **2026-03-27** (~5 weeks before today). Last commit refactored composite-statistics collection to use `jax.lax.scan`. There's one **uncommitted edit** in [tests/test_composite_statistics.py](tests/test_composite_statistics.py) that parameterizes `num_steps` and bumps the default to 5000 — looks like the user was about to kick off a longer sweep and stopped. The big open question from when work paused: **composites plateau at ~5–8 members**, and we had three hypotheses but never definitively isolated the cause.

## What this project is (one-liner refresher)

JAX/GPU 2D particle simulator where everything has a half-life. Particles fuse into composites whose properties are determined by a hash over the sorted multiset of member species (Sayama-style hash chemistry). Goal: emergent autocatalytic sets / autopoiesis. Live render via ModernGL+pygame.

Full context: [CLAUDE.md](CLAUDE.md), [README.md](README.md), [PLAN.md](PLAN.md).

## Where we left off

### Last commit
`29d75cf` (2026-03-27) — replaced a Python for-loop in [tests/test_composite_statistics.py](tests/test_composite_statistics.py) with `jax.lax.scan` for stats collection. Pure refactor, no behavior change.

### Uncommitted local change
[tests/test_composite_statistics.py:817](tests/test_composite_statistics.py#L817) — `test_composite_statistics()` now takes `num_steps` as a parameter (was hard-coded `600`); `__main__` calls it with `num_steps=5000`. Small signature change, suggests a long-run sweep was being prepped. Decide: commit this and run, or drop it.

### Last *running* state (per [docs/PHASE1_ANALYSIS.md](docs/PHASE1_ANALYSIS.md))
- Steady 45–50 FPS at 2k particles, ~1.1–1.3 ms/step
- Composites form readily, peak max-size = **8 members**, plateau around step 1000
- C+C fusion rate ~0.065/step, mostly 2+2 → 4 mergers
- JIT compile ~5 s; profiling adds negligible runtime overhead

## Big open question (the main "nub")

**Why does max composite size plateau at 5–8?** Three hypotheses, none confirmed:

1. **BE threshold too conservative** — `fusion_threshold=0.2` may filter out C+C mergers once member multisets get diverse. *Suggested test:* sweep threshold 0.2 → 0.1 → 0.05.
2. **Hash chemistry is biased** — the polynomial hash may produce unfavorable BE values for the multi-species combinations needed to keep growing. *Suggested test:* histogram BE across all reachable multisets.
3. **Spatial isolation** — composites drift apart and don't encounter each other. *Suggested test:* track inter-composite distances over time.

The composite-statistics sweep in [tests/test_composite_statistics.py](tests/test_composite_statistics.py) was set up specifically to attack hypothesis #1 across `(fusion_threshold, interaction_radius, composite_size_decay_scale)` triples. The 5000-step bump in the uncommitted edit was likely the next experiment.

## Project arc — commit summary by theme

Recent work clusters into a few coherent phases. (Use `git log` for fine-grained detail; this is just the skeleton.)

### Phase A — Initial build & first-run debugging (early)
`342b2c9` initial working sim → `d755a47` first 4 perf bottleneck fixes (30 ms → 3.9 ms/step). README + CLAUDE.md added.

### Phase B — Polarity, UI, events, stats panel
`c7142af` polarity chemistry + UI overlay + event sprites + live stats →
`452171b` N-particle composites + sparklines + histogram fix →
`07ade6c` decay-model overhaul + FPS recovery →
`086e9e1` **10–100× perf jump** (commutative hash, COM-spring bonds, async pipeline).

### Phase C — Live tuning UX + fusion-scan optimization
`bb7d951` log-scale sliders, frame profiling →
`42d4acc` renderer.update 50 ms → 10 ms →
`2b44ab9` UI polish (per-slider reset, 0.1×–10× range) →
`065eec8` argsort → cumsum compaction in fusion scan (−4.8 ms) →
`b0c049f` removed dead-particle machinery (alive mask, half_life field on particles, apply_particle_decay) →
`da2811a` `lax.switch` → `jnp.where` in fusion = 4× speedup.

### Phase D — Phase-1 instrumentation (2026-03-26 → 03-27)
`f90f203` design doc for composite reaction network evolution →
`243642d` Phase-1 plan →
`edd8465`–`746b4ac` profiler module + integration into step/main/renderer →
`fafa07c` **bug fix**: metrics collection was running inside JIT and silently dropping data; moved outside JIT boundary →
`8c7e39b`/`322be89`/`3dc950a` Phase-1 verification + proper C+C fusion detection →
`3567c7b`–`890dddc` README_PROFILER / README_TESTS / README_PERFORMANCE docs.

### Phase E — Composite statistics analysis (last sessions before pause)
`f62fafc` composite-statistics sweep + HTML report scaffolding →
`0098ed1` add `composite_size_decay_scale` axis, sortable tables, interactive histograms →
`29d75cf` `jax.lax.scan` refactor for the stats loop.

Generated reports sit in [tests/reports/](tests/reports/) — six HTML files from 2026-03-27.

## Performance baseline (as of 2026-03-27)

From [tests/README_PERFORMANCE.md](tests/README_PERFORMANCE.md):

| Phase | mean ms | % total |
|---|---|---|
| build_cell_list | 0.30 | 1.9% |
| find_all_neighbors | 1.52 | 9.5% |
| compute_all_forces | 1.36 | 8.5% |
| compute_bond_forces | 1.71 | 10.7% |
| **attempt_fusion** | **8.81** | **55.1%** ← bottleneck |
| apply_composite_decay | 1.99 | 12.4% |
| energy_conservation | 0.31 | 2.0% |
| sum (separate) | 16.0 | |
| **fused step** | **11.6** | (XLA saves 4.4 ms) |

`attempt_fusion` is the dominant cost. Worth knowing if growth-related optimizations push fusion rates higher.

## Suggested ways to pick up

In rough priority order:

1. **Decide on the uncommitted edit.** Either commit + run the 5000-step sweep, or revert and start fresh with a sharper experiment design.
2. **Attack the size plateau directly.** Hypothesis #1 (BE threshold) is cheapest to test — the existing sweep already covers it; just lower `fusion_threshold` further and look at the HTML report. If plateau persists at threshold=0.05, escalate to #2 or #3.
3. **Pick up the Phase-2 design** from `f90f203` ("composite reaction network evolution spec"). Phase 2 was meant to add free-particle absorption into existing composites — explicitly designed to break the plateau. Worth re-reading that spec before designing new experiments.
4. **`max_composite_size` audit.** Per the user's design philosophy memory: this is supposed to be a JAX buffer constraint only, *not* a physics cap. Worth verifying nothing has crept in that treats it as a hard limit on growth.

## Useful pointers

- Memory records live at [Windows path] `/mnt/c/Users/Heysoos/.claude/projects/C--Users-Heysoos-Documents-Pycharm-Projects-halflife-particle/memory/` — five files covering project context, user background, WSL command pattern, composite design philosophy, subagent naming. WSL-side memory dir for this project is empty.
- All runtime tuning lives in [halflife/config.py](halflife/config.py) (`SimConfig` frozen dataclass).
- Full project roadmap and known-issues list: [PLAN.md](PLAN.md).
- Phase-1 deep-dive: [docs/PHASE1_ANALYSIS.md](docs/PHASE1_ANALYSIS.md).

## Nubs

- ?? `tests/__pycache__/` and `halflife/__pycache__/profiler.cpython-310.pyc` show as untracked/modified — confirm they're in `.gitignore` and stop showing in `git status`.
- ?? Six HTML sweep reports from 2026-03-27 in [tests/reports/](tests/reports/) — none ever opened in this recap. Worth eyeballing the most recent (`composite_statistics_20260327_211754.html`) before designing a new experiment; it may already answer hypothesis #1.
- ?? Phase-1 profiler bug history: `fafa07c` moved metrics collection outside JIT. Check whether anything else still tries to mutate Python state from inside a JIT'd function.
