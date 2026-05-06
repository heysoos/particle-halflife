# Render/update pipeline audit

**Date:** 2026-05-06
**Topic:** At 10k particles, frame split shows `sim=1.4ms update=28-36ms render=15-16ms` (≈18-20fps). Audit where time actually goes and what's worth attacking.

## Summary

The user's read — "time moved from sim into update/render" — is **misleading because of how the timer is wired**. `sim_ms` only measures JAX dispatch (non-blocking), and `renderer.update()` ends up absorbing the GPU sync stall via `jax.device_get`. The simulation is still the dominant cost; it's just hiding inside the wrong column. Real CPU update is ~3-8ms; real render is ~15ms, mostly HUD rebuilt every frame.

## Headline finding: the timing readout is misleading

[main.py:217-237](../halflife/main.py#L217-L237) wraps timers around three calls:

- `t0_sim` → `run_n(...)` is **JAX dispatch only** — returns immediately. `sim_ms ≈ 1.4ms` is dispatch overhead, not GPU work.
- `t0_update` → `renderer.update(pending_state)` calls `jax.device_get(...)` at [renderer.py:527](../halflife/renderer.py#L527), which **blocks** until the GPU finishes computing `pending_state`. So `update_ms` = GPU-wait + actual CPU work.
- `t0_render` → genuine CPU/GL/HUD work.

The async overlap design (`next_pending` dispatched at [main.py:219](../halflife/main.py#L219), then `update(pending_state)` syncs while GPU works on N+1) is real — but the readout makes it look like sim got cheap and update got expensive, which it didn't.

## Bottleneck ranking

1. **GPU simulation step at 10k (~25-35ms), hidden in `update_ms`.** The 2k-particle benchmark in [tests/README_PERFORMANCE.md](../tests/README_PERFORMANCE.md) (2026-03-27) measured a full step at 11.74ms with `attempt_fusion` at 55% (8.81ms), `apply_composite_decay` at 12%, `compute_bond_forces` at 11%, `find_all_neighbors` at 10%, `compute_all_forces` at 9%. Linear-in-N extrapolation lands ~50-60ms/step at 10k, matching what we see in `update_ms`.

2. **HUD rebuilt every frame (~5-10ms suspected).** [renderer.py:787-964](../halflife/renderer.py#L787-L964) re-renders all buttons + text + sparklines + histogram + slider panel every frame, then `pygame.image.tostring` encodes 3.6 MB and `_hud_texture.write` uploads it. `font.render` is a known pygame slow path. Prior render-optimization commit `42d4acc` brought render 50→10ms — we're at 15-16 now, regrowth is from added HUD elements (sparklines, histogram, params panel).

3. **Bond VBO build (~1-3ms).** [renderer.py:576-615](../halflife/renderer.py#L576-L615) — `np.broadcast_to`, `np.triu_indices(max_n)`, ravel/mask, periodic wrap, then write a `(2·n_pairs, 6)` float32 VBO. With `max_n=64` worst case, `triu_indices` produces 2,016 pair-slots × n_alive_composites; allocating this per frame is wasteful even though only a fraction are valid.

4. **Misc CPU update (sub-ms each).** Histogram, event-tuple churn, `comp_alive.copy()`.

## Recommended diagnostics (read-only, do these first)

These confirm the ranking before any code change goes in:

1. **Split GPU-wait from CPU-update.** Add `pending_state.position.block_until_ready()` (or a single-scalar `device_get`) just before `t0_update` at [main.py:227](../halflife/main.py#L227). Whatever interval that adds is true `gpu_wait_ms`; the rest is real CPU work.
2. **Run `python -m halflife.profiler --n 10000`** — gives per-phase GPU breakdown at the actual problem scale (the 2k baseline doesn't extrapolate cleanly because fusion's `max_neighbors=256` clamp may behave differently at higher density).
3. **Time `_render_hud_surface` and `_hud_texture.write` separately.** If together >5ms, HUD caching is the highest-leverage UI fix.
4. **Confirm vsync state.** `pygame.OPENGL | pygame.DOUBLEBUF` doesn't enable vsync by default but driver settings can.

## Optimization candidates (after diagnostics)

Ordered by expected payoff.

**High-leverage:**

- **HUD frame-skip / cache.** Rebuild the HUD texture every ~6-10 frames instead of every frame. Most of it (buttons, ticks, slider chrome) doesn't change frame-to-frame.
- **Reduce `device_get` payload at [renderer.py:524-534](../halflife/renderer.py#L524-L534).** Currently transfers `comp_members` and `comp_count` for the full `MAX_COMPOSITES × MAX_COMPOSITE_SIZE` array. Gather alive-only on device, then transfer.
- **Move bond pair-building into JAX.** `triu_indices` + mask + periodic-wrap is pure data manipulation; doing it in a JIT'd function would let XLA overlap it with the rest of the step.

**Sim-level (orthogonal, but biggest absolute win):**

- Fusion is the dominant phase at every scale. Reducing `max_neighbors` (256) when density permits, or coarsening the cell list — already in scope of the existing `tests/README_PERFORMANCE.md` plans, not new work.

**Easy/cheap:**

- Verify the `if self._show_stats:` gate at [renderer.py:832](../halflife/renderer.py#L832) really skips all the histogram/sparkline drawing when off.
- Replace per-frame `comp_alive.copy()` at [renderer.py:724](../halflife/renderer.py#L724) with a swap of two pre-allocated buffers.

## Files referenced

- [halflife/main.py](../halflife/main.py) — frame loop and timer wiring (217-237)
- [halflife/renderer.py](../halflife/renderer.py) — `update()` 515-735, `render()` 740-769, `_render_hud_surface()` 787-964
- [halflife/profiler.py](../halflife/profiler.py) — per-phase GPU profiler
- [tests/test_performance.py](../tests/test_performance.py) — 2k baseline benchmarks
- [tests/README_PERFORMANCE.md](../tests/README_PERFORMANCE.md) — phase breakdown
- [notes/2026-05-05-project-status-recap.md](2026-05-05-project-status-recap.md) — prior 50→10ms render optimization (commit `42d4acc`)

## Nubs

- ?? confirm fusion's `max_neighbors=256` clamp isn't actually saturating at 10k — would change the linear extrapolation
- pygame text rendering: pre-render static glyph atlases (button labels, axis ticks) once instead of per-frame `font.render`
- if `pygame.image.tostring` is the dominant HUD cost, try `moderngl.Texture` direct from a numpy buffer view of the surface (skip the encode step)
