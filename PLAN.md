# Half-Life Particle Simulator — Project Roadmap

## Status: All core modules written — ready to run and test

---

## Phase 0: Project Setup ✅

- [x] Architecture designed and approved
- [x] `CLAUDE.md` — project context
- [x] `PLAN.md` — this file
- [x] `requirements.txt`
- [x] `halflife/__init__.py`

---

## Phase 1: Foundation Modules ✅

- [x] `halflife/config.py` — SimConfig frozen dataclass
- [x] `halflife/state.py` — ParticleState, CompositeState, WorldState + initialize_world()
- [x] `halflife/utils.py` — hash_multiset(), find_free_slots(), boundary helpers
- [ ] Unit test: hash determinism, multiset order invariance

---

## Phase 2: Physics Core ✅

- [x] `halflife/spatial.py` — build_cell_list(), find_all_neighbors()
- [x] `halflife/interactions.py` — InteractionParams, pairwise_force(), compute_all_forces()
- [x] `halflife/step.py` — full simulation step (forces + chemistry integrated)
- [ ] Verify: particles move, interact by species, stay in bounds
- [ ] Profile JIT compile time and step throughput

---

## Phase 3: Visualization ✅

- [x] `halflife/renderer.py` — ModernGL setup, shaders, VBO pipeline, point sprites
- [x] `halflife/main.py` — pygame event loop, keyboard controls, FPS display
- [x] Species color palette (HSV), size scaling from mass, brightness by speed
- [x] Composite viz: bonds mode (GL_LINES) + merged mode (large point) — toggle `B`
- [ ] Verify: real-time visualization of particle physics

---

## Phase 4: Chemistry Engine ✅

- [x] `halflife/chemistry.py` — hash_to_products(), attempt_fusion(), apply_decay(), apply_composite_decay()
- [x] `halflife/energy.py` — energy tracking, soft conservation
- [x] Integrated into `step.py`
- [ ] Verify: fusion, composite formation, decay, fission, energy roughly conserved

---

## NEXT: First Run and Debugging ✅ (partial)

- [x] Install dependencies
- [x] Fix chemistry.py bug: shape mismatch in `apply_composite_decay` center-of-mass calc
- [x] All imports OK, JIT compiles, simulation step runs
- [x] Composites forming (~11 after 10 steps), ~30ms/step on 1000-particle config
- [ ] Run full GUI: `python -m halflife.main` (requires display — run locally in PyCharm)
- [ ] Verify physics looks plausible (particles cluster by species, move around)
- [ ] Tune parameters in config.py if behavior is degenerate

---

## Performance Investigation (done after first ~2 FPS run)

Four bottlenecks identified and fixed:

| # | Bottleneck | Root cause | Fix | Speedup |
|---|-----------|-----------|-----|---------|
| 1 | `attempt_fusion` scan | `lax.scan` over all N=20K particles; each step carries full `CompositeState` | Pre-filter to `max_fusions_per_step=100` candidates before scan | ~200x on scan |
| 2 | `find_free_slots` | O(N²): vmapped N `jnp.min()` reductions of size N | Replace with O(N log N) `jnp.sort` on candidate array | ~N/log(N) |
| 3 | `compute_bond_forces` | Iterates ALL 5K composites × 28 pairs every step, even when dead | Gate on `config.use_bond_forces` (default False) | ~140K ops/step saved |
| 4 | Default config size | `max_particles=20K` (10x benchmark size), causing all O(N) ops to be 10x worse | Reduced to `max_particles=4K, num_particles_init=2K, max_composites=500` | ~10x |

Result: 3.9ms/step → ~258 FPS potential (at default 2K particles, no bond forces)

Remaining known perf issues (not yet addressed):
- `find_neighbors_for_particle`: `pack_slot` vmap is O(max_neighbors × max_candidates)
  per particle — an O(N × K²) total, could be simplified to O(N × K)
- Bond forces still O(max_composites) when re-enabled — needs alive-composite filtering

---

## Phase 5: Polish and Optimization

- [ ] Verify FPS is good in full GUI run
- [ ] Async rendering overlap (simulate N+1 while rendering N)
- [ ] Scale up to larger particle counts once perf baseline is solid
- [ ] Profile with jax.profiler if FPS drops when chemistry becomes active
- [ ] Interactive parameter controls (live sliders)
- [ ] Statistics overlay (species distribution, composite count, energy histogram)
- [ ] Frame recording (save to disk for videos)

---

## Phase 6: Evolution and Emergence

- [ ] Group fitness metrics
- [ ] Spatial compartmentalization detection
- [ ] Parameter sweep infrastructure
- [ ] NCA-style learned update rules on internal state vector
- [ ] Mass conservation mode (FlowLenia / reintegration tracking inspired)
- [ ] Optimization loops (evolve interaction matrices for specific goals)

---

## Known Issues / Notes

- Fixed: `chemistry.py` — shape mismatch in `apply_composite_decay` CoM calc
- Fixed: `renderer.py` — unused `u_window_size` uniform removed (GLSL optimizes it away)
- Fixed: `attempt_fusion` scan O(N) → O(max_fusions_per_step)
- Fixed: `find_free_slots` O(N²) → O(N log N)
- Fixed: bond forces gated off by default (`use_bond_forces=False`)

---

## Implementation Log

### 2026-03-14
- Architecture designed after researching: Hash Chemistry (Sayama), FlowLenia/Particle Life/NCAs/Boids, JAX+rendering options
- Key decisions: 2D, configurable boundaries, toggleable composite viz, ModernGL+pygame
- Phase 0 files created
