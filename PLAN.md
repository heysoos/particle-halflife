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

## Phase 5: Polish and Optimization

- [ ] Async rendering overlap (simulate N+1 while rendering N)
- [ ] Profile with jax.profiler, identify bottlenecks
- [ ] Interactive parameter controls
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

- Fixed: `chemistry.py:312` — `apply_composite_decay` computed center-of-mass by multiplying
  `particles.position (N,2)` by `members mask (M,1)` — shape mismatch. Fixed to gather
  member positions via indexed lookup before summing.

---

## Implementation Log

### 2026-03-14
- Architecture designed after researching: Hash Chemistry (Sayama), FlowLenia/Particle Life/NCAs/Boids, JAX+rendering options
- Key decisions: 2D, configurable boundaries, toggleable composite viz, ModernGL+pygame
- Phase 0 files created
