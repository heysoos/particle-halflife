# Million-Particle Scaling — Design

**Date:** 2026-07-30
**Status:** Design (approved for spec review)
**Target:** 1–2M particles with full chemistry at 60 fps; 5M+ at interactive rates
**Hardware:** RTX 3080 Laptop (8 GB, GA104, compute 8.6), 95 W power cap

---

## 1. Why the current architecture cannot get there

### 1.1 Measured baseline

All numbers measured on the target machine, default `SimConfig`, `angle_mode="off"`,
`emit_events=False` (i.e. *easier* than the live app, which enables both).

Fixed world 200×112.5 — density grows with N:

| N | step | steps/s | peak VRAM |
|---|---|---|---|
| 5,000 | 6.01 ms | 166 | 0.11 GB |
| 10,000 | 7.54 ms | 133 | 0.33 GB |
| 20,000 | 11.77 ms | 85 | 0.45 GB |
| 40,000 | 21.76 ms | 46 | 0.88 GB |
| 80,000 | 156.51 ms | 6.4 | 1.88 GB |

Constant density — world area scales with N:

| N | world | step | steps/s |
|---|---|---|---|
| 5,000 | 200×112 | 6.10 ms | 164 |
| 20,000 | 400×225 | 11.60 ms | 86 |
| 80,000 | 800×450 | 38.41 ms | 26 |
| 200,000 | 1265×711 | **OOM** (3.11 GiB single allocation failed) | — |

Two facts set the agenda:

1. **At 80k, holding density constant is 4.1× faster** than letting it grow (38.4 ms vs 156.5 ms).
   Cost tracks *neighbors per particle*, not particle count.
2. **The wall at 200k is memory, not speed.** An independent run at N=1e6 reported
   `Can't reduce memory use below 1.09GiB ... only reduced to 12.25GiB, down from 17.51GiB`
   before failing. 3.6 GB of that is *static* allocation.

### 1.2 The padding tax — and the caps that actually bind

**Corrected 2026-07-30.** An earlier version of this section claimed composites average 8.3
members against a 256 cap, implying the cap could be cut ~30×. That measurement was taken on
an under-developed state. Measured at **steady state (1000 steps, clean GPU)**:

| | N=20,000 | N=5,000 | cap | verdict |
|---|---|---|---|---|
| cell occupancy | mean 57.1, p99 388, **max 453** | mean 14.3, max 141 | `cell_capacity=64` | **23.1% / 5.4% of cells OVERFLOW** |
| neighbors/particle | mean 178.7, p99 256, max 256 | mean 105.7, max 256 | `max_neighbors=256` | **4.7% / 0.2% saturate** |
| composite size | mean 5.7, p99 152, max 256 | mean 13.9, p99 184, max 255 | `max_composite_size=256` | binding at the tail |
| edges/composite | mean 5.6, p99 191, max 333 | mean 15.5, p99 228, max 307 | `e_max=512` | not binding |

**Both interaction caps bind at N=20,000, and shrinking them would corrupt the physics.**
The composite-size distribution is extremely heavy-tailed — mean 5.7 but p99 152 and max
pinned at the 256 cap — so the padding is not free headroom to reclaim; it is buffer for a
real tail. The mean/cap ratio is a misleading statistic here.

This changes the conclusion: the padding cost is real for *memory* at N=1e6 (neighbor list
1 GB, `members` 512 MB, `edges` 2 GB — a composite is 5,149 B, 161× a particle), but it
**cannot be fixed by lowering the caps**. It requires the representation change in §4.1,
where cost tracks the real tail rather than a worst-case bound.

### 1.2b Correctness: the sim is lossy today, at two levels

Independent of scaling, at the sizes actually being run:
- **23.1% of grid cells overflow `cell_capacity=64` at N=20,000** (max occupancy 453, 7× the
  cap); 5.4% at N=5,000. Particles past slot 64 are invisible to forces, fusion and ring closure.
- **4.7% of particles saturate `max_neighbors=256` at N=20,000** (mean 178.7), so their force
  sums are truncated too.

Both must be fixed regardless of which scaling path is chosen, and neither is a performance
issue — they are silent physics corruption.

### 1.3 The transient tax

`find_all_neighbors` ([spatial.py:129-209](../../../halflife/spatial.py#L129-L209)) allocates
**24.6 KiB of transient per particle** — (N, 9·cell_capacity) = (N,576) candidate arrays and
(N,576,2) position gathers, compacted down to (N,256). ≥5.6 GB live at N=1e6.
It evaluates all 576 slots per particle regardless of occupancy, to find ~45 real neighbors.

### 1.4 Other structural costs

- **5 full-N sorts/permutations per step.** Two sort all N elements to extract 64 and 16 items.
- **A 3.07M-element radix sort every step** for the VSEPR angle list (2·C·e_max), measured at
  2.6 ms — ~43% of the entire step at N=5,000. Scales with C·E, not N.
- **6+ full (C, e_max) grid traversals per step** (degree, bond forces, scission incl. a
  (C,512) RNG draw, liquid-drop, radius of gyration) — 9.2M edge-slot visits for typically
  <10k real bonds.
- **256 sequential `fori_loop` iterations per step** (BFS tree, subtree sums, descendant mask,
  reachable mask) — a fixed launch-latency floor.

### 1.4b Real frame profile at N=20,000 (the sizes actually being run)

**Measured 2026-07-30** on a developed state (1000 steps, 1,737 alive composites), real
`Renderer`, real GL context, idle GPU. Simulation fully awaited before timing the renderer so
async overlap cannot smear sim time into the render number:

| phase | ms | share |
|---|---|---|
| simulation — 8 steps, fully awaited | **82.0** | **89%** |
| `renderer.update()` (bonds mode) | 14.3 | 6% |
| `renderer.render()` | 4.7 | 5% |
| **total frame** | **91.9** | → **10.9 fps** |

Cross-checked against the live app at the same size, which reports 8.0–9.2 fps.

**The simulation is 89% of the frame. The renderer is not the bottleneck at this scale.**

The apparent contradiction between "~98 steps/s" and "~9 fps" is entirely
`steps_per_frame = 8` ([main.py:160](../../../halflife/main.py#L160)): each rendered frame
advances the sim 8 steps at ~10.2 ms each. Lowering it trades simulation progress for frame
rate directly, and is bound to the `-` key already.

This does **not** contradict §1.5: the render path costs 75 ms at N=1e6 and becomes dominant
there. Both are true at their own scale — render work grows with N far faster than the
per-frame sim work does. Fix the sim for today; fix the render path for the scale target.

**Anomaly worth a separate look:** `renderer.update()` in *merged* mode measures **47.6 ms**,
3.3× *slower* than bonds mode's 14.3 ms, despite merged mode skipping the bond-vertex loop
entirely. The merged branch ([renderer.py:1427](../../../halflife/renderer.py#L1427)) is doing
something unexpectedly expensive. Not on the critical path, but it is backwards.

### 1.5 The renderer

`renderer.py:1197` does `jax.device_get` of **19 arrays in full every frame**, including the
entire (C, e_max, 2) edge array and (C, M) member array regardless of how many are alive.
Then per-frame numpy over N (colors, norms, clips), then `.tobytes()` — a full extra CPU copy.
Then a **Python `for` loop over every alive composite** to build bond vertices.

Measured cost of that path in isolation, before OpenGL touches anything:

| N | current host path | on-device density splat |
|---|---|---|
| 100,000 | 4.90 ms | 3.67 ms |
| 1,000,000 | **75.48 ms** | 7.15 ms |
| 2,000,000 | **145.29 ms** | 4.06 ms |

Above ~350k the render path alone exceeds the entire 60 fps frame budget.

**Important nuance:** drawing is *not* the problem. Measured on this GPU at 1280×720,
5M additively-blended 1px points cost **2.86 ms**. Moving 5M positions costs 24.7 ms.
**Data movement is 8.6× more expensive than rasterisation.** The fix is to stop moving data,
not to draw more cleverly.

### 1.6 Correctness note (pre-existing, unrelated to scaling)

Measured max cell occupancy is **372 against `cell_capacity=64`**. Particles past slot 64 are
silently invisible to forces, fusion and ring closure by step ~2000. The sim is already lossy
at 5k particles. Any rewrite must not reproduce this; any interim fix should address it.

---

## 2. Platform decision

### 2.1 What is and isn't possible under WSL2

Verified first-hand on this machine through the real pygame + moderngl stack:

| capability | WSL2 status |
|---|---|
| WSLg default GL adapter | **Intel UHD iGPU**, GL 4.1 — *not* the RTX 3080 |
| `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` | RTX 3080, GL 4.2 |
| GLSL version string | capped at 4.20 |
| Compute shaders | **WORK** via `#extension GL_ARB_compute_shader : require` |
| SSBOs | **WORK** via `GL_ARB_shader_storage_buffer_object` |
| Zero-copy SSBO-as-VBO | **WORKS** (same `moderngl.Buffer` for storage + vertex array) |
| Float atomics in GLSL | **NO** (`GL_NV_shader_atomic_float` absent) — integer only |
| CUDA ↔ OpenGL interop | **IMPOSSIBLE** — `CUDA error 304`; NVIDIA's CUDA-on-WSL guide §5.2 states *"OpenGL-CUDA Interop is not yet supported"* |
| Vulkan | no ICD installed |
| JAX on native Windows | **does not exist** — Linux-only CUDA wheels |

The binding consequence: **JAX ⟹ WSL2 ⟹ a mandatory host round-trip, permanently.**
There is no path from a JAX device array into a GL buffer without going through the host.

Measured round-trip, with a hard cliff at 32 MB:

| payload | D2H | GL upload | total | fps ceiling |
|---|---|---|---|---|
| 1M × vec2 fp32 (8 MB) | 2.11 ms | 1.10 ms | 3.22 ms | 311 |
| 5M × vec2 fp32 (40 MB) | 24.68 ms | 3.10 ms | 27.78 ms | 36 |
| 5M × vec2 **fp16** (20 MB) | 4.78 ms | — | — | — |

D2H is linear at ~4.5 GB/s up to 32 MB, then collapses (32 MB → 7.0 ms, 40 MB → 23.7 ms).
Chunking does not help. **fp16 positions are worth 5×** and keep you under the cliff.

### 2.2 Decision: NVIDIA Warp on native Windows

**Chosen.** Rationale:

- **Measured on this GPU.** A particle-life force kernel with `wp.HashGrid` at current density
  (~45 neighbors/particle): 1M at **5.59 ms (179 steps/s)**, 5M at **32.12 ms (31 steps/s)**.
- **Warp ships `win_amd64` wheels.** On native Windows you get working
  `RegisteredGLBuffer` zero-copy into existing ModernGL VBOs — the host round-trip disappears.
  JAX cannot follow; Warp can.
- **Migration discipline already paid.** Warp has no dynamic containers either, so the
  fixed-size-array + `-1` sentinel + mask idiom carries over unchanged. But real `if`/`for`
  control flow returns, which makes `graph.py` (`bfs_tree`, `subtree_sums`) and `chemistry.py`'s
  conflict batching *simpler*, not harder.
- **Incremental migration is possible.** `wp.from_jax` / `wp.to_jax` are zero-copy, so hot
  kernels can move one at a time rather than big-bang.

Rejected alternatives:

- **Raw ModernGL compute shaders** — highest ceiling (a validated prototype hit 5M at 16.14 ms
  = 62 fps on this GPU), but writing `chemistry.py`'s 2,189 lines of graph algorithms in GLSL
  with no printf and no debugger is a different order of pain. Keep as a fallback for the
  render path only.
- **Taichi** — effectively abandoned. Zero commits to `master` since 2025-07-30; last release
  v1.7.4 (2025-07-31); 924 open issues. GGUI broken under WSL2 ([taichi#8055](https://github.com/taichi-dev/taichi/issues/8055), open 3 years).
- **CuPy RawKernel / cuda-python / PyCUDA** — blocked under WSL2 by the same error 304, and
  on native Windows Warp dominates them on ergonomics. `numba-cuda` is explicitly in
  maintenance mode.
- **wgpu-py** — compute pipelines fail here (`DownlevelFlags(COMPUTE_SHADERS) ... not supported`);
  v0.31.1+ needs Python 3.11 (this project is on 3.10.12).

**Critical implementation constraint discovered during benchmarking:** spatial sorting is
**mandatory, not an optimisation**. Unsorted, Warp at 1M runs at 10.7 steps/s — no better than
JAX at 200k. Sorted, 179 steps/s. That is a **16.8× swing at 1M and 21.1× at 5M**. Reorder
`pos`/`vel`/`species` into cell order every step via
`wp.utils.radix_sort_pairs(keys, values, count, begin_bit=0, end_bit=None)` plus gather kernels.
Grid build itself is cheap (0.61 ms at 1M).

Two `wp.HashGrid` gotchas: `HashGrid(dim_x, dim_y, dim_z)` sizes *hash buckets*, not spatial
extent; and queries return points outside the radius on hash collisions, so the kernel must
re-check distance itself.

---

## 3. Target configuration

Chosen regime: **bigger world *and* denser.** Anchor point for design:

| | value |
|---|---|
| N | 1–2M (full chemistry), 5M (reduced mode) |
| world area | ~20× current |
| density | ~10× current (2.22 particles/unit²) |
| neighbors within `interaction_radius`=8 | **~447 per particle** |
| naive pair evaluations | 447M/step at N=1e6 |

447M pairs/step at 60 fps is 27 G pair-evaluations/s. That is *near* the ceiling of a
GROMACS-class cluster kernel on this hardware, with no margin for chemistry. **Direct pairwise
summation does not close at this density.** This is what makes §4.2 load-bearing rather than
optional.

---

## 4. Architecture

### 4.1 Data layout

**Particles — structure of arrays, mixed precision.**

| field | dtype | bytes | @2M |
|---|---|---|---|
| `position` | fp32 × 2 | 8 | 16 MB |
| `velocity` | fp16 × 2 | 4 | 8 MB |
| `species` | uint8 | 1 | 2 MB |
| `composite_label` | int32 | 4 | 8 MB |
| `mass`, `energy`, `age` | fp16 | 6 | 12 MB |
| **total** | | **23 B** | **46 MB** |

Position stays fp32 for world-coordinate accuracy. Velocity and derived scalars go fp16 —
this is the same split `par-particle-life` uses, and it is why their buffers are separate.
(A further option, if bandwidth binds: unorm16 positions *relative to cell origin*, as
bleuje's physarum does — 0.02 px resolution at 1280 wide. Not needed at 2M; noted for 5M+.)

**Bonds — an edge list, not per-composite padded grids.** This is the single highest-leverage
change in the whole design.

```
edges        (E, 2) int32      E ≈ 2N        →  32 MB at 2M particles
edge_species (E, 2) uint8      (cached)      →   8 MB
```

versus the current `(C, 256)` members + `(C, 512, 2)` edges at 5,149 B/composite. Holding the
current pool ratio C ≈ 0.6·N, that is **3.1 GB at 1M particles and 15.4 GB at 5M** — on its own
enough to exceed the card. A **~100× reduction from one data-structure change**, and it deletes
`max_composite_size` and `e_max` as concepts entirely.

Composites become *implicit*: a `composite_label` per particle, derived from connected
components over the edge list. Per-composite aggregates come from segmented reductions over
labels, not from iterating a padded pool.

**Total simulation state at 2M particles: well under 200 MB on an 8 GB card.**
Memory stops being the binding constraint.

### 4.2 Forces — P³M split

Split the force at `r_split ≈ 2` (just above `fusion_radius=1.5`, comfortably above
`repulsion_radius=0.8`):

**Short range, `r < r_split` — exact, on a tight cell list.**
Covers the hard core and all fusion contact. At 10× density that is
π·4·2.22 ≈ **28 neighbors per particle** instead of 447 — a **16× reduction in exact pair work**.
Cell size `r_split/2 = 1`, 5×5 stencil (see §4.3 for why r/2).

**Long range, `r_split < r < 8` — on a mesh, density-independent.**

```
F_long(i) = Σ_s  a[s_i, s] · (∇W * ρ_s)(x_i)
```

- Scatter each particle's species into S density fields `ρ_s` (bilinear/cloud-in-cell deposit,
  integer `atomicAdd` — note GLSL has no float atomics here, so accumulate fixed-point).
- Convolve each field once with the radial kernel `W` (separable stencil, or FFT).
- Gather per particle: S multiply-adds against the precomputed gradient fields.

Cost is `O(N + S·G)` where G is grid cells — **independent of density**. At the target
configuration: grid at h=1 over ~900k unit² = 900k cells × 4 species × 4 B = **14.4 MB**;
a separable 17-tap convolution is ~60M taps. Negligible against 447M direct pairs.

**Preserving per-species-pair force shape.** The current kernel has per-pair `peak_fraction`
and `cutoff_fraction`, so the radial shape differs per (s_i, s_j) and cannot be factored out of
a single convolution. Solution: express each pair kernel in a small basis of B fixed radial
kernels (e.g. B=3 Gaussians at different radii),
`F_long(i) = Σ_b Σ_s c_b[s_i,s] · (∇W_b * ρ_s)(x_i)`, giving S·B convolutions.
At S=4, B=3 that is 12 separable convolutions of a 900k-cell field — still trivial.

This is standard practice (P³M in cosmological N-body; hybrid particle-field MD), and it is the
formulation Particle Lenia and FlowLenia — already in this project's inspiration list —
point directly at.

**Physics caveat to validate:** the mesh long-range term is a mean-field approximation. With
~447 neighbors in the long-range band the discrete sum is already effectively smooth, so grid
error should sit well below the existing discreteness noise. This needs an A/B check against
the direct kernel at current N before it is trusted (see §6).

### 4.3 Neighbor search

Per the literature and the Warp benchmark, in priority order:

1. **Morton-reorder the particle arrays every step** (two-level: cells in Morton order,
   particles Morton-ordered within cell, cell boundaries coinciding with the search grid).
   Worth **16.8–21.1×** as measured in Warp on this GPU; independently measured at ~2× by two
   other groups. This is not optional.
2. **Do not materialise a neighbor list.** [Bramas et al. 2024](https://arxiv.org/pdf/2406.16091),
   studying exactly this regime (30–40 neighbors, few particles per cell), found the fastest
   NVIDIA strategy is **one thread per particle looping neighbor cells directly, no shared
   memory, no stored list** — all classical shared-memory tiling schemes lost. This is also the
   bandwidth-optimal choice: ~2.6 GB/step materialised vs ~200 MB/step traversed.
3. **Cell size = `r_split/2`.** In 2D, cell=r gives a 3×3 stencil covering 9r² for a useful πr²
   — 35% efficiency. cell=r/2 gives 5×5 over 6.25r² — **50%**. cell=r/3 gives 58%, diminishing.
   r/2 is the sweet spot, matching both Hoetzlein and the LAMMPS GPU default.
4. **Build the cell list by boundary detection, not sorting.** After Morton reordering, cell
   membership is already contiguous — one thread per particle comparing with its predecessor.
   If a sort is needed, use a single-radix counting sort on exact bins (Hoetzlein: 5–10× over
   4-pass byte radix, 15 kernels → 4).

### 4.4 Chemistry at scale

With composites as an edge list, every current `(C, e_max)` grid traversal becomes linear in
*real* edges:

| operation | current | redesigned |
|---|---|---|
| `compute_degree` | O(C·E) scatter over padded grid | O(E) scatter-add over edge list |
| `compute_edge_bond_forces` | O(C·E) = 1.54M slots | O(E), one thread per real edge |
| radius of gyration | O(C·M), (C,256,2) arrays | segmented reduction by label, O(N) |
| bond scission | O(C·E) grid + (C,E) RNG | O(E), one thread per edge |
| angle list | 3.07M-element sort/step | counting sort over E, O(E) |
| liquid-drop half-life | O(C·E) + O(C·M) | O(N + E) segmented reductions |

**Connected components: mostly, don't.** This was an open question; ALIEN answers it (§7.1).
Their core physics — bond forces, angle forces, fusion, scission — is *purely local* (a cell
plus its ≤6 bonded neighbors) and needs no global connectivity at all. Global labels are
computed only where genuinely required, and then:

- by **approximate label propagation** (`atomicMin` + pointer chasing), with a hardcoded 30-hop
  cap, launched exactly 3 times (≤90 hops) and explicitly **not run to convergence** — the
  source comment is literally `// Heuristics to cover connected cells`;
- **only every 3rd timestep**, and only when the feature needing it is enabled.

Adopt the same shape: identify which consumers actually need a global label (liquid-drop
half-life, radius of gyration, per-composite aggregates) and run an approximate, capped,
every-Kth-step relabel for them, rather than an exact per-step pass.

**Per-composite aggregates live at the root, with no composite array.** ALIEN `atomicAdd`s
COM, velocity, angular momentum and angular mass onto the *member with the lowest index*.
Periodic-boundary COM is handled by a 2-bit mask recording whether the cluster touches the
left/upper third of the world, then shifting members past the two-thirds line. That is a
cheaper and simpler answer than a segmented reduction over sorted labels — adopt it.

**The expensive graph surgery does not scale with N and is not a problem.** BFS spanning trees,
subtree sums for fission cut-scoring, and reachability for scission already run on a *compacted
batch* — `max_fissions_per_step=64`, `max_scissions_per_step=32`. Measured steady-state demand
is ~15 fissions/step. That is a fixed cost that amortises away as N grows. Keep the batching.

### 4.5 Rendering

On native Windows: `RegisteredGLBuffer` maps the Warp position/color buffers directly as GL
vertex buffers. **Zero copy, no host round-trip.** Draw as `GL_POINTS`.

Measured on this GPU at 1280×720: 5M 1px additively-blended points = **2.86 ms**. Rendering is
not the bottleneck once the data stops moving. Sprite size is the only real fillrate lever
(4px → 5.09 ms, 8px → 10.59 ms at 5M).

A compute-shader density splat (`atomicAdd` into an image, then a fullscreen tonemap) is a
*later* optimisation, not a day-one need — published gains are 3.5–136× over `GL_POINTS`
([Schütz et al.](https://arxiv.org/pdf/2204.01287)) and biggest in clumpy distributions, which
composites produce by design. Revisit if point size must grow.

The bond renderer needs rework regardless: the current Python loop over alive composites is
fatal at scale. With an edge list it becomes a single `GL_LINES` draw over the edge buffer,
with no host involvement.

---

## 4.6 Lessons from ALIEN (chrxh/alien)

[ALIEN](https://github.com/chrxh/alien) is a CUDA artificial-life simulator solving a very
similar problem — cells with an explicit bond graph, angle constraints, fusion, decay,
emergent structures — at large scale. 103k LOC total, 19.6k LOC of CUDA. It is the closest
existing analogue to this project and the most useful reference found.

### 4.6.0 Caveat: ALIEN's demonstrated scale is ~158k particles, not millions

The README's entire performance claim is one unqualified sentence ("optimized for large-scale
real-time simulations with millions of particles") — no GPU, no fps, no cell count, and nothing
in 15 releases of notes, the issues, or the HN thread. Digging harder produced the following,
and it matters:

- **The author's own flagship shipped preset is 157,764 particles** on a 5000×1500 world
  (`Evolution Presets/Hanging Garden`, retrieved from the live `api.alien-project.org` catalogue
  the in-app browser uses). The second and only other current preset is 40,261. **The demonstrated
  scale is ~1.6 × 10⁵ — roughly 6× below "millions."**
- **Published TPS numbers exist but carry no cell count.** The author A/B-benchmarks on an
  RTX 4090 headless at **~200–250 TPS** (PRs #707/#708/#709), on a sim described only as
  "the SPH-heavy test simulation." Not convertible to throughput, not comparable across projects.
- **VRAM is ~2,228 B per live cell** (derived by compiling the actual structs: 512 B `Object`
  + 368 B `NeuralNet` + alignment, heap **double-buffered**, plus 404 B of per-slot arrays), so
  1M cells ≈ 2.07 GiB *before* the 3× array growth slack. Plus ~166 MB of fixed precomputed RNG
  tables and 12 B per world grid unit (86 MB for a 5000×1500 world, even empty).
- **Population growth, not world size, is what kills it.** A third-party report describes
  30 fps degrading to 4 seconds/frame overnight as offspring count evolved from ~5 into the
  tens of thousands — a 120× slowdown from population alone.
- The README's stated "compute capability 6.0+" is **stale**: `develop` throws below CC 7.5,
  which explains the GTX 10xx failure reports.
- v5-alpha is explicitly not yet performance-tuned (author, discussion #810, 2026-07).

**The reframe this forces.** Comparing interaction counts rather than particle counts:
ALIEN at 158k particles with ≤10 neighbors each ≈ 1.6M interactions/step; this project at
20k particles with ~45 neighbors ≈ 0.9M interactions/step. **The two are within ~2× of each
other on raw pairwise work.** ALIEN runs it at ~200–250 steps/s on a 4090 against this
project's 85 steps/s on a 3080 — call it 3–5× faster per interaction once hardware is
discounted, not the 100× the headline implies.

So ALIEN's apparent scale advantage is mostly **a much lower neighbor count** (interaction
range 1.6 on a 1.0 grid, ≤10 candidates per cell) rather than a fundamentally better engine.
That is genuinely encouraging for the targets in §3 — but it means **ALIEN is an architecture
reference, not an existence proof that millions run at interactive rates.** No such proof was
found. Treat §4.6's techniques as well-motivated engineering, and keep the independent
evidence in §7 (bleuje's 5.77M at 60 fps on an RTX 2060; Hoetzlein's 2.1M at 12 fps) as the
actual scale precedent.

### 4.6.1 The historical validation

ALIEN v2.0 had explicit rigid-body **cell-cluster objects**. v3.0 **deleted them** in favour of
per-cell soft-body springs. `MapSectionCollector.cuh` survives as entirely commented-out dead
code still referencing `Cluster*`, `List<Cluster*>`, `DynamicMemory*`.

They had this project's `CompositeState`, at scale, and removed it. That is independent
confirmation of §4.1's edge-list decision.

### 4.6.2 Phase-index amortisation — the highest-leverage idea, and it applies today

ALIEN runs a **22-kernel baseline step, and a 49-kernel step every 3rd timestep**. Angle
forces, inner friction, connected components and *all* chemistry are gated on
`timestep % 3 == 0` (`TIMESTEPS_PER_CELL_FUNCTION = 3`). This ships in production.

Direct candidates here, in cost order:
- **VSEPR angle forces** — currently 2.6 ms/step, ~43% of the step at N=5,000, driven by a
  3.07M-element radix sort for the angle list.
- **Liquid-drop half-life** recomputation (`(C,E)` bond sum + `(C,M)` radius of gyration).
- **Bond scission** (`(C,E)` grid plus a `(C,E)` RNG draw).

This fits the existing `static_argnums` model exactly: add a phase index as a static argument
and XLA caches K variants — the same mechanism ALIEN gets from caching `cudaGraphExec_t` keyed
on a config struct containing `timestepMod3`. **This is applicable to the current JAX codebase
before any rewrite**, and is the one item that could pay off immediately.

### 4.6.3 A compact hot mirror for the neighbor scan

ALIEN keeps a 40-byte `LightObject` mirror of only the fields the neighbor scan touches
(`pos`, `vel`, `density`, `type`, `self`, chain pointer, `numConnections`, `flags`), guarded by
`static_assert(sizeof(LightObject) == 40)` and the comment *"growing it directly costs memory
bandwidth in the hot SPH kernels"*. Against a 512-byte `Object`, that is a **12.8× bandwidth
reduction on the O(N·k) inner loop**.

This project is already SoA, so the mechanism differs, but the principle holds and is testable:
the force kernel must read a minimal `(position, species)` working set, never gather from every
state array. Assert the packed width in a test so it cannot silently grow.

### 4.6.4 Angle constraints via `angleFromPrevious` on sorted edges

ALIEN stores each cell's connections in **circular angular order**, each carrying
`angleFromPrevious` — the rest-angle gap to the previous connection, summing to 360°. Angle
forces then apply only to *consecutive* pairs: **O(degree) instead of O(C(degree,2))**.

This deletes `build_angle_list` and its `(N, C(max_valence,2), 3)` triple array outright — the
triples become implicit as `(i-1, i)`. It is a structural simplification, not a micro-optimisation.

Two further details worth taking: the penalty is **linear in |Δθ|, not quadratic** (a
constant-magnitude restoring torque, far more stable at high stiffness), and the force is
tangential-only via 90° rotations with **no `atan2` in the hot path**.

Keep the VSEPR *physics* (emergent rest angle from chord-Coulomb repulsion suits hash chemistry
better than ALIEN's stored, genome-settable angles) but adopt their *data structure*.

### 4.6.5 Deferred structural operations — already validated

Nothing in ALIEN mutates topology during physics. Fusion, bond deletion and cell death all
enqueue into a bounded per-step queue sized `1000 + numObjects/2` that **silently drops on
overflow**, drained afterwards by five substep kernels. Bond deletion is lock-free (atomic
bitmask mark, then each object compacts its own ≤6 connections); addition uses a
**pointer-ordered non-blocking double lock** that drops and retries next step on contention.

This is `max_fissions_per_step` / `max_scissions_per_step` deferral, independently arrived at
and validated at much larger scale. Keep it. One refinement to adopt: ALIEN compacts work into
**one dense queue per cell type**, so each kernel processes only its own list rather than all N
with a divergent branch — directly applicable to per-reaction-channel masking here.

### 4.6.6 Rendering — cull before you transfer

**ALIEN's default build round-trips through the host**; CUDA-GL interop is opt-in via
`--interop`, and even then there are 9 blocking `cudaDeviceSynchronize()` calls per frame.
Interop is not what makes it fast.

What makes it fast is **GPU-side frustum culling plus atomic compaction before extraction**, so
the VBO only ever contains on-screen entities. At 5M particles with ~20k visible that is a
~250× cut in transfer — and it is implementable in JAX today as a `where` + `cumsum` compaction,
independent of any interop plumbing. Pair it with drawing a **constant** vertex count and
pushing the surplus off-screen (`gl_Position = vec4(-2,-2,-2,1)`) so the host never needs the
count and no sync is required.

Their LOD trick is also worth taking: below zoom 5, **bonded cells stop being drawn as sprites
entirely** — bond lines and fills become the low-LOD representation, while free particles stay
and are drawn *larger*. Kills moiré and saves fragment shading exactly when on-screen particle
count peaks.

### 4.6.7 What does NOT transfer

- **Their force kernel is easier than this one.** Species-agnostic SPH pressure/viscosity,
  interaction range 1.6 on a 1.0 grid, a 5×5 = 25-cell scan with ≤10 candidates each. There is
  **no species-pair force matrix anywhere in ALIEN** (the only colour matrix is
  `attackerFoodChainColorMatrix`, used for predation). This project's radius-8, ~45-neighbor,
  species-indexed kernel will not inherit their throughput by copying their structures.
- **Their bond chemistry is simpler.** Bond breaking is "distance > 3.6 → dissolve *all* of that
  cell's bonds", plus a probabilistic force threshold. No hash-derived bond energies, no
  Arrhenius, no liquid-drop fissility, no max-binding-energy cut. **This project is ahead here;
  there is nothing to import.**
- **The 1×1 grid with a hard 10-per-cell cap silently drops neighbors.** Acceptable only because
  they enforce `minObjectDistance = 0.3` and clamp forces hard. Copying it into a sim that
  genuinely wants ~45 neighbors would corrupt the physics — and note §1.6, where this project
  already has exactly that bug.
- **They abandoned determinism.** The RNG is a 160 MB precomputed table read through one global
  atomic ring index, so which thread gets which number depends on scheduling. Combined with
  `atomicAdd` float accumulation and best-effort lock/queue drops, ALIEN is not reproducible.
  **Do not follow this** — the threaded `rng_key` is a research asset for an open-ended-evolution
  project, not a nicety.
- **The semi-space copying GC is the wrong tool.** It costs 4× VRAM for the live set and exists
  only because their edges are raw pointers into a heap of variable-size objects. Index-addressed
  fixed-size arrays need at most a prefix-sum permutation.

### 4.6.8 Miscellaneous techniques worth noting

- **fp32 everywhere in physics; fp64 only for global reductions** (`externalEnergy`, statistics).
  A clean rule to adopt verbatim.
- Launch geometry is **constant and independent of N** (16384 blocks, grid-stride loops), which
  is what makes the `cudaGraphExec_t` cache viable. ~49 kernel launches per step ≈ 250–320 µs of
  pure launch overhead — a reminder that kernel count, not just kernel cost, is a budget.
- The neighbor kernel is **one block per particle, one thread per grid cell** (block size set to
  exactly the scan rectangle, 25 threads), reduced with cooperative-groups warp reductions.
  Contrast with [Bramas et al.](https://arxiv.org/pdf/2406.16091) (§4.3), who found
  thread-per-particle fastest on NVIDIA at this neighbor count — ALIEN's choice suits their much
  smaller per-particle neighbor count. **Benchmark both; do not assume.**
- They also use a deliberate one-step-lag on density (*"Optimization: using the density from
  last time step"*) — the same trick already used here for `rep_pe`.

---

## 5. Phasing

Each phase leaves the project runnable. **This spec is the umbrella design; Phases 1–4 are each
large enough to warrant their own implementation plan**, written when that phase starts rather
than up front.

**Phase 0 — free wins, hours, no architecture change.**
- `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` — measured **10.5× on draw at 1M** (8.16 → 0.78 ms).
  One env var, zero code impact. Do this regardless of everything else.
- fp16 positions on the render transfer path — 5× on D2H at 5M, and stays under the 32 MB cliff.
- Delete the per-frame numpy work in `renderer.py` (colors, norms, clips move to the shader).
- Fix the `cell_capacity=64` vs occupancy-372 correctness bug.

**Phase 0b — wins that need no rewrite.** Applicable to the current JAX codebase:
- **Phase-index amortisation** of angle forces, liquid-drop half-life and scission (§4.6.2).
  Measured: the VSEPR path costs **2.3 ms of a 10.2 ms step at N=20,000 (23%)** — 7.9 ms with
  `angle_mode="off"` vs 10.2 ms with `vsepr`. Running it every 3rd step is worth roughly
  1.18×, not the 1.3–1.5× claimed earlier. Real but modest; validate composite geometry via
  the analysis pipeline before keeping it.
- **Frustum culling + compaction before the render readback** (§4.6.6). A `where` + `cumsum`
  in JAX, no interop needed, and the single largest cut to the 75 ms host path at 1M.
- **Constant vertex count** with off-screen surplus, removing the host-side count sync.
- **Kill the Python loop over alive composites** in `renderer.py:1288`. At N=20,000 there are
  **~1,700–1,970 alive composites**, so that loop runs ~1,900 Python iterations *per frame*.
  This is a strong candidate for the app feeling slower than the 10.2 ms step time implies.

**Do NOT lower `max_composite_size`, `e_max`, or `max_neighbors`** — §1.2 measures all of the
interaction caps as binding at steady state. An earlier draft of this spec recommended cutting
`max_composite_size` to 16 on the basis of a mean of 8.3; that measurement was premature and
the recommendation was wrong.

**Measurement discipline note.** `make_run_n_steps`' `n` is a `lax.scan` length, so **every
distinct chunk size triggers a fresh ~15–20 s XLA compile**. Timing a chunk size that has not
been compiled measures compile time amortised over the chunk. This produced fake "1,215 ms/step"
figures that disagreed with the true ~10 ms by 100×. Always compile the exact chunk length once,
untimed, before timing anything — and check `nvidia-smi` for competing GPU work first.

**Phase 1 — Warp prototype, validate before committing.**
Port only the neighbor search + short-range force path to Warp, **with spatial sorting from the
start**, keeping everything else in JAX via zero-copy `wp.from_jax`. Validate the measured
179 steps/s at 1M against the real chemistry. This is the go/no-go gate for the rewrite.

**Phase 2 — edge-list composites.**
Replace `CompositeState` with the edge list + connected-components labelling. Rewrite the
chemistry kernels against it. Largest single piece of work (~2,200 lines) and the one that
removes the memory wall.

**Phase 3 — P³M long-range forces.**
Add the mesh path with the radial-basis decomposition. A/B against the direct kernel at current
N to quantify the mean-field error before enabling by default.

**Phase 4 — native Windows + zero-copy rendering.**
Move the interactive app to native Windows Python, wire `RegisteredGLBuffer`, delete the host
round-trip. Single `GL_LINES` draw for bonds.

**Effort estimate:** ~4,300 lines rewritten (`chemistry.py` 2,189, `step.py` 624, `state.py` 320,
`interactions.py` 214, `spatial.py` 209 — deleted outright, `graph.py` 137);
~3,800 lines preserved (`renderer.py`, `main.py`, `render/*`); ~2,300 lines of `analysis/`
largely unaffected.

---

## 6. Success criteria and validation

| criterion | target |
|---|---|
| 1M particles, 10× density, full chemistry | ≥ 60 fps |
| 2M particles, 10× density, full chemistry | ≥ 30 fps |
| 5M particles, reduced mode | ≥ 30 fps |
| VRAM at 2M | < 2 GB (leaves headroom on 8 GB) |
| cell-list losslessness | zero silently-dropped neighbors at target density |

**Physics-equivalence gates** (each must pass before the corresponding phase is enabled by
default):

- Warp force kernel vs JAX `compute_all_forces`: per-particle force agreement to fp32 tolerance
  on an identical warmed state.
- Edge-list chemistry vs current: identical fusion/fission event sequences from a fixed seed
  over ≥1000 steps.
- P³M vs direct: composite size distribution, mean bond length, and the Tier-5 openendedness
  metrics statistically indistinguishable over a 3000-step run at current N.

The existing `halflife/analysis/` diagnostic pipeline is the instrument for the third gate —
run `--scenario current_experiment` before and after and diff the reports.

---

## 7. Reference points

| system | scale | hardware | notes |
|---|---|---|---|
| Sage Jenson physarum | 5–10M agents, real-time | GTX 1070 | field-mediated, no neighbor search |
| bleuje interactive-physarum | 5.77M @ 60 fps | RTX 2060 | 12 B/agent bit-packed; open source |
| bleuje, larger grid | 13.1M | RTX 4060 | 1920×1088 |
| Hoetzlein SPH | 2.1M @ 12 fps | GTX Titan (2013) | counting-sort cell list |
| HOOMD-blue LJ | ~3M @ 100 neighbors | 3 GB GPU | hand-tuned CUDA; the honest ceiling |
| par-particle-life | caps neighbors above 200k | — | *deliberately approximate* above ~50k |
| ALIEN, author's shipped preset | **157,764 particles** | unstated | ~200–250 TPS headless on a 4090, cell count unpublished |
| **this design (target)** | **1–2M full chemistry** | **RTX 3080 Laptop 8 GB** | P³M + edge list + Warp |

Jenson's numbers are not a like-for-like target: his agents never interact with each other
directly, so there is no neighbor search at all — per agent it is 3 texture reads and one
deposit. This project's per-particle work is intrinsically ~20–50× heavier. The realistic
ambition is HOOMD-class scale with richer chemistry, not physarum-class scale.

---

## 8. Open questions

1. ~~**Connected components cadence.**~~ **Resolved by §4.6.1/§4.6.4** — ALIEN's answer is an
   approximate, hop-capped, every-3rd-step label propagation, with per-composite aggregates
   accumulated at the lowest-index member rather than in a composite array. Remaining sub-question:
   which consumers here genuinely need a global label, and can any of them be made local?
2. **Radial basis size B.** How many fixed kernels are needed to reproduce the per-pair
   `peak_fraction`/`cutoff_fraction` shapes to acceptable fidelity? Fit offline against the
   current kernel; likely B=3, to be confirmed.
3. **`max_fusions_per_step` at 10× density.** Fusion candidate density scales with particle
   density; the current budget of 64 was tuned at ~4.7 fusions/step. Needs re-measurement.
4. **Whether 5M "reduced mode" drops chemistry entirely or only the expensive extras**
   (angles, ring closure, liquid-drop). Defer until Phase 3 numbers exist.
