---
name: Shader port design
description: Design notes for a separate shader-based reimplementation of the half-life simulator — what porting the algorithms entails, how the new app would run, and how to drive it from UI/audio.
type: project
---

# Shader port design

**Date:** 2026-05-07
**Topic:** Notes on what would be involved in building a *separate* project that takes the algorithms of this simulator and implements them in GPU shaders. The current JAX project stays as-is — these notes are about a hypothetical sibling project, what porting the math involves, how its host loop would run, and how external inputs (UI, audio/music) would feed parameters in.

## Summary

The algorithms in [step.py](../halflife/step.py), [chemistry.py](../halflife/chemistry.py), [spatial.py](../halflife/spatial.py), and [interactions.py](../halflife/interactions.py) are already shaped like a GPU kernel pipeline — JAX just hides the GPU plumbing. A shader-based reimplementation in a separate project would translate those kernels into GLSL/WGSL, store state in SSBOs or textures instead of JAX arrays, and use the current Python files purely as a *reference design* (algorithmic spec, not code to share). The payoffs of having such a sibling project are portability (browser-friendly demos, distributable binaries with no Python+CUDA dependency), every-frame state ↔ render integration without GPU↔CPU bounces, and the freedom to drive everything at frame rate from external signals like audio. The cost is that fusion/decay/cell-list logic that JAX hides behind clean Python becomes explicit atomics and ping-pong texture/SSBO dances. This is a multi-week greenfield effort, not a refactor.

## Three plausible target stacks

Listed in order of how much new infrastructure each requires:

1. **Compute shaders + a thin host loop** — pick a graphics layer (ModernGL, raw OpenGL via GLFW, sokol_gfx, or wgpu-py) and write GLSL `#version 430 compute` kernels. State lives in SSBOs (shader storage buffer objects), one per field of the equivalent of [state.py](../halflife/state.py). Simplest path to validating the algorithms work in a shader environment.
2. **WebGL2 fragment-shader sim** — encode state into floating-point textures, advance via fragment-shader passes that write to ping-pong FBOs. No compute shaders, no atomics — runs on any browser. Best for "publish a web demo, embed in a page, run on phones." But almost certainly a dead end for this project's chemistry: atomic-free fusion/fission are nearly impossible to express cleanly without compute or storage textures.
3. **WebGPU compute** — same shape as option 1 but in WGSL with browser delivery. Modern, but the ecosystem is still maturing in 2026 and atomics on f32 are limited.

A reasonable build order if this gets picked up: option 1 first (proves the algorithms translate, with the fastest dev loop on desktop), then port the same kernels to WGSL for option 3 once the design is validated.

## Phase mapping: JAX → GLSL compute

The 9 phases in [step.py:151-244](../halflife/step.py#L151-L244) translate as follows. The Python files are referenced as the algorithmic spec — the new project would re-implement, not import.

| Phase | Reference (Python) | GLSL compute equivalent |
|------|---------------------|-------------------------|
| 1. Cell list | [spatial.py](../halflife/spatial.py) — `build_cell_list` writes `(cells_x, cells_y, capacity)` int32 array | One-pass shader: `imageAtomicAdd` to bump per-cell counts, then scatter particle IDs. Needs `GL_ARB_shader_image_load_store`. |
| 2. Neighbor query | `find_all_neighbors` materialises `(N, max_neighbors)` array | Same array as SSBO. Compute shader: one thread per particle, walks 9 cells. |
| 3. Forces | [interactions.py](../halflife/interactions.py) `compute_all_forces` — vmapped pairwise reductions | One thread per particle, accumulate force in registers. The kernel is already pure math — translates almost line-for-line. |
| 4. Integrate | Vector ops in [step.py:191-200](../halflife/step.py#L191-L200) | Trivial. |
| 5. Boundary | `apply_boundary` | Trivial. |
| 6. **Fusion** | [chemistry.py](../halflife/chemistry.py) — slot allocation via `find_free_slots` + scan | **Hardest.** Needs `atomicAdd` on a "next free composite slot" counter, then atomic CAS to claim slots. Multiple pairs racing for the same composite slot need an arbitration pass. |
| 7. **Decay** | `apply_composite_decay` | Per-composite thread, RNG, `imageStore(-1)` to free slot. Tractable. |
| 8. Energy | `compute_total_energy` | Parallel reduction (subgroup ops or two-pass). |
| 9. Counters | Increments | Trivial. |

Phases 1-5 and 8-9 are essentially mechanical translations. Phase 6 (fusion) is where most of the real engineering effort lives — JAX's abstractions are hiding nontrivial parallel-allocation logic that has to be made explicit in shader land.

### Hash chemistry stays simple

The polynomial rolling hash in [chemistry.py](../halflife/chemistry.py) is just integer arithmetic — moves to GLSL untouched. Reaction-rule lookup remains implicit (same species multiset → same hash → same properties). This is one of the design choices in the original project that makes it especially port-friendly: there's no lookup table to translate, just a function.

## How the new app would run

Once the kernels exist, the host loop is roughly:

```python
ctx = moderngl.create_context(require=430)
sim_program = ctx.compute_shader(open('shaders/step.glsl').read())
state_ssbo  = ctx.buffer(reserve=N * sizeof_particle_struct)
# ... bind SSBOs to binding points 0..K
sim_program['u_dt'].value = config.dt
sim_program.run(group_x=ceil(N / 64))   # one workgroup per 64 particles
ctx.memory_barrier()                    # ensure writes visible to next pass
```

A renderer pass on top of the simulation pass is then very cheap: **the same SSBO that the compute shader writes can be bound directly as a vertex buffer for the rendering pass** — zero CPU↔GPU traffic. The frame budget is just `compute(sim) + draw(particles)`, both on-GPU. No Python ↔ GPU sync per frame, which is structurally different from how the current JAX project is wired (it has to `np.asarray(...)` the GPU state out to CPU before the renderer can touch it; see [2026-05-06-render-update-pipeline-audit.md](2026-05-06-render-update-pipeline-audit.md)).

**Web delivery** (option 2 or 3): bundle with [wgpu-py](https://github.com/pygfx/wgpu-py) for desktop testing, then ship WGSL → run in browser. The HUD becomes HTML/CSS overlay; mouse/keyboard handled in JS.

## Driving it: UI and audio inputs

This is where a shader-native version becomes genuinely interesting — every uniform is a knob you can wiggle at frame rate, with no JIT recompile or GPU sync to negotiate.

### UI

The runtime knobs in the original ([state.py:226-257](../halflife/state.py#L226-L257)'s `PhysicsParams`) translate directly to one `uniform float` per knob. UI sets uniforms before each compute dispatch. No recompilation, instant response.

Three UI delivery options for the new project:

- **Pygame** — straightforward if the host is Python and the goal is "looks like the existing app." Slider widgets writing into uniform values.
- **Dear ImGui** via [pyimgui](https://github.com/pyimgui/pyimgui) — nicer dev-time UI, draws on top of the moderngl context. Probably the most pleasant option for a desktop-only build.
- **HTML/JS** — required for browser delivery (option 2 or 3). Standard `<input type="range">` → uniform.

### Audio / music as a control signal

The interesting one. Pipeline:

```
microphone or audio file
  → FFT (web audio: AnalyserNode; desktop: sounddevice + numpy.fft)
  → feature extraction (band energies, onset detection, spectral centroid, beat phase)
  → uniform updates per frame
```

Concrete mappings worth trying:

| Audio feature | Shader uniform | Effect |
|---|---|---|
| Bass energy (20-200 Hz RMS) | `attraction_scale` | Drops bump clustering. |
| Treble energy | `damping` (inverse) | Hi-hats reduce damping → particles get jittery. |
| Onset / transient | One-shot pulse on `binding_energy_scale` | Snare hits trigger fusion bursts. |
| Spectral centroid | `polarity_fusion_scale` | Bright sounds prefer polarized bonds. |
| Beat phase (0..1) | `repulsion_radius` modulation | Pulsing breathing motion locked to tempo. |
| Mel-band vector (8-16 dim) | `attraction[i,j]` matrix entries | Music rewires chemistry per moment. |

The last row is the most ambitious: feed an audio embedding into the *interaction matrix* rather than a single scalar. The matrix has `(num_species)²` entries — 144 for a 12-species universe — easily covered by a 16-band mel envelope projected through a fixed random matrix. This makes the universe's chemistry literally a function of the music being played.

Implementation note: keep audio analysis on a dedicated thread (or Web Audio worklet) at ~60 Hz update rate, write into a single ring-buffer or atomic uniform set; main render thread reads. Don't FFT inside the render loop.

### MIDI / OSC

For live performance, [`mido`](https://mido.readthedocs.io/) (desktop) or `WebMIDI` (browser). Each CC controller maps to one uniform. Small addition once the uniform plumbing is there, and gives a "modular synth for chemistry" interface that is genuinely novel.

## Risks and likely pain points

- **Fission RNG correlation.** XLA's PRNG is splittable and stateless; GLSL has no native PRNG. Most ports use a hash of `(particle_id, step, salt)` as input to a `pcg`-style hash. Quality is fine for this use case but it's an extra ~30 lines per shader.
- **Atomic contention during fusion.** At high density many pairs race for free composite slots. The reference Python implementation works around this with [`utils.find_free_slots`](../halflife/utils.py) + a sequential scan; in compute shaders the analogue is `atomicAdd` on a counter, which serializes only the slot grab, not the work. Should be fine, but profile early.
- **Debuggability collapses.** No `print`, no `jax.debug.print`, no Python REPL on intermediate state. Build a "dump SSBO to PNG" debug pass *before* writing the chemistry shader, not after.
- **No JIT warmup.** GLSL compiles in milliseconds; WGSL similar. Quality-of-life win over the current 10-30s warm-up.
- **Validating correctness.** The reference Python project has [tests/](../tests/) covering hash chemistry, spatial indexing, etc. The shader project has none of that built in. The realistic path is **differential testing**: run the Python sim and the shader sim from the same seed and config, compare state arrays after N steps, accept divergence below a tolerance. That keeps the JAX project useful as an oracle without coupling the two.

## Reasonable starting scope

If this gets picked up, don't try to land all 9 phases at once. A 100-particle compute-shader sandbox that does **only Phases 3+4+5** (forces, integration, boundary) — no fusion, no decay, no cell list (use brute-force O(N²) for now) — proves out the parts that aren't algorithmic:

- SSBO ↔ vertex-buffer aliasing for zero-copy render.
- Uniforms-per-frame UI loop responsiveness.
- Audio feature → uniform pipeline (drop in a basic FFT, map bass to attraction).

If those three pieces feel good, the rest is mostly mechanical translation of the kernels. If any of them don't, the bigger build wouldn't have either.

## Nubs

- ?? does ModernGL 5.x expose `compute_shader` cleanly, or are there driver/version traps worth knowing before committing to it as the host layer? Worth a quick check.
- ?? at what particle count does brute-force O(N²) start to choke on a typical GPU — sets the bar for when the cell-list port becomes mandatory in the new project.
- ?? would feeding audio into the `params.attraction` matrix directly be too destructive (loses the human-tuned baseline)? Maybe a separate `audio_attraction` matrix added on top preserves the dial-in.
- look at FlowLenia / Lenia shader implementations for prior art on fragment-shader-only continuous-state sims — relevant if option 2 is ever revisited for a non-chemistry experiment in the new project.
