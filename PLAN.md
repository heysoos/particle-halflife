# Half-Life Particle Simulator — Project Roadmap

## Status (2026-07-30)

Core simulator is feature-complete through the covalent-chemistry era: edge-based bonds
with valence saturation and ring closure, bond-cut binary fission, per-bond kinetic +
thermal scission, liquid-drop fissility half-lives, VSEPR/harmonic angle locking, HDR
renderer with pan/zoom + inspector, video recording, and the composite diagnostic /
open-endedness analysis pipeline. Runs at 5k particles by default (~80k practical
ceiling). Two designs are specced but **not yet implemented**: kinetic-energy-coupled
reactions and million-particle scaling (see Active Roadmap).

> ⚠️ The simulation-recording work (`halflife/render/recorder.py`, tests, renderer/HUD
> changes) is in the working tree but **uncommitted** as of this writing.

---

## Spec / Plan Index (docs/superpowers/)

Audited 2026-07-30 against the working tree. "Superseded" = implemented, then replaced
by a later mechanism; the doc remains as historical record.

| Doc | Status |
|---|---|
| `specs/2026-03-26-composite-reaction-network-design.md` | **Partial / superseded.** Phase 1 (instrumentation) and Phase 2 (free-particle absorption) implemented. Phase 3 (multi-product C+C → 1–5 products, polarity blending) never built — superseded by binary bond-cut fission; polarity machinery removed entirely (`a33593b`). Phase 4 (heredity/lineage tracking) still open — kept as a roadmap idea. |
| `plans/2026-03-26-phase1-verification-instrumentation.md` | **Implemented** (`halflife/profiler.py`, `--enable-profiling`, size stats). |
| `specs/2026-05-07-hash-fission-design.md` + plan | **Implemented, then superseded** by the 2026-06-12 bond-cut fission redesign (the old slot-order `_hash_to_partition` path minted long bonds). |
| `specs/2026-05-14-covalent-bonds-design.md` + plan | **Implemented** (`bond_mode="edges"`, hash-derived rest lengths, ring closure). |
| `specs/2026-05-14-hdr-rendering-design.md` + plan | **Implemented** (RGBA16F FBOs, ACES tonemap, OKLCh palette, trails). |
| `plans/2026-05-14-pan-zoom-inspector.md` | **Implemented** (`render/camera.py`, inspector panel in `render/hud.py`). |
| `specs/2026-05-30-composite-diagnostic-design.md` + plan | **Implemented** (`halflife/analysis/`, HTML reports, cache). |
| `plans/2026-06-12-fission-bond-breaking-redesign.md` | **Implemented** (bond-cut fission, `_apply_binary_splits`, bond scission). |
| `specs/2026-06-12-openendedness-dashboard-design.md` + plan | **Implemented** (Tier 5: novelty, Hill diversity, turnover, topology split). |
| `specs/2026-06-12-vsepr-angle-kernel-design.md` + plan | **Implemented** (`angle_mode="vsepr"/"harmonic"`, runtime `k_angle`). |
| `specs/2026-06-13-liquid-drop-disruption-design.md` + plan | **Implemented** (Coulomb-analog `disruption_scale·n²/R_g`, runtime sliders). |
| `specs/2026-06-14-kinetic-energy-reactions-design.md` + plan | **NOT implemented.** Spec + plan only (commits `95004bb`, `ab824a7`); zero code landed. Next up — see Active Roadmap. |
| `specs/2026-07-29-simulation-recording-design.md` | **Implemented** (uncommitted; see Status note above). |
| `specs/2026-07-30-million-particle-scaling-design.md` | **NOT implemented.** Design only; of Phase 0's "free wins" just the MESA env var pre-exists. See Active Roadmap. |

---

## Active Roadmap

### 1. Kinetic-energy-coupled reactions (spec + plan ready)

`docs/superpowers/specs/2026-06-14-kinetic-energy-reactions-design.md` — all nine plan
tasks pending: energy-ledger sign fix, `endothermic_fission_mode` (KE_rel + Q ≥ 0 gate
replacing the hard `forbid_endothermic_fission`), momentum-conserving signed-kick rework
of `_apply_binary_splits`, hash-derived fusion activation barriers, born-hot bonds,
config flags + `main.py` opt-in, `tests/test_kinetic_reactions.py`.

### 2. Million-particle scaling (design only)

`docs/superpowers/specs/2026-07-30-million-particle-scaling-design.md` — phased path
from the ~80k ceiling to 1–2M:
- **Phase 0 — free wins (mostly not done):** fp16 render transfers, move numpy
  color/brightness work into shaders, fix the cell-capacity silent-drop bug
  (`cell_capacity=64` vs measured occupancy ~372 at scale).
- **Phase 1:** Warp integration + Morton spatial reordering (est. 16.8–21.1×).
- **Phase 2:** edge-list composites (flat `(E,2)` array + connected-component labels,
  replacing `(C,256)` / `(C,512,2)` padding).
- **Phase 3:** P³M long-range mesh forces.
- **Phase 4:** zero-copy GL interop + single-draw GL_LINES bonds.

### Longer-horizon ideas (no spec yet)

- **Composite heredity / lineage tracking** — the surviving piece of the 2026-03-26
  reaction-network design (Phase 4): `parent_ids` on composites, lineage analysis,
  selection-pressure metrics. Fits the project's evolutionary-dynamics goal.
- Group fitness metrics
- Spatial compartmentalization detection
- NCA-style learned update rules — would need to restore the `internal` field removed in `b0c049f`
- Mass conservation mode (FlowLenia / reintegration-tracking inspired)
- Optimization loops (evolve interaction matrices for specific goals)

---

## Resolved / Culled (2026-07-30 audit)

The 2026-05-06 "Next Session" TODOs are all resolved and removed from the active list:

- **Polarity scaling rethink** → mooted: polarity/`attr_mod` machinery removed entirely (`a33593b`).
- **Interaction range + rep trick** → done: `interaction_radius` 4.0 → 8.0; the
  rep-only fusion gate was replaced by nearest member-member contact + per-particle
  valence in the 2026-06-12 fusion rework (rep survives only as a dedup key).
- **`cell_capacity` overflow** → bumped 8 → 64 (fine at 5k; flagged again as
  insufficient at 1M+ in the scaling spec, Phase 0).
- **Force-kernel audit** → resolved: kernel now uses per-pair `peak_fraction` /
  `cutoff_fraction` (`InteractionParams`); the dead `r_attract` arg is gone. The
  "sketch LJ/Lenia alternative kernels" experiment was never run — revive on demand.
- **Phase 5: frame recording** → done (video recording, 2026-07-29 spec).
- **Phase 5: scale to 10k+** → folded into the million-particle scaling design above.
- **Phase 6: parameter sweep infrastructure** → superseded by the `halflife/analysis/`
  diagnostic pipeline (the old `test_composite_statistics.py` HTML sweeps predate it).

---

## Implementation Log

Detailed history is in `git log`; thematic recap in [notes/2026-05-05-project-status-recap.md](notes/2026-05-05-project-status-recap.md). High-level arc:

- **Phase A — Initial build & first-run debugging.** Modules written; first run hit 4 perf bottlenecks, fixed → 30ms → 3.9ms/step.
- **Phase B — Polarity chemistry, UI, events, stats.** Live HUD, event sprites, sparklines, polarity-based fusion preference and stability. Closed with a 10–100× perf jump (`086e9e1`: commutative hash, COM-spring bonds, async pipeline).
- **Phase C — Live tuning UX + fusion-scan optimization.** Log-scale sliders, fusion scan rewrites, dead-particle machinery removed (`b0c049f`), `lax.switch` → `jnp.where` (4× speedup).
- **Phase D — Profiling instrumentation (2026-03).** Profiler module, C+C fusion detection, baseline performance docs.
- **Phase E — Composite statistics analysis (2026-03-27).** HTML sweep reports over `(fusion_threshold, interaction_radius, composite_size_decay_scale)` (six runs in `tests/reports/`).
- **Phase F — Physics audit (2026-05-05/06).** Diagnosed degenerate-kernel issue at user's UI settings, dead-code cleanup (`98abb0f`), fusion_radius bump 1.0→4.0 (`86eb78f`), discovered cell_capacity overflow bug. Notes: `notes/2026-05-05-*`, `notes/2026-05-06-*`.
- **Phase G — Fission + covalent chemistry (2026-05 → 2026-06).** Hash fission, then
  edge-based covalent bonds with valence/free-bond saturation and ring closure; bond-cut
  binary fission redesign (no minted edges, Q-value kicks); per-bond kinetic + thermal
  scission; liquid-drop fissility half-lives with runtime disruption/cohesion sliders;
  VSEPR/harmonic angle locking.
- **Phase H — Rendering + analysis (2026-05 → 2026-07).** HDR pipeline (RGBA16F, ACES,
  OKLCh palette, trails), pan/zoom camera + particle inspector, composite diagnostic
  pipeline + open-endedness dashboard (`halflife/analysis/`), event sprites from real
  kernel events, video recording with realtime pacing + HUD hiding.

Performance baseline (2026-03-27): 11.6 ms/step fused, 85 steps/sec at 2k particles. Bottleneck was `attempt_fusion` at 55% of step time. Full breakdown in [tests/README_PERFORMANCE.md](tests/README_PERFORMANCE.md).
