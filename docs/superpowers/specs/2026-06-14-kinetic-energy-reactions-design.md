# Kinetic-Energy-Coupled Reactions (Exo / Endothermic)

**Date:** 2026-06-14
**Status:** Design approved, pending implementation plan

## Motivation

Today the chemistry model never reads particle velocity. Fusion is gated only by
proximity, hash-derived binding-energy threshold, and valence; fission is a
stochastic half-life roll whose only energetic check is a hard veto on
energy-costing splits (`forbid_endothermic_fission`). "Temperature" exists solely
as the fixed `bond_temperature` constant feeding the Arrhenius scission channel —
decoupled from how fast things are actually moving.

This makes the model a pure *potential-energy-landscape* chemistry. Real chemistry
is kinetic-energy-coupled: fast collisions react and slow ones bounce (activation
energy), reactions either release energy (exothermic) or require it (endothermic),
and temperature *emerges* from the velocity distribution rather than being dialed in.

This spec adds that coupling, in a way that is **energy-conserving** and reuses the
existing fission and scission machinery wherever possible.

## Core principle

Every reaction obeys energy conservation:

```
KE_after = KE_before + Q,    Q = ΣBE(products) − ΣBE(reactants)
```

with binding energy `BE` treated as a positive quantity (more bound → larger BE),
so the physical total energy is `KE − BE` (bound states sit lower).

- **Q > 0 (exothermic):** binding energy is released *into* kinetic energy. No
  kinetic precondition from energetics (an activation barrier may still gate it).
- **Q < 0 (endothermic):** the deficit `|Q|` must be *paid out of* kinetic energy.
  Allowed only if enough KE is available, and it *consumes* that KE.

The familiar special case — full dissociation `AB → A + B` — has `Q = −BE(AB)`, so it
requires `KE ≥ BE(AB)`. That is the user's original "KE must at least equal the bond
energy" intuition, correctly scoped to the *breaking* direction. Fusion of free
particles has `Q = +BE(AB) ≥ 0` and is therefore exothermic with no KE floor.

### Which KE — the 2-body reduction

A composite has no single velocity; its members each move independently. The KE a
reaction may spend is the **relative motion of the two prospective fragments**, not
the total internal agitation. For a split into fragments `p0`, `p1`:

```
M0 = Σ mᵢ (i∈p0),   V0 = (Σ mᵢ vᵢ) / M0          (fragment mass + COM velocity)
M1, V1   likewise for p1
μ      = M0·M1 / (M0 + M1)                        (reduced mass)
KE_rel = ½ μ |V0 − V1|²                           (the reaction's energy budget)
```

Each member's motion *relative to its own fragment's COM* is internal vibration and
stays with that fragment, untouched. Total momentum is conserved because the pair's
COM velocity `(M0·V0 + M1·V1)/(M0+M1)` is never altered. This is the most physical
choice (only motion along the split can break the bond) and unifies cleanly with the
existing COM-axis separation kick.

## Scope

Covers **both** fission and fusion. All new behavior ships behind config flags that
default to current behavior; the live app (`main.py`) opts in. Existing tests,
headless runs, and analysis presets are unchanged unless they explicitly enable the
flags.

Out of scope (YAGNI): re-enabling the soft-energy thermostat; radiating fusion heat
to neighbors (the "born-hot bond" channel is used instead); any change to the
liquid-drop half-life law, ring closure, or angle locking.

---

## Part 1 — Energy-conserving fission

### Why the existing system is already 90% ready

`apply_composite_decay` (`halflife/chemistry.py`) already:

1. Rolls each composite against its half-life → `fissions` mask (thermal activation).
2. In `choose_cut(k)`, scores every bond cut by `be0 + be1`, picks the best, and
   computes **`q = score[v] − BE[parent]`** — this is already `Q = ΣBE(products) − BE(parent)`.
3. Gates with `if forbid_endothermic_fission: fires &= (q ≥ 0)` — a hard wall at `Q < 0`.
4. Applies `kick = max(q, 0)` via `_apply_binary_splits`.

So the Q-value and the exact decision point already exist. Endothermic fission is
**lowering the hard wall to a conditional one**. Two surgical changes; the BFS,
cut-scoring, member/edge compaction, and slot writes are all untouched.

### Change 1 — the gate

In the new `endothermic_fission_mode == "energy_gated"` mode, the veto becomes:

```
fires &= (KE_rel + q ≥ 0)
```

A composite may break an unfavorable bond **iff its two fragments are separating fast
enough to pay `|q|`**. Hot/agitated composites can; cold ones cannot. Because
`choose_cut` already selects the *highest-Q* cut, an unaffordable best cut means every
cut is unaffordable — the composite simply survives and re-rolls next step. No
fallback search is needed.

### Change 2 — the signed kick

`kick = max(q, 0)` (which only ever *adds* a non-negative separation velocity) becomes
a **signed relative-velocity update**. Set the fragments' post-split relative speed from

```
½ μ |v_rel_after|² = KE_rel + q     →     |v_rel_after| = sqrt(2 (KE_rel + q) / μ)
```

then rescale `V0, V1` about the conserved pair COM velocity to hit it. Exothermic
splits (`q > 0`) speed the fragments apart; endothermic splits (`q < 0`) slow them,
draining `|q|`.

**Direction:** along the existing relative-velocity vector `V0 − V1` when it is
non-negligible; otherwise fall back to the COM-position axis (the current code's
convention), which exactly recovers today's exothermic-from-rest kick.

### `_apply_binary_splits` rework

`_apply_binary_splits` already computes fragment COMs from *positions* (`per_split`).
Add the parallel computation of fragment **masses** and **COM velocities** (from
`particles.mass` and `particles.velocity`), then replace the additive-kick step
(`per_member`, currently `velocity.at[pid].add(kick)`) with the signed `v_rel` rescale
described above. The `kick_energy` argument becomes a signed `q` per row; the shape and
the rest of the function's contract are unchanged, so the bond-scission caller (which
passes zero kick) continues to work without an energy change.

---

## Part 2 — Fusion: activation energy + born-hot bonds

### Activation gate

Add one condition to the fusion candidate gate in `check_neighbor`
(`halflife/chemistry.py`), AND-ed alongside the existing `be_eff > fusion_threshold`,
overflow, and valence checks:

```
KE_rel ≥ E_a(s_i, s_j)
```

where `KE_rel = ½ μ |v_i − v_j|²` for the contacting pair (reduced mass from the two
particles' masses), and `E_a` is a new **hash-derived per-species-pair activation
energy**: same commutative additive pair hash as bond energy / rest length, re-mixed
with its own distinct Fibonacci stream (decorrelated from BE, valence, rest-length,
and bond-energy), scaled to `[0, activation_energy_scale]`. Pre-computed into an
`(S, S)` matrix like `compute_bond_energy_matrix`.

Both fusion paths (matching mode `_fusion_apply_matching` and the legacy
`fusion_scan_body` scan) consume this same candidate gate, so adding the term in
`check_neighbor` covers both modes at once.

### Born-hot bonds (energy disposal)

A two-body association cannot conserve both energy and momentum on its own — the
released `KE_rel + Q` must go *somewhere*. We store it as **vibration of the new bond**:
the two contact particles are given a relative velocity carrying `KE_rel + Q` along
their bond axis (COM velocity preserved → momentum conserved).

If that vibration exceeds what the bond can hold (its dissociation energy `E_b`), the
**existing scission system breaks it the next step** — the bond stretches, strain
exceeds `E_b`, scission fires. So a collision too violent to stick fuses for an instant
and then snaps apart: the "third-body problem" resolves itself with **no new code in
the scission path**. Survivable vibration is bled off over subsequent steps by the
sim's existing damping, spreading heat to neighbors.

---

## Part 3 — Bookkeeping & configuration

### Energy-ledger sign fix

`halflife/energy.py:compute_total_energy` currently returns `KE + BE`, the wrong sign:
under it, forming a bond would *cost* kinetic energy — the opposite of what the fission
kick does. Flip to `KE − BE` so the recorded `total_energy` diagnostic is actually
conserved by the new reactions. `apply_soft_energy_conservation` stays commented out as
it is today; this change only makes the readout honest. Note its internal
`target_ke = target − BE` arithmetic assumes the *old* `KE + BE` sign and would be
wrong under the new convention — so we leave a warning comment on that function rather
than silently flip it, since re-enabling the thermostat is out of scope here.

### New config knobs (`halflife/config.py`)

All default to **current behavior**:

| Knob | Default | Meaning |
|---|---|---|
| `endothermic_fission_mode` | `"forbid"` | `"forbid"` = today's hard `Q ≥ 0` wall; `"energy_gated"` = the `KE_rel + Q ≥ 0` pay-to-break rule. Replaces the False-branch meaning of `forbid_endothermic_fission`. |
| `enable_fusion_activation` | `False` | Turns on the `KE_rel ≥ E_a` fusion gate and born-hot disposal. |
| `activation_energy_scale` | `1.0` (tuning seed) | Ceiling for hash-derived `E_a`. Starting value, retuned against observed fuse/scission churn once the live app runs. |

`forbid_endothermic_fission` is **subsumed** by `endothermic_fission_mode`: migrate
existing references (`True → "forbid"`). Decide during planning whether to keep the
old boolean as a deprecated alias or remove it outright (only internal callers and
the analysis presets reference it).

`main.py:build_config` sets `endothermic_fission_mode="energy_gated"` and
`enable_fusion_activation=True` for the live app — same opt-in pattern as `emit_events`,
`angle_mode`, and `stability_mode`. Headless/test configs keep the defaults.

Because `config` is `static_argnums`, the gates are selected with Python `if`, so XLA
traces only the live branch — zero runtime cost when off.

### Testing

New unit tests (run on CPU per the project's GPU-contention note):

- Reduced-mass / `KE_rel` helper: correct value and symmetry.
- Endothermic split **allowed when hot, refused when cold** (same composite, two KE states).
- **Momentum conserved** across both fission directions (exo and endo).
- Energy bookkeeping: `KE_after − KE_before ≈ Q` for a controlled split.
- Fusion activation gate **blocks a slow pair, admits a fast pair** (same species).
- A too-fast fusion **fuses then scissions** within a couple of steps (born-hot path).
- `compute_total_energy` sign: a fusion lowers `KE − BE`-consistent total by ~0 (conserved).

Existing chemistry / analysis suites must stay **green unchanged** with flags at their
defaults.

## Files touched

- `halflife/chemistry.py` — `_apply_binary_splits` (fragment masses/velocities, signed
  kick), `apply_composite_decay` (energy-gated veto), `check_neighbor` (activation gate),
  new `_hash_to_activation_energy` + `compute_activation_energy_matrix`.
- `halflife/energy.py` — sign fix in `compute_total_energy`.
- `halflife/config.py` — new knobs; subsume `forbid_endothermic_fission`.
- `halflife/main.py` — `build_config` opts in.
- `halflife/step.py` — thread the activation-energy matrix if it is built once and passed
  (mirroring `r_rest` / bond-energy matrices); confirm during planning.
- `tests/` — new unit tests as above.

## Open items for the plan

- `activation_energy_scale` default — choose once we can run the live app and observe the
  fuse/scission churn rate.
- Whether `forbid_endothermic_fission` survives as a deprecated alias or is removed.
- Exact threshold below which `|V0 − V1|` is "negligible" and the kick falls back to the
  COM-position axis.
