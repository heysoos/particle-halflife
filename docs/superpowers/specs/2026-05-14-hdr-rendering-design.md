# HDR rendering, OKLCh palette, and trails — design specification

**Date:** 2026-05-14
**Status:** Design

## Goal

Replace the current pure-HSV palette and direct-to-framebuffer LDR pipeline with a
**linear-light HDR scene buffer + ACES filmic tonemap composite**, an
**OKLCh-derived perceptually-uniform palette**, and an optional **ping-pong
accumulation-buffer trail effect** with user-controlled decay, particle size, and
alpha multipliers.

This is purely a renderer change. No simulation logic is touched. No new physics
parameters. The simulation continues to feed the same `(position, species, mass,
velocity)` arrays to the renderer each frame.

## Motivation

1. **Colors look like Mondrian, not a chemistry sim.** The palette in
   [utils.py:132](../../../halflife/utils.py#L132) generates HSV with `hue =
   i/num_species`. With `num_species=12` that lands exactly on pure red, pure
   green, pure blue — perceptually nonlinear and harsh. OKLCh fixes this by
   sampling hues in a perceptually-uniform space at constrained lightness and
   chroma.
2. **No tonemapping → no headroom.** The particle fragment shader writes
   `color * brightness` directly to the framebuffer and clips at 1.0. Fast
   particles just go flat instead of feeling bright. An HDR buffer + ACES gives
   smooth highlight rolloff for free, and is a prerequisite for trails to look
   like glow rather than mud.
3. **Trails are the most-requested missing visual feature** for a particle sim of
   this kind. A feedback-buffer trail with three tunable sliders gives a wide
   visual range from "subtle motion blur" to "long phosphorescent comet tails."

## Architecture

### Render pipeline (current vs new)

**Current:**
```
clear default FB → draw particles → draw bonds → draw events → draw HUD → flip
```

**New:**
```
[trails on?]
  ├── yes: feedback-decay pass (sample prev trail FBO × decay → curr trail FBO)
  └── no:  clear scene FBO

draw particles → draw bonds → draw events     (all into scene/trail FBO, RGBA16F)
tonemap composite pass (scene FBO → default FB, ACES + sRGB OETF)
draw HUD on default FB
flip
```

The HUD continues to render in LDR/sRGB space on top of the tonemapped output.
Bond and event shaders are unchanged structurally — they just render into the
HDR scene buffer instead of the default framebuffer.

### Color space conventions

After this change:
- **`species_colors`** stored as **linear sRGB** in `[0, 1]`. Particle / bond
  shaders read them as-is and write to the linear HDR FBO.
- **`background_color`** in `SimConfig` is reinterpreted as linear sRGB. The
  existing default `(0.05, 0.05, 0.08, 1.0)` is dark enough that gamma-vs-linear
  reading is barely visible; if it ends up looking off we adjust the default.
- **Tonemap pass output** is gamma-encoded sRGB ready for the display, with no
  further gamma correction.

ACES tonemap function (Krzysztof Narkowicz's fit, ~10 lines GLSL):
```glsl
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
```

sRGB OETF (gamma encoding for display) applied after `aces()`:
```glsl
vec3 linear_to_srgb(vec3 x) {
    return mix(1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, x * 12.92, step(x, vec3(0.0031308)));
}
```

### OKLCh palette

Replace `make_species_colors` in [utils.py:132](../../../halflife/utils.py#L132)
with an OKLCh sampler. Math from Björn Ottosson's reference.

```python
def make_species_colors(num_species: int) -> np.ndarray:
    """Perceptually-uniform palette via OKLCh. Returns linear-sRGB (N, 3) float32."""
    L = 0.72   # lightness — bright enough to be visible against dark bg
    C = 0.13   # chroma — pops without being neon
    colors = np.zeros((num_species, 3), dtype=np.float32)
    for i in range(num_species):
        # Slight L/C jitter every other species so adjacent hues separate even more
        Li = L + (0.04 if (i % 2 == 0) else -0.04)
        Ci = C + (0.02 if (i % 3 != 2) else -0.02)
        h_deg = (i * 360.0 / num_species + 20.0) % 360.0   # 20° offset to avoid pure red at i=0
        colors[i] = oklch_to_linear_srgb(Li, Ci, h_deg)
    return np.clip(colors, 0.0, 1.0)
```

Helper `oklch_to_linear_srgb` uses the standard OKLab → linear-sRGB matrices.
Out-of-gamut hues are clipped to [0, 1] (acceptable for the moderate chroma
chosen).

### Trail feedback pipeline

Two screen-sized `RGBA16F` textures + framebuffers: `trail_a` and `trail_b`. A
flip-flop index `_trail_idx` selects "current" each frame.

Per frame:
1. `curr = trail_fbos[idx]`, `prev = trail_fbos[1 - idx]`
2. If trails on:
   - Bind `curr`, run a fullscreen-quad shader that samples `prev` and writes
     `texel * u_decay` (no clear).
3. If trails off:
   - Bind `curr`, `glClear` to background color.
4. Draw particles / bonds / events into `curr` with current blend mode
   (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`). When trails are on, multiply the
   particle's vertex-shader `in_size` by `u_trail_size_mult` and the fragment
   alpha by `u_trail_alpha_mult` (uniforms wired from renderer state). When
   trails are off, both multipliers are 1.0.
5. Bind default FB, run tonemap composite reading from `curr`.
6. `_trail_idx = 1 - _trail_idx`.

Decay range exposed to the slider: `[0.0, 0.999]` (linear). 0.0 = full clear
(equivalent to trails-off), 0.95 = pleasant short tails, 0.99 = long
phosphorescent tails. Default: 0.95.

### UI

**One new toggle button**, appended to the existing button strip in
[renderer.py:386](../../../halflife/renderer.py#L386):

| Label    | Action          | Behavior                            |
|----------|-----------------|-------------------------------------|
| Trails   | `toggle_trails` | Enable/disable trail feedback pass  |

**Three new sliders**, appended to the params panel after a group separator:

| Field                  | Label        | Range           | Default | Format    |
|------------------------|--------------|-----------------|---------|-----------|
| `trail_decay`          | trail decay  | `[0.0, 0.999]`  | 0.95    | `{:.3f}`  |
| `trail_particle_size`  | trail size   | `[0.25, 2.0]`   | 1.0     | `{:.2f}`  |
| `trail_particle_alpha` | trail alpha  | `[0.1, 1.0]`    | 1.0     | `{:.2f}`  |

The size and alpha sliders affect the **live particles** that are rendered each
frame (so the user can dial particles small/dim while letting accumulated trails
dominate the image, or vice versa). Confirmed reading from brainstorm.

The render sliders write to a separate `_render_settings` dict on the renderer
rather than `_physics_updates`. The `Slider` class is extended with an optional
`target_dict` arg passed at construction; updates go to whichever dict each
slider was wired to. Main loop is untouched.

The Trails button label flips between `"Trails"` (off) and `"Trails ON"` (on),
matching the existing pattern for Events / Params buttons.

### Files touched

| File                        | Change                                                |
|-----------------------------|-------------------------------------------------------|
| `halflife/utils.py`         | Replace `make_species_colors` with OKLCh sampler; add `oklch_to_linear_srgb` helper |
| `halflife/renderer.py`      | HDR scene FBO, tonemap shader, ping-pong trail FBOs, decay shader, particle uniforms for trail size/alpha, Trails toggle button, three trail sliders, render-settings dict |
| `halflife/config.py`        | (Optional) tweak `background_color` default if linear reinterpretation looks wrong |
| `tests/`                    | New `test_palette.py` — sanity-check OKLCh palette: shape, dtype, in-range, all distinct |

No changes to:
- `halflife/state.py` (besides `get_species_colors`, which just calls
  `make_species_colors` — unchanged signature)
- `halflife/step.py`, `halflife/chemistry.py`, `halflife/spatial.py`,
  `halflife/interactions.py` — simulation untouched
- `halflife/main.py` — renderer construction signature unchanged; renderer
  owns its own trail/render state

## Testing strategy

This is a visual feature. Automated tests cover the small testable surface
(palette math); the rest is visual smoke-testing.

**Unit tests** (`tests/test_palette.py`):
- `make_species_colors(N)` returns `(N, 3) float32`, all values in `[0, 1]`.
- All rows distinct (no two species share a color).
- `oklch_to_linear_srgb` roundtrips a known reference value within tolerance
  (e.g. OKLCh(0.5, 0.0, 0.0) → linear-sRGB ~mid-gray).

**Visual smoke checks** (manual, in the live sim):
- Launch sim → particles look harmonious, no eye-stabbing pure red/green/blue.
- Toggle "Trails" → trail effect appears/disappears cleanly with no visual
  artifacts.
- Drag `trail_decay`: 0.0 = no trails, 0.95 = short tails, 0.99 = long tails.
- Drag `trail size` / `trail alpha`: live particles change size/opacity
  smoothly. Trails (the accumulated layer) are unaffected.
- Bright clusters during fusion now glow softly (ACES rolloff) instead of
  clipping to white.
- HUD remains crisp and readable on top of tonemapped output.

## Out of scope

- **Bloom.** Discussed in brainstorm as approach C, rejected for marginal payoff
  over ACES + trails on this dense scene.
- **Per-particle trail history** (stamped line segments). The feedback-buffer
  approach gives the look the user wants more cheaply.
- **Dithering / temporal AA** on the 16F → 8-bit composite. Unlikely to be
  visible at this color depth and palette.
- **HUD tonemapping.** HUD continues to draw in LDR sRGB on top; it doesn't
  need HDR.
- **Per-species size/brightness overrides.** All species use the same global
  size/alpha multipliers.

## Risks / open questions

1. **Background color reinterpretation.** Current `(0.05, 0.05, 0.08, 1.0)` is
   already very dark; reinterpreting as linear makes it darker (gamma-decoded
   it's ~0.002). Should be fine but if the visible background ends up
   completely black, retune the default.
2. **WSL OpenGL FBO support.** Renderer already uses ModernGL custom shaders
   and one offscreen texture (HUD), so RGBA16F FBOs should work. If `f2` dtype
   isn't supported in the WSL OpenGL stack, fall back to RGBA8 — ACES still
   helps (smoother rolloff in the LDR range), trails just lose some glow
   fidelity.
3. **Performance.** Two extra fullscreen-quad passes per frame (trail decay +
   tonemap composite) on a 1280×720-ish window is microseconds — negligible
   next to particle rasterization. No projected regression.
