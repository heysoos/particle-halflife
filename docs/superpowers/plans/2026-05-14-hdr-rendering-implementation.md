# HDR Rendering, OKLCh Palette, and Trails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the renderer's pure-HSV palette and direct-to-framebuffer LDR pipeline with an OKLCh-derived perceptually-uniform palette, an HDR RGBA16F scene framebuffer composited via ACES filmic tonemap + sRGB OETF, and an optional ping-pong accumulation-buffer trail effect with user-controlled decay / particle size / particle alpha sliders.

**Architecture:** Linear-light rendering into a half-float scene FBO; per-frame ACES tonemap composite to the default framebuffer; trail effect via two ping-pong RGBA16F framebuffers with a uniform-driven decay multiplier; species colors precomputed in linear sRGB via OKLCh sampling; HUD continues to render in LDR on top of the tonemapped output.

**Tech Stack:** Python 3.10, JAX (untouched), ModernGL 5.12, pygame 2.6.1, numpy 2.2.6. All work is in `halflife/utils.py`, `halflife/renderer.py`, and `tests/test_palette.py`.

**Spec:** [docs/superpowers/specs/2026-05-14-hdr-rendering-design.md](../specs/2026-05-14-hdr-rendering-design.md)

---

## Environment & Setup

This project runs under WSL Ubuntu with a venv at `.venv/`. Claude Code is running natively in WSL, so activate the venv directly — no `wsl bash -c` wrapper.

**Running tests:**
```bash
source .venv/bin/activate && JAX_PLATFORMS=cpu pytest tests/test_palette.py -v
```

`JAX_PLATFORMS=cpu` keeps tests off the GPU so they don't contend with a live sim. Palette tests don't touch JAX directly but `halflife/state.py` imports it at module-load time, so the env var prevents a slow GPU init.

**Running the live sim** (for visual checks):
```bash
source .venv/bin/activate && python -m halflife.main
```

**Comment preservation rule (from `CLAUDE.md`):** when editing existing functions in `renderer.py` or `utils.py`, **do not delete existing comments** that are not specifically describing code you are removing. Be surgical. Only the obsolete comments inside `make_species_colors` (the HSV-specific ones) are eligible for deletion — they describe code that is itself being replaced.

**Git commits** require inline identity (no global git config in WSL):
```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add <files>
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "..."
```

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `halflife/utils.py` | OKLCh→linear-sRGB helper; perceptually-uniform palette generator. | Modify |
| `halflife/renderer.py` | HDR scene FBO; ACES tonemap shader & pass; ping-pong trail FBOs & decay shader; particle-shader trail-mode uniforms; Trails toggle button; three trail sliders; render-settings dict; Slider class extension to support arbitrary target dicts. | Modify |
| `tests/test_palette.py` | Unit tests for OKLCh palette: shape, dtype, range, all-distinct, roundtrip a known reference. | Create |

No other files are touched. `halflife/state.py` calls `make_species_colors` through `get_species_colors` and the signature is preserved.

---

## Task 1: OKLCh palette helper + `make_species_colors` rewrite (TDD)

**Files:**
- Modify: `halflife/utils.py:130-146` (the `# ── Color Palette ──` block, function `make_species_colors`)
- Create: `tests/test_palette.py`

### Step 1: Write the failing test

- [ ] Create `tests/test_palette.py` with the following content:

```python
"""
tests/test_palette.py — OKLCh species palette sanity checks.

The renderer expects make_species_colors(N) to return a (N, 3) float32 array of
linear-sRGB values in [0, 1] with every row distinct. oklch_to_linear_srgb is
the underlying conversion; we roundtrip a known neutral reference to catch
matrix typos.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from halflife.utils import make_species_colors, oklch_to_linear_srgb


def test_palette_shape_and_dtype():
    colors = make_species_colors(12)
    assert colors.shape == (12, 3)
    assert colors.dtype == np.float32


def test_palette_in_unit_range():
    colors = make_species_colors(32)
    assert (colors >= 0.0).all()
    assert (colors <= 1.0).all()


def test_palette_rows_distinct():
    # With moderate chroma and 24 hues spaced 15° apart, every species must be
    # visibly different — no two rows should be identical.
    colors = make_species_colors(24)
    seen = {tuple(row) for row in colors}
    assert len(seen) == 24


def test_oklch_neutral_is_grayscale():
    # OKLCh with C=0 (no chroma) at any hue should round-trip to a neutral
    # gray in linear sRGB: all three channels equal.
    rgb = oklch_to_linear_srgb(0.5, 0.0, 0.0)
    assert rgb.shape == (3,)
    assert abs(float(rgb[0]) - float(rgb[1])) < 1e-4
    assert abs(float(rgb[1]) - float(rgb[2])) < 1e-4
    # And mid-lightness should land near linear-sRGB ≈ 0.18 (mid gray in linear)
    assert 0.10 < float(rgb[0]) < 0.30


def test_oklch_known_red():
    # OKLCh(0.628, 0.258, 29.234°) is the reference value for sRGB pure red.
    # We allow loose tolerance because our impl clips to gamut and uses float32.
    rgb = oklch_to_linear_srgb(0.628, 0.258, 29.234)
    # Red channel should dominate clearly.
    assert float(rgb[0]) > float(rgb[1])
    assert float(rgb[0]) > float(rgb[2])
    assert float(rgb[0]) > 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
source .venv/bin/activate && JAX_PLATFORMS=cpu pytest tests/test_palette.py -v
```

Expected: `ImportError: cannot import name 'oklch_to_linear_srgb' from 'halflife.utils'`. All tests fail at collection.

### Step 3: Implement `oklch_to_linear_srgb` and rewrite `make_species_colors`

- [ ] Open `halflife/utils.py` and replace the function `make_species_colors` (currently spanning lines ~132–146) **and add `oklch_to_linear_srgb` above it**. Preserve the surrounding `# ── Color Palette ──` section header comment. The new content for the section is:

```python
# ── Color Palette ────────────────────────────────────────────────────────────

def oklch_to_linear_srgb(L: float, C: float, h_deg: float) -> np.ndarray:
    """
    Convert OKLCh (L, C, hue°) → linear sRGB. Returns (3,) float32 in [0, 1]
    after clamping; out-of-gamut OKLCh inputs are clipped (acceptable for the
    moderate-chroma palette used here).

    Math from Björn Ottosson (2020) https://bottosson.github.io/posts/oklab/
    """
    import math
    h_rad = math.radians(h_deg)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    # OKLab → cone-response basis (cubic root applied in inverse)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    # Cone responses → linear sRGB
    r =  4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_ = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    out = np.array([r, g, b_], dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def make_species_colors(num_species: int) -> np.ndarray:
    """
    Generate a perceptually-uniform palette for num_species species using OKLCh.
    Returns: (num_species, 3) float32 *linear-sRGB* in [0, 1].

    The renderer composites with ACES tonemap + sRGB OETF, so colors stay in
    linear space all the way through the scene FBO. Adjacent species get
    slightly different lightness and chroma so neighboring hues separate
    visually as well as chromatically.
    """
    L_base = 0.72   # target lightness — bright against dark background
    C_base = 0.13   # moderate chroma — pops without being neon
    HUE_OFFSET = 20.0  # offsets the wheel so species 0 isn't pure-red
    colors = np.zeros((num_species, 3), dtype=np.float32)
    for i in range(num_species):
        Li = L_base + (0.04 if (i % 2 == 0) else -0.04)
        Ci = C_base + (0.02 if (i % 3 != 2) else -0.02)
        h_deg = (i * 360.0 / num_species + HUE_OFFSET) % 360.0
        colors[i] = oklch_to_linear_srgb(Li, Ci, h_deg)
    return colors
```

The old HSV-specific block can be deleted (those comments describe the code being replaced — that's the one exception allowed by the comment-preservation rule). The section header `# ── Color Palette ──` and the `# ── Misc JAX Helpers ──` block below it stay.

### Step 4: Run the tests to verify they pass

- [ ] Run:

```bash
source .venv/bin/activate && JAX_PLATFORMS=cpu pytest tests/test_palette.py -v
```

Expected: all 5 tests PASS.

### Step 5: Commit

- [ ] Run:

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/utils.py tests/test_palette.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): OKLCh-derived perceptually-uniform species palette

Replace the HSV palette in make_species_colors with an OKLCh sampler.
The HSV palette landed exactly on pure red / green / blue at num_species=12;
OKLCh at fixed L≈0.72 and C≈0.13 with mild per-species L/C jitter gives a
muted-but-saturated set that reads as cohesive.

Colors are returned in linear sRGB — the renderer is moving to an HDR scene
FBO + ACES tonemap composite where linear is the right input space."
```

---

## Task 2: HDR scene FBO + ACES tonemap composite pass

This task introduces the HDR pipeline. After this commit, the live sim's
rendered image goes through a half-float framebuffer and an ACES filmic
tonemap before reaching the display. Trails are not introduced yet.

**Files:**
- Modify: `halflife/renderer.py` (add shader source, scene FBO, tonemap pass; rewrite `render()`)

### Step 1: Add ACES tonemap shaders to the shader-source block in `renderer.py`

- [ ] Below the existing `HUD_FRAGMENT_SHADER` constant (around line 136) and above the `EVENT_VERTEX_SHADER` block, add:

```python
# Tonemap composite: sample HDR scene texture, apply ACES filmic curve,
# convert linear → sRGB display gamma. Output replaces what would have been
# the direct framebuffer write of the LDR pipeline. Bg/HUD compositing
# happens *after* this pass.
TONEMAP_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

TONEMAP_FRAGMENT_SHADER = """
#version 330

in  vec2 v_uv;
uniform sampler2D scene_tex;
out vec4 fragColor;

// Narkowicz 2015 ACES fit — close to the full ACES RRT+ODT for the
// brightness range we render in, ~10 ALU ops.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Linear sRGB → display sRGB (gamma encoding).
vec3 linear_to_srgb(vec3 x) {
    vec3 lo = x * 12.92;
    vec3 hi = 1.055 * pow(max(x, vec3(1e-6)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, step(x, vec3(0.0031308)));
}

void main() {
    vec3 hdr    = texture(scene_tex, v_uv).rgb;
    vec3 mapped = aces(hdr);
    fragColor   = vec4(linear_to_srgb(mapped), 1.0);
}
"""
```

### Step 2: Build the scene FBO and tonemap program in `Renderer.__init__`

- [ ] Find the section in `Renderer.__init__` that constructs the HUD shader (currently around line 343 in `renderer.py`, the `# ── HUD shader …` block). **After** the HUD block but **before** the `# ── Event sprite shader …` block, insert:

```python
        # ── HDR scene framebuffer + tonemap composite ─────────────────────────
        # The scene (particles + bonds + events) is rendered into an RGBA16F
        # framebuffer so we have headroom above 1.0. The tonemap pass samples
        # this texture, applies an ACES filmic curve, and writes sRGB-gamma
        # output to the default framebuffer. The HUD is drawn on top of the
        # tonemapped output in LDR space.
        self._scene_tex = self.ctx.texture(
            (config.window_width, config.window_height), 4, dtype='f2'
        )
        self._scene_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._scene_fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])

        self.tonemap_prog = self.ctx.program(
            vertex_shader=TONEMAP_VERTEX_SHADER,
            fragment_shader=TONEMAP_FRAGMENT_SHADER,
        )
        # Reuse the HUD fullscreen-quad VBO geometry — same -1..1 NDC quad.
        self._tonemap_vao = self.ctx.vertex_array(
            self.tonemap_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )
```

Note: the HUD quad VBO is already created higher up in `__init__` (the `quad_verts` block) so we can reuse it. Reusing avoids a duplicate buffer.

### Step 3: Rewrite `Renderer.render` to use the HDR pipeline

- [ ] Replace the body of `Renderer.render` (currently around line 882–921 in `renderer.py`) with the following. **Keep the docstring**.

```python
    def render(self, fps: float, step_count: int, n_alive: int):
        """Clear screen, draw scene, draw HUD, flip buffers."""
        bg = self.config.background_color

        # ── Scene pass into HDR FBO ──────────────────────────────────────────
        self._scene_fbo.use()
        self.ctx.clear(*bg)

        # Particles
        if self._n_particles_to_draw > 0:
            self.particle_vao.render(moderngl.POINTS, vertices=self._n_particles_to_draw)

        # Bonds
        if self.composite_mode == self.MODE_BONDS and self._n_bond_vertices > 0:
            self.ctx.line_width = 1.0
            self.bond_vao.render(moderngl.LINES, vertices=self._n_bond_vertices)

        # Event sprites
        if self._show_events and self._n_event_vertices > 0:
            self.event_prog['u_world_size'].value = (
                self.config.world_width, self.config.world_height
            )
            self._event_vao.render(moderngl.POINTS, vertices=self._n_event_vertices)

        # ── Tonemap composite to default framebuffer ─────────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._scene_tex.use(location=0)
        self.tonemap_prog['scene_tex'].value = 0
        self._tonemap_vao.render(moderngl.TRIANGLES, vertices=6)

        # ── HUD overlay (LDR, on top of tonemapped scene) ────────────────────
        # HUD overlay — only re-render the pygame surface and re-upload the
        # texture when the HUD has actually changed. The fullscreen-quad GL
        # blit always runs (it's cheap; reads the cached texture).
        # Stats panel updates every frame (sparklines, FPS), so force dirty
        # while it's shown.
        if self._show_stats:
            self._hud_dirty = True

        if self._hud_dirty:
            self._render_hud_surface(fps)
            surf_data = pygame.image.tostring(self._hud_surface, 'RGBA', True)
            self._hud_texture.write(surf_data)
            self._hud_dirty = False

        self._hud_texture.use(location=0)
        self.hud_prog['hud_tex'].value = 0
        self._hud_quad_vao.render(moderngl.TRIANGLES, vertices=6)

        pygame.display.flip()
```

The four-line "HUD overlay — only re-render…" comment is preserved verbatim from the original `render()`. The new outer `# ── HUD overlay (LDR, …) ──` section header is added on top; nothing is deleted.

### Step 4: Release the new GL resources in `Renderer.close`

- [ ] In `Renderer.close` (currently around line 1180), insert these lines anywhere within the release block (next to the other VBO/VAO/program releases is fine):

```python
        self._scene_fbo.release()
        self._scene_tex.release()
        self._tonemap_vao.release()
        self.tonemap_prog.release()
```

### Step 5: Visual smoke check

- [ ] Run the live sim:

```bash
source .venv/bin/activate && python -m halflife.main
```

Expected:
- Sim starts without GL errors (no traceback in stdout).
- Particles render with the new OKLCh palette — no eye-stabbing pure RGB primaries.
- Bright clusters during fusion roll off smoothly rather than clipping to flat color.
- HUD buttons / hint text remain crisp at the top/bottom of the window.
- Toggling Stats / Events / Bonds / Pause / Params works as before.

If you see a black screen or GL error, common fixes:
- Confirm `'f2'` dtype is supported in your moderngl build. If not, swap `dtype='f2'` for `dtype='f4'` in the `_scene_tex` line (still HDR, just wider).
- Confirm `self._hud_quad_vbo` exists before the HDR scene block in `__init__`. The HUD shader section creates it; the HDR block must come after.

Close the sim window (Q or window close).

### Step 6: Commit

- [ ] Run:

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): HDR scene FBO + ACES tonemap composite

Scene (particles + bonds + events) renders to a half-float RGBA16F
framebuffer, then a fullscreen tonemap pass applies a Narkowicz-fit
ACES curve and converts linear sRGB → display sRGB before reaching
the default framebuffer. HUD continues to draw on top in LDR.

Gives the renderer headroom above 1.0 so velocity-boosted particles
roll off gracefully instead of clipping, and is the prerequisite for
the trails effect which needs a feedback-friendly HDR buffer."
```

---

## Task 3: Ping-pong trail FBOs + decay shader + particle trail uniforms

After this task, trails are visible in the live sim with hardcoded settings.
The UI is not wired up yet. Toggling trails on/off and tuning their look
happens in Task 4.

**Files:**
- Modify: `halflife/renderer.py` (add decay shader, ping-pong FBOs, render-settings dict, particle shader uniforms; update `render()`)

### Step 1: Add the decay shader source

- [ ] In the shader-source block of `renderer.py`, immediately after the `TONEMAP_FRAGMENT_SHADER` definition added in Task 2, append:

```python
# Trail decay: full-screen pass that reads the previous trail texture and
# writes texel × u_decay into the current trail FBO. Acts as both "clear"
# and "fade." u_decay = 0 ⇒ full clear; u_decay = 0.95 ⇒ short tails;
# u_decay → 1.0 ⇒ near-infinite trails (will saturate eventually).
TRAIL_DECAY_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

TRAIL_DECAY_FRAGMENT_SHADER = """
#version 330

in  vec2 v_uv;
uniform sampler2D src_tex;
uniform float u_decay;
out vec4 fragColor;

void main() {
    fragColor = texture(src_tex, v_uv) * u_decay;
}
"""
```

### Step 2: Add particle trail uniforms to the existing particle shaders

- [ ] In the `PARTICLE_VERTEX_SHADER` source (currently at the top of `renderer.py`, around line 50), add `u_size_mult` and `u_alpha_mult` uniforms and apply them. Replace the entire `PARTICLE_VERTEX_SHADER` constant with:

```python
PARTICLE_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec4  in_color;
in float in_size;

out vec4 v_color;

uniform vec2  u_world_size;
uniform float u_size_mult;
uniform float u_alpha_mult;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = in_size * u_size_mult;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha_mult);
}
"""
```

The fragment shader is unchanged.

### Step 3: Build the trail FBOs, decay program, and render-settings dict in `__init__`

- [ ] In `Renderer.__init__`, immediately after the HDR scene FBO block (the block that creates `self._scene_tex` and `self.tonemap_prog` from Task 2) and before the `# ── Event sprite shader …` block, insert:

```python
        # ── Trail accumulation (ping-pong RGBA16F FBOs) ──────────────────────
        # Two half-float framebuffers swapped each frame. When trails are on,
        # each frame's scene is composited on top of (decay × previous frame)
        # to produce smooth exponentially-fading streaks. When trails are off,
        # the scene FBO from the previous block is used directly and these
        # remain idle.
        self._trail_texs = [
            self.ctx.texture((config.window_width, config.window_height), 4, dtype='f2')
            for _ in range(2)
        ]
        for tex in self._trail_texs:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._trail_fbos = [
            self.ctx.framebuffer(color_attachments=[tex]) for tex in self._trail_texs
        ]
        self._trail_idx = 0

        self.trail_decay_prog = self.ctx.program(
            vertex_shader=TRAIL_DECAY_VERTEX_SHADER,
            fragment_shader=TRAIL_DECAY_FRAGMENT_SHADER,
        )
        self._trail_decay_vao = self.ctx.vertex_array(
            self.trail_decay_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )

        # ── Render-settings dict ─────────────────────────────────────────────
        # Trail-related state lives here, separate from PhysicsParams. Defaults
        # match the slider defaults defined in Task 4. UI sliders write into
        # this dict; the renderer reads from it each frame.
        self._render_settings = {
            'trails_on':            False,
            'trail_decay':          0.95,
            'trail_particle_size':  1.0,
            'trail_particle_alpha': 1.0,
        }
```

### Step 4: Update `render()` to ping-pong through trail FBOs and apply particle uniforms

- [ ] Replace the body of `Renderer.render` again (defined in Task 2 Step 3) with this version. The differences from Task 2: trail decay pass replaces the direct clear when trails are on; tonemap reads from the current trail FBO instead of `_scene_fbo`; particle program receives `u_size_mult` and `u_alpha_mult` uniforms.

```python
    def render(self, fps: float, step_count: int, n_alive: int):
        """Clear screen, draw scene, tonemap, draw HUD, flip buffers."""
        bg     = self.config.background_color
        rs     = self._render_settings
        on     = bool(rs['trails_on'])
        decay  = float(rs['trail_decay']) if on else 0.0
        sz_m   = float(rs['trail_particle_size'])  if on else 1.0
        al_m   = float(rs['trail_particle_alpha']) if on else 1.0

        curr_fbo = self._trail_fbos[self._trail_idx]
        prev_tex = self._trail_texs[1 - self._trail_idx]

        # ── Decay / clear into the current trail FBO ─────────────────────────
        curr_fbo.use()
        if on:
            # Read previous frame, write previous × decay. No clear before —
            # the shader fully covers the framebuffer.
            prev_tex.use(location=0)
            self.trail_decay_prog['src_tex'].value = 0
            self.trail_decay_prog['u_decay'].value = decay
            self._trail_decay_vao.render(moderngl.TRIANGLES, vertices=6)
        else:
            # Trails off → reset the FBO each frame.
            self.ctx.clear(*bg)

        # ── Scene pass (into current trail FBO) ──────────────────────────────
        # Particle size/alpha multipliers live on the particle program so the
        # user can dial live particles down while accumulated trails dominate.
        self.particle_prog['u_size_mult'].value  = sz_m
        self.particle_prog['u_alpha_mult'].value = al_m

        if self._n_particles_to_draw > 0:
            self.particle_vao.render(moderngl.POINTS, vertices=self._n_particles_to_draw)

        # Bonds
        if self.composite_mode == self.MODE_BONDS and self._n_bond_vertices > 0:
            self.ctx.line_width = 1.0
            self.bond_vao.render(moderngl.LINES, vertices=self._n_bond_vertices)

        # Event sprites
        if self._show_events and self._n_event_vertices > 0:
            self.event_prog['u_world_size'].value = (
                self.config.world_width, self.config.world_height
            )
            self._event_vao.render(moderngl.POINTS, vertices=self._n_event_vertices)

        # ── Tonemap composite to default framebuffer ─────────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._trail_texs[self._trail_idx].use(location=0)
        self.tonemap_prog['scene_tex'].value = 0
        self._tonemap_vao.render(moderngl.TRIANGLES, vertices=6)

        # ── HUD overlay (LDR, on top of tonemapped scene) ────────────────────
        # HUD overlay — only re-render the pygame surface and re-upload the
        # texture when the HUD has actually changed. The fullscreen-quad GL
        # blit always runs (it's cheap; reads the cached texture).
        # Stats panel updates every frame (sparklines, FPS), so force dirty
        # while it's shown.
        if self._show_stats:
            self._hud_dirty = True

        if self._hud_dirty:
            self._render_hud_surface(fps)
            surf_data = pygame.image.tostring(self._hud_surface, 'RGBA', True)
            self._hud_texture.write(surf_data)
            self._hud_dirty = False

        self._hud_texture.use(location=0)
        self.hud_prog['hud_tex'].value = 0
        self._hud_quad_vao.render(moderngl.TRIANGLES, vertices=6)

        pygame.display.flip()

        # Swap ping-pong index for next frame
        self._trail_idx = 1 - self._trail_idx
```

### Step 5: Retire the unused `_scene_fbo` from Task 2

- [ ] The single-FBO `_scene_fbo` from Task 2 is now superseded by the ping-pong trail FBOs. Remove these three lines from `__init__` (they were added in Task 2 Step 2):

```python
        self._scene_tex = self.ctx.texture(
            (config.window_width, config.window_height), 4, dtype='f2'
        )
        self._scene_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._scene_fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])
```

Keep the `self.tonemap_prog` and `self._tonemap_vao` definitions — they're still used (tonemap pass now reads from the current trail texture instead).

- [ ] Remove the matching releases from `Renderer.close`:

```python
        self._scene_fbo.release()
        self._scene_tex.release()
```

(`self._tonemap_vao.release()` and `self.tonemap_prog.release()` stay.)

- [ ] Add releases for the new trail resources in `Renderer.close`:

```python
        for fbo in self._trail_fbos:
            fbo.release()
        for tex in self._trail_texs:
            tex.release()
        self._trail_decay_vao.release()
        self.trail_decay_prog.release()
```

### Step 6: Hardcode `trails_on = True` temporarily and visual smoke check

- [ ] In `__init__`, temporarily change the `_render_settings` initialization so trails are on by default for this test:

```python
        self._render_settings = {
            'trails_on':            True,   # TEMPORARY — reverted in Task 4
            'trail_decay':          0.95,
            'trail_particle_size':  1.0,
            'trail_particle_alpha': 1.0,
        }
```

- [ ] Run the sim:

```bash
source .venv/bin/activate && python -m halflife.main
```

Expected:
- Particles leave visible exponentially-fading trails.
- Trail length feels "short tail" (~half-second visible decay) at 0.95.
- No GL error in stdout.
- The image otherwise composites correctly (HUD on top, etc.).

If trails look wrong (too short / too long / not visible), confirm the `u_decay` value reaches the shader by adding a `print(decay)` line in `render` temporarily — it should read 0.95.

- [ ] Revert the temporary `trails_on: True` back to `False`:

```python
            'trails_on':            False,
```

Re-run the sim and verify it now looks like Task 2's output (no trails).

### Step 7: Commit

- [ ] Run:

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): ping-pong trail FBOs + decay shader + particle trail uniforms

Two RGBA16F framebuffers ping-ponged each frame. When trails are on a
fullscreen decay pass writes (previous × u_decay) into the current FBO
before the scene draws on top; when off the FBO is cleared normally.

Particle shader gets u_size_mult and u_alpha_mult so live particles can
be dialed independently of the accumulated trail. Defaults are 1.0 and
trails off so behavior matches the previous task until the UI lands."
```

---

## Task 4: UI — Trails toggle button + three trail sliders

This task wires the existing renderer state to user controls. It extends the
`Slider` class to support arbitrary target dicts, adds a `_render_sliders`
list, and appends a "Trails" toggle button to the existing button strip.

**Files:**
- Modify: `halflife/renderer.py` (Slider class; button list; slider construction; handlers; `_render_hud_surface` for button label)

### Step 1: Extend the `Slider` class to accept a target-dict reference

- [ ] In `renderer.py`, find the `Slider` class (around line 178) and update its `__init__` signature to accept an optional `target_dict` argument. Replace the signature line and add the assignment. The change is small and surgical:

Old `__init__` signature:
```python
    def __init__(self, label: str, field: str, default_value: float,
                 track_rect: pygame.Rect, fmt: str = "{:.3f}",
                 linear_range=None):
```

New `__init__` signature:
```python
    def __init__(self, label: str, field: str, default_value: float,
                 track_rect: pygame.Rect, fmt: str = "{:.3f}",
                 linear_range=None, target_dict: dict = None):
```

And inside `__init__`, immediately after the `self._linear_range = linear_range` line, add:

```python
        # When set, this slider writes into target_dict[field] instead of the
        # renderer's generic _physics_updates dict. Used to keep render-only
        # sliders out of the PhysicsParams update pipeline.
        self._target_dict = target_dict
```

Add a property to expose it:

```python
    @property
    def target_dict(self) -> dict:
        return self._target_dict
```

(Place this right next to the existing `field` and `value` properties.)

### Step 2: Add the Trails toggle button to the button strip

- [ ] In `Renderer.__init__`, find the button-list construction (around line 386, `# ── HUD buttons: …`). Append a `"Trails"` entry to the list passed to `enumerate(...)`. The full updated list is:

```python
        for k, (label, action) in enumerate([
            ("Pause",       "pause"),
            ("Bonds",       "toggle_bonds"),
            ("Events",      "toggle_events"),
            ("Trails",      "toggle_trails"),
            ("Reset",       "reset"),
            ("Params",      "toggle_params"),
            ("Reroll All",  "reroll_all"),
            ("Reroll IC",   "reroll_particles"),
            ("Reroll Chem", "reroll_chemistry"),
        ]):
```

### Step 3: Add a `toggle_trails` method on `Renderer`

- [ ] Next to `toggle_events` (around line 533), add:

```python
    def toggle_trails(self):
        self._render_settings['trails_on'] = not self._render_settings['trails_on']
        self._hud_dirty = True
```

### Step 4: Update the toggle-button label in `_render_hud_surface`

- [ ] In `_render_hud_surface`, find the per-button label-flip chain (around line 947, the `if action == 'pause': … elif action == 'toggle_events': … elif action == 'toggle_params':` block). Add a new branch for `'toggle_trails'` between `'toggle_events'` and `'toggle_params'`:

```python
            elif action == 'toggle_events':
                display_label = "Events ON" if self._show_events else "Events"
            elif action == 'toggle_trails':
                display_label = "Trails ON" if self._render_settings['trails_on'] else "Trails"
            elif action == 'toggle_params':
                display_label = "Params ON" if self._show_params else "Params"
```

### Step 5: Append three trail sliders to the params panel

- [ ] In `Renderer.__init__`, find the `slider_specs` list (around line 426). At the end of the list, **after** the existing `("spring_k", …)` entry and before the closing `]`, insert a group separator and the three trail sliders. The full tail of the list becomes:

```python
            ("spring_k",                 "spring k",    _phys("spring_k"),             "{:.1f}", None),
            None,
            # ── Rendering / trails ──────────────────────────────────────────
            ("trail_decay",          "trail decay", 0.95, "{:.3f}", (0.0, 0.999)),
            ("trail_particle_size",  "trail size",  1.0,  "{:.2f}", (0.25, 2.0)),
            ("trail_particle_alpha", "trail alpha", 1.0,  "{:.2f}", (0.1,  1.0)),
        ]
```

### Step 6: Wire slider construction to point trail sliders at `_render_settings`

- [ ] In the same `__init__`, find the loop that builds `self._sliders` from `slider_specs` (around line 446). The render-only fields need their `Slider` instance to carry `target_dict=self._render_settings`. The simplest way: detect by field name. Replace the loop body to look like:

```python
        group_gap = 14
        self._sliders = []
        row_y = slider_start_y
        RENDER_FIELDS = {'trail_decay', 'trail_particle_size', 'trail_particle_alpha'}
        for spec in slider_specs:
            if spec is None:
                row_y += group_gap
                continue
            field, label, default, fmt, lin = spec
            track = pygame.Rect(panel_x + 4, row_y + 18, slider_track_w, 8)
            target = self._render_settings if field in RENDER_FIELDS else None
            self._sliders.append(
                Slider(label, field, default, track, fmt, linear_range=lin, target_dict=target)
            )
            row_y += slider_row_h
```

### Step 7: Update slider handlers to dispatch to the right dict

- [ ] In `handle_mousedown_slider` (around line 556), the global "Reset Params" branch and each per-slider write currently push to `self._physics_updates`. Make each write inspect the slider's `target_dict` and write there if set; otherwise write to `_physics_updates`. The full updated method:

```python
    def handle_mousedown_slider(self, pos) -> bool:
        """Start dragging a slider if pos hits a handle, or reset if a reset button hit."""
        if not self._show_params:
            return False
        if self._params_reset_rect.collidepoint(pos):
            for s in self._sliders:
                s.reset()
                if s.target_dict is not None:
                    s.target_dict[s.field] = s.value
                else:
                    self._physics_updates[s.field] = s.value
            self._hud_dirty = True
            return True
        for slider in self._sliders:
            if slider.hit_reset(pos):
                slider.reset()
                if slider.target_dict is not None:
                    slider.target_dict[slider.field] = slider.value
                else:
                    self._physics_updates[slider.field] = slider.value
                self._hud_dirty = True
                return True
        for slider in self._sliders:
            if slider.hit_handle(pos):
                self._dragging_slider = slider
                slider.handle_drag(pos)
                if slider.target_dict is not None:
                    slider.target_dict[slider.field] = slider.value
                else:
                    self._physics_updates[slider.field] = slider.value
                self._hud_dirty = True
                return True
        return False
```

- [ ] Update `handle_mousemotion` (around line 581) the same way:

```python
    def handle_mousemotion(self, pos) -> None:
        if self._dragging_slider is not None:
            self._dragging_slider.handle_drag(pos)
            if self._dragging_slider.target_dict is not None:
                self._dragging_slider.target_dict[self._dragging_slider.field] = self._dragging_slider.value
            else:
                self._physics_updates[self._dragging_slider.field] = self._dragging_slider.value
            self._hud_dirty = True
```

### Step 8: Wire the Trails button click to `toggle_trails` in `main.py`

The `Renderer.handle_click` method returns the action string for a button click, and `main.py` dispatches on that string. We need to confirm `'toggle_trails'` is dispatched. Find the click-dispatch block in `halflife/main.py` (search for `toggle_events` or `toggle_bonds`).

- [ ] Find this block in `halflife/main.py`:

```python
                    elif action == "toggle_events":
                        renderer.toggle_events()
```

Immediately after it, add:

```python
                    elif action == "toggle_trails":
                        renderer.toggle_trails()
```

(If your local copy of `main.py` uses a `dict` or `match` dispatch instead, follow the existing pattern.)

### Step 9: Visual smoke check

- [ ] Run the sim:

```bash
source .venv/bin/activate && python -m halflife.main
```

Expected:
- Sim starts with trails OFF (no fading effect).
- Click the "Trails" button — label flips to "Trails ON", trails appear with default 0.95 decay.
- Open the Params panel — three trail sliders visible at the bottom of the list with a small gap above them.
- Drag `trail decay` slider:
  - 0.0 → no visible trails even with Trails ON.
  - 0.95 → short comet tails.
  - 0.99 → long phosphorescent tails.
- Drag `trail size`:
  - 0.25 → live particles shrink to tiny points.
  - 2.0 → live particles balloon up.
  - The trail layer (accumulated faded particles) is unchanged in size; only the freshly-drawn particles move.
- Drag `trail alpha`:
  - 0.1 → live particles nearly invisible.
  - 1.0 → live particles at full opacity.
- Click "Trails" again → trails disappear, button reverts to "Trails".
- Per-slider reset (↺) returns each slider to its default.
- Global "Reset Params" still works and does not crash; trail sliders also reset.
- Physics sliders (repulsion, attract, dt, …) still apply to the simulation as before — confirm by dragging `damping` and seeing the sim respond.

### Step 10: Commit

- [ ] Run:

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py halflife/main.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): Trails toggle button + decay/size/alpha sliders

A new 'Trails' button in the left button strip toggles the accumulation
pass on and off. Three sliders at the bottom of the params panel (trail
decay, trail size, trail alpha) tune the look at runtime — decay
controls the feedback multiplier; size and alpha modulate the live
particles drawn each frame so accumulated trails can dominate.

Slider class gains an optional target_dict so render-only sliders write
to renderer-local state instead of the PhysicsParams update buffer."
```

---

## Task 5: Final visual smoke pass and cleanup

**Files:** None modified — this is a verification pass.

- [ ] Run the live sim:

```bash
source .venv/bin/activate && python -m halflife.main
```

- [ ] Walk through this checklist. Every item should pass with no GL errors:
  - [ ] Colors look harmonious — no eye-stabbing pure RGB primaries with `num_species=12`.
  - [ ] Bright fusion events glow softly instead of clipping to flat color.
  - [ ] Pause / Resume button works.
  - [ ] Bonds / Merged / None mode cycle works.
  - [ ] Events button toggles ring sprites on/off.
  - [ ] Trails button toggles trails on/off; label flips correctly.
  - [ ] Stats panel toggles and renders correctly on top of tonemapped scene.
  - [ ] Params panel toggles. All sliders draggable. Per-slider reset (↺) works.
  - [ ] Global "Reset Params" resets physics sliders AND trail sliders.
  - [ ] Reroll All / IC / Chem buttons re-init the world without GL errors.
  - [ ] Saving a screenshot (S key) produces a PNG that matches the on-screen image.
  - [ ] FPS hasn't regressed dramatically — should be within 10% of pre-change FPS at the same `num_particles`.

- [ ] If anything in the checklist fails, fix it before declaring done. Common issues:
  - Black screen on launch → check that `self._hud_quad_vbo` is created before any `vertex_array` reuses it.
  - Trails persist forever → confirm `_trail_idx = 1 - self._trail_idx` swap is at the end of `render()`.
  - Trail sliders affect physics → confirm `RENDER_FIELDS` set in Task 4 Step 6 lists all three field names exactly.
  - "Reset Params" doesn't reset trail sliders → confirm Task 4 Step 7 patch uses `s.target_dict` in the global reset loop.

- [ ] No further commit needed unless cleanups were required.

---

## Self-review notes

**Spec coverage check:**
- OKLCh palette → Task 1.
- HDR scene FBO + ACES tonemap composite → Task 2.
- Ping-pong trail FBOs with decay shader → Task 3.
- Particle size/alpha trail uniforms → Task 3 (shader) + Task 4 (UI wiring).
- Trails toggle button → Task 4.
- Three trail sliders → Task 4.
- Render-settings dict separate from PhysicsParams → Task 3 (introduction) + Task 4 (wiring).
- Test for OKLCh palette → Task 1.

**Type-consistency check:**
- Field names used in `_render_settings` (`trails_on`, `trail_decay`, `trail_particle_size`, `trail_particle_alpha`) match the slider `field` strings in Task 4 Step 5 and the `RENDER_FIELDS` set in Task 4 Step 6. Verified consistent across tasks.
- Shader uniform names (`u_decay`, `u_size_mult`, `u_alpha_mult`, `scene_tex`, `src_tex`) match between shader source and Python `.value` accesses.

**Placeholder scan:** No "TBD" / "TODO" / "implement later" — every step has executable code or commands.

**Risks acknowledged:**
- `dtype='f2'` may need fallback to `'f4'` in some WSL OpenGL builds (called out in Task 2 Step 5).
- Background color is reinterpreted as linear sRGB; for the default `(0.05, 0.05, 0.08)` this is barely visible. Tunable post-implementation if needed.
