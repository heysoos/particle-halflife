"""
Real-time ModernGL + pygame renderer for the Half-Life simulator.

Renders particles as point sprites (circular blobs), colored by species
and sized by mass. Composites can be shown in two modes (toggle with B):
  - Bonds mode:  GL_LINES connecting composite member particles
  - Merged mode: single large point at the composite center of mass

Clickable HUD overlay (rendered as a pygame surface → OpenGL texture):
  Buttons along the left edge; stats panel (toggle) on the right; event
  sprites (expanding rings) at fusion/fission/spawn/decay sites.

Data flow per frame:
  1. np.asarray() — transfer JAX GPU arrays to CPU numpy
  2. Pack vertex data: (x, y, r, g, b, a, size) per particle
  3. vbo.write() — upload to GPU via OpenGL
  4. Draw particles (GL_POINTS), bonds (GL_LINES), events (GL_POINTS)
  5. Render HUD: pygame surface → moderngl texture → fullscreen quad
  6. pygame.display.flip()
"""

import collections
from typing import NamedTuple, Optional

import numpy as np
import pygame
import moderngl
import jax

from halflife.config import SimConfig
from halflife.state import WorldState, get_species_colors, initialize_physics_params
from halflife.profiler import ProfileMetrics
from halflife.render.widgets import Slider
from halflife.render.camera import Camera
from halflife.render.hud import HUDPainter


# ── CPU-side state snapshot ───────────────────────────────────────────────────
# Per-frame CPU copy of the simulation arrays that the renderer needs for
# things that run *after* update() but before the next step — selection
# hit-testing, the inspector panel, the highlight ring. Built once at the top
# of update() via a single batched jax.device_get(...) so we pay one CUDA
# sync + one DMA per frame instead of fifteen.
#
# Field names drop the "_last_" prefix the individual instance attributes
# used to carry — the tuple itself is the "last-frame" marker.
class CPUStateSnapshot(NamedTuple):
    positions:           np.ndarray
    velocities:          np.ndarray
    species:             np.ndarray
    mass:                np.ndarray
    energy:              np.ndarray
    age:                 np.ndarray
    comp_id:             np.ndarray
    comp_members:        np.ndarray
    comp_count:          np.ndarray
    comp_alive:          np.ndarray
    comp_species_hash:   np.ndarray
    comp_binding_energy: np.ndarray
    comp_half_life:      np.ndarray
    comp_age:            np.ndarray
    comp_free_bonds:     np.ndarray


# ── Bond rendering cap ────────────────────────────────────────────────────────
# Maximum number of forward-slot bonds emitted per composite member in
# "bonds" view. For each member at slot i, bonds are emitted to slots
# (i+1, i+2, …, i+MAX_BONDS_PER_PARTICLE). Each particle ends up touched by
# at most ~2× this value's worth of bonds (forward + incoming). Pure visual
# cap — does not affect physics. Bumping this above ~5 starts to undo the
# performance win on big composites.
MAX_BONDS_PER_PARTICLE = 3

# Composite-size histogram refresh cadence. The chart only updates every
# this many frames (and only while the stats panel is open). Lower = more
# responsive, higher = cheaper.
HIST_COMPUTE_INTERVAL = 10


# ── GLSL Shaders ──────────────────────────────────────────────────────────────

PARTICLE_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec4  in_color;
in float in_size;

out vec4 v_color;

uniform vec2  u_world_size;
uniform vec2  u_view_center;   // world point at screen center
uniform float u_view_scale;    // 1.0 = default; >1 zooms in
uniform float u_size_mult;
uniform float u_alpha_mult;

void main() {
    // Camera: (pos - center) * scale, then re-center on (world_size / 2) so
    // the result still lives in [0, world_size] when scale=1 and center is
    // the world midpoint — keeps the no-camera default visually identical.
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = in_size * u_size_mult * u_view_scale;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha_mult);
}
"""

PARTICLE_FRAGMENT_SHADER = """
#version 330

in  vec4 v_color;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = length(coord) * 2.0;
    float alpha = 1.0 - smoothstep(0.6, 1.0, r);
    float brightness = 1.0 - smoothstep(0.0, 0.5, r) * 0.3;
    fragColor = vec4(v_color.rgb * brightness, v_color.a * alpha);
}
"""

BOND_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec4 in_color;

out vec4 v_color;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""

BOND_FRAGMENT_SHADER = """
#version 330

in  vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""

# Shared fullscreen-quad vertex shader: emit NDC pos straight through and
# derive UVs from it. Reused by every fullscreen-quad pass (HUD, tonemap,
# trail decay) — three separate copies used to exist and were byte-identical.
#
# No Y-flip in the shader. For the HUD pass this is intentional: the pygame
# surface is uploaded with pygame.image.tostring(..., True) which pre-flips
# the pixel data, so sampling with v_uv aligned to in_pos works. For tonemap
# and trail-decay the texture is an OpenGL FBO attachment that is already in
# OpenGL convention — also no flip needed.
FULLSCREEN_QUAD_VS = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

# HUD: fullscreen quad with pygame surface as texture
HUD_FRAGMENT_SHADER = """
#version 330

in  vec2 v_uv;
uniform sampler2D hud_tex;
out vec4 fragColor;

void main() {
    fragColor = texture(hud_tex, v_uv);
}
"""

# Selection highlight: a single point sprite stamped at the selected
# particle's world position. Soft white ring drawn on top of bonds/events
# in the fresh-overlay FBO so it's always visible regardless of zoom.
HIGHLIGHT_VERTEX_SHADER = """
#version 330

in vec2 in_position;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;
uniform float u_size_px;     // ring outer diameter in screen pixels

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_size_px;
}
"""

HIGHLIGHT_FRAGMENT_SHADER = """
#version 330

out vec4 fragColor;

void main() {
    float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
    // Ring at r ≈ 0.85, ~0.08 thick, soft falloff
    float ring = 1.0 - clamp(abs(r - 0.85) / 0.08, 0.0, 1.0);
    if (ring < 0.02) discard;
    fragColor = vec4(1.0, 1.0, 1.0, ring);
}
"""


# Tonemap composite: sample HDR scene texture, apply ACES filmic curve,
# convert linear → sRGB display gamma. Output replaces what would have been
# the direct framebuffer write of the LDR pipeline. Bg/HUD compositing
# happens *after* this pass. Vertex shader is the shared FULLSCREEN_QUAD_VS.
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
    vec4 src    = texture(scene_tex, v_uv);
    vec3 mapped = aces(src.rgb);
    // Pass-through alpha so the same shader works for the opaque trail pass
    // (caller disables blending; alpha doesn't matter) and the alpha-blended
    // fresh-overlay pass (caller enables SRC_ALPHA blending and source alpha
    // becomes the coverage of bonds + event sprites over the trail layer).
    fragColor   = vec4(linear_to_srgb(mapped), src.a);
}
"""

# Trail decay: full-screen pass that reads the previous trail texture and
# writes texel × u_decay into the current trail FBO. Acts as both "clear"
# and "fade." u_decay = 0 ⇒ full clear; u_decay = 0.95 ⇒ short tails;
# u_decay → 1.0 ⇒ near-infinite trails (will saturate eventually). Vertex
# shader is the shared FULLSCREEN_QUAD_VS.
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

# Event sprites: expanding rings at fusion/fission/spawn/decay sites
EVENT_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec3  in_color;
in float in_age;

out vec3  v_color;
out float v_age;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = mix(60.0, 20.0, in_age) * u_view_scale;
    v_color = in_color;
    v_age   = in_age;
}
"""

EVENT_FRAGMENT_SHADER = """
#version 330

in  vec3  v_color;
in  float v_age;
out vec4  fragColor;

void main() {
    float r    = length(gl_PointCoord - vec2(0.5)) * 2.0;
    float ring = 1.0 - abs(r - v_age) / 0.12;
    ring = clamp(ring, 0.0, 1.0) * (1.0 - v_age);
    fragColor = vec4(v_color, ring);
}
"""



# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    ModernGL + pygame renderer. One instance per simulation run.

    After initialization, call update(state) + render(fps, step_count, n_alive)
    each frame.
    """

    MODE_BONDS  = "bonds"
    MODE_MERGED = "merged"
    MODE_NONE   = "none"   # plain particles, no bond overlay / no COM blob

    # Button background tints, keyed by an action substring or sentinel.
    # Pause shows the alternate tint when self._paused is True. Looked up by
    # HUDPainter._button_bg_color so HUDPainter.paint stays tidy.
    BUTTON_BG = {
        'default':      (40, 55, 80, 200),
        'reset':        (90, 40, 40, 200),
        'reroll':       (60, 40, 80, 200),   # any action starting with 'reroll_'
        'pause':        (40, 80, 50, 200),   # play/idle state
        'pause_active': (80, 60, 30, 200),   # actually paused
    }

    # Total stats-panel height in pixels. Derived from the row layout inside
    # HUDPainter._paint_stats_panel:
    #   4 static rows (16px each)
    # + 6 spark-stat rows (15px label + 18px sparkline = 33px each)
    # + 4px gap
    # + 18px histogram header
    # + 64px chart
    # + 20px tick row
    # + 10px bottom padding
    # Used both for the panel itself AND by the inspector panel to know how
    # far down to slide when the stats panel is open. The two had drifted
    # apart as literals in the past, so it lives here as a single source.
    STATS_PANEL_H = 4 * 16 + 6 * 33 + 4 + 18 + 64 + 20 + 10

    def __init__(self, config: SimConfig, metrics: ProfileMetrics = None):
        self.config = config
        self.metrics = metrics
        self.composite_mode = self.MODE_BONDS

        # ── pygame + OpenGL context ──────────────────────────────────────────
        pygame.init()
        pygame.display.set_caption("Half-Life Particle Simulator")
        # No RESIZABLE: avoids texture/surface resize complexity
        pygame.display.set_mode(
            (config.window_width, config.window_height),
            pygame.OPENGL | pygame.DOUBLEBUF
        )
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Particle shader ──────────────────────────────────────────────────
        self.particle_prog = self.ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER,
            fragment_shader=PARTICLE_FRAGMENT_SHADER,
        )
        self.particle_prog['u_world_size'].value = (config.world_width, config.world_height)

        self._particle_vertex_size = 7  # floats: x,y,r,g,b,a,size
        _max_draw = config.num_particles + config.max_composites
        self._particle_buf_size = _max_draw * self._particle_vertex_size * 4

        self.particle_vbo = self.ctx.buffer(reserve=self._particle_buf_size)
        # Pre-allocated CPU staging buffer — avoids per-frame heap allocations
        self._part_vbuf = np.zeros((_max_draw, self._particle_vertex_size), dtype=np.float32)
        self.particle_vao = self.ctx.vertex_array(
            self.particle_prog,
            [(self.particle_vbo, '2f 4f 1f', 'in_position', 'in_color', 'in_size')],
        )

        # ── Bond shader ──────────────────────────────────────────────────────
        self.bond_prog = self.ctx.program(
            vertex_shader=BOND_VERTEX_SHADER,
            fragment_shader=BOND_FRAGMENT_SHADER,
        )
        self.bond_prog['u_world_size'].value = (config.world_width, config.world_height)

        # Allocate for up to ~16 members per composite on average (all-pairs = 120 pairs).
        # max_composite_size=64 worst-case would be 184 MB; this keeps it ~23 MB while
        # covering the common case. Bonds beyond the cap are silently dropped.
        avg_pairs_per_comp = 16 * 15 // 2  # 120
        max_bonds = config.max_composites * avg_pairs_per_comp
        self._bond_vertex_size = 6  # floats: x,y,r,g,b,a
        self._bond_buf_size = max_bonds * 2 * self._bond_vertex_size * 4
        self.bond_vbo = self.ctx.buffer(reserve=self._bond_buf_size)
        self.bond_vao = self.ctx.vertex_array(
            self.bond_prog,
            [(self.bond_vbo, '2f 4f', 'in_position', 'in_color')],
        )

        # ── HUD shader (pygame surface → fullscreen quad) ────────────────────
        self.hud_prog = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VS,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        # Fullscreen quad: 2 triangles (6 vertices)
        quad_verts = np.array([
            -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,  1.0, -1.0,  1.0,
        ], dtype=np.float32)
        self._hud_quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self._hud_quad_vao = self.ctx.vertex_array(
            self.hud_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )
        self._hud_surface = pygame.Surface(
            (config.window_width, config.window_height), pygame.SRCALPHA
        )
        self._hud_texture = self.ctx.texture(
            (config.window_width, config.window_height), 4
        )
        self._hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ── HDR scene framebuffer + tonemap composite ─────────────────────────
        # The scene (particles + bonds + events) is rendered into an RGBA16F
        # framebuffer so we have headroom above 1.0. The tonemap pass samples
        # this texture, applies an ACES filmic curve, and writes sRGB-gamma
        # output to the default framebuffer. The HUD is drawn on top of the
        # tonemapped output in LDR space.
        self.tonemap_prog = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VS,
            fragment_shader=TONEMAP_FRAGMENT_SHADER,
        )
        # Reuse the HUD fullscreen-quad VBO geometry — same -1..1 NDC quad.
        self._tonemap_vao = self.ctx.vertex_array(
            self.tonemap_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )

        # ── Trail accumulation (ping-pong RGBA16F FBOs) ──────────────────────
        # Two half-float framebuffers swapped each frame. When trails are on,
        # each frame's scene is composited on top of (decay × previous frame)
        # to produce smooth exponentially-fading streaks. When trails are off,
        # the FBO is cleared normally before the scene draws. The tonemap pass
        # always reads from the current trail FBO regardless of trail state.
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
            vertex_shader=FULLSCREEN_QUAD_VS,
            fragment_shader=TRAIL_DECAY_FRAGMENT_SHADER,
        )
        self._trail_decay_vao = self.ctx.vertex_array(
            self.trail_decay_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )

        # ── Fresh-scene FBO (non-trailing elements) ──────────────────────────
        # Bond lines and event sprites are drawn into this FBO and cleared
        # every frame. It's tonemapped on top of the trail layer with alpha
        # blending so bonds don't smear into colored ribbons when trails are
        # on and event rings don't leave ghost trails after they expire.
        self._fresh_tex = self.ctx.texture(
            (config.window_width, config.window_height), 4, dtype='f2'
        )
        self._fresh_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fresh_fbo = self.ctx.framebuffer(color_attachments=[self._fresh_tex])

        # ── Selection highlight ring ──────────────────────────────────────────
        # Single-vertex point sprite stamped at the selected particle's world
        # position. Rendered into the fresh-overlay FBO so it sits above
        # bonds and events; ring size is fixed in screen pixels so the
        # selection stays visible at any zoom.
        self.highlight_prog = self.ctx.program(
            vertex_shader=HIGHLIGHT_VERTEX_SHADER,
            fragment_shader=HIGHLIGHT_FRAGMENT_SHADER,
        )
        self.highlight_prog['u_world_size'].value = (config.world_width, config.world_height)
        self._highlight_vbo = self.ctx.buffer(reserve=2 * 4)   # one vec2 float32
        self._highlight_vao = self.ctx.vertex_array(
            self.highlight_prog,
            [(self._highlight_vbo, '2f', 'in_position')],
        )

        # ── Render-settings dict ─────────────────────────────────────────────
        # Trail-related state lives here, separate from PhysicsParams. Defaults
        # match the slider defaults defined in Task 4. UI sliders write into
        # this dict; the renderer reads from it each frame.
        self._render_settings = {
            'trails_on':           True,
            'trail_decay':         0.75,
            'particle_size_mult':  0.60,
            'particle_alpha_mult': 0.20,
        }

        # ── Camera (pan + zoom) ──────────────────────────────────────────────
        # Pan/zoom state and conversions live in halflife/render/camera.py.
        # Defaults: view_center = world midpoint, view_scale = 1.0 (identical
        # to the no-camera frame). main.py drives it via self.camera.zoom_at,
        # .pan_by, and .reset; the renderer pushes its uniforms onto every
        # world-space program at the top of render().
        self.camera = Camera(config)

        # ── Event sprite shader ──────────────────────────────────────────────
        self.event_prog = self.ctx.program(
            vertex_shader=EVENT_VERTEX_SHADER,
            fragment_shader=EVENT_FRAGMENT_SHADER,
        )
        self.event_prog['u_world_size'].value = (config.world_width, config.world_height)

        # Programs that read the camera uniforms (u_view_center, u_view_scale).
        # self.camera.push_uniforms() pushes the current view onto every one
        # of them each frame; u_world_size is already set above and doesn't
        # change.
        self._world_space_progs = [
            self.particle_prog,
            self.bond_prog,
            self.highlight_prog,
            self.event_prog,
        ]

        self._event_max = 200
        self._event_vbo = self.ctx.buffer(reserve=self._event_max * 6 * 4)
        self._event_vao = self.ctx.vertex_array(
            self.event_prog,
            [(self._event_vbo, '2f 3f 1f', 'in_position', 'in_color', 'in_age')],
        )
        self._n_event_vertices = 0
        # Each event: (x, y, r, g, b, birth_sim_time, lifetime)
        self._events = []

        # ── Color palette ─────────────────────────────────────────────────────
        self.species_colors = get_species_colors(config)

        # ── HUD buttons: (label, pygame.Rect, action_str) ────────────────────
        btn_w, btn_h, gap = 108, 26, 4
        btn_x = 8
        self._buttons = []
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
            rect = pygame.Rect(btn_x, 8 + k * (btn_h + gap), btn_w, btn_h)
            self._buttons.append((label, rect, action))
        self._n_buttons = len(self._buttons)
        # Stats button lives in the top-right corner
        self._stats_btn_rect = pygame.Rect(config.window_width - btn_w - 8, 8, btn_w, btn_h)

        # ── Physics sliders ───────────────────────────────────────────────────
        self._show_params = False
        self._dragging_slider = None
        self._physics_updates = {}
        panel_x = btn_x + btn_w + 8   # panel opens to the RIGHT of the button strip
        slider_track_w = 200
        # First slider starts 8px below the bottom of the last button in the strip
        slider_start_y = 8 + self._n_buttons * (btn_h + gap) + 8
        slider_row_h = 38
        # Slider specs grouped by relevance. None entries mark group breaks
        # and add an extra gap between groups (no slider drawn for None).
        #
        # Slider defaults are sourced from the PhysicsParams struct the sim
        # actually starts with \u2014 initialize_physics_params(config) reads
        # SimConfig where applicable and falls back to hardcoded values for
        # the few fields not yet in config. This keeps the slider knob (and
        # log-scale 1\u00d7 pivot) aligned with the running simulation: change a
        # SimConfig value, restart, and the slider reflects it.
        physics_defaults = initialize_physics_params(config)
        def _phys(field: str) -> float:
            return float(getattr(physics_defaults, field))
        # Cache layout dims so set_bond_mode() can rebuild the physics slider
        # list with the same geometry after a mode swap.
        self._slider_panel_x      = panel_x
        self._slider_track_w      = slider_track_w
        self._slider_start_y      = slider_start_y
        self._slider_row_h        = slider_row_h
        self._slider_group_gap    = 14
        self._slider_phys_default = _phys

        self._rebuild_physics_sliders(config.bond_mode)
        self._params_reset_rect = pygame.Rect(panel_x + 4, slider_start_y - 26, 100, 20)

        # ── Render sliders (trail panel — toggled via gear nub on Trails) ────
        # Live in their own panel, mutually exclusive with the physics params
        # panel. The trail panel reuses the same X column so it never overflows
        # at narrow window widths. Each slider writes into self._render_settings.
        self._show_render_params = False
        render_slider_specs = [
            ("trail_decay",         "trail decay",   0.95, "{:.3f}", (0.0, 0.999)),
            ("particle_size_mult",  "particle size", 1.0,  "{:.2f}", (0.25, 2.0)),
            ("particle_alpha_mult", "particle alpha", 1.0, "{:.2f}", (0.1,  1.0)),
        ]
        self._render_sliders = []
        r_row_y = slider_start_y
        for field, label, default, fmt, lin in render_slider_specs:
            track = pygame.Rect(panel_x + 4, r_row_y + 18, slider_track_w, 8)
            self._render_sliders.append(
                Slider(label, field, default, track, fmt,
                       linear_range=lin, target_dict=self._render_settings)
            )
            r_row_y += slider_row_h
        self._render_slider_content_h = r_row_y - slider_start_y
        self._render_params_reset_rect = pygame.Rect(panel_x + 4, slider_start_y - 26, 100, 20)

        # Gear nub on the Trails button — small region on its right edge that
        # opens the render-settings panel separately from the Params button.
        for _lbl, _rect, _act in self._buttons:
            if _act == 'toggle_trails':
                gear_w = 18
                self._trails_gear_rect = pygame.Rect(
                    _rect.right - gear_w, _rect.top, gear_w, _rect.height
                )
                break

        # ── Runtime state ─────────────────────────────────────────────────────
        self._n_particles_to_draw = 0
        self._n_bond_vertices = 0
        self._font = pygame.font.SysFont('monospace', 13)
        self._clock = pygame.time.Clock()

        self._show_stats  = False
        self._show_events = True
        self._paused      = False   # mirror of main loop paused state

        # ── Particle selection (click-to-inspect) ─────────────────────────────
        # _selected_idx is the particle index of the current selection, or -1
        # for none. The cached snapshot is rebuilt every frame from the
        # CPU-side state arrays so the inspector panel reads stay live.
        self._selected_idx = -1
        self._selected_snapshot = None
        # Pick radius in WORLD UNITS — independent of zoom, so clicks have the
        # same feel at all zoom levels.
        self._select_radius_world = 2.5
        # Cached per-frame state arrays (populated at the top of update()).
        # None until the first update() runs. Single CPUStateSnapshot tuple
        # so adding a new cached array touches one definition (the NamedTuple
        # at the top of this file) instead of an init block + an assignment
        # block + every read site.
        self._cpu_state: Optional[CPUStateSnapshot] = None
        # Per-species valence cache — built lazily on first inspector access.
        self._species_valence = None

        # Close-button rect on the inspector panel. Recomputed each frame
        # inside HUDPainter._paint_inspector; gated by _selected_idx >= 0 so
        # a stale (0,0,0,0)-area rect is harmless before the first panel draw.
        self._inspector_close_rect = pygame.Rect(0, 0, 18, 18)

        # HUD dirty flag: when False, skip the full pygame redraw + texture
        # upload and just blit the cached texture. The stats panel updates
        # every frame (sparklines, FPS), so it forces dirty=True while shown;
        # all other UI changes (toggles, slider drags, button label flips)
        # explicitly set dirty=True via the mutator that caused them.
        self._hud_dirty = True

        self._prev_comp_alive = None

        self._stats_alive    = 0
        self._stats_free     = 0
        self._stats_n_comp   = 0
        self._stats_n_unique = 0   # # of distinct species_hash values across alive composites
        self._stats_energy   = 0.0
        self._stats_step     = 0
        self._stats_sim_time = 0.0
        self._stats_hist     = np.zeros(config.max_composite_size, dtype=np.int32)
        # Histogram x-axis state — see the histogram block in
        # HUDPainter._paint_stats_panel. Expansion is instant (so data never
        # overflows the chart) but shrinking only happens every
        # HIST_AXIS_SHRINK_FRAMES frames to suppress jitter from churning
        # small composites.
        self._hist_size_max_cached = 2
        self._hist_axis_age        = 0
        # np.histogram on alive_counts runs once per N frames — the bar chart
        # readout doesn't need 60-120 Hz refresh.
        self._hist_compute_age     = 0

        # Event counters
        self._fusion_total   = 0
        self._decay_total    = 0   # composite decays (fissions)
        self._fusion_rate    = 0.0
        self._decay_rate     = 0.0
        self._event_history  = collections.deque(maxlen=300)

        # Sparkline buffers
        SPARK_LEN = 150
        self._spark_free   = collections.deque(maxlen=SPARK_LEN)
        self._spark_comp   = collections.deque(maxlen=SPARK_LEN)
        self._spark_unique = collections.deque(maxlen=SPARK_LEN)
        self._spark_energy = collections.deque(maxlen=SPARK_LEN)
        self._spark_fusion = collections.deque(maxlen=SPARK_LEN)
        self._spark_decay  = collections.deque(maxlen=SPARK_LEN)

        # ── HUD painter ──────────────────────────────────────────────────────
        # Owns the pygame-surface drawing (buttons, panels, inspector). Holds
        # a back-reference to this Renderer; all state still lives here.
        # Instantiated last so every field it reads (sliders, stats, etc.)
        # has been initialized above.
        self.hud = HUDPainter(self)

    # ── Public interface ──────────────────────────────────────────────────────

    def toggle_composite_mode(self):
        """Cycle bonds → merged → none → bonds."""
        cycle = {
            self.MODE_BONDS:  self.MODE_MERGED,
            self.MODE_MERGED: self.MODE_NONE,
            self.MODE_NONE:   self.MODE_BONDS,
        }
        self.composite_mode = cycle.get(self.composite_mode, self.MODE_BONDS)
        self._hud_dirty = True

    def _rebuild_physics_sliders(self, bond_mode: str) -> None:
        """
        (Re)build self._sliders for the active bond_mode. The bond-related
        slots are the only thing that changes:
          edges       → k_bond (harmonic stiffness) + r_rest_scale (length)
          star_spring → spring_k (COM star spring) only
          off         → no bond slots
        All other physics sliders (forces, fusion chemistry, dt, damping)
        are unchanged across modes.
        """
        _phys = self._slider_phys_default
        if bond_mode == "edges":
            bond_slots = [
                ("k_bond",       "bond k",      _phys("k_bond"),       "{:.1f}", None),
                # r_rest_scale = 1.0 is the hash-determined chemistry; the
                # linear range lets the user tighten (<1) or loosen (>1) all
                # bonds uniformly without changing the per-pair structure.
                ("r_rest_scale", "bond length", _phys("r_rest_scale"), "{:.2f}", (0.3, 2.0)),
            ]
        elif bond_mode == "star_spring":
            bond_slots = [
                ("spring_k", "spring k", _phys("spring_k"), "{:.1f}", None),
            ]
        else:  # "off"
            bond_slots = []

        slider_specs = [
            # ── Force kernel shape ────────────────────────────────────────────
            # (field, label, default, fmt, linear_range or None)
            ("repulsion_strength",       "repulsion",   _phys("repulsion_strength"),   "{:.2f}", None),
            ("repulsion_radius",         "repulse r",   _phys("repulsion_radius"),     "{:.2f}", None),
            ("attraction_scale",         "attract",     _phys("attraction_scale"),     "{:.2f}", None),
            ("r_cutoff_scale",           "attract r",   _phys("r_cutoff_scale"),       "{:.2f}", None),
            None,
            # ── Fusion chemistry ──────────────────────────────────────────────
            ("fusion_threshold",         "fuse thresh", _phys("fusion_threshold"),     "{:.3f}", None),
            ("binding_energy_scale",     "bind energy", _phys("binding_energy_scale"), "{:.3f}", None),
            None,
            # ── Particle dynamics ─────────────────────────────────────────────
            ("dt",                       "dt",          _phys("dt"),                   "{:.4f}", (0.001, 0.1)),
            ("damping",                  "damping",     _phys("damping"),              "{:.4f}", (0.0, 1.0)),
        ]
        slider_specs.extend(bond_slots)

        panel_x        = self._slider_panel_x
        slider_track_w = self._slider_track_w
        slider_start_y = self._slider_start_y
        slider_row_h   = self._slider_row_h
        group_gap      = self._slider_group_gap

        self._sliders = []
        row_y = slider_start_y
        for spec in slider_specs:
            if spec is None:
                row_y += group_gap
                continue
            field, label, default, fmt, lin = spec
            track = pygame.Rect(panel_x + 4, row_y + 18, slider_track_w, 8)
            self._sliders.append(
                Slider(label, field, default, track, fmt, linear_range=lin)
            )
            row_y += slider_row_h
        # Total content height (last row's bottom relative to slider_start_y).
        self._slider_content_h = row_y - slider_start_y

    def set_bond_mode(self, new_mode: str, new_config) -> None:
        """
        Reflect a runtime bond_mode change in renderer-owned state:
          - update self.config so the bond-emission branch and HUD badge see it
          - rebuild the physics sliders to expose the right stiffness knob
          - mark the HUD dirty so the badge / sliders repaint
        Called by main.py's M-key handler.
        """
        self.config = new_config
        self._rebuild_physics_sliders(new_mode)
        self._hud_dirty = True

    def toggle_stats(self):
        self._show_stats = not self._show_stats
        # When the panel opens, force the histogram to recompute on the very
        # next update() so the chart shows live data instead of a snapshot
        # from before it was hidden. (The throttle counter would otherwise
        # wait up to HIST_COMPUTE_INTERVAL-1 frames.)
        if self._show_stats:
            self._hist_compute_age = HIST_COMPUTE_INTERVAL
        self._hud_dirty = True

    def toggle_events(self):
        self._show_events = not self._show_events
        self._hud_dirty = True

    def toggle_trails(self):
        self._render_settings['trails_on'] = not self._render_settings['trails_on']
        self._hud_dirty = True

    def set_paused(self, paused: bool):
        """Keep the renderer in sync with the main loop's pause state."""
        if self._paused != paused:
            self._hud_dirty = True
        self._paused = paused

    def handle_click(self, pos) -> str | None:
        """Return action string if a button was clicked, else None."""
        if self._stats_btn_rect.collidepoint(pos):
            return "toggle_stats"
        # Inspector close button — only meaningful when a selection is active,
        # so the (0,0,18,18) default rect is harmless before the first panel
        # draw populates real coordinates.
        if self._selected_idx >= 0 and self._inspector_close_rect.collidepoint(pos):
            self.clear_selection()
            return "clear_selection"
        # Gear nub overlays the right edge of the Trails button. Check it
        # before the buttons list so the gear region wins over the underlying
        # toggle_trails hit.
        if self._trails_gear_rect.collidepoint(pos):
            return "toggle_render_params"
        for _label, rect, action in self._buttons:
            if rect.collidepoint(pos):
                return action
        return None

    def toggle_params(self) -> None:
        # Physics and render panels are mutually exclusive — opening one closes
        # the other so they never overlap at narrow window widths.
        self._show_params = not self._show_params
        if self._show_params:
            self._show_render_params = False
        self._hud_dirty = True

    def toggle_render_params(self) -> None:
        self._show_render_params = not self._show_render_params
        if self._show_render_params:
            self._show_params = False
        self._hud_dirty = True

    def handle_mousedown_slider(self, pos) -> bool:
        """Start dragging a slider if pos hits a handle, or reset if a reset button hit."""
        # Only one of the two panels can be open at a time (mutually exclusive).
        # Iterate whichever panel is active; both panels share the same X column
        # so we never need to consider both lists at once.
        if self._show_params:
            sliders = self._sliders
            reset_rect = self._params_reset_rect
        elif self._show_render_params:
            sliders = self._render_sliders
            reset_rect = self._render_params_reset_rect
        else:
            return False

        if reset_rect.collidepoint(pos):
            for s in sliders:
                s.reset()
                s.commit(self._physics_updates)
            self._hud_dirty = True
            return True
        for slider in sliders:
            if slider.hit_reset(pos):
                slider.reset()
                slider.commit(self._physics_updates)
                self._hud_dirty = True
                return True
        for slider in sliders:
            if slider.hit_handle(pos):
                self._dragging_slider = slider
                slider.handle_drag(pos)
                slider.commit(self._physics_updates)
                self._hud_dirty = True
                return True
        return False

    def handle_mousemotion(self, pos) -> None:
        if self._dragging_slider is not None:
            self._dragging_slider.handle_drag(pos)
            self._dragging_slider.commit(self._physics_updates)
            self._hud_dirty = True

    def handle_mouseup(self) -> None:
        self._dragging_slider = None

    def get_physics_updates(self) -> dict:
        """Return accumulated {field: float} updates and clear the buffer."""
        updates = dict(self._physics_updates)
        self._physics_updates.clear()
        return updates

    # ── Camera (pan + zoom) interface ─────────────────────────────────────────
    # Camera state and its methods live in halflife/render/camera.py and are
    # accessed via self.camera (zoom_at, pan_by, reset, screen_to_world,
    # push_uniforms). Callers go through the Camera directly instead of
    # delegating through Renderer.

    # ── Particle selection (click-to-inspect) ─────────────────────────────────

    def select_at(self, sx: int, sy: int) -> None:
        """Pick the nearest particle within a world-radius of screen (sx, sy).

        Reads the cached positions stashed during the last update() so no
        extra GPU sync is needed. A click further than _select_radius_world
        from every particle clears the selection (out-of-radius dismiss).
        """
        if self._cpu_state is None:
            return
        wx, wy = self.camera.screen_to_world(sx, sy)
        pos = self._cpu_state.positions
        # Periodic min-image so a click near a wrap edge can still pick a
        # particle that visually looks close but lives on the other side.
        delta = pos - np.array([wx, wy], dtype=pos.dtype)
        self._wrap_min_image(delta)
        d2 = (delta * delta).sum(axis=1)
        best = int(np.argmin(d2))
        if d2[best] <= self._select_radius_world ** 2:
            self._selected_idx = best
        else:
            self._selected_idx = -1
        self._hud_dirty = True

    def clear_selection(self) -> None:
        """Drop the current particle selection (used by the × close button)."""
        self._selected_idx = -1
        self._selected_snapshot = None
        self._hud_dirty = True

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, state: WorldState):
        """
        Transfer simulation state to GPU buffers for rendering.
        Called once per frame before render().
        """
        config = self.config
        particles  = state.particles
        composites = state.composites

        # Single batched GPU→CPU transfer — one CUDA sync + one DMA instead of 13
        (pos, vel, species, mass, p_energy, p_age, comp_id,
         comp_members, comp_count, comp_alive, comp_species_hash,
         comp_binding_energy, comp_half_life, comp_age, comp_free_bonds,
         comp_edges, comp_edge_count,
         total_energy, step_count, sim_time) = jax.device_get((
            particles.position, particles.velocity,
            particles.species,
            particles.mass,     particles.energy, particles.age,
            particles.composite_id,
            composites.members, composites.member_count, composites.alive,
            composites.species_hash,
            composites.binding_energy, composites.half_life, composites.age,
            composites.free_bonds,
            composites.edges, composites.edge_count,
            state.total_energy, state.step_count, state.time,
        ))

        # ── Cache for inspector panel / hit-test ──────────────────────────────
        # Stashed for select_at() and HUDPainter.refresh_selected_snapshot(),
        # both of which run after this method returns. All these arrays are
        # already on CPU at this point — no extra transfer cost.
        self._cpu_state = CPUStateSnapshot(
            positions=pos, velocities=vel, species=species,
            mass=mass, energy=p_energy, age=p_age,
            comp_id=comp_id,
            comp_members=comp_members, comp_count=comp_count,
            comp_alive=comp_alive, comp_species_hash=comp_species_hash,
            comp_binding_energy=comp_binding_energy,
            comp_half_life=comp_half_life, comp_age=comp_age,
            comp_free_bonds=comp_free_bonds,
        )

        # ── Particle tracking ────────────────────────────────────────────────
        # While a particle is selected AND the camera is zoomed in, recentre
        # the view on it each frame. Gated on view_scale > 1 because at default
        # zoom (or zoomed out) the whole world is visible and "tracking" would
        # just be a no-op clamp back to the world midpoint.
        if self._selected_idx >= 0 and self.camera.view_scale > 1.0:
            sel_pos = pos[self._selected_idx]
            self.camera.follow(float(sel_pos[0]), float(sel_pos[1]))

        # ── Particle vertex data ──────────────────────────────────────────────
        alive_idx = np.arange(len(pos))
        n_alive   = len(alive_idx)

        if n_alive > 0:
            p_pos    = pos[alive_idx]
            p_sp     = species[alive_idx]
            p_mass   = mass[alive_idx]
            p_comp   = comp_id[alive_idx]

            colors = self.species_colors[p_sp]
            speed  = np.linalg.norm(vel[alive_idx], axis=-1)
            brightness = np.clip(0.5 + speed / config.init_speed, 0.5, 1.5)
            colors = np.clip(colors * brightness[:, None], 0.0, 1.0)

            if self.composite_mode == self.MODE_MERGED:
                alpha = np.where(p_comp >= 0, 0.15, 1.0).astype(np.float32)
            else:
                alpha = np.ones(n_alive, dtype=np.float32)

            size = np.clip(
                config.point_size_min + np.log1p(p_mass) * 3.0,
                config.point_size_min, config.point_size_max
            ).astype(np.float32)

            # In-place write to pre-allocated buffer — no heap allocations
            buf = self._part_vbuf[:n_alive]
            buf[:, 0:2] = p_pos
            buf[:, 2:5] = colors
            buf[:, 5]   = alpha
            buf[:, 6]   = size
            self.particle_vbo.write(buf.tobytes())
            self._n_particles_to_draw = n_alive
        else:
            self._n_particles_to_draw = 0

        # ── Bond / merged vertex data ─────────────────────────────────────────
        self._n_bond_vertices = 0
        alive_comp_idx = np.where(comp_alive)[0]

        if self.composite_mode == self.MODE_BONDS and len(alive_comp_idx) > 0:
            # Strictly bond_mode-driven so toggling 'off' actually clears the
            # mesh. Previously this fell back to "draw whatever edges happen
            # to be populated" which painted stale edges across the screen
            # when the user toggled away from edges mode.
            #   "edges"       — draw the real edges array
            #   "star_spring" — draw the forward-slot heuristic (composites
            #                   exist but have no edge structure in this mode)
            #   "off"         — draw no bonds at all
            mode = self.config.bond_mode

            if mode == "edges":
                # Collect all valid edges across alive composites.
                edge_pairs = []
                for c in alive_comp_idx:
                    ec = int(comp_edge_count[c])
                    if ec == 0:
                        continue
                    edge_pairs.append(comp_edges[c, :ec])
                if edge_pairs:
                    all_edges = np.concatenate(edge_pairs, axis=0)  # (E_total, 2)
                    mem_a = all_edges[:, 0]
                    mem_b = all_edges[:, 1]
                    pos_a = pos[mem_a]
                    dx    = pos[mem_b] - pos_a
                    self._wrap_min_image(dx)
                    pos_b = pos_a + dx

                    # Constant alpha for now — edges are by construction sparse.
                    alpha = 0.7

                    n_pairs = len(all_edges)
                    bond_verts = np.empty((n_pairs * 2, 6), dtype=np.float32)
                    bond_verts[0::2, :2] = pos_a
                    bond_verts[1::2, :2] = pos_b
                    bond_verts[0::2, 2:5] = self.species_colors[species[mem_a]]
                    bond_verts[1::2, 2:5] = self.species_colors[species[mem_b]]
                    bond_verts[0::2, 5] = alpha
                    bond_verts[1::2, 5] = alpha
                    n_bytes = min(bond_verts.nbytes, self._bond_buf_size)
                    self.bond_vbo.write(bond_verts.tobytes()[:n_bytes])
                    self._n_bond_vertices = n_bytes // (self._bond_vertex_size * 4)
                # else: zero edges → _n_bond_vertices stays 0

            elif mode == "star_spring":
                # ── Legacy forward-slot heuristic ────────────────────────────
                # Composites exist (held by COM spring) but carry no edge
                # structure in this mode. Draw bonds to the next K members in
                # slot order as a visualization approximation.
                C_idx = alive_comp_idx
                max_n = int(comp_count[C_idx].max())
                if max_n >= 2:
                    mb  = comp_members[C_idx, :max_n]   # (n_comps, max_n)
                    cnt = comp_count[C_idx]              # (n_comps,)
                    # Deterministic per-particle bond cap: for each member at slot
                    # i, emit bonds to slots (i+1, i+2, ..., i+K). Each particle
                    # ends up touched by at most ~2K bonds. Cost is O(K·N) per
                    # composite vs the old O(N²) — large composites no longer
                    # explode the bond count. Slot order is stable across frames
                    # (fusion preserves it), so bonds don't flicker.
                    K = max(1, int(MAX_BONDS_PER_PARTICLE))
                    ii_grid = np.arange(max_n)
                    offsets = np.arange(1, K + 1)
                    ii_full = np.broadcast_to(ii_grid[:, None], (max_n, K)).ravel()
                    jj_full = (ii_grid[:, None] + offsets[None, :]).ravel()
                    # Drop pairs where the j-slot is past the widest composite —
                    # those can't be valid for any composite this frame.
                    in_range = jj_full < max_n
                    ii = ii_full[in_range]
                    jj = jj_full[in_range]
                    # valid: both pair indices in-range and member slots non-negative
                    valid = (ii[None, :] < cnt[:, None]) & (jj[None, :] < cnt[:, None])
                    mem_a = mb[:, ii].ravel()
                    mem_b = mb[:, jj].ravel()
                    # per-pair member-count of owning composite (broadcast cnt across pairs)
                    cnt_per_pair = np.broadcast_to(cnt[:, None], (len(C_idx), len(ii))).ravel()
                    valid = valid.ravel() & (mem_a >= 0) & (mem_b >= 0)
                    mem_a = mem_a[valid]
                    mem_b = mem_b[valid]
                    cnt_per_pair = cnt_per_pair[valid]

                    if len(mem_a) > 0:
                        pos_a = pos[mem_a]
                        dx    = pos[mem_b] - pos_a
                        self._wrap_min_image(dx)
                        pos_b = pos_a + dx

                        # Alpha falls off with composite size (sqrt) so cliquey large
                        # composites — n*(n-1)/2 bonds — don't dominate the view.
                        # Dimers (n=2) keep the historic 0.5; n=16 ≈ 0.18.
                        n_eff = np.maximum(cnt_per_pair.astype(np.float32), 2.0)
                        alpha = 0.5 * np.sqrt(2.0 / n_eff)

                        n_pairs = len(mem_a)
                        bond_verts = np.empty((n_pairs * 2, 6), dtype=np.float32)
                        bond_verts[0::2, :2] = pos_a
                        bond_verts[1::2, :2] = pos_b
                        bond_verts[0::2, 2:5] = self.species_colors[species[mem_a]]
                        bond_verts[1::2, 2:5] = self.species_colors[species[mem_b]]
                        bond_verts[0::2, 5] = alpha
                        bond_verts[1::2, 5] = alpha
                        n_bytes = min(bond_verts.nbytes, self._bond_buf_size)
                        self.bond_vbo.write(bond_verts.tobytes()[:n_bytes])
                        self._n_bond_vertices = n_bytes // (self._bond_vertex_size * 4)

        elif self.composite_mode == self.MODE_MERGED and len(alive_comp_idx) > 0:
            C_idx = alive_comp_idx
            max_n = int(comp_count[C_idx].max())
            mb    = comp_members[C_idx, :max_n]     # (n_comps, max_n)
            cnt   = comp_count[C_idx]               # (n_comps,)
            # valid mask: slot index in-range and member index non-negative
            vm    = (mb >= 0) & (np.arange(max_n)[None, :] < cnt[:, None])
            safe_m = np.where(vm, mb, 0)             # safe gather indices (no -1)
            vm_f  = vm.astype(np.float32)

            g_pos  = pos[safe_m]                                     # (n_comps, max_n, 2)
            g_mass = mass[safe_m] * vm_f                             # (n_comps, max_n)
            g_col  = self.species_colors[species[safe_m]]            # (n_comps, max_n, 3)
            sum_vm = vm_f.sum(1, keepdims=True) + 1e-8
            # Periodic-aware center of mass. A naive average breaks when a
            # composite straddles a wrap boundary (mean of x=0.1 and x=199.9
            # lands at 100 — middle of the world). Fix: take member slot 0 as
            # an anchor, express every other member position as its
            # minimum-image displacement from the anchor, average, then wrap
            # back into the world. Slot 0 is always populated for alive
            # composites (fusion packs members densely from index 0).
            ref_pos = g_pos[:, 0, :]                                 # (n_comps, 2)
            rel     = g_pos - ref_pos[:, None, :]                    # (n_comps, max_n, 2)
            self._wrap_min_image(rel)
            com_rel = (rel * vm_f[:, :, None]).sum(1) / sum_vm
            com     = ref_pos + com_rel                              # (n_comps, 2)
            if config.boundary_mode == "periodic":
                com[:, 0] = np.mod(com[:, 0], config.world_width)
                com[:, 1] = np.mod(com[:, 1], config.world_height)
            tmass  = g_mass.sum(1)
            acolor = (g_col * vm_f[:, :, None]).sum(1) / sum_vm      # (n_comps, 3)
            sz = np.clip(
                config.point_size_min + np.log1p(tmass) * 4.0,
                config.point_size_min, config.point_size_max * 1.5
            )
            n_m  = len(C_idx)
            slot = self._part_vbuf[n_alive : n_alive + n_m]
            slot[:, 0:2] = com
            slot[:, 2:5] = acolor
            slot[:, 5]   = 1.0
            slot[:, 6]   = sz
            self.particle_vbo.write(self._part_vbuf[:n_alive + n_m].tobytes())
            self._n_particles_to_draw = n_alive + n_m

        # ── Stats ─────────────────────────────────────────────────────────────
        self._stats_alive    = len(pos)
        self._stats_free     = int(np.sum(comp_id < 0))
        self._stats_n_comp   = int(np.sum(comp_alive))
        self._stats_energy   = float(total_energy)
        self._stats_step     = int(step_count)
        self._stats_sim_time = float(sim_time)

        # Unique-type count is cheap (np.unique on ~3k uint32s is microseconds)
        # and feeds the per-frame "Unique:" sparkline, so it runs every frame.
        if self._stats_n_comp > 0:
            self._stats_n_unique = int(np.unique(comp_species_hash[comp_alive]).size)
        else:
            self._stats_n_unique = 0

        # Composite-size histogram is heavier and only feeds the bar chart.
        # Skip entirely when the stats panel is hidden — no one's looking at
        # it — and otherwise recompute every HIST_COMPUTE_INTERVAL frames so
        # the chart still feels live without paying for a refresh the eye
        # can't track at 60-120 Hz.
        if self._show_stats:
            self._hist_compute_age += 1
            if self._hist_compute_age >= HIST_COMPUTE_INTERVAL:
                self._hist_compute_age = 0
                if self._stats_n_comp > 0:
                    alive_counts = comp_count[comp_alive]
                    self._stats_hist, _ = np.histogram(
                        alive_counts,
                        bins=np.arange(1, config.max_composite_size + 2)
                    )
                else:
                    self._stats_hist = np.zeros(config.max_composite_size, dtype=np.int32)

        # ── Event detection ───────────────────────────────────────────────────
        current_sim_time  = self._stats_sim_time
        comp_lifetime_secs = 50.0 * config.dt
        part_lifetime_secs = 20.0 * config.dt

        # Expire old events
        self._events = [
            ev for ev in self._events
            if current_sim_time - ev[5] < ev[6]
        ]

        if self._show_events and self._prev_comp_alive is not None:
            new_comps  = ~self._prev_comp_alive & comp_alive
            dead_comps = self._prev_comp_alive  & ~comp_alive

            budget = max(0, self._event_max - len(self._events))

            # Fusion events (gold)
            for c in np.where(new_comps)[0][:min(40, budget // 3)]:
                n       = comp_count[c]
                mems    = comp_members[c, :n]
                valid_m = mems[mems >= 0]
                if len(valid_m) > 0:
                    com = self._periodic_com(pos[valid_m])
                    self._events.append(
                        (float(com[0]), float(com[1]),
                         1.0, 0.85, 0.0, current_sim_time, comp_lifetime_secs)
                    )

            # Fission events (cyan)
            for c in np.where(dead_comps)[0][:min(40, budget // 3)]:
                n       = comp_count[c]
                mems    = comp_members[c, :n]
                valid_m = mems[mems >= 0]
                if len(valid_m) > 0:
                    com = self._periodic_com(pos[valid_m])
                    self._events.append(
                        (float(com[0]), float(com[1]),
                         0.0, 1.0, 1.0, current_sim_time, comp_lifetime_secs)
                    )

            # Hard cap
            self._events = self._events[-self._event_max:]

        # Accumulate event counts (composite-level only; free particles don't decay)
        if self._prev_comp_alive is not None:
            n_fusion = int(np.sum(~self._prev_comp_alive & comp_alive))
            n_decay  = int(np.sum(self._prev_comp_alive  & ~comp_alive))
            self._fusion_total += n_fusion
            self._decay_total  += n_decay
            self._event_history.append(
                (self._stats_sim_time, n_fusion, n_decay)
            )
            recent = [e for e in self._event_history
                      if e[0] >= self._stats_sim_time - 0.5]
            if len(recent) >= 2:
                dt = max(0.01, recent[-1][0] - recent[0][0])
                self._fusion_rate = sum(e[1] for e in recent) / dt
                self._decay_rate  = sum(e[2] for e in recent) / dt

        # Sparklines
        self._spark_free.append(self._stats_free)
        self._spark_comp.append(self._stats_n_comp)
        self._spark_unique.append(self._stats_n_unique)
        self._spark_energy.append(self._stats_energy)
        self._spark_fusion.append(self._fusion_rate)
        self._spark_decay.append(self._decay_rate)

        self._prev_comp_alive = comp_alive.copy()

        # ── Build event VBO ───────────────────────────────────────────────────
        self._n_event_vertices = 0
        if self._show_events and self._events:
            ev_data = []
            for (x, y, r, g, b, bt, lt) in self._events:
                age = float(np.clip((current_sim_time - bt) / max(lt, 1e-8), 0.0, 1.0))
                ev_data.append([x, y, r, g, b, age])
            ev_arr = np.array(ev_data, dtype=np.float32)
            n_bytes = min(ev_arr.nbytes, self._event_max * 6 * 4)
            self._event_vbo.write(ev_arr.flatten().tobytes()[:n_bytes])
            self._n_event_vertices = len(ev_data)

        # ── Inspector snapshot ────────────────────────────────────────────────
        # Reads from the cached arrays stashed at the top of this method.
        self.hud.refresh_selected_snapshot()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, fps: float, step_count: int, n_alive: int):
        """Clear screen, draw scene, tonemap, draw HUD, flip buffers."""
        bg     = self.config.background_color
        rs     = self._render_settings
        on     = bool(rs['trails_on'])
        decay  = float(rs['trail_decay']) if on else 0.0
        # Particle size/alpha multipliers always apply, regardless of trail
        # state — they modulate how the live particles render, not the trail
        # itself. Trail decay is the only trails-gated setting.
        sz_m   = float(rs['particle_size_mult'])
        al_m   = float(rs['particle_alpha_mult'])

        # Push camera uniforms to every world-space program (particle, bond,
        # event, highlight). Adding a new world-space program means just
        # appending to self._world_space_progs in __init__.
        self.camera.push_uniforms(self._world_space_progs)

        curr_fbo = self._trail_fbos[self._trail_idx]
        prev_tex = self._trail_texs[1 - self._trail_idx]

        # ── Particle trail pass (into the current trail FBO) ─────────────────
        # Trails apply ONLY to particles — bonds connect *current* composite
        # positions and event rings have their own age-based fade, so they're
        # drawn fresh into a separate FBO below. This keeps bond lines from
        # smearing into colored scribbles and stops event rings from leaving
        # ghost halos in the trail buffer.
        curr_fbo.use()
        # Decay/clear writes are full overwrites — blending MUST be off here.
        # If left on, the destination keeps stale data from two frames ago
        # (we ping-pong, so curr_fbo currently holds frame N-2's content) and
        # the trail buffer never fully fades, slowly filling with residue.
        self.ctx.disable(moderngl.BLEND)
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

        # Re-enable alpha blending for the particle pass (particles have a
        # smoothstep alpha falloff at the sprite edge).
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.particle_prog['u_size_mult'].value  = sz_m
        self.particle_prog['u_alpha_mult'].value = al_m

        if self._n_particles_to_draw > 0:
            self.particle_vao.render(moderngl.POINTS, vertices=self._n_particles_to_draw)

        # ── Fresh overlay pass (bonds + events, no trail) ────────────────────
        self._fresh_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)   # fully transparent

        # Bonds
        if self.composite_mode == self.MODE_BONDS and self._n_bond_vertices > 0:
            self.ctx.line_width = 1.0
            self.bond_vao.render(moderngl.LINES, vertices=self._n_bond_vertices)

        # Event sprites. u_world_size was already set in __init__ and never
        # changes; camera uniforms are pushed at the top of render() via
        # self.camera.push_uniforms(), so this branch just issues the draw.
        if self._show_events and self._n_event_vertices > 0:
            self._event_vao.render(moderngl.POINTS, vertices=self._n_event_vertices)

        # Selection highlight ring. Camera uniforms are already on the
        # highlight program (pushed via self.camera at the top of render()).
        if self._selected_idx >= 0 and self._cpu_state is not None:
            sel_pos = self._cpu_state.positions[self._selected_idx].astype(np.float32)
            self._highlight_vbo.write(sel_pos.tobytes())
            # Constant screen-pixel diameter so the ring stays visible even
            # on tiny zoomed-out particles. ~26 px works at default windowing.
            self.highlight_prog['u_size_px'].value = 26.0
            self._highlight_vao.render(moderngl.POINTS, vertices=1)

        # ── Tonemap composite to default framebuffer ─────────────────────────
        # Two passes through the same tonemap shader:
        #   1) Trail layer — written opaquely (blending disabled). Trail-FBO
        #      alpha may have drifted toward 0 under repeated decay, so we
        #      can't trust it for compositing; instead we just write fully.
        #   2) Fresh layer — alpha-blended on top so transparent areas keep
        #      the underlying trail visible and bond/event coverage maps to
        #      source.alpha emitted by the tonemap shader.
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.ctx.disable(moderngl.BLEND)
        self._trail_texs[self._trail_idx].use(location=0)
        self.tonemap_prog['scene_tex'].value = 0
        self._tonemap_vao.render(moderngl.TRIANGLES, vertices=6)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._fresh_tex.use(location=0)
        self.tonemap_prog['scene_tex'].value = 0
        self._tonemap_vao.render(moderngl.TRIANGLES, vertices=6)

        # ── HUD overlay (LDR, on top of tonemapped scene) ────────────────────
        # HUD overlay — only re-render the pygame surface and re-upload the
        # texture when the HUD has actually changed. The fullscreen-quad GL
        # blit always runs (it's cheap; reads the cached texture).
        # Stats panel updates every frame (sparklines, FPS), so force dirty
        # while it's shown. Inspector panel likewise — position/velocity move
        # every frame while a particle is selected.
        if self._show_stats or self._selected_idx >= 0:
            self._hud_dirty = True

        if self._hud_dirty:
            self.hud.paint(fps)
            surf_data = pygame.image.tostring(self._hud_surface, 'RGBA', True)
            self._hud_texture.write(surf_data)
            self._hud_dirty = False

        self._hud_texture.use(location=0)
        self.hud_prog['hud_tex'].value = 0
        self._hud_quad_vao.render(moderngl.TRIANGLES, vertices=6)

        pygame.display.flip()

        # Swap ping-pong index for next frame
        self._trail_idx = 1 - self._trail_idx

    # ── Private helpers ───────────────────────────────────────────────────────

    def _wrap_min_image(self, delta: np.ndarray) -> None:
        """In-place periodic min-image: subtract integer multiples of world
        size so each component of `delta[..., 0:2]` falls in [-W/2, W/2].

        No-op when boundary_mode != "periodic". Accepts any array shape where
        the last axis is the spatial (x, y) pair, so the same helper handles
        1D-per-component selects, (N, 2) bond deltas, and (n_comps, max_n, 2)
        member-relative arrays.
        """
        if self.config.boundary_mode != "periodic":
            return
        W = self.config.world_width
        H = self.config.world_height
        delta[..., 0] -= W * np.round(delta[..., 0] / W)
        delta[..., 1] -= H * np.round(delta[..., 1] / H)

    def _periodic_com(self, positions_2d: np.ndarray) -> np.ndarray:
        """Periodic-aware center of mass for an (n, 2) array of positions.

        Anchor on positions_2d[0]; compute each member's min-image displacement
        relative to that anchor; average the displacements; add back to the
        anchor; wrap into the world. Naive np.mean breaks when the composite
        straddles a wrap boundary (members at x=0.1 and x=199.9 average to
        x=100, planting the marker in the middle of an empty region).
        """
        config = self.config
        ref = positions_2d[0]
        rel = positions_2d - ref
        self._wrap_min_image(rel)
        com = ref + rel.mean(axis=0)
        if config.boundary_mode == "periodic":
            com = np.array([
                com[0] % config.world_width,
                com[1] % config.world_height,
            ])
        return com


    # ── Legacy HUD (title bar only) ───────────────────────────────────────────

    def _update_title(self, fps: float, step_count: int, n_alive: int):
        pygame.display.set_caption(
            f"Half-Life | FPS:{fps:.0f} | Step:{step_count:,} | Alive:{n_alive:,}"
        )

    def close(self):
        """Release all GPU resources."""
        self.particle_vbo.release()
        self.particle_vao.release()
        self.bond_vbo.release()
        self.bond_vao.release()
        self.particle_prog.release()
        self.bond_prog.release()
        self._hud_quad_vbo.release()
        self._hud_quad_vao.release()
        self._hud_texture.release()
        self.hud_prog.release()
        self._event_vbo.release()
        self._event_vao.release()
        self.event_prog.release()
        for fbo in self._trail_fbos:
            fbo.release()
        for tex in self._trail_texs:
            tex.release()
        self._trail_decay_vao.release()
        self.trail_decay_prog.release()
        self._fresh_fbo.release()
        self._fresh_tex.release()
        self._highlight_vbo.release()
        self._highlight_vao.release()
        self.highlight_prog.release()
        self._tonemap_vao.release()
        self.tonemap_prog.release()
        self.ctx.release()
        pygame.quit()
