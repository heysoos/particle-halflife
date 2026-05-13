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
import numpy as np
import pygame
import moderngl
import jax

from halflife.config import SimConfig
from halflife.state import WorldState, get_species_colors, initialize_physics_params
from halflife.profiler import ProfileMetrics


# ── Bond rendering cap ────────────────────────────────────────────────────────
# Maximum number of forward-slot bonds emitted per composite member in
# "bonds" view. For each member at slot i, bonds are emitted to slots
# (i+1, i+2, …, i+MAX_BONDS_PER_PARTICLE). Each particle ends up touched by
# at most ~2× this value's worth of bonds (forward + incoming). Pure visual
# cap — does not affect physics. Bumping this above ~5 starts to undo the
# performance win on big composites.
MAX_BONDS_PER_PARTICLE = 3


# ── GLSL Shaders ──────────────────────────────────────────────────────────────

PARTICLE_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec4  in_color;
in float in_size;

out vec4 v_color;

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = in_size;
    v_color = in_color;
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

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
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

# HUD: fullscreen quad with pygame surface as texture
HUD_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
    // No Y-flip needed: pygame surface is pre-flipped on upload
}
"""

HUD_FRAGMENT_SHADER = """
#version 330

in  vec2 v_uv;
uniform sampler2D hud_tex;
out vec4 fragColor;

void main() {
    fragColor = texture(hud_tex, v_uv);
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

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = mix(60.0, 20.0, in_age);
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


# ── Slider ────────────────────────────────────────────────────────────────────

class Slider:
    """Horizontal slider widget — log-scale multiplier (0.1×–10×) or optional linear range."""

    EXPO_MIN, EXPO_MAX = -1.0, 1.0   # 0.1× to 10× the default value

    def __init__(self, label: str, field: str, default_value: float,
                 track_rect: pygame.Rect, fmt: str = "{:.3f}",
                 linear_range=None):
        self._label = label
        self._field = field
        self._default = float(default_value)
        self._track_rect = track_rect
        self._fmt = fmt
        self._linear_range = linear_range
        self._reset_rect = pygame.Rect(track_rect.right + 4, track_rect.centery - 7, 14, 14)
        # Initialize exponent so that value == default
        if linear_range is not None:
            lo, hi = linear_range
            t = (self._default - lo) / max(hi - lo, 1e-8)
            self._exponent = self.EXPO_MIN + float(np.clip(t, 0.0, 1.0)) * (self.EXPO_MAX - self.EXPO_MIN)
        else:
            self._exponent = 0.0  # 1× = default

    @property
    def field(self) -> str:
        return self._field

    @property
    def value(self) -> float:
        if self._linear_range is not None:
            lo, hi = self._linear_range
            t = (self._exponent - self.EXPO_MIN) / (self.EXPO_MAX - self.EXPO_MIN)
            return lo + float(np.clip(t, 0.0, 1.0)) * (hi - lo)
        return self._default * (10.0 ** self._exponent)

    def reset(self) -> None:
        if self._linear_range is not None:
            lo, hi = self._linear_range
            t = (self._default - lo) / max(hi - lo, 1e-8)
            self._exponent = self.EXPO_MIN + float(np.clip(t, 0.0, 1.0)) * (self.EXPO_MAX - self.EXPO_MIN)
        else:
            self._exponent = 0.0

    def hit_reset(self, pos) -> bool:
        return self._reset_rect.collidepoint(pos)

    def _handle_x(self) -> int:
        t = (self._exponent - self.EXPO_MIN) / (self.EXPO_MAX - self.EXPO_MIN)
        return int(self._track_rect.left + t * self._track_rect.width)

    def draw(self, surface: pygame.Surface, font) -> None:
        r = self._track_rect
        if self._linear_range is not None:
            label_str = f"{self._label}: {self._fmt.format(self.value)}"
        else:
            mult = 10.0 ** self._exponent
            label_str = f"{self._label}: {self._fmt.format(self.value)} ({mult:.2f}\u00d7)"
        txt = font.render(label_str, True, (190, 215, 255))
        surface.blit(txt, (r.left, r.top - 14))
        # Track
        pygame.draw.rect(surface, (50, 60, 80, 220), r, border_radius=3)
        # Fill
        hx = self._handle_x()
        fill_r = pygame.Rect(r.left, r.top, hx - r.left, r.height)
        if fill_r.width > 0:
            pygame.draw.rect(surface, (60, 130, 200, 220), fill_r, border_radius=3)
        # Handle
        pygame.draw.circle(surface, (180, 210, 255), (hx, r.centery), 6)
        # Per-slider reset button (↺)
        is_default = abs(self._exponent - (self.EXPO_MIN + (
            (self._default - self._linear_range[0]) / max(self._linear_range[1] - self._linear_range[0], 1e-8)
            if self._linear_range else 0.5
        ) * (self.EXPO_MAX - self.EXPO_MIN))) < 0.01 if self._linear_range else abs(self._exponent) < 0.01
        bg_col = (40, 25, 25, 200) if not is_default else (25, 30, 40, 200)
        pygame.draw.rect(surface, bg_col, self._reset_rect, border_radius=3)
        pygame.draw.rect(surface, (120, 80, 80, 180), self._reset_rect, 1, border_radius=3)
        lbl = font.render("\u21ba", True, (200, 140, 140) if not is_default else (80, 90, 110))
        surface.blit(lbl, (self._reset_rect.centerx - lbl.get_width() // 2,
                            self._reset_rect.centery - lbl.get_height() // 2))

    def hit_handle(self, pos) -> bool:
        # Whole track is draggable, not just the knob — clicking anywhere on
        # the bar grabs the slider and snaps the knob to that x.
        r = self._track_rect
        return (r.left <= pos[0] <= r.right and
                abs(pos[1] - r.centery) <= 10)

    def handle_drag(self, pos) -> float:
        r = self._track_rect
        t = (pos[0] - r.left) / max(r.width, 1)
        self._exponent = self.EXPO_MIN + float(max(0.0, min(1.0, t))) * (self.EXPO_MAX - self.EXPO_MIN)
        return self.value


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    ModernGL + pygame renderer. One instance per simulation run.

    After initialization, call update(state) + render(fps, step_count, n_alive)
    each frame.
    """

    MODE_BONDS  = "bonds"
    MODE_MERGED = "merged"

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
            vertex_shader=HUD_VERTEX_SHADER,
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

        # ── Event sprite shader ──────────────────────────────────────────────
        self.event_prog = self.ctx.program(
            vertex_shader=EVENT_VERTEX_SHADER,
            fragment_shader=EVENT_FRAGMENT_SHADER,
        )
        self.event_prog['u_world_size'].value = (config.world_width, config.world_height)

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
        slider_specs = [
            # \u2500\u2500 Force kernel shape \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            # (field, label, default, fmt, linear_range or None)
            ("repulsion_strength",       "repulsion",   _phys("repulsion_strength"),   "{:.2f}", None),
            ("repulsion_radius",         "repulse r",   _phys("repulsion_radius"),     "{:.2f}", None),
            ("attraction_scale",         "attract",     _phys("attraction_scale"),     "{:.2f}", None),
            ("r_cutoff_scale",           "attract r",   _phys("r_cutoff_scale"),       "{:.2f}", None),
            None,
            # \u2500\u2500 Fusion chemistry \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            ("fusion_threshold",         "fuse thresh", _phys("fusion_threshold"),     "{:.3f}", None),
            ("binding_energy_scale",     "bind energy", _phys("binding_energy_scale"), "{:.3f}", None),
            None,
            # \u2500\u2500 Particle dynamics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            ("dt",                       "dt",          _phys("dt"),                   "{:.4f}", (0.001, 0.1)),
            ("damping",                  "damping",     _phys("damping"),              "{:.4f}", (0.0, 1.0)),
            ("spring_k",                 "spring k",    _phys("spring_k"),             "{:.1f}", None),
        ]
        group_gap = 14  # extra pixels inserted at each None sentinel
        self._sliders = []
        row_y = slider_start_y
        for spec in slider_specs:
            if spec is None:
                row_y += group_gap
                continue
            field, label, default, fmt, lin = spec
            track = pygame.Rect(panel_x + 4, row_y + 18, slider_track_w, 8)
            self._sliders.append(Slider(label, field, default, track, fmt, linear_range=lin))
            row_y += slider_row_h
        # Total content height (last row's bottom relative to slider_start_y).
        self._slider_content_h = row_y - slider_start_y
        self._params_reset_rect = pygame.Rect(panel_x + 4, slider_start_y - 26, 100, 20)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._n_particles_to_draw = 0
        self._n_bond_vertices = 0
        self._font = pygame.font.SysFont('monospace', 13)
        self._clock = pygame.time.Clock()

        self._show_stats  = False
        self._show_events = True
        self._paused      = False   # mirror of main loop paused state

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

    # ── Public interface ──────────────────────────────────────────────────────

    def toggle_composite_mode(self):
        """Cycle between bonds and merged visualization modes."""
        if self.composite_mode == self.MODE_BONDS:
            self.composite_mode = self.MODE_MERGED
        else:
            self.composite_mode = self.MODE_BONDS
        self._hud_dirty = True

    def toggle_stats(self):
        self._show_stats = not self._show_stats
        self._hud_dirty = True

    def toggle_events(self):
        self._show_events = not self._show_events
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
        for _label, rect, action in self._buttons:
            if rect.collidepoint(pos):
                return action
        return None

    def toggle_params(self) -> None:
        self._show_params = not self._show_params
        self._hud_dirty = True

    def handle_mousedown_slider(self, pos) -> bool:
        """Start dragging a slider if pos hits a handle, or reset if a reset button hit."""
        if not self._show_params:
            return False
        if self._params_reset_rect.collidepoint(pos):
            for s in self._sliders:
                s.reset()
            self._physics_updates.update({s.field: s.value for s in self._sliders})
            self._hud_dirty = True
            return True
        for slider in self._sliders:
            if slider.hit_reset(pos):
                slider.reset()
                self._physics_updates[slider.field] = slider.value
                self._hud_dirty = True
                return True
        for slider in self._sliders:
            if slider.hit_handle(pos):
                self._dragging_slider = slider
                slider.handle_drag(pos)
                self._physics_updates[slider.field] = slider.value
                self._hud_dirty = True
                return True
        return False

    def handle_mousemotion(self, pos) -> None:
        if self._dragging_slider is not None:
            self._dragging_slider.handle_drag(pos)
            self._physics_updates[self._dragging_slider.field] = self._dragging_slider.value
            self._hud_dirty = True

    def handle_mouseup(self) -> None:
        self._dragging_slider = None

    def get_physics_updates(self) -> dict:
        """Return accumulated {field: float} updates and clear the buffer."""
        updates = dict(self._physics_updates)
        self._physics_updates.clear()
        return updates

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
        (pos, vel, species, mass, _energy, comp_id,
         comp_members, comp_count, comp_alive, comp_species_hash,
         total_energy, step_count, sim_time) = jax.device_get((
            particles.position, particles.velocity,
            particles.species,
            particles.mass,     particles.energy,
            particles.composite_id,
            composites.members, composites.member_count, composites.alive,
            composites.species_hash,
            state.total_energy, state.step_count, state.time,
        ))

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
                    if config.boundary_mode == "periodic":
                        dx[:, 0] -= config.world_width  * np.round(dx[:, 0] / config.world_width)
                        dx[:, 1] -= config.world_height * np.round(dx[:, 1] / config.world_height)
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
            if config.boundary_mode == "periodic":
                rel[..., 0] -= config.world_width  * np.round(rel[..., 0] / config.world_width)
                rel[..., 1] -= config.world_height * np.round(rel[..., 1] / config.world_height)
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

        if self._stats_n_comp > 0:
            alive_counts = comp_count[comp_alive]
            self._stats_hist, _ = np.histogram(
                alive_counts,
                bins=np.arange(1, config.max_composite_size + 2)
            )
            # Distinct composite "types": each species_hash identifies one
            # multiset of member species, so the unique count is the size of
            # the chemical zoo currently alive. np.unique on ~3k uint32s is
            # microseconds — safe to call every frame.
            self._stats_n_unique = int(np.unique(comp_species_hash[comp_alive]).size)
        else:
            self._stats_hist = np.zeros(config.max_composite_size, dtype=np.int32)
            self._stats_n_unique = 0

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
                    ex = float(np.mean(pos[valid_m, 0]))
                    ey = float(np.mean(pos[valid_m, 1]))
                    self._events.append(
                        (ex, ey, 1.0, 0.85, 0.0, current_sim_time, comp_lifetime_secs)
                    )

            # Fission events (cyan)
            for c in np.where(dead_comps)[0][:min(40, budget // 3)]:
                n       = comp_count[c]
                mems    = comp_members[c, :n]
                valid_m = mems[mems >= 0]
                if len(valid_m) > 0:
                    ex = float(np.mean(pos[valid_m, 0]))
                    ey = float(np.mean(pos[valid_m, 1]))
                    self._events.append(
                        (ex, ey, 0.0, 1.0, 1.0, current_sim_time, comp_lifetime_secs)
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

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, fps: float, step_count: int, n_alive: int):
        """Clear screen, draw scene, draw HUD, flip buffers."""
        bg = self.config.background_color
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

    # ── HUD surface drawing ───────────────────────────────────────────────────

    def _draw_sparkline(self, surface, data, x, y, w, h, color):
        if len(data) < 2:
            return
        arr = np.array(data, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            hi = lo + 1
        pts = [
            (x + int(k * w / (len(arr) - 1)),
             y + h - int((v - lo) / (hi - lo) * h))
            for k, v in enumerate(arr)
        ]
        pygame.draw.lines(surface, color, False, pts, 1)

    def _render_hud_surface(self, fps: float):
        """Draw buttons and optional stats panel onto the transparent HUD surface."""
        surface = self._hud_surface
        surface.fill((0, 0, 0, 0))  # clear to transparent
        font    = self._font

        # ── Buttons ──────────────────────────────────────────────────────────
        for label, rect, action in self._buttons:
            # Dynamic label
            display_label = label
            if action == 'pause':
                display_label = "Resume" if self._paused else "Pause"
            elif action == 'toggle_bonds':
                display_label = "Merged" if self.composite_mode == self.MODE_BONDS else "Bonds"
            elif action == 'toggle_events':
                display_label = "Events ON" if self._show_events else "Events"
            elif action == 'toggle_params':
                display_label = "Params ON" if self._show_params else "Params"

            # Background
            if action in ('reset',):
                bg_col = (90, 40, 40, 200)
            elif action == 'pause':
                bg_col = (40, 80, 50, 200) if not self._paused else (80, 60, 30, 200)
            elif action.startswith('reroll_'):
                bg_col = (60, 40, 80, 200)
            else:
                bg_col = (40, 55, 80, 200)
            pygame.draw.rect(surface, bg_col, rect, border_radius=4)
            pygame.draw.rect(surface, (100, 140, 200, 180), rect, 1, border_radius=4)

            # Text
            txt = font.render(display_label, True, (210, 230, 255))
            surface.blit(txt, (rect.centerx - txt.get_width() // 2,
                                rect.centery - txt.get_height() // 2))

        # Stats button (top-right corner)
        stats_rect = self._stats_btn_rect
        stats_bg = (40, 55, 80, 200)
        pygame.draw.rect(surface, stats_bg, stats_rect, border_radius=4)
        pygame.draw.rect(surface, (100, 140, 200, 180), stats_rect, 1, border_radius=4)
        stats_lbl = "Stats ON" if self._show_stats else "Stats"
        txt = font.render(stats_lbl, True, (210, 230, 255))
        surface.blit(txt, (stats_rect.centerx - txt.get_width() // 2,
                            stats_rect.centery - txt.get_height() // 2))

        # ── Stats panel ───────────────────────────────────────────────────────
        if self._show_stats:
            config = self.config
            # panel_h: 4 static rows (16px) + 6 spark-stat rows (15+18px) +
            #          gap + header (18px) + chart (64px) + ticks (20px) + bottom (10px)
            panel_h = (4 * 16 + 6 * 33 + 4 + 18 + 64 + 20 + 10)
            panel_w = 215
            panel_x = config.window_width - panel_w - 8
            panel_y = self._stats_btn_rect.bottom + 4

            panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
            pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)

            y_off = panel_y + 6

            # Static lines (no sparkline)
            for line in [
                f"FPS:        {fps:.1f}",
                f"Step:       {self._stats_step:,}",
                f"Sim time:   {self._stats_sim_time:.1f}",
                f"Particles:  {self._stats_alive:,}",
            ]:
                txt = font.render(line, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 16

            # Spark-stat rows: label on one line, sparkline below it
            spark_w = panel_w - 16
            spark_h = 14
            spark_x = panel_x + 8
            for label, spark_data, color in [
                (f"Free:       {self._stats_free:,}",     self._spark_free,   (80, 180, 255)),
                (f"Composites: {self._stats_n_comp:,}",   self._spark_comp,   (120, 220, 120)),
                (f"Unique:     {self._stats_n_unique:,}", self._spark_unique, (200, 140, 220)),
                (f"Energy:     {self._stats_energy:.0f}", self._spark_energy, (220, 180, 80)),
            ]:
                txt = font.render(label, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 15
                self._draw_sparkline(surface, spark_data, spark_x, y_off, spark_w, spark_h, color)
                y_off += 18

            # Spark-stat rows for fusions and decays
            y_off += 4
            for label, spark_data, color in [
                (f"Fusions: {self._fusion_total:,} ({self._fusion_rate:.1f}/s)", self._spark_fusion, (220, 170, 60)),
                (f"Decays:  {self._decay_total:,}  ({self._decay_rate:.1f}/s)",  self._spark_decay,  (80, 200, 220)),
            ]:
                txt = font.render(label, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 15
                self._draw_sparkline(surface, spark_data, spark_x, y_off, spark_w, spark_h, color)
                y_off += 18

            # Histogram — vertical bar chart
            y_off += 4
            txt = font.render("Composite sizes:", True, (160, 185, 230))
            surface.blit(txt, (panel_x + 6, y_off))
            y_off += 18

            chart_w = panel_w - 16
            chart_h = 64
            chart_x = panel_x + 8
            chart_y = y_off
            # X-axis auto-zooms to the largest live composite so the chart
            # always uses the full chart_w (rather than reserving space for a
            # max-size composite that may never form again after a reroll).
            # Bounded below by a 2-size floor (so empty/early states aren't
            # a single bar) and above by config.max_composite_size (the
            # absolute physical cap). When size_max is large, bins are
            # *widened* (bin_width > 1) so we never exceed MAX_BINS_HIST
            # bars — 1px bars + 1px gaps always fit inside chart_w.
            MAX_BINS_HIST = 100
            # hist[i] = count of composites with member_count == i+1, so the
            # largest live size = (highest non-zero index) + 1.
            nz = np.flatnonzero(self._stats_hist) if len(self._stats_hist) else np.array([], dtype=np.int32)
            largest_live = int(nz.max()) + 1 if len(nz) > 0 else 2
            size_max  = max(2, min(largest_live, config.max_composite_size))
            bin_width = max(1, -(-size_max // MAX_BINS_HIST))   # ceil(size_max / MAX_BINS_HIST)
            n_bins    = -(-size_max // bin_width)               # ceil(size_max / bin_width)
            bar_w     = max(1, (chart_w - n_bins) // max(1, n_bins))

            # Aggregate hist[i] (= count of composites with member_count == i+1)
            # into n_bins of bin_width consecutive sizes. Right-pad with zeros
            # so the reshape divides evenly without shifting counts.
            padded  = np.zeros(n_bins * bin_width, dtype=np.int64)
            src_len = min(len(self._stats_hist), padded.size)
            padded[:src_len] = self._stats_hist[:src_len]
            binned    = padded.reshape(n_bins, bin_width).sum(axis=1)
            max_count = max(1, int(binned.max()))

            for b in range(n_bins):
                count = int(binned[b])
                if count == 0:
                    continue
                bh = max(1, int(chart_h * count / max_count))
                bx = chart_x + b * (bar_w + 1)
                pygame.draw.rect(surface, (60, 140, 220, 200),
                                 pygame.Rect(bx, chart_y + chart_h - bh, bar_w, bh))

            pygame.draw.line(surface, (80, 110, 160),
                             (chart_x, chart_y + chart_h),
                             (chart_x + chart_w, chart_y + chart_h), 1)
            # Ticks label the upper-edge size of selected bins. Bin b spans
            # sizes [b*bin_width + 1, (b+1)*bin_width]; ticks are evenly
            # spaced along the size axis so the rightmost tick reads size_max.
            raw_ticks = np.linspace(bin_width, size_max, 5).astype(int)
            seen = set()
            for tick in raw_ticks:
                tick = int(tick)
                if tick in seen:
                    continue
                seen.add(tick)
                b = (tick - 1) // bin_width
                if 0 <= b < n_bins:
                    tx = chart_x + b * (bar_w + 1)
                    lbl = font.render(str(tick), True, (120, 150, 190))
                    surface.blit(lbl, (tx - lbl.get_width() // 2, chart_y + chart_h + 2))
            y_off = chart_y + chart_h + 20

            # Composite size metrics (if profiling enabled)
            if self.metrics is not None:
                max_comp_size = self.metrics.max_composite_size_observed
                num_samples = len(self.metrics.composite_size_samples)

                y_off += 4
                txt = font.render(f"Max composite: {max_comp_size} members", True, (160, 185, 230))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 16

                if num_samples > 0:
                    recent_samples = self.metrics.composite_size_samples[-10:]  # Last 10 samples
                    recent_max_sizes = [s[1] for s in recent_samples]
                    avg_recent = sum(recent_max_sizes) / len(recent_max_sizes)
                    txt = font.render(f"Recent avg max: {avg_recent:.1f}", True, (160, 185, 230))
                    surface.blit(txt, (panel_x + 6, y_off))
                    y_off += 16

        # ── Params panel ──────────────────────────────────────────────────────
        if self._show_params:
            btn_w, btn_h, gap = 108, 26, 4
            panel_x = 8 + btn_w + 8   # right of the button strip
            slider_start_y = 8 + self._n_buttons * (btn_h + gap) + 8
            panel_w = 244   # track(200) + gap(4) + reset-btn(14) + margins
            panel_h = self._slider_content_h + 10 + 26
            panel_rect = pygame.Rect(panel_x - 4, slider_start_y - 30, panel_w, panel_h)
            pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)
            # Reset Params button
            reset_rect = self._params_reset_rect
            pygame.draw.rect(surface, (80, 30, 30, 200), reset_rect, border_radius=4)
            pygame.draw.rect(surface, (150, 80, 80, 180), reset_rect, 1, border_radius=4)
            reset_txt = font.render("Reset Params", True, (255, 160, 160))
            surface.blit(reset_txt, (reset_rect.centerx - reset_txt.get_width() // 2,
                                     reset_rect.centery - reset_txt.get_height() // 2))
            for slider in self._sliders:
                slider.draw(surface, font)

        # ── Bottom key hint ───────────────────────────────────────────────────
        hint = "[Space] pause  [+/-] speed  [B] viz  [R] reset  [Q] quit"
        hint_surf = font.render(hint, True, (120, 140, 160))
        surface.blit(hint_surf,
                     (self.config.window_width // 2 - hint_surf.get_width() // 2,
                      self.config.window_height - hint_surf.get_height() - 4))

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
        self.ctx.release()
        pygame.quit()
